"""
RAG评估框架

基于RAGAS思想实现的检索增强生成（RAG）评估工具，
从检索质量和生成质量两个维度评估合同审查系统的RAG管线。

评估指标：
1. Context Precision - 相关文档是否排在靠前的位置
2. Context Recall - 是否检索到了所有必需的文档
3. Faithfulness - LLM回答是否忠实于检索到的上下文（需要--full）
4. Answer Relevancy - LLM回答是否与问题相关（需要--full）
5. MRR (Mean Reciprocal Rank) - 第一个相关文档的排名倒数的均值

运行方式：
    python deploy/rag_eval.py          # 仅检索指标（无需LLM调用）
    python deploy/rag_eval.py --full   # 包含LLM-as-judge指标
    python deploy/rag_eval.py --top_k 5  # 自定义检索数量
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

# 确保项目根目录在导入路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import get_openai_client, get_model_config
from knowledge.build_kb import search_similar, VECTOR_STORE_PATH

logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, 'tests', 'rag_golden_dataset.json')


def load_golden_dataset(path: str = GOLDEN_DATASET_PATH) -> list[dict]:
    """加载评估用的黄金标准数据集。

    Args:
        path: 数据集JSON文件路径。

    Returns:
        包含question、expected_answer、relevant_doc_sources的字典列表。

    Raises:
        FileNotFoundError: 数据集文件不存在时抛出。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"黄金标准数据集不存在: {path}\n"
            f"请先创建 tests/rag_golden_dataset.json"
        )

    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # 支持两种格式：直接列表 或 带 dataset 字段的包装结构
    if isinstance(raw, dict) and 'dataset' in raw:
        dataset = raw['dataset']
    elif isinstance(raw, list):
        dataset = raw
    else:
        raise ValueError(f"数据集格式不正确，期望列表或含 'dataset' 字段的字典")

    logger.info("已加载黄金标准数据集: %d 条评估样本", len(dataset))
    return dataset


def check_vector_store_exists() -> bool:
    """检查向量知识库是否已构建。

    Returns:
        True表示知识库文件存在，False表示不存在。
    """
    store_path = os.path.join(VECTOR_STORE_PATH, 'legal_kb.json')
    return os.path.exists(store_path)


def retrieve_for_query(question: str, top_k: int = 5) -> list[dict]:
    """对单个问题执行检索，返回检索结果。

    Args:
        question: 用户查询问题。
        top_k: 返回的最相似文档块数量。

    Returns:
        检索结果列表，每个结果包含text、source、similarity字段。
    """
    return search_similar(question, top_k=top_k)


def calc_context_precision(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """计算上下文精度。

    衡量检索结果中相关文档是否排在靠前位置。使用加权精度：
    对每个位置k，如果该位置的文档是相关的，则贡献 precision@k / 相关文档总数。

    Args:
        retrieved_sources: 检索结果的来源文件列表（按相似度降序排列）。
        relevant_sources: 黄金标准中标注的相关文件列表。

    Returns:
        上下文精度得分，范围[0, 1]。
    """
    if not relevant_sources or not retrieved_sources:
        return 0.0

    relevant_set = set(relevant_sources)
    hits = 0
    precision_sum = 0.0

    for k, source in enumerate(retrieved_sources, start=1):
        if source in relevant_set:
            hits += 1
            precision_at_k = hits / k
            precision_sum += precision_at_k

    # 除以相关文档数量进行归一化
    num_relevant_in_results = min(len(relevant_set), len(retrieved_sources))
    if num_relevant_in_results == 0:
        return 0.0

    return precision_sum / len(relevant_set)


def calc_context_recall(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """计算上下文召回率。

    衡量所有标注的相关文档是否都被检索到了。

    Args:
        retrieved_sources: 检索结果的来源文件列表。
        relevant_sources: 黄金标准中标注的相关文件列表。

    Returns:
        召回率得分，范围[0, 1]。
    """
    if not relevant_sources:
        return 1.0

    relevant_set = set(relevant_sources)
    retrieved_set = set(retrieved_sources)

    recalled = relevant_set & retrieved_set
    return len(recalled) / len(relevant_set)


def calc_reciprocal_rank(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """计算倒数排名（Reciprocal Rank）。

    找到第一个相关文档的排名位置，返回其倒数。

    Args:
        retrieved_sources: 检索结果的来源文件列表（按相似度降序排列）。
        relevant_sources: 黄金标准中标注的相关文件列表。

    Returns:
        倒数排名得分。如果没有检索到相关文档，返回0.0。
    """
    relevant_set = set(relevant_sources)

    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant_set:
            return 1.0 / rank

    return 0.0


def judge_faithfulness(
    question: str,
    answer: str,
    context_texts: list[str],
    client: Any,
    model: str,
) -> float:
    """使用LLM-as-judge评估回答的忠实度。

    评估LLM的回答是否忠实地基于检索到的上下文，而非编造信息。

    Args:
        question: 用户提出的问题。
        answer: LLM生成的回答。
        context_texts: 检索到的上下文文本列表。
        client: OpenAI兼容客户端实例。
        model: 使用的模型名称。

    Returns:
        忠实度得分，范围[0, 1]。
    """
    context_str = "\n---\n".join(context_texts)

    prompt = f"""你是一个严格的评估专家。请评估以下"回答"是否忠实地基于提供的"上下文"内容。

忠实度的含义：回答中的每一个事实性陈述都能在上下文中找到依据，没有编造信息。

【问题】
{question}

【上下文】
{context_str}

【回答】
{answer}

请按照以下标准打分（只回复一个0到1之间的数字，不要其他内容）：
- 1.0: 回答完全忠实于上下文，所有陈述都有依据
- 0.7-0.9: 回答大部分忠实，少量推理性延伸但合理
- 0.4-0.6: 回答部分基于上下文，但有一些无法从上下文推导的内容
- 0.1-0.3: 回答大部分内容缺乏上下文依据
- 0.0: 回答与上下文完全无关或完全编造

评分:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError, IndexError) as e:
        logger.warning("忠实度评估解析失败: %s, 原始返回: %s", e, score_text if 'score_text' in dir() else 'N/A')
        return 0.0
    except Exception as e:
        logger.warning("忠实度评估API调用失败: %s", e)
        return 0.0


def judge_answer_relevancy(
    question: str,
    answer: str,
    client: Any,
    model: str,
) -> float:
    """使用LLM-as-judge评估回答的相关性。

    评估LLM的回答是否与用户问题相关、是否回答了用户的问题。

    Args:
        question: 用户提出的问题。
        answer: LLM生成的回答。
        client: OpenAI兼容客户端实例。
        model: 使用的模型名称。

    Returns:
        相关性得分，范围[0, 1]。
    """
    prompt = f"""你是一个严格的评估专家。请评估以下"回答"与"问题"的相关性。

相关性的含义：回答是否直接针对问题进行了回答，信息是否有用。

【问题】
{question}

【回答】
{answer}

请按照以下标准打分（只回复一个0到1之间的数字，不要其他内容）：
- 1.0: 回答完全针对问题，信息完整且有用
- 0.7-0.9: 回答基本针对问题，但可能遗漏了部分要点
- 0.4-0.6: 回答部分相关，但有较多偏题内容
- 0.1-0.3: 回答与问题关系不大
- 0.0: 回答完全偏题或无意义

评分:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError, IndexError) as e:
        logger.warning("相关性评估解析失败: %s", e)
        return 0.0
    except Exception as e:
        logger.warning("相关性评估API调用失败: %s", e)
        return 0.0


def generate_answer(
    question: str,
    context_texts: list[str],
    client: Any,
    model: str,
) -> str:
    """基于检索上下文生成回答。

    模拟RAG管线中的生成步骤，用于后续评估。

    Args:
        question: 用户提出的问题。
        context_texts: 检索到的上下文文本列表。
        client: OpenAI兼容客户端实例。
        model: 使用的模型名称。

    Returns:
        LLM生成的回答文本。
    """
    context_str = "\n---\n".join(context_texts)

    prompt = f"""你是一个专业的合同法律顾问。请根据以下参考资料回答用户问题。
只根据提供的资料回答，不要添加资料中没有的信息。

【参考资料】
{context_str}

【问题】
{question}

请给出简洁专业的回答:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("生成回答失败: %s", e)
        return ""


def run_retrieval_eval(
    dataset: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """运行检索阶段的评估（不需要LLM调用）。

    对每个评估样本执行检索，计算精度、召回率和倒数排名。

    Args:
        dataset: 黄金标准数据集。
        top_k: 检索返回的文档块数量。

    Returns:
        每个样本的评估结果列表。
    """
    results = []

    for i, sample in enumerate(dataset):
        question = sample['question']
        relevant_sources = sample['relevant_doc_sources']

        logger.info("[%d/%d] 检索评估: %s", i + 1, len(dataset), question[:40])

        # 执行检索
        retrieved = retrieve_for_query(question, top_k=top_k)
        retrieved_sources = [r['source'] for r in retrieved]

        # 计算检索指标
        precision = calc_context_precision(retrieved_sources, relevant_sources)
        recall = calc_context_recall(retrieved_sources, relevant_sources)
        rr = calc_reciprocal_rank(retrieved_sources, relevant_sources)

        result = {
            'question': question,
            'relevant_sources': relevant_sources,
            'retrieved_sources': retrieved_sources,
            'retrieved_texts': [r['text'] for r in retrieved],
            'similarities': [r['similarity'] for r in retrieved],
            'context_precision': precision,
            'context_recall': recall,
            'reciprocal_rank': rr,
        }
        results.append(result)

        logger.info("  检索到: %s", retrieved_sources)
        logger.info("  精度=%.3f  召回=%.3f  RR=%.3f", precision, recall, rr)

    return results


def run_llm_eval(
    dataset: list[dict],
    retrieval_results: list[dict],
) -> list[dict]:
    """运行LLM-as-judge评估（需要LLM调用）。

    对每个样本生成回答，然后用LLM评估忠实度和相关性。

    Args:
        dataset: 黄金标准数据集。
        retrieval_results: 检索阶段的评估结果。

    Returns:
        补充了faithfulness和answer_relevancy字段的结果列表。
    """
    client = get_openai_client()
    config = get_model_config()
    model = config['model']

    logger.info("使用模型 %s 进行LLM-as-judge评估", model)

    for i, (sample, result) in enumerate(zip(dataset, retrieval_results)):
        question = sample['question']
        context_texts = result['retrieved_texts']

        logger.info("[%d/%d] LLM评估: %s", i + 1, len(dataset), question[:40])

        # 生成回答
        answer = generate_answer(question, context_texts, client, model)
        result['generated_answer'] = answer

        if not answer:
            result['faithfulness'] = 0.0
            result['answer_relevancy'] = 0.0
            logger.warning("  生成回答为空，跳过LLM评估")
            continue

        logger.info("  生成回答: %s...", answer[:60])

        # 评估忠实度
        faithfulness = judge_faithfulness(question, answer, context_texts, client, model)
        result['faithfulness'] = faithfulness

        # 评估相关性
        relevancy = judge_answer_relevancy(question, answer, client, model)
        result['answer_relevancy'] = relevancy

        logger.info("  忠实度=%.3f  相关性=%.3f", faithfulness, relevancy)

    return retrieval_results


def print_summary_table(results: list[dict], full_eval: bool = False) -> None:
    """打印评估结果汇总表。

    Args:
        results: 评估结果列表。
        full_eval: 是否包含LLM评估指标。
    """
    import numpy as np

    # 计算各指标均值
    precisions = [r['context_precision'] for r in results]
    recalls = [r['context_recall'] for r in results]
    rrs = [r['reciprocal_rank'] for r in results]

    avg_precision = float(np.mean(precisions))
    avg_recall = float(np.mean(recalls))
    mrr = float(np.mean(rrs))

    # 打印逐条结果
    print("\n" + "=" * 90)
    print("RAG 评估结果 - 逐条详情")
    print("=" * 90)

    if full_eval:
        header = (f"{'#':<4} {'问题':<28} {'精度':<8} {'召回':<8} "
                  f"{'RR':<8} {'忠实度':<8} {'相关性':<8}")
    else:
        header = f"{'#':<4} {'问题':<28} {'精度':<8} {'召回':<8} {'RR':<8}"

    print(header)
    print("-" * 90)

    for i, r in enumerate(results):
        question_short = r['question'][:26]
        if len(r['question']) > 26:
            question_short += '..'

        if full_eval:
            faith = r.get('faithfulness', 0.0)
            relev = r.get('answer_relevancy', 0.0)
            row = (f"{i+1:<4} {question_short:<28} {r['context_precision']:<8.3f} "
                   f"{r['context_recall']:<8.3f} {r['reciprocal_rank']:<8.3f} "
                   f"{faith:<8.3f} {relev:<8.3f}")
        else:
            row = (f"{i+1:<4} {question_short:<28} {r['context_precision']:<8.3f} "
                   f"{r['context_recall']:<8.3f} {r['reciprocal_rank']:<8.3f}")

        print(row)

    # 打印汇总
    print("\n" + "=" * 90)
    print("RAG 评估结果 - 汇总")
    print("=" * 90)

    print(f"\n{'指标':<30} {'得分':<10} {'说明'}")
    print("-" * 70)
    print(f"{'Context Precision (上下文精度)':<30} {avg_precision:<10.4f} 相关文档是否排在前列")
    print(f"{'Context Recall (上下文召回)':<30} {avg_recall:<10.4f} 是否检索到所有必需文档")
    print(f"{'MRR (平均倒数排名)':<30} {mrr:<10.4f} 首个相关文档的排名质量")

    if full_eval:
        faiths = [r.get('faithfulness', 0.0) for r in results]
        relevs = [r.get('answer_relevancy', 0.0) for r in results]
        avg_faith = float(np.mean(faiths))
        avg_relev = float(np.mean(relevs))

        print(f"{'Faithfulness (忠实度)':<30} {avg_faith:<10.4f} 回答是否忠实于上下文")
        print(f"{'Answer Relevancy (回答相关性)':<30} {avg_relev:<10.4f} 回答是否针对问题")

    print("-" * 70)

    # 综合评分
    scores = [avg_precision, avg_recall, mrr]
    if full_eval:
        scores.extend([avg_faith, avg_relev])
    overall = float(np.mean(scores))
    print(f"{'综合得分':<30} {overall:<10.4f}")

    # 质量判定
    print()
    if overall >= 0.8:
        print("评定: 优秀 - RAG管线质量良好")
    elif overall >= 0.6:
        print("评定: 良好 - RAG管线基本可用，部分指标可优化")
    elif overall >= 0.4:
        print("评定: 一般 - RAG管线存在明显不足，建议调优")
    else:
        print("评定: 较差 - RAG管线需要重大改进")

    print(f"\n评估样本数: {len(results)}")
    print("=" * 90)


def save_eval_report(results: list[dict], full_eval: bool = False) -> str:
    """将评估结果保存为JSON报告。

    Args:
        results: 评估结果列表。
        full_eval: 是否包含LLM评估指标。

    Returns:
        报告文件保存路径。
    """
    import numpy as np

    report = {
        'summary': {
            'num_samples': len(results),
            'context_precision': float(np.mean([r['context_precision'] for r in results])),
            'context_recall': float(np.mean([r['context_recall'] for r in results])),
            'mrr': float(np.mean([r['reciprocal_rank'] for r in results])),
        },
        'details': [],
    }

    if full_eval:
        report['summary']['faithfulness'] = float(
            np.mean([r.get('faithfulness', 0.0) for r in results])
        )
        report['summary']['answer_relevancy'] = float(
            np.mean([r.get('answer_relevancy', 0.0) for r in results])
        )

    for r in results:
        detail = {
            'question': r['question'],
            'relevant_sources': r['relevant_sources'],
            'retrieved_sources': r['retrieved_sources'],
            'context_precision': r['context_precision'],
            'context_recall': r['context_recall'],
            'reciprocal_rank': r['reciprocal_rank'],
        }
        if full_eval:
            detail['faithfulness'] = r.get('faithfulness', 0.0)
            detail['answer_relevancy'] = r.get('answer_relevancy', 0.0)
            detail['generated_answer'] = r.get('generated_answer', '')
        report['details'].append(detail)

    report_dir = os.path.join(PROJECT_ROOT, 'docs')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'rag_eval_report.json')

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("评估报告已保存到: %s", report_path)
    return report_path


def main() -> None:
    """RAG评估主流程。"""
    parser = argparse.ArgumentParser(description='RAG管线评估工具')
    parser.add_argument(
        '--full', action='store_true',
        help='运行完整评估，包含LLM-as-judge指标（faithfulness、answer_relevancy）',
    )
    parser.add_argument(
        '--top_k', type=int, default=5,
        help='检索返回的文档块数量（默认: 5）',
    )
    parser.add_argument(
        '--dataset', type=str, default=GOLDEN_DATASET_PATH,
        help='黄金标准数据集路径',
    )
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    )

    logger.info("=" * 50)
    logger.info("RAG 管线评估")
    logger.info("=" * 50)
    logger.info("模式: %s", "完整评估（含LLM-as-judge）" if args.full else "仅检索指标")
    logger.info("Top-K: %d", args.top_k)

    # 前置检查：向量知识库是否存在
    if not check_vector_store_exists():
        print("\n[错误] 向量知识库尚未构建！")
        print("请先运行以下命令构建知识库：")
        print("  python knowledge/build_kb.py")
        print()
        sys.exit(1)

    # 加载数据集
    try:
        dataset = load_golden_dataset(args.dataset)
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        sys.exit(1)

    # 阶段1: 检索评估
    logger.info("阶段1: 检索质量评估")
    retrieval_results = run_retrieval_eval(dataset, top_k=args.top_k)

    # 阶段2: LLM评估（可选）
    if args.full:
        logger.info("阶段2: LLM-as-judge 评估")
        retrieval_results = run_llm_eval(dataset, retrieval_results)

    # 输出结果
    print_summary_table(retrieval_results, full_eval=args.full)

    # 保存报告
    report_path = save_eval_report(retrieval_results, full_eval=args.full)
    print(f"\n详细报告已保存到: {report_path}")


if __name__ == '__main__':
    main()
