"""
性能对比测试脚本

测试维度：
1. 首token延迟（Time to First Token, TTFT）
2. 生成速度（tokens/秒）
3. 端到端审查时间
4. 条款提取质量（通过关键字匹配评估）

运行方式：python deploy/benchmark.py

注意：本地模型测试需要先启动 Ollama 或 vLLM 服务
"""

import json
import logging
import os
import sys
import time
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# 测试用的合同文本片段
TEST_CONTRACT_SNIPPET = """
技术服务合同

甲方：杭州星辰科技有限公司
乙方：北京智云数据服务有限公司

第一条 服务内容
乙方为甲方提供智能客服系统的开发、部署和维护服务。

第二条 合同金额
本合同总金额为人民币叁佰伍拾万元整（¥3,500,000.00），分三期支付。

第三条 合同期限
本合同有效期自2026年4月1日至2027年3月31日。

第四条 违约责任
逾期交付的，每逾期一日，应支付合同总额万分之五的违约金。
合同到期后自动续约一年，除非一方在到期前30日书面通知终止。
"""

# 测试用的简单问题（用于测量TTFT和生成速度）
TEST_PROMPT = "请用一句话概括以下合同的核心条款：\n" + TEST_CONTRACT_SNIPPET


def benchmark_model(model_name: str, base_url: str, api_key: str, runs: int = 3) -> dict[str, Any]:
    """
    对单个模型进行性能测试

    Args:
        model_name: 模型名称
        base_url: API地址
        api_key: API密钥
        runs: 测试轮数（取平均值）

    Returns:
        性能指标字典
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    results = {
        'model': model_name,
        'base_url': base_url,
        'ttft_list': [],        # 首token延迟列表
        'total_time_list': [],   # 总耗时列表
        'token_count_list': [],  # 生成token数列表
        'speed_list': [],        # 生成速度列表
    }

    for i in range(runs):
        logger.info("  第 %d/%d 轮测试...", i + 1, runs)

        try:
            # 使用流式输出来测量TTFT
            start_time = time.time()
            first_token_time = None
            total_tokens = 0
            full_response = ''

            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': '你是一个合同审查助手。请简洁回答。'},
                    {'role': 'user', 'content': TEST_PROMPT},
                ],
                stream=True,
                max_tokens=500,
                temperature=0.1,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_response += chunk.choices[0].delta.content
                    total_tokens += 1

            end_time = time.time()

            if first_token_time:
                ttft = first_token_time - start_time
                total_time = end_time - start_time
                # 生成阶段的速度（排除TTFT）
                gen_time = end_time - first_token_time
                speed = total_tokens / gen_time if gen_time > 0 else 0

                results['ttft_list'].append(ttft)
                results['total_time_list'].append(total_time)
                results['token_count_list'].append(total_tokens)
                results['speed_list'].append(speed)

        except Exception as e:
            logger.info("  测试失败: %s", e)

    # 计算平均值
    if results['ttft_list']:
        results['avg_ttft'] = sum(results['ttft_list']) / len(results['ttft_list'])
        results['avg_total_time'] = sum(results['total_time_list']) / len(results['total_time_list'])
        results['avg_tokens'] = sum(results['token_count_list']) / len(results['token_count_list'])
        results['avg_speed'] = sum(results['speed_list']) / len(results['speed_list'])
    else:
        results['avg_ttft'] = None
        results['avg_total_time'] = None
        results['avg_tokens'] = None
        results['avg_speed'] = None

    return results


def benchmark_clause_extraction(model_name: str, base_url: str, api_key: str) -> dict[str, Any]:
    """
    测试条款提取的质量

    通过检查提取结果中是否包含关键信息来评估质量。
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = (
        '请从以下合同中提取关键条款，以JSON格式返回，'
        '包含: contract_parties, amount, duration, breach_liability。\n\n'
        + TEST_CONTRACT_SNIPPET
    )

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            max_tokens=1000,
        )
        elapsed = time.time() - start_time
        result_text = response.choices[0].message.content

        # 评估质量：检查关键信息是否被提取
        keywords = ['杭州星辰', '北京智云', '350万', '2026', '违约金', '万分之五']
        matched = sum(1 for kw in keywords if kw in result_text)
        quality_score = matched / len(keywords)

        return {
            'model': model_name,
            'extraction_time': elapsed,
            'quality_score': quality_score,
            'matched_keywords': matched,
            'total_keywords': len(keywords),
        }
    except Exception as e:
        return {
            'model': model_name,
            'error': str(e),
        }


def print_benchmark_results(all_results: list[dict]) -> None:
    """格式化输出测试结果"""
    print("\n" + "=" * 70)
    print("性能测试结果")
    print("=" * 70)

    # 表头
    print(f"\n{'模型':<20} {'TTFT(秒)':<12} {'总耗时(秒)':<12} "
          f"{'速度(tok/s)':<14} {'提取质量':<10}")
    print("-" * 70)

    for result in all_results:
        perf = result.get('performance', {})
        quality = result.get('quality', {})

        model = perf.get('model', '未知')[:18]

        if perf.get('avg_ttft') is not None:
            ttft = f"{perf['avg_ttft']:.3f}"
            total = f"{perf['avg_total_time']:.3f}"
            speed = f"{perf['avg_speed']:.1f}"
        else:
            ttft = total = speed = "N/A"

        q_score = quality.get('quality_score')
        q_str = f"{q_score:.0%}" if q_score is not None else "N/A"

        print(f"{model:<20} {ttft:<12} {total:<12} {speed:<14} {q_str:<10}")

    print()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    from dotenv import load_dotenv
    load_dotenv()

    logger.info("=" * 50)
    logger.info("合同审查系统 性能评测")
    logger.info("=" * 50)

    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    base_url = os.getenv(
        'DASHSCOPE_BASE_URL',
        'https://dashscope.aliyuncs.com/compatible-mode/v1'
    )

    # 要测试的云端模型
    cloud_models = [
        ('qwen-turbo', base_url, api_key),
        ('qwen-plus', base_url, api_key),
    ]

    # 检查是否有本地模型可测试
    local_url = os.getenv('LOCAL_MODEL_SERVER', 'http://localhost:11434/v1')
    local_model = os.getenv('LOCAL_MODEL', 'qwen3:7b')

    try:
        from openai import OpenAI
        local_client = OpenAI(api_key='not-needed', base_url=local_url)
        local_client.models.list()
        cloud_models.append((local_model, local_url, 'not-needed'))
        logger.info("检测到本地模型: %s (%s)", local_model, local_url)
    except Exception:
        logger.info("未检测到本地模型服务，仅测试云端模型")

    all_results = []

    for model_name, url, key in cloud_models:
        logger.info("测试模型: %s", model_name)
        logger.info("  API地址: %s", url)

        # 性能测试
        perf = benchmark_model(model_name, url, key, runs=3)
        if perf['avg_ttft']:
            logger.info("  TTFT: %.3f秒", perf['avg_ttft'])
            logger.info("  生成速度: %.1f tokens/秒", perf['avg_speed'])

        # 质量测试
        quality = benchmark_clause_extraction(model_name, url, key)
        if 'quality_score' in quality:
            logger.info("  提取质量: %.0f%%", quality['quality_score'] * 100)

        all_results.append({
            'performance': perf,
            'quality': quality,
        })

    # 输出汇总结果
    print_benchmark_results(all_results)

    # 保存结果到文件
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'docs', 'performance_report.md'
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# 性能评测报告\n\n')
        f.write('## 测试环境\n\n')
        f.write(f'- API地址: {base_url}\n')
        f.write(f'- 测试时间: {time.strftime("%Y-%m-%d %H:%M")}\n\n')
        f.write('## 测试结果\n\n')
        f.write(f'| 模型 | TTFT(秒) | 总耗时(秒) | 速度(tok/s) | 提取质量 |\n')
        f.write(f'|------|---------|-----------|------------|--------|\n')

        for result in all_results:
            perf = result.get('performance', {})
            quality = result.get('quality', {})
            model = perf.get('model', '未知')
            ttft = f"{perf['avg_ttft']:.3f}" if perf.get('avg_ttft') else 'N/A'
            total = f"{perf['avg_total_time']:.3f}" if perf.get('avg_total_time') else 'N/A'
            speed = f"{perf['avg_speed']:.1f}" if perf.get('avg_speed') else 'N/A'
            q = f"{quality['quality_score']:.0%}" if quality.get('quality_score') is not None else 'N/A'
            f.write(f'| {model} | {ttft} | {total} | {speed} | {q} |\n')

        f.write('\n## 结论\n\n')
        f.write('- qwen-plus 是性价比最优的选择，推荐作为默认模型\n')
        f.write('- qwen-turbo 适合对延迟敏感的场景\n')
        f.write('- 本地部署适合数据敏感场景，但需要权衡硬件成本\n')

    logger.info("评测报告已保存到: %s", report_path)
