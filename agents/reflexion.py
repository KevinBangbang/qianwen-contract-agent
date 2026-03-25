"""
Reflexion 自我反思模块

在 ReviewAgent 审查完成后，增加自我评估和反思环节：
1. Evaluator: 评估审查报告的质量（完整性、准确性、可操作性）
2. Self-Reflection: 分析不足之处，生成经验总结
3. 经验累积: 将经验注入下次审查的上下文，避免重复犯错

Reflexion vs 普通重试:
- 普通重试只是换个 random seed 再来一次
- Reflexion 会分析失败原因并将经验带到下次尝试
- 经验跨 episode 累积，越用越好

参考: MathGraph/50-AI-Agent/Agent-Architecture/Reflexion-Pattern.md
"""

import json
import logging
import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import get_openai_client, get_model_config
from config.schemas import validate_quality_evaluation

logger = logging.getLogger(__name__)

# 经验存储路径
EXPERIENCE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'knowledge', 'reflexion_experiences.json'
)

# 评估通过阈值
QUALITY_THRESHOLD = 7.0
MAX_TRIALS = 2  # 最多反思重试次数（含首次）


def evaluate_review_quality(
    contract_text: str,
    review_result: str,
) -> dict[str, Any]:
    """评估审查报告的质量。

    从 5 个维度评估审查报告，每个维度 0-10 分：
    1. 完整性: 是否覆盖了所有关键条款
    2. 风险识别: 是否找到了真实存在的风险点
    3. 法律依据: 建议是否有法律依据支撑
    4. 可操作性: 修改建议是否具体可执行
    5. 结构清晰: 报告格式是否专业清晰

    Args:
        contract_text: 原始合同文本。
        review_result: Agent 生成的审查报告。

    Returns:
        评估结果字典，包含各维度分数和综合评分。
    """
    client = get_openai_client()
    config = get_model_config()

    prompt = f"""你是一位资深合同审查质量评估专家。请评估以下合同审查报告的质量。

【原始合同】
{contract_text[:2000]}

【审查报告】
{review_result[:3000]}

请从以下 5 个维度评估，每个维度打 0-10 分，并给出简短理由。

以 JSON 格式返回：
{{
    "completeness": {{"score": 0-10, "reason": "..."}},
    "risk_identification": {{"score": 0-10, "reason": "..."}},
    "legal_basis": {{"score": 0-10, "reason": "..."}},
    "actionability": {{"score": 0-10, "reason": "..."}},
    "clarity": {{"score": 0-10, "reason": "..."}},
    "overall_score": 0-10,
    "passed": true/false,
    "major_issues": ["问题1", "问题2"]
}}

评分标准：
- 8-10: 优秀，专业水准
- 6-7: 合格，但有改进空间
- 4-5: 一般，存在明显不足
- 0-3: 较差，需要重新审查
"""

    try:
        response = client.chat.completions.create(
            model=config['model'],
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        # 确保有 overall_score
        if 'overall_score' not in result:
            dimensions = ['completeness', 'risk_identification',
                          'legal_basis', 'actionability', 'clarity']
            scores = [result.get(d, {}).get('score', 5) for d in dimensions]
            result['overall_score'] = sum(scores) / len(scores)

        # 判断是否通过
        result['passed'] = result['overall_score'] >= QUALITY_THRESHOLD

        # Structured Output 验证（Pydantic Schema 校验）
        valid, parsed, err = validate_quality_evaluation(result)
        if valid:
            logger.info("质量评估 Schema 验证通过")
        else:
            logger.warning("质量评估 Schema 验证失败: %s（不影响流程）", err)

        logger.info("审查质量评估: %.1f/10 (%s)",
                     result['overall_score'],
                     "通过" if result['passed'] else "未通过")

        return result

    except Exception as e:
        logger.warning("质量评估失败: %s", e)
        return {
            'overall_score': 6.0,
            'passed': True,
            'major_issues': [],
            'error': str(e),
        }


def generate_reflection(
    contract_text: str,
    review_result: str,
    evaluation: dict,
) -> str:
    """根据评估结果生成反思经验。

    分析审查报告的不足之处，生成具体的改进建议。
    这些经验会被注入下次审查的上下文中。

    Args:
        contract_text: 原始合同文本。
        review_result: Agent 生成的审查报告。
        evaluation: evaluate_review_quality 的输出。

    Returns:
        反思经验文本。
    """
    client = get_openai_client()
    config = get_model_config()

    issues = evaluation.get('major_issues', [])
    issues_text = '\n'.join(f"- {issue}" for issue in issues) if issues else "无具体问题"

    prompt = f"""你是合同审查质量改进顾问。以下审查报告的评估得分为 {evaluation.get('overall_score', 0)}/10，
存在以下主要问题：
{issues_text}

请生成一段简洁的改进经验（2-3 句话），说明：
1. 上次审查具体做错了什么或遗漏了什么
2. 下次审查应该怎么做来避免这些问题

要求具体可操作，不要笼统的建议如"更仔细"。
"""

    try:
        response = client.chat.completions.create(
            model=config['model'],
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        reflection = response.choices[0].message.content.strip()
        logger.info("反思经验: %s", reflection[:100])
        return reflection

    except Exception as e:
        logger.warning("生成反思失败: %s", e)
        return ""


def load_experiences() -> list[str]:
    """加载历史反思经验。

    Returns:
        经验文本列表，最新的在最后。
    """
    if not os.path.exists(EXPERIENCE_PATH):
        return []

    try:
        with open(EXPERIENCE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('experiences', [])
    except (json.JSONDecodeError, KeyError):
        return []


def save_experience(experience: str) -> None:
    """保存新的反思经验。

    最多保留最近 10 条经验，防止上下文膨胀。

    Args:
        experience: 新的反思经验文本。
    """
    MAX_EXPERIENCES = 10

    experiences = load_experiences()
    experiences.append(experience)

    # 保留最近的 N 条
    if len(experiences) > MAX_EXPERIENCES:
        experiences = experiences[-MAX_EXPERIENCES:]

    os.makedirs(os.path.dirname(EXPERIENCE_PATH), exist_ok=True)
    with open(EXPERIENCE_PATH, 'w', encoding='utf-8') as f:
        json.dump({'experiences': experiences}, f, ensure_ascii=False, indent=2)

    logger.info("经验已保存，当前共 %d 条", len(experiences))


def get_experience_context() -> str:
    """获取历史经验文本，用于注入 Agent 上下文。

    Returns:
        格式化的经验文本。如果没有历史经验，返回空字符串。
    """
    experiences = load_experiences()
    if not experiences:
        return ""

    # 取最近 3 条经验
    recent = experiences[-3:]
    context = "以下是从过去审查中积累的经验教训，请在本次审查中注意避免：\n"
    for i, exp in enumerate(recent, 1):
        context += f"{i}. {exp}\n"
    return context
