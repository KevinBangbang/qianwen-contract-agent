"""
Agent编排器

负责判断输入类型并路由到对应的Agent：
- PDF文本文件 -> 直接交给 review_agent 审查
- 扫描件图片 -> 先交给 ocr_agent 识别，再交给 review_agent 审查
- 纯文本 -> 直接交给 review_agent 审查

编排模式：串行路由（if-else判断输入类型，依次处理）
比起用另一个Agent做路由决策，if-else路由更简单、更可控、延迟更低。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

from agents.review_agent import create_review_agent
from agents.ocr_agent import run_ocr
from agents.reflexion import (
    evaluate_review_quality,
    generate_reflection,
    save_experience,
    get_experience_context,
    QUALITY_THRESHOLD,
    MAX_TRIALS,
)
from agents.guardrails import GuardrailChain

# 支持的图片格式
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
# 支持的文档格式
DOC_EXTENSIONS = {'.pdf', '.txt'}


def detect_input_type(file_path: str) -> str:
    """
    检测输入文件的类型

    返回值：
    - 'image': 图片文件，需要OCR处理
    - 'pdf': PDF文件，先尝试文本提取
    - 'text': 纯文本文件，直接读取
    - 'unknown': 不支持的格式
    """
    if not file_path or not os.path.exists(file_path):
        return 'unknown'

    ext = os.path.splitext(file_path)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        return 'image'
    elif ext == '.pdf':
        return 'pdf'
    elif ext == '.txt':
        return 'text'
    else:
        return 'unknown'


def process_contract(file_path: Optional[str] = None, text: Optional[str] = None) -> Generator:
    """
    处理合同输入的主入口

    编排逻辑：
    1. 如果传入了文本，直接审查
    2. 如果传入了文件路径，根据类型路由
       - 图片 -> OCR -> 审查
       - PDF/TXT -> 读取 -> 审查

    Args:
        file_path: 合同文件路径（可选）
        text: 合同文本内容（可选）

    Returns:
        生成器，逐步产出Agent的推理过程
    """
    # 情况1: 直接传入了文本
    if text:
        logger.info("[编排器] 输入类型: 纯文本，直接交给审查Agent")
        yield from _run_review(text)
        return

    # 情况2: 传入了文件路径
    if not file_path:
        yield [{'role': 'assistant', 'content': '请提供合同文件路径或文本内容'}]
        return

    input_type = detect_input_type(file_path)
    logger.info(f"[编排器] 文件: {os.path.basename(file_path)}, 类型: {input_type}")

    if input_type == 'image':
        # 图片文件：先OCR再审查
        logger.info("[编排器] 图片文件，先执行OCR识别...")
        ocr_text = run_ocr(file_path)
        if not ocr_text:
            yield [{'role': 'assistant', 'content': 'OCR识别失败，请检查图片质量'}]
            return
        logger.info(f"[编排器] OCR完成，识别文本长度: {len(ocr_text)} 字")
        logger.info("[编排器] 将识别结果交给审查Agent...")
        yield from _run_review(ocr_text)

    elif input_type == 'pdf':
        # PDF文件：让review_agent使用contract_parser工具处理
        logger.info("[编排器] PDF文件，交给审查Agent处理（Agent会自动调用解析工具）")
        prompt = (
            f'请审查以下合同文件。文件路径: {file_path}\n'
            f'请先使用contract_parser工具解析文件，然后进行全面审查。'
        )
        yield from _run_review_with_prompt(prompt)

    elif input_type == 'text':
        # 纯文本文件：读取后审查
        logger.info("[编排器] 文本文件，读取后交给审查Agent")
        with open(file_path, 'r', encoding='utf-8') as f:
            contract_text = f.read()
        yield from _run_review(contract_text)

    else:
        yield [{
            'role': 'assistant',
            'content': f'不支持的文件格式: {os.path.splitext(file_path)[1]}\n'
                       f'支持的格式: PDF, TXT, PNG, JPG, JPEG, BMP, TIFF'
        }]


def _run_agent(messages: list[dict[str, str]]) -> Generator:
    """创建review_agent并执行消息列表。"""
    agent = create_review_agent()
    yield from agent.run(messages=messages)


def _run_review(contract_text: str) -> Generator:
    """用review_agent审查合同文本（含 Guardrails + Reflexion）。

    完整流程：
    1. 输入护栏：检查合同文本合法性（长度、非空）
    2. 首次审查：注入历史经验 -> Agent 执行 -> 输出报告
    3. 成本护栏：监控 LLM / 工具调用次数和超时
    4. 质量评估：Evaluator 对报告打分（5维度）
    5. 输出护栏：检查报告结构完整性
    6. 如果不达标：Self-Reflection 生成经验 -> 注入上下文 -> 重新审查
    7. 最多重试 MAX_TRIALS 次
    """
    # 初始化护栏链
    guardrail = GuardrailChain()

    # Layer 1: 输入护栏
    ok, msg = guardrail.check_input(contract_text)
    if not ok:
        logger.warning("[护栏] 输入检查未通过: %s", msg)
        yield [{'role': 'assistant', 'content': f'⚠ 输入检查未通过: {msg}'}]
        return

    guardrail.cost.reset()

    # 获取历史经验
    experience_context = get_experience_context()

    base_prompt = (
        '请对以下合同进行全面审查，'
        '提取关键条款，检查风险点，并生成审查报告。\n\n'
    )

    if experience_context:
        base_prompt = experience_context + '\n' + base_prompt

    best_response = None

    for trial in range(MAX_TRIALS):
        logger.info("[Reflexion] 第 %d/%d 次审查", trial + 1, MAX_TRIALS)

        # 执行审查
        messages = [{
            'role': 'user',
            'content': base_prompt + contract_text
        }]

        response = []
        for chunk in _run_agent(messages):
            response = chunk
            yield chunk  # 流式输出给前端

        best_response = response

        # 提取 Agent 最终回复文本
        review_text = ''
        for msg in response:
            if msg.get('role') == 'assistant' and msg.get('content'):
                review_text += msg['content']

        if not review_text:
            logger.warning("[Reflexion] Agent 未生成有效回复，跳过评估")
            break

        # Layer 3: 输出护栏
        out_ok, out_msg, out_details = guardrail.check_output(review_text)
        if not out_ok:
            logger.warning("[护栏] 输出检查未通过: %s", out_msg)
            # 输出护栏不阻断流程，但记录警告

        # 成本护栏检查
        guardrail.cost.record_llm_call(input_tokens=len(contract_text))
        cost_ok, cost_msg = guardrail.check_cost()
        if not cost_ok:
            logger.warning("[护栏] 成本超限: %s，停止重试", cost_msg)
            break

        # 质量评估
        logger.info("[Reflexion] 评估审查质量...")
        evaluation = evaluate_review_quality(contract_text, review_text)

        if evaluation.get('passed', True):
            logger.info("[Reflexion] 审查质量通过 (%.1f/10)，无需重试",
                        evaluation.get('overall_score', 0))
            break

        # 未通过：生成反思经验
        if trial < MAX_TRIALS - 1:
            logger.info("[Reflexion] 质量未达标 (%.1f/10)，生成反思经验...",
                        evaluation.get('overall_score', 0))
            reflection = generate_reflection(contract_text, review_text, evaluation)
            if reflection:
                save_experience(reflection)
                # 将反思注入下次审查的提示
                base_prompt = (
                    f'上次审查的反思：{reflection}\n\n'
                    f'请基于以上经验改进审查。'
                    f'请对以下合同进行全面审查，'
                    f'提取关键条款，检查风险点，并生成审查报告。\n\n'
                )
        else:
            logger.info("[Reflexion] 已达最大重试次数，使用当前结果")


def _run_review_with_prompt(prompt: str) -> Generator:
    """用自定义prompt运行review_agent。"""
    # Reflexion 经验也注入到自定义 prompt 中
    experience_context = get_experience_context()
    if experience_context:
        prompt = experience_context + '\n' + prompt
    yield from _run_agent([{'role': 'user', 'content': prompt}])


if __name__ == '__main__':
    # 测试编排器：使用样本合同
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'tests', 'sample_contracts', 'sample_contract.txt'
    )

    if os.path.exists(sample_path):
        logger.info("=" * 50)
        logger.info("合同审查系统 - 编排器测试")
        logger.info("=" * 50)

        response = []
        for chunk in process_contract(file_path=sample_path):
            response = chunk

        logger.info("审查结果:")
        for msg in response:
            if msg.get('role') == 'assistant' and msg.get('content'):
                logger.info(msg['content'])
    else:
        logger.error(f"样本文件不存在: {sample_path}")
