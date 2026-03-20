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
    """用review_agent审查合同文本。"""
    messages = [{
        'role': 'user',
        'content': f'请对以下合同进行全面审查，'
                   f'提取关键条款，检查风险点，并生成审查报告。\n\n'
                   f'{contract_text}'
    }]
    yield from _run_agent(messages)


def _run_review_with_prompt(prompt: str) -> Generator:
    """用自定义prompt运行review_agent。"""
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
