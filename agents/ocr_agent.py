"""
多模态OCR Agent

专门处理合同扫描件的Agent。
使用Qwen-VL模型对图片进行OCR识别，输出结构化文本。
识别完成后可以将文本传递给review_agent进行审查。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logger = logging.getLogger(__name__)

from qwen_agent.agents import Assistant

from config.model_config import get_model_config
from config.prompts import OCR_AGENT_SYSTEM_PROMPT

# 导入合同解析工具（包含OCR功能）
import tools.contract_parser  # noqa: F401


def create_ocr_agent() -> Assistant:
    """
    创建OCR Agent

    使用VL模型专门处理图片输入。
    只注册contract_parser工具，专注于文档识别任务。
    """
    config = get_model_config()

    llm_cfg = {
        # OCR任务使用VL模型
        'model': config['vl_model'],
        'model_server': config['base_url'],
        'api_key': config['api_key'],
    }

    agent = Assistant(
        llm=llm_cfg,
        function_list=['contract_parser'],
        name='文档OCR助手',
        description='合同扫描件OCR识别Agent，将图片转换为结构化文本',
        system_message=OCR_AGENT_SYSTEM_PROMPT,
    )

    return agent


def run_ocr(image_path: str) -> str:
    """
    对图片执行OCR识别

    Args:
        image_path: 图片文件路径

    Returns:
        识别出的文本内容
    """
    agent = create_ocr_agent()

    messages = [{
        'role': 'user',
        'content': f'请识别以下合同图片中的文字内容，'
                   f'保持原文结构输出。文件路径: {image_path}'
    }]

    response = []
    for chunk in agent.run(messages=messages):
        response = chunk

    # 提取Agent返回的文本内容
    for msg in response:
        if msg.get('role') == 'assistant' and msg.get('content'):
            return msg['content']

    return ''


if __name__ == '__main__':
    logger.info("OCR Agent 已就绪")
    logger.info("使用方式: run_ocr('path/to/contract_image.png')")
