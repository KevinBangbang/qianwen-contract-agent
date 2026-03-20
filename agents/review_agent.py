"""
主审查Agent

基于Qwen-Agent的Assistant类构建，注册所有5个自定义工具。
使用ReAct范式进行多步推理，自主决定工具调用顺序：
  解析合同 -> 提取条款 -> 风险检查 -> 计算验证 -> 生成报告

ReAct = Reasoning + Acting 循环
模型先"思考"该做什么，然后选择工具执行，观察结果后继续思考。
"""

import logging
import os
import sys

# 确保项目根目录在导入路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

from qwen_agent.agents import Assistant

from config.model_config import get_model_config
from config.prompts import REVIEW_AGENT_SYSTEM_PROMPT

# 导入工具模块，触发 @register_tool 注册
import tools.contract_parser  # noqa: F401
import tools.clause_extractor  # noqa: F401
import tools.risk_checker  # noqa: F401
import tools.amount_calculator  # noqa: F401
import tools.report_generator  # noqa: F401


def create_review_agent() -> Assistant:
    """
    创建合同审查Agent

    Agent的配置核心：
    1. llm: 指定使用的模型和调用方式
    2. function_list: 注册可用的工具列表
    3. system_message: 系统提示词，决定Agent的行为模式
    """
    config = get_model_config()

    # 模型配置
    # 使用OpenAI兼容接口调用DashScope
    llm_cfg = {
        'model': config['model'],
        'model_server': config['base_url'],
        'api_key': config['api_key'],
    }

    # 注册的工具列表
    # Agent会根据任务需要自主选择调用哪些工具
    tool_list = [
        'contract_parser',      # 合同解析
        'clause_extractor',     # 条款提取
        'risk_checker',         # 风险检查
        'amount_calculator',    # 金额计算
        'report_generator',     # 报告生成
    ]

    # 创建Agent实例
    agent = Assistant(
        llm=llm_cfg,
        function_list=tool_list,
        name='合同审查助手',
        description='专业的智能合同审查Agent，支持PDF和扫描件，'
                    '自动提取条款、检查风险、生成审查报告',
        system_message=REVIEW_AGENT_SYSTEM_PROMPT,
    )

    return agent


def run_review(contract_text: str) -> list:
    """
    执行合同审查

    接收合同文本，运行审查Agent，返回审查结果。
    Agent会自主执行完整的审查流程。
    """
    agent = create_review_agent()

    messages = [{
        'role': 'user',
        'content': f'请对以下合同进行全面审查，'
                   f'提取关键条款，检查风险点，并生成审查报告。\n\n'
                   f'{contract_text}'
    }]

    # agent.run() 返回一个生成器，逐步产出Agent的推理过程
    response = []
    for chunk in agent.run(messages=messages):
        response = chunk

    return response


if __name__ == '__main__':
    # 快速测试：用样本合同运行审查
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'tests', 'sample_contracts', 'sample_contract.txt'
    )

    if os.path.exists(sample_path):
        with open(sample_path, 'r', encoding='utf-8') as f:
            contract_text = f.read()

        logger.info("开始合同审查...")
        result = run_review(contract_text)

        # 打印Agent的最终回复
        for msg in result:
            if msg.get('role') == 'assistant' and msg.get('content'):
                logger.info(msg['content'])
    else:
        logger.error(f"样本合同文件不存在: {sample_path}")
