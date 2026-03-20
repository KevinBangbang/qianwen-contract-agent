"""
条款提取工具

接收合同全文，调用千问模型进行结构化信息提取。
提取的条款包括：合同主体、标的、金额、期限、违约责任、争议解决等。

这是一个典型的"LLM驱动的工具"：工具本身的逻辑很简单，
核心能力来自prompt engineering，让大模型理解合同并提取结构化数据。
"""

import json
import logging

from qwen_agent.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)

from config.prompts import CLAUSE_EXTRACTION_PROMPT


@register_tool('clause_extractor')
class ClauseExtractor(BaseTool):
    description = '从合同全文中提取关键条款，包括合同主体、标的、金额、期限、违约责任、争议解决等，以JSON格式返回'
    parameters = [{
        'name': 'contract_text',
        'type': 'string',
        'description': '合同全文文本内容',
        'required': True,
    }]

    def call(self, params: str, **kwargs) -> str:
        """
        调用千问模型从合同文本中提取关键条款

        工作原理：
        1. 将合同文本填入精心设计的提示词模板
        2. 发送给千问模型
        3. 模型返回JSON格式的条款信息
        4. 验证并返回结果
        """
        params = self._verify_json_format_args(params)
        contract_text = params['contract_text']

        if not contract_text or not contract_text.strip():
            return json.dumps(
                {'success': False, 'error': '合同文本为空'},
                ensure_ascii=False
            )

        try:
            from config.model_config import get_openai_client, get_model_config
            client = get_openai_client()
            config = get_model_config()

            # 使用预定义的提取prompt模板
            prompt = CLAUSE_EXTRACTION_PROMPT.format(
                contract_text=contract_text
            )

            response = client.chat.completions.create(
                model=config['model'],
                messages=[
                    {
                        'role': 'system',
                        'content': '你是一位专业的合同分析助手。'
                                   '请严格按照要求提取合同条款，以JSON格式返回。'
                                   '如果某个字段在合同中未找到，填写"未提及"。'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.1,  # 低温度确保输出稳定，提取任务不需要创造性
                max_tokens=2000,
                # 要求返回JSON格式
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content

            # 尝试解析模型返回的JSON
            try:
                clauses = json.loads(result_text)
            except json.JSONDecodeError:
                # 如果模型返回的不是有效JSON，包装成结果返回
                clauses = {'raw_extraction': result_text}

            return json.dumps({
                'success': True,
                'clauses': clauses
            }, ensure_ascii=False)

        except Exception as e:
            logger.error("条款提取失败: %s", e)
            return json.dumps(
                {'success': False, 'error': f'条款提取失败: {str(e)}'},
                ensure_ascii=False
            )
