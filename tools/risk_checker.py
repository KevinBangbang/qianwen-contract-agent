"""
风险检查工具

对照知识库中的法律法规和风险条款模板，检查合同条款是否存在风险。
这个工具结合了RAG检索和LLM推理：
1. 先从知识库中检索相关的法律条文（RAG检索）
2. 将检索到的法律依据和内置风险规则合并
3. 再让LLM对比合同条款和法律要求，找出风险点
"""

import json
import logging
from typing import Any

from qwen_agent.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)

from config.prompts import RISK_CHECK_PROMPT

# 内置的常见风险条款模板，作为基础检查规则
# RAG检索结果会和这些规则合并，提供更全面的法律依据
DEFAULT_RISK_RULES = [
    {
        'category': '无限连带责任',
        'description': '合同中要求一方承担无限连带担保责任，风险极高',
        'level': '高',
    },
    {
        'category': '自动续约',
        'description': '合同到期后自动续约，且未设置明确的终止条件',
        'level': '中',
    },
    {
        'category': '单方面变更',
        'description': '一方有权单方面变更合同条款，另一方无权异议',
        'level': '高',
    },
    {
        'category': '违约金过高',
        'description': '违约金超过合同标的额的30%，可能被认定为过高',
        'level': '中',
    },
    {
        'category': '管辖权不利',
        'description': '约定的争议管辖法院对一方明显不利',
        'level': '低',
    },
    {
        'category': '付款条件模糊',
        'description': '付款时间、方式、条件等约定不明确',
        'level': '中',
    },
    {
        'category': '知识产权归属不明',
        'description': '合同涉及知识产权但未明确约定归属',
        'level': '中',
    },
    {
        'category': '保密义务不对等',
        'description': '保密义务仅约束一方，或保密期限设置不合理',
        'level': '低',
    },
]


@register_tool('risk_checker')
class RiskChecker(BaseTool):
    description = '对合同条款进行风险检查，对照法律法规和风险模板，识别潜在风险点并给出风险等级和修改建议'
    parameters = [{
        'name': 'clauses',
        'type': 'string',
        'description': '需要检查的合同条款内容，JSON格式的条款信息',
        'required': True,
    }]

    def call(self, params: str, **kwargs) -> str:
        """
        检查合同条款中的风险点

        流程：
        1. 从知识库中检索与条款相关的法律条文（RAG）
        2. 将检索结果和内置风险规则合并为参考依据
        3. 让LLM基于这些依据分析合同条款的风险
        """
        params = self._verify_json_format_args(params)
        clauses = params['clauses']

        # 如果输入是字符串，尝试解析为结构化数据
        if isinstance(clauses, str):
            try:
                clauses_data = json.loads(clauses)
            except json.JSONDecodeError:
                clauses_data = clauses
        else:
            clauses_data = clauses

        try:
            from config.model_config import get_openai_client, get_model_config
            client = get_openai_client()
            config = get_model_config()

            # 构建内置风险规则文本
            rules_text = '\n'.join(
                f"- [{r['level']}风险] {r['category']}: {r['description']}"
                for r in DEFAULT_RISK_RULES
            )

            # RAG检索：从知识库中获取与条款相关的法律依据
            rag_references = self._retrieve_legal_references(clauses_data)
            if rag_references:
                rules_text = rag_references + '\n\n常见风险规则:\n' + rules_text

            # 使用风险检查prompt模板
            prompt = RISK_CHECK_PROMPT.format(
                legal_references=rules_text,
                clause_content=json.dumps(clauses_data, ensure_ascii=False, indent=2)
                if isinstance(clauses_data, (dict, list))
                else str(clauses_data)
            )

            response = client.chat.completions.create(
                model=config['model'],
                messages=[
                    {
                        'role': 'system',
                        'content': '你是一位资深的合同法律顾问。'
                                   '请基于提供的法律参考和风险规则，'
                                   '对合同条款进行专业的风险评估。'
                                   '以JSON格式返回风险评估结果。'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.2,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content

            try:
                risk_result = json.loads(result_text)
            except json.JSONDecodeError:
                risk_result = {'raw_analysis': result_text}

            return json.dumps({
                'success': True,
                'risk_assessment': risk_result
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps(
                {'success': False, 'error': f'风险检查失败: {str(e)}'},
                ensure_ascii=False
            )

    def _retrieve_legal_references(self, clauses_data: Any, top_k: int = 3) -> str:
        """
        从知识库中检索相关法律条文

        将合同条款作为查询，从向量知识库中检索最相关的法律文档片段。
        这就是RAG中的Retrieval环节。
        """
        try:
            from knowledge.build_kb import search_similar

            # 将条款内容转换为检索查询文本
            if isinstance(clauses_data, dict):
                query = json.dumps(clauses_data, ensure_ascii=False)
            else:
                query = str(clauses_data)

            # 如果查询文本太长，截取关键部分
            if len(query) > 500:
                query = query[:500]

            results = search_similar(query, top_k=top_k)
            if not results:
                return ''

            # 格式化检索结果
            references = '知识库检索到的相关法律依据:\n'
            for i, r in enumerate(results):
                references += (
                    f"\n[来源: {r['source']}, 相似度: {r['similarity']:.2f}]\n"
                    f"{r['text']}\n"
                )

            return references

        except Exception as e:
            # 如果知识库未构建或检索失败，回退到仅使用内置规则
            logger.warning("知识库检索失败，回退到内置规则: %s", e)
            return ''
