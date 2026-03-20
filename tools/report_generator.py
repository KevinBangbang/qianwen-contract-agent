"""
审查报告生成工具

整合所有审查结果（条款提取、风险检查、金额计算），
生成结构化的合同审查报告。支持Markdown格式输出。

这是审查流程的最后一步，负责将零散的分析结果组织成
清晰、专业的审查报告，方便人工复核。
"""

import json
from datetime import datetime

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('report_generator')
class ReportGenerator(BaseTool):
    description = '合同审查报告生成工具，整合条款提取、风险检查等结果，生成结构化的Markdown格式审查报告'
    parameters = [{
        'name': 'contract_info',
        'type': 'object',
        'description': '合同基本信息，包含文件名、解析结果等',
        'required': True,
    }, {
        'name': 'clauses',
        'type': 'object',
        'description': '提取的合同条款信息',
        'required': True,
    }, {
        'name': 'risk_assessment',
        'type': 'object',
        'description': '风险检查结果',
        'required': True,
    }, {
        'name': 'calculations',
        'type': 'object',
        'description': '金额和日期计算结果（可选）',
        'required': False,
    }]

    def call(self, params: str, **kwargs) -> str:
        """
        生成合同审查报告

        将所有分析结果整合为一份结构化的Markdown报告。
        报告包括：基本信息、条款摘要、风险评估、修改建议。
        """
        params = self._verify_json_format_args(params)

        contract_info = params.get('contract_info', {})
        clauses = params.get('clauses', {})
        risk_assessment = params.get('risk_assessment', {})
        calculations = params.get('calculations', {})

        try:
            report = self._build_report(
                contract_info, clauses, risk_assessment, calculations
            )

            return json.dumps({
                'success': True,
                'report': report,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                'success': False,
                'error': f'报告生成失败: {str(e)}'
            }, ensure_ascii=False)

    def _build_report(
        self,
        contract_info: dict,
        clauses: dict,
        risk_assessment: dict,
        calculations: dict,
    ) -> str:
        """
        构建Markdown格式的审查报告

        报告结构：
        1. 报告头部（标题、时间、基本信息）
        2. 合同条款摘要
        3. 风险评估结果
        4. 金额计算验证
        5. 综合建议
        """
        now = datetime.now().strftime('%Y年%m月%d日 %H:%M')
        sections = []

        # 报告标题
        sections.append('# 合同审查报告\n')
        sections.append(f'**审查时间**: {now}\n')

        # 基本信息
        sections.append('## 一、合同基本信息\n')
        if isinstance(contract_info, dict):
            for key, value in contract_info.items():
                if value:
                    sections.append(f'- **{key}**: {value}')
        sections.append('')

        # 条款摘要
        sections.append('## 二、关键条款摘要\n')
        if isinstance(clauses, dict):
            for key, value in clauses.items():
                if value and value != '未提及':
                    sections.append(f'### {key}\n')
                    sections.append(f'{value}\n')
        elif isinstance(clauses, str):
            sections.append(clauses)
        sections.append('')

        # 风险评估
        sections.append('## 三、风险评估\n')
        risk_items = []
        if isinstance(risk_assessment, dict):
            # 尝试从不同格式的风险结果中提取信息
            items = risk_assessment.get('risks', [])
            if not items:
                items = risk_assessment.get('risk_items', [])
            if not items:
                items = risk_assessment.get('risk_points', [])

            if items and isinstance(items, list):
                # 按风险等级排序：高 > 中 > 低
                level_order = {'高': 0, '中': 1, '低': 2}
                items.sort(
                    key=lambda x: level_order.get(
                        x.get('level', x.get('risk_level', '低')), 3
                    )
                )

                for item in items:
                    level = item.get('level', item.get('risk_level', '未知'))
                    desc = item.get('description', item.get('detail', ''))
                    suggestion = item.get('suggestion', item.get('advice', ''))

                    # 用不同标记区分风险等级
                    level_mark = {'高': '🔴', '中': '🟡', '低': '🟢'}.get(level, '⚪')

                    risk_line = f'- {level_mark} **[{level}风险]** {desc}'
                    if suggestion:
                        risk_line += f'\n  - 建议: {suggestion}'
                    risk_items.append(risk_line)

            if risk_items:
                sections.extend(risk_items)
            else:
                # 如果没有结构化的风险项，直接输出原始内容
                sections.append(json.dumps(
                    risk_assessment, ensure_ascii=False, indent=2
                ))
        sections.append('')

        # 金额计算验证
        if calculations:
            sections.append('## 四、金额计算验证\n')
            if isinstance(calculations, dict):
                for key, value in calculations.items():
                    sections.append(f'- **{key}**: {value}')
            sections.append('')

        # 综合评价
        sections.append('## 五、综合建议\n')
        # 统计各等级风险数量
        high_count = sum(
            1 for item in risk_items
            if '高风险' in item
        )
        medium_count = sum(
            1 for item in risk_items
            if '中风险' in item
        )
        low_count = sum(
            1 for item in risk_items
            if '低风险' in item
        )

        if high_count > 0:
            sections.append(
                f'本合同存在 **{high_count}** 项高风险问题，'
                '**建议暂缓签署**，待修改完善后再行签订。\n'
            )
        elif medium_count > 0:
            sections.append(
                f'本合同存在 **{medium_count}** 项中等风险问题，'
                '建议与对方协商修改相关条款后签署。\n'
            )
        else:
            sections.append('本合同整体风险较低，可考虑签署。\n')

        sections.append(f'风险统计: 高风险{high_count}项 / '
                        f'中风险{medium_count}项 / 低风险{low_count}项\n')

        sections.append('---\n')
        sections.append('*本报告由智能合同审查系统自动生成，仅供参考，不构成法律意见。*')

        return '\n'.join(sections)
