"""
工具模块

包含合同审查所需的5个自定义工具，每个工具都继承 qwen_agent.tools.BaseTool。
工具列表：
- contract_parser: 合同解析（PDF提取 + OCR）
- clause_extractor: 条款提取
- risk_checker: 风险检查
- amount_calculator: 金额/日期计算
- report_generator: 审查报告生成
"""

__all__ = [
    "ContractParser",
    "ClauseExtractor",
    "RiskChecker",
    "AmountCalculator",
    "ReportGenerator",
]
