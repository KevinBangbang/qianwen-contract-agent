"""
结构化输出 Schema 定义

参考 MathGraph/50-AI-Agent/Prompt-Engineering/Output-Formatting.md，
使用 Pydantic 模型约束 LLM 输出格式，解决自由文本无法被程序解析的问题。

所有 Agent 和工具的输出 Schema 集中定义在此，便于维护和复用。
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# 枚举类型
# ============================================================

class RiskLevel(str, Enum):
    """风险等级。"""
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


class ContractType(str, Enum):
    """合同类型。"""
    SALES = "买卖合同"
    SERVICE = "服务合同"
    LABOR = "劳动合同"
    LEASE = "租赁合同"
    TECHNOLOGY = "技术合同"
    PARTNERSHIP = "合作合同"
    NDA = "保密协议"
    OTHER = "其他"


# ============================================================
# 条款提取结构
# ============================================================

class ContractParty(BaseModel):
    """合同主体信息。"""
    name: str = Field(description="主体名称")
    role: str = Field(description="角色（如甲方、乙方）")
    is_identified: bool = Field(default=True, description="是否明确标识")


class AmountInfo(BaseModel):
    """金额信息。"""
    total_amount: Optional[str] = Field(default=None, description="合同总金额")
    currency: str = Field(default="人民币", description="币种")
    payment_schedule: Optional[str] = Field(default=None, description="付款安排")


class ClauseExtraction(BaseModel):
    """条款提取结果。"""
    contract_parties: list[ContractParty] = Field(description="合同主体列表")
    contract_type: ContractType = Field(description="合同类型")
    subject_matter: str = Field(description="合同标的描述")
    amount: Optional[AmountInfo] = Field(default=None, description="金额信息")
    duration: Optional[str] = Field(default=None, description="合同期限")
    payment_terms: Optional[str] = Field(default=None, description="付款条件")
    breach_liability: Optional[str] = Field(default=None, description="违约责任")
    dispute_resolution: Optional[str] = Field(default=None, description="争议解决方式")
    special_clauses: list[str] = Field(default_factory=list, description="特殊条款")


# ============================================================
# 风险检查结构
# ============================================================

class RiskItem(BaseModel):
    """单个风险项。"""
    category: str = Field(description="风险类别")
    level: RiskLevel = Field(description="风险等级")
    description: str = Field(description="风险描述")
    clause_reference: Optional[str] = Field(default=None, description="涉及的合同条款原文")
    legal_basis: Optional[str] = Field(default=None, description="法律依据")
    suggestion: str = Field(description="修改建议")


class RiskAssessment(BaseModel):
    """风险评估结果。"""
    overall_risk_level: RiskLevel = Field(description="整体风险等级")
    risk_items: list[RiskItem] = Field(description="风险项列表")
    risk_summary: str = Field(description="风险总结")
    total_high_risks: int = Field(default=0, description="高风险数量")
    total_medium_risks: int = Field(default=0, description="中风险数量")
    total_low_risks: int = Field(default=0, description="低风险数量")


# ============================================================
# 审查报告结构
# ============================================================

class ReviewReport(BaseModel):
    """合同审查报告完整结构。"""
    title: str = Field(default="合同审查报告", description="报告标题")
    contract_type: ContractType = Field(description="合同类型")
    parties: list[ContractParty] = Field(description="合同主体")
    clause_summary: ClauseExtraction = Field(description="条款提取摘要")
    risk_assessment: RiskAssessment = Field(description="风险评估结果")
    overall_opinion: str = Field(description="总体审查意见")
    recommendations: list[str] = Field(description="修改建议列表")


# ============================================================
# 质量评估结构（用于 Reflexion）
# ============================================================

class DimensionScore(BaseModel):
    """单维度评分。"""
    score: float = Field(ge=0, le=10, description="分数 0-10")
    reason: str = Field(description="评分理由")


class QualityEvaluation(BaseModel):
    """审查质量评估结果。"""
    completeness: DimensionScore = Field(description="完整性评分")
    risk_identification: DimensionScore = Field(description="风险识别评分")
    legal_basis: DimensionScore = Field(description="法律依据评分")
    actionability: DimensionScore = Field(description="可操作性评分")
    clarity: DimensionScore = Field(description="结构清晰度评分")
    overall_score: float = Field(ge=0, le=10, description="综合评分")
    passed: bool = Field(description="是否通过质量阈值")
    major_issues: list[str] = Field(default_factory=list, description="主要问题")


# ============================================================
# 工具函数
# ============================================================

def get_review_report_schema() -> dict:
    """获取审查报告的 JSON Schema，用于 LLM structured output。"""
    return ReviewReport.model_json_schema()


def get_risk_assessment_schema() -> dict:
    """获取风险评估的 JSON Schema。"""
    return RiskAssessment.model_json_schema()


def get_quality_evaluation_schema() -> dict:
    """获取质量评估的 JSON Schema。"""
    return QualityEvaluation.model_json_schema()


def validate_risk_assessment(data: dict) -> tuple[bool, Optional[RiskAssessment], Optional[str]]:
    """验证风险评估 JSON 是否符合 Schema。

    Args:
        data: LLM 输出的 JSON 字典。

    Returns:
        (通过, 解析结果, 错误消息) 元组。
    """
    try:
        result = RiskAssessment.model_validate(data)
        return True, result, None
    except Exception as e:
        return False, None, str(e)


def validate_quality_evaluation(data: dict) -> tuple[bool, Optional[QualityEvaluation], Optional[str]]:
    """验证质量评估 JSON 是否符合 Schema。

    Args:
        data: LLM 输出的 JSON 字典。

    Returns:
        (通过, 解析结果, 错误消息) 元组。
    """
    try:
        result = QualityEvaluation.model_validate(data)
        return True, result, None
    except Exception as e:
        return False, None, str(e)
