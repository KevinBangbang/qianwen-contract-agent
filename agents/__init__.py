"""
Agent模块

包含合同审查系统的Agent编排逻辑。
- review_agent: 主审查Agent，多步推理
- ocr_agent: 多模态OCR Agent
- orchestrator: Agent编排器，负责路由
"""

__all__ = [
    "create_review_agent",
    "run_review",
    "create_ocr_agent",
    "run_ocr",
    "process_contract",
    "detect_input_type",
]
