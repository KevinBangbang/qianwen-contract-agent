"""
Agent 护栏模块

参考 MathGraph/50-AI-Agent/Safety/Agent-Guardrails.md，实现纵深防御：
Layer 1: 输入护栏 — 长度限制、空内容检查
Layer 2: 成本控制 — Token 预算、工具调用次数、超时限制
Layer 3: 输出护栏 — 结构验证、幻觉标记、置信度检查

设计原则：
- 护栏不应过严影响可用性，只拦截明显异常
- 所有护栏触发事件记录日志，用于后续调优
- 不替代 prompt engineering，两者互补
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ============================================================
# Layer 1: 输入护栏
# ============================================================

@dataclass
class InputGuardrailConfig:
    """输入护栏配置。"""
    max_length: int = 50000          # 最大输入长度（字符数）
    min_length: int = 10             # 最小输入长度
    blocked_patterns: list[str] = field(default_factory=list)  # 屏蔽的关键词模式


class InputGuardrail:
    """输入内容验证护栏。

    检查用户输入是否在合理范围内：
    - 长度限制：防止超长输入导致 token 爆炸
    - 最短长度：过短的文本不是有效合同
    - 内容检查：基础的输入合法性校验
    """

    def __init__(self, config: Optional[InputGuardrailConfig] = None):
        self.config = config or InputGuardrailConfig()

    def check(self, text: str) -> tuple[bool, str]:
        """验证输入内容。

        Args:
            text: 用户输入的文本。

        Returns:
            (通过, 消息) 元组。通过为 True 表示输入合法。
        """
        if not text or not text.strip():
            return False, "输入内容为空，请提供合同文本"

        text_len = len(text.strip())

        if text_len < self.config.min_length:
            return False, f"输入内容过短（{text_len}字），请提供完整的合同文本"

        if text_len > self.config.max_length:
            return False, (
                f"输入内容过长（{text_len}字，上限{self.config.max_length}字）。"
                f"请分段提交或提供摘要"
            )

        logger.debug("[输入护栏] 通过: %d 字", text_len)
        return True, "OK"


# ============================================================
# Layer 2: 成本控制护栏
# ============================================================

@dataclass
class CostGuardrailConfig:
    """成本控制配置。"""
    max_llm_calls: int = 20          # 单次审查最大 LLM 调用次数
    max_tool_calls: int = 15         # 单次审查最大工具调用次数
    max_time_seconds: float = 300    # 单次审查最大耗时（秒）
    max_input_tokens: int = 100000   # 单次审查最大输入 token 数（估算）


class CostGuardrail:
    """成本和资源控制护栏。

    防止 Agent 陷入无限循环或过度消耗资源：
    - LLM 调用次数限制
    - 工具调用次数限制
    - 时间限制
    - Token 预算估算
    """

    def __init__(self, config: Optional[CostGuardrailConfig] = None):
        self.config = config or CostGuardrailConfig()
        self.reset()

    def reset(self) -> None:
        """重置计数器，每次新任务开始时调用。"""
        self._llm_calls = 0
        self._tool_calls = 0
        self._start_time = time.time()
        self._estimated_tokens = 0

    def record_llm_call(self, input_tokens: int = 0) -> None:
        """记录一次 LLM 调用。"""
        self._llm_calls += 1
        self._estimated_tokens += input_tokens

    def record_tool_call(self) -> None:
        """记录一次工具调用。"""
        self._tool_calls += 1

    def check(self) -> tuple[bool, str]:
        """检查是否超出成本限制。

        Returns:
            (通过, 消息) 元组。
        """
        elapsed = time.time() - self._start_time

        if self._llm_calls >= self.config.max_llm_calls:
            msg = f"LLM 调用次数超限 ({self._llm_calls}/{self.config.max_llm_calls})"
            logger.warning("[成本护栏] %s", msg)
            return False, msg

        if self._tool_calls >= self.config.max_tool_calls:
            msg = f"工具调用次数超限 ({self._tool_calls}/{self.config.max_tool_calls})"
            logger.warning("[成本护栏] %s", msg)
            return False, msg

        if elapsed >= self.config.max_time_seconds:
            msg = f"审查超时 ({elapsed:.0f}s/{self.config.max_time_seconds}s)"
            logger.warning("[成本护栏] %s", msg)
            return False, msg

        if self._estimated_tokens >= self.config.max_input_tokens:
            msg = f"Token 预算超限 (~{self._estimated_tokens}/{self.config.max_input_tokens})"
            logger.warning("[成本护栏] %s", msg)
            return False, msg

        return True, "OK"

    def get_usage_report(self) -> dict[str, Any]:
        """获取当前资源使用报告。"""
        elapsed = time.time() - self._start_time
        return {
            'llm_calls': self._llm_calls,
            'tool_calls': self._tool_calls,
            'elapsed_seconds': round(elapsed, 1),
            'estimated_tokens': self._estimated_tokens,
        }


# ============================================================
# Layer 3: 输出护栏
# ============================================================

@dataclass
class OutputGuardrailConfig:
    """输出护栏配置。"""
    min_output_length: int = 100       # 最短输出长度
    required_sections: list[str] = field(default_factory=lambda: [
        '风险',  # 审查报告必须包含风险分析
    ])
    hallucination_markers: list[str] = field(default_factory=lambda: [
        '我不确定',
        '可能存在',
        '无法确认',
    ])


class OutputGuardrail:
    """输出内容验证护栏。

    检查 Agent 输出是否满足质量要求：
    - 最短长度：防止空洞回复
    - 必要章节：审查报告必须包含风险分析等核心内容
    - 幻觉标记：检测低置信度表述并标注
    """

    def __init__(self, config: Optional[OutputGuardrailConfig] = None):
        self.config = config or OutputGuardrailConfig()

    def check(self, output: str) -> tuple[bool, str, dict]:
        """验证输出内容。

        Args:
            output: Agent 生成的输出文本。

        Returns:
            (通过, 消息, 详情) 元组。
        """
        details: dict[str, Any] = {
            'output_length': len(output),
            'missing_sections': [],
            'hallucination_warnings': [],
        }

        if not output or len(output.strip()) < self.config.min_output_length:
            return False, "输出内容过短，审查报告不完整", details

        # 检查必要章节
        for section in self.config.required_sections:
            if section not in output:
                details['missing_sections'].append(section)

        if details['missing_sections']:
            msg = f"审查报告缺少必要内容: {', '.join(details['missing_sections'])}"
            logger.warning("[输出护栏] %s", msg)
            return False, msg, details

        # 检测幻觉标记
        for marker in self.config.hallucination_markers:
            if marker in output:
                details['hallucination_warnings'].append(marker)

        if details['hallucination_warnings']:
            logger.info("[输出护栏] 检测到低置信度表述: %s",
                        details['hallucination_warnings'])

        return True, "OK", details


# ============================================================
# 组合护栏链
# ============================================================

class GuardrailChain:
    """护栏链：将输入、成本、输出护栏串联成完整的安全管线。

    使用方式：
        chain = GuardrailChain()

        # 审查前检查输入
        ok, msg = chain.check_input(contract_text)
        if not ok:
            return error_response(msg)

        # 审查中监控成本
        chain.cost.record_llm_call()
        ok, msg = chain.check_cost()

        # 审查后检查输出
        ok, msg, details = chain.check_output(review_text)
    """

    def __init__(
        self,
        input_config: Optional[InputGuardrailConfig] = None,
        cost_config: Optional[CostGuardrailConfig] = None,
        output_config: Optional[OutputGuardrailConfig] = None,
    ):
        self.input_guard = InputGuardrail(input_config)
        self.cost = CostGuardrail(cost_config)
        self.output_guard = OutputGuardrail(output_config)

    def check_input(self, text: str) -> tuple[bool, str]:
        """输入阶段护栏检查。"""
        return self.input_guard.check(text)

    def check_cost(self) -> tuple[bool, str]:
        """成本阶段护栏检查。"""
        return self.cost.check()

    def check_output(self, output: str) -> tuple[bool, str, dict]:
        """输出阶段护栏检查。"""
        return self.output_guard.check(output)

    def reset(self) -> None:
        """重置成本计数器。"""
        self.cost.reset()

    def get_report(self) -> dict:
        """获取护栏运行报告。"""
        return {
            'cost_usage': self.cost.get_usage_report(),
        }
