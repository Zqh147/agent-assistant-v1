from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# 子 Agent 的名称
AgentName = Literal[
    "knowledge_agent",
    "order_agent",
    "ticket_agent",
    "supervisor",
]


# Agent 执行结果状态
AgentStatus = Literal[
    "success",
    "pending_confirmation",
    "needs_clarification",
    "parameter_conflict",
    "unsupported_product",
    "no_evidence",
    "error",
]


@dataclass
class AgentTask:
    """Supervisor 发给子 Agent 的任务。"""

    task: str
    product_scope: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task.strip():
            raise ValueError("task must not be blank")


@dataclass
class AgentResult:
    """子 Agent 返回给 Supervisor 的统一结果。"""

    agent: AgentName
    status: AgentStatus
    summary: str

    evidence: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
