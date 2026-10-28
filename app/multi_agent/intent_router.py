"""LLM intent routing for the multi-agent workflow.

The model only proposes a task type. Every field that matters for
execution (order ID, device scope, whether execution may start) is
re-checked here against the user's original message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Mapping

from langchain_core.messages import AIMessage, HumanMessage

from .dynamic_supervisor import ORDER_ID_PATTERN, extract_requested_order_id


TASK_TYPES = (
    "order_only",
    "knowledge_only",
    "order_knowledge",
    "order_ticket",
    "order_knowledge_ticket",
)
ORDER_TASK_TYPES = frozenset(TASK_TYPES) - {"knowledge_only"}
SCOPE_FIELDS = ("product_model", "hardware_version", "region")

RouteIntent = Literal[
    "order_only",
    "knowledge_only",
    "order_knowledge",
    "order_ticket",
    "order_knowledge_ticket",
    "unclear",
]
RouteStatus = Literal["routed", "needs_clarification"]


ROUTER_PROMPT = """
你是网络设备售后系统的意图路由器，只负责判断用户想做什么，不回答问题。
你必须调用 route_request 工具给出判断，不要直接输出文字。

task_type 取值：
- order_only：只查询订单或订单关联的设备信息（型号、状态、签收、发货）。
- knowledge_only：设备技术问题（设置、功能、故障排查），消息中没有订单号。
- order_knowledge：有订单号，并询问该设备的技术或排查问题，没有要求报修或退货。
- order_ticket：有订单号（或明显想针对某个订单），要求维修或退货申请，没有要求先排查。
- order_knowledge_ticket：要求先查技术资料或排查，再申请维修或退货。
- unclear：问候、闲聊、与售后无关、产品比较、工单进度查询，或无法判断。

判断规则：
1. 按用户真实意图判断；用户明确拒绝某件事（例如"不想报修"）时，不要把它算进意图。
2. 缺少订单号不影响判断意图，例如"路由器坏了能退吗"仍是 order_ticket。
3. order_id 只填写消息中原样出现的订单号，没有就填空字符串。
4. product_model、hardware_version、region 只填写消息中原样出现的内容，没有就填空字符串，不要推测。
5. reason 用一句中文说明判断依据。
"""


ROUTE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "route_request",
        "description": "Report the routing decision for the user's message.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": [*TASK_TYPES, "unclear"],
                },
                "order_id": {
                    "type": "string",
                    "description": "Order number exactly as written in the message, or empty.",
                },
                "product_model": {
                    "type": "string",
                    "description": "Device model as written in the message, or empty.",
                },
                "hardware_version": {
                    "type": "string",
                    "description": "Hardware version as written in the message, or empty.",
                },
                "region": {
                    "type": "string",
                    "description": "Sales region as written in the message, or empty.",
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence explaining the decision.",
                },
            },
            "required": ["task_type", "reason"],
        },
    },
}


CLARIFY_UNCLEAR = (
    "我可以帮你查询订单、解答设备技术问题，或准备维修/退货申请。"
    "请说明你想办理哪一项。"
)
CLARIFY_ORDER = "请提供一个有效的订单号（例如 DEMO-1001），每次只处理一个订单。"
CLARIFY_SCOPE = "请提供完整的设备型号、硬件版本和销售地区，例如 Archer AX10 V1 US。"
CLARIFY_ORDER_IN_KNOWLEDGE = (
    "你的问题里包含订单号。请确认是要查询该订单设备的技术问题，"
    "还是只咨询通用设备问题。"
)


@dataclass
class RouteDecision:
    """Router output after program-side validation."""

    intent: RouteIntent
    status: RouteStatus
    reason: str
    order_id: str | None = None
    product_scope: dict[str, str] | None = None
    clarification: str | None = None
    model_intent: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def task_type(self) -> str | None:
        """Executable task type, or None when execution must not start."""
        if self.status != "routed" or self.intent == "unclear":
            return None
        return self.intent

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- device scope helpers -------------------------------------------------

_MODEL = re.compile(r"(?<![A-Za-z0-9])AX\s?(\d{2,4})(?![0-9])", re.IGNORECASE)
_VERSION = re.compile(r"(?<![A-Za-z0-9])V(\d+(?:\.\d+)?)(?![0-9.])", re.IGNORECASE)
# A closed list: a generic two-letter pattern would also match "AX" in
# "AX10" or "AP" in "AP 模式".
_REGION_CODES = ("US", "EU", "UK", "CA", "CN", "JP", "AU", "DE", "FR", "RU", "TW", "BR")
_CODE_GROUP = "|".join(_REGION_CODES)
_REGION = re.compile(rf"(?<![A-Za-z0-9])({_CODE_GROUP})(?![A-Za-z0-9])")
# Lowercase codes are ordinary English words ("help us"), so they only count
# directly after a hardware version, as in "AX20 V1 us".
_REGION_AFTER_VERSION = re.compile(
    rf"(?<![A-Za-z0-9])V\d+(?:\.\d+)?\s*({_CODE_GROUP})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_REGION_ALIASES = {
    "美版": "US",
    "美国版": "US",
    "美规": "US",
    "欧版": "EU",
    "欧规": "EU",
    "国行": "CN",
}


def _regions_in(text: str) -> set[str]:
    found = {match.upper() for match in _REGION.findall(text)}
    found |= {match.upper() for match in _REGION_AFTER_VERSION.findall(text)}
    found |= {code for alias, code in _REGION_ALIASES.items() if alias in text}
    return found


def _normalize_region(value: str) -> str | None:
    value = value.strip()
    if value in _REGION_ALIASES:
        return _REGION_ALIASES[value]
    return value.upper() if value.upper() in _REGION_CODES else None


def _find_unique(pattern: re.Pattern, text: str) -> str | None:
    values = {match.upper() for match in pattern.findall(text)}
    return next(iter(values)) if len(values) == 1 else None


def scope_from_model_output(message: str, args: Mapping[str, Any]) -> dict[str, str] | None:
    """Accept a device scope only if every part literally appears in the message.

    The model's values are compared against what a regex finds in the original
    text, so a hallucinated model, version or region is rejected.
    """

    text_without_orders = ORDER_ID_PATTERN.sub(" ", message)

    model_in_text = _find_unique(_MODEL, text_without_orders)
    version_in_text = _find_unique(_VERSION, text_without_orders)
    regions = _regions_in(text_without_orders)
    region_in_text = next(iter(regions)) if len(regions) == 1 else None

    claimed_model = _find_unique(_MODEL, str(args.get("product_model") or ""))
    claimed_version = _find_unique(_VERSION, str(args.get("hardware_version") or ""))
    claimed_region = _normalize_region(str(args.get("region") or ""))

    if not (model_in_text and version_in_text and region_in_text):
        return None
    if (claimed_model, claimed_version, claimed_region) != (
        model_in_text,
        version_in_text,
        region_in_text,
    ):
        return None

    return {
        "product_model": f"Archer AX{model_in_text}",
        "hardware_version": f"V{version_in_text}",
        "region": region_in_text,
    }


def _complete_scope(scope: Mapping[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(scope, Mapping):
        return None
    if set(scope) != set(SCOPE_FIELDS):
        return None
    if any(not isinstance(scope[key], str) or not scope[key].strip() for key in SCOPE_FIELDS):
        return None
    return {key: scope[key].strip() for key in SCOPE_FIELDS}


# --- router ---------------------------------------------------------------


class IntentRouter:
    """One LLM call proposes the task type; Python decides if it may run."""

    def __init__(self, llm):
        self.llm = llm
        self.schemas = [ROUTE_TOOL_SCHEMA]

    def propose(self, message: str) -> AIMessage:
        response = self.llm.invoke(
            [HumanMessage(content=message)],
            self.schemas,
            system_prompt=ROUTER_PROMPT,
        )
        if not isinstance(response, AIMessage):
            raise TypeError("LLM must return an AIMessage")
        return response

    def route(
        self,
        message: str,
        *,
        public_product_scope: Mapping[str, str] | None = None,
    ) -> RouteDecision:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must not be blank")
        message = message.strip()

        response = self.propose(message)
        return validate_route(
            message,
            response,
            public_product_scope=public_product_scope,
        )


def _unclear(reason: str, *, model_intent: str | None = None, warnings=None) -> RouteDecision:
    return RouteDecision(
        intent="unclear",
        status="needs_clarification",
        reason=reason,
        clarification=CLARIFY_UNCLEAR,
        model_intent=model_intent,
        warnings=list(warnings or []),
    )


def validate_route(
    message: str,
    response: AIMessage,
    *,
    public_product_scope: Mapping[str, str] | None = None,
) -> RouteDecision:
    """Turn the model's proposal into a decision the program can trust."""

    calls = [
        call for call in (response.tool_calls or [])
        if call.get("name") == "route_request"
    ]

    if len(calls) != 1:
        return _unclear(
            "路由器没有给出唯一的路由判断。",
            warnings=["router_no_decision"],
        )

    args = calls[0].get("args")
    if not isinstance(args, dict):
        return _unclear("路由参数格式不合法。", warnings=["router_invalid_args"])

    intent = args.get("task_type")
    reason = str(args.get("reason") or "").strip() or "未提供理由。"

    if intent not in (*TASK_TYPES, "unclear"):
        return _unclear(
            "路由器返回了未知的任务类型。",
            model_intent=str(intent),
            warnings=["router_unknown_task_type"],
        )

    if intent == "unclear":
        return _unclear(reason, model_intent=intent)

    warnings: list[str] = []
    order_in_text = extract_requested_order_id(message)
    claimed_order = str(args.get("order_id") or "").strip().upper() or None

    # Order tasks: the order ID always comes from the original text.
    if intent in ORDER_TASK_TYPES:
        if order_in_text is None:
            return RouteDecision(
                intent=intent,
                status="needs_clarification",
                reason=reason,
                clarification=CLARIFY_ORDER,
                model_intent=intent,
            )

        if claimed_order is not None and claimed_order != order_in_text:
            warnings.append("model_order_id_mismatch")

        return RouteDecision(
            intent=intent,
            status="routed",
            reason=reason,
            order_id=order_in_text,
            model_intent=intent,
            warnings=warnings,
        )

    # knowledge_only
    if ORDER_ID_PATTERN.search(message):
        return RouteDecision(
            intent=intent,
            status="needs_clarification",
            reason=reason,
            clarification=CLARIFY_ORDER_IN_KNOWLEDGE,
            model_intent=intent,
        )

    scope = _complete_scope(public_product_scope)
    if scope is None:
        scope = scope_from_model_output(message, args)
        if scope is None and any(args.get(key) for key in SCOPE_FIELDS):
            warnings.append("model_scope_not_in_message")

    if scope is None:
        return RouteDecision(
            intent=intent,
            status="needs_clarification",
            reason=reason,
            clarification=CLARIFY_SCOPE,
            model_intent=intent,
            warnings=warnings,
        )

    return RouteDecision(
        intent=intent,
        status="routed",
        reason=reason,
        product_scope=scope,
        model_intent=intent,
        warnings=warnings,
    )
