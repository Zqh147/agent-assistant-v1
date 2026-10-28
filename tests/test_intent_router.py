"""Intent router: the model proposes, the program validates."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.multi_agent.intent_router import (
    IntentRouter,
    ROUTE_TOOL_SCHEMA,
    ROUTER_PROMPT,
    TASK_TYPES,
    scope_from_model_output,
    validate_route,
)


SCOPE = {
    "product_model": "Archer AX10",
    "hardware_version": "V1",
    "region": "US",
}


def route_call(**args):
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "route_request",
            "args": {"reason": "测试", **args},
            "id": "call-route",
            "type": "tool_call",
        }],
    )


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, messages, tools, *, system_prompt=None):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "system_prompt": system_prompt,
        })
        return self.response


def test_router_uses_only_route_tool_and_own_prompt():
    llm = FakeLLM(route_call(task_type="order_only", order_id="DEMO-1001"))

    IntentRouter(llm).route("订单 DEMO-1001 是什么型号？")

    call = llm.calls[0]
    assert call["tools"] == [ROUTE_TOOL_SCHEMA]
    assert call["system_prompt"] == ROUTER_PROMPT
    assert call["messages"][0].content == "订单 DEMO-1001 是什么型号？"


def test_schema_enum_matches_task_types():
    enum = ROUTE_TOOL_SCHEMA["function"]["parameters"]["properties"]["task_type"]["enum"]
    assert enum == [*TASK_TYPES, "unclear"]


@pytest.mark.parametrize("task_type", [
    "order_only",
    "order_knowledge",
    "order_ticket",
    "order_knowledge_ticket",
])
def test_order_tasks_are_routed_with_order_from_text(task_type):
    decision = validate_route(
        "DEMO-1001 设备掉线",
        route_call(task_type=task_type, order_id="demo-1001"),
    )

    assert decision.status == "routed"
    assert decision.task_type == task_type
    assert decision.order_id == "DEMO-1001"
    assert decision.warnings == []


def test_order_task_without_order_id_needs_clarification():
    decision = validate_route(
        "路由器坏了能退吗？",
        route_call(task_type="order_ticket"),
    )

    assert decision.intent == "order_ticket"
    assert decision.status == "needs_clarification"
    assert decision.task_type is None
    assert "订单号" in decision.clarification


def test_two_order_ids_need_clarification():
    decision = validate_route(
        "DEMO-1001 和 DEMO-1002 哪个是 AX20？",
        route_call(task_type="order_only", order_id="DEMO-1001"),
    )

    assert decision.status == "needs_clarification"
    assert decision.order_id is None


def test_model_cannot_choose_a_different_order():
    decision = validate_route(
        "查一下 DEMO-1001 的状态",
        route_call(task_type="order_only", order_id="DEMO-1004"),
    )

    assert decision.status == "routed"
    assert decision.order_id == "DEMO-1001"
    assert "model_order_id_mismatch" in decision.warnings


def test_knowledge_scope_from_message():
    decision = validate_route(
        "AX10 V1 US 怎么修改 Wi-Fi 密码？",
        route_call(
            task_type="knowledge_only",
            product_model="AX10",
            hardware_version="v1",
            region="us",
        ),
    )

    assert decision.status == "routed"
    assert decision.task_type == "knowledge_only"
    assert decision.product_scope == SCOPE


def test_knowledge_scope_not_in_message_is_rejected():
    decision = validate_route(
        "AX10 怎么设置 PPPoE？",
        route_call(
            task_type="knowledge_only",
            product_model="Archer AX10",
            hardware_version="V1",
            region="US",
        ),
    )

    assert decision.status == "needs_clarification"
    assert decision.product_scope is None
    assert "model_scope_not_in_message" in decision.warnings


def test_knowledge_scope_mismatch_is_rejected():
    decision = validate_route(
        "Archer AX10 V1 US 怎么开启访客网络？",
        route_call(
            task_type="knowledge_only",
            product_model="Archer AX20",
            hardware_version="V1",
            region="US",
        ),
    )

    assert decision.status == "needs_clarification"


def test_client_scope_takes_priority():
    decision = validate_route(
        "怎么开启访客网络？",
        route_call(task_type="knowledge_only"),
        public_product_scope=SCOPE,
    )

    assert decision.status == "routed"
    assert decision.product_scope == SCOPE


def test_knowledge_only_with_order_id_needs_clarification():
    decision = validate_route(
        "DEMO-1001 怎么开 AP 模式？",
        route_call(task_type="knowledge_only"),
    )

    assert decision.status == "needs_clarification"
    assert decision.task_type is None


def test_unclear_intent():
    decision = validate_route("你好", route_call(task_type="unclear"))

    assert decision.intent == "unclear"
    assert decision.status == "needs_clarification"
    assert decision.clarification


@pytest.mark.parametrize("response,warning", [
    (AIMessage(content="这是订单查询。"), "router_no_decision"),
    (route_call(task_type="refund_now"), "router_unknown_task_type"),
    (
        AIMessage(content="", tool_calls=[{
            "name": "call_ticket_agent",
            "args": {},
            "id": "x",
            "type": "tool_call",
        }]),
        "router_no_decision",
    ),
])
def test_invalid_model_output_falls_back_to_unclear(response, warning):
    decision = validate_route("DEMO-1001 帮我报修", response)

    assert decision.intent == "unclear"
    assert decision.task_type is None
    assert warning in decision.warnings


@pytest.mark.parametrize("message,expected", [
    ("Archer AX10 V1 US 怎么设置", SCOPE),
    ("AX20 V1 US 开 AP 模式", {"product_model": "Archer AX20", "hardware_version": "V1", "region": "US"}),
    ("AX10 V1 开 AP 模式", None),
    ("AX10 AX20 V1 US 哪个好", None),
])
def test_scope_extraction_only_trusts_text(message, expected):
    args = {"product_model": "AX10", "hardware_version": "V1", "region": "US"}
    if expected:
        args = {"product_model": expected["product_model"], "hardware_version": "V1", "region": "US"}

    assert scope_from_model_output(message, args) == expected


def test_routing_cases_file_is_well_formed():
    path = Path(__file__).resolve().parent.parent / "data" / "eval" / "routing_cases.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]

    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["intent"] in (*TASK_TYPES, "unclear")
        assert case["status"] in ("routed", "needs_clarification")
        assert case["message"].strip()
        if case["intent"] == "unclear":
            assert case["status"] == "needs_clarification"


def test_perfect_model_reproduces_labeled_status():
    """If the model's intent is right, validation must yield the labeled status."""
    import re

    path = Path(__file__).resolve().parent.parent / "data" / "eval" / "routing_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]

    def literal(pattern, text):
        match = re.search(pattern, text)
        return match.group(0) if match else ""

    mismatches = []
    for case in cases:
        text = case["message"]
        response = route_call(
            task_type=case["intent"],
            order_id=literal(r"DEMO-\d+", text),
            product_model=literal(r"AX\d+", text),
            hardware_version=literal(r"V\d+", text),
            region=literal(r"\bUS\b", text),
        )
        decision = validate_route(text, response)
        if decision.status != case["status"]:
            mismatches.append((case["id"], decision.status, case["status"]))

    assert mismatches == []
