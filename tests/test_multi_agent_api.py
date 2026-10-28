
"""Deterministic tests for the multi-agent HTTP boundary."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.multi_agent.http_router as http_router


@pytest.fixture
def api_client(monkeypatch):
    observed = {}

    class StubSupervisor:
        def __init__(self, llm):
            observed["llm"] = llm

        def run(self, **kwargs):
            observed["run"] = kwargs

            return SimpleNamespace(
                status="pending_confirmation",
                summary="演示工单草稿已创建，等待确认。",
                evidence=[{"source": "manual"}],
                data={
                    "ticket_id": "TKT-TEST123",
                    "tool_trace": [
                        {
                            "step": 1,
                            "tool": "call_order_agent",
                            "status": "success",
                        },
                        {
                            "step": 2,
                            "tool": "call_ticket_agent",
                            "status": "pending_confirmation",
                        },
                    ],
                },
                error=None,
            )

    monkeypatch.setattr(
        http_router,
        "DynamicSupervisor",
        StubSupervisor,
    )

    monkeypatch.setattr(
        http_router,
        "OrderAgent",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        http_router,
        "KnowledgeAgent",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        http_router,
        "TicketAgent",
        lambda **kwargs: object(),
    )

    def create_tools(**kwargs):
        observed["trusted_context"] = kwargs[
            "trusted_context"
        ]
        observed["public_product_scope"] = kwargs[
            "public_product_scope"
        ]
        return object()

    monkeypatch.setattr(
        http_router,
        "SupervisorTools",
        create_tools,
    )

    def fake_get_ticket(ticket_id, customer_id):
        if not observed.get("draft_present", True):
            return None

        return {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "thread_id": observed["trusted_context"]["thread_id"],
            "status": "draft",
        }

    fake_agent = SimpleNamespace(
        settings=SimpleNamespace(
            support_mode="demo",
            demo_customer_id="demo-customer-001",
        ),
        support_service=SimpleNamespace(
            store=SimpleNamespace(
                get_ticket=fake_get_ticket,
            ),
        ),
        llm=object(),
    )

    app = FastAPI()
    app.include_router(
        http_router.create_multi_agent_router(fake_agent)
    )

    return TestClient(app), observed, fake_agent


def test_multi_agent_chat_success(api_client):
    client, observed, _ = api_client

    response = client.post(
        "/multi-agent/chat",
        json={
            "message": (
                "订单 DEMO-1001 设备频繁掉线，"
                "请准备维修工单草稿。"
            ),
            "thread_id": "day05-test-001",
            "task_type": "order_ticket",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "pending_confirmation"
    assert body["ticket_id"] == "TKT-TEST123"
    assert body["thread_id"] == "day05-test-001"

    assert len(body["tool_trace"]) == 2

    assert observed["run"]["task_type"] == "order_ticket"
    assert observed["run"]["max_steps"] == 8

    assert observed["trusted_context"]["customer_id"] == (
        "demo-customer-001"
    )

    assert observed["trusted_context"]["thread_id"] == (
        "day05-test-001"
    )
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "message": "",
            "thread_id": "test-001",
            "task_type": "order_ticket",
        },
        {
            "message": "查询订单",
            "thread_id": "",
            "task_type": "order_ticket",
        },
        {
            "message": "查询订单",
            "thread_id": "test-001",
            "task_type": "unknown_task",
        },
        {
            "message": "查询订单",
            "thread_id": "test-001",
            "task_type": "order_ticket",
            "customer_id": "another-customer",
        },
        {
            "message": "设备是否支持 QoS？",
            "thread_id": "test-001",
            "task_type": "knowledge_only",
        },
        {
            "message": "准备工单",
            "thread_id": "test-001",
            "task_type": "order_ticket",
            "public_product_scope": {
                "product_model": "Archer AX10",
                "hardware_version": "V1",
                "region": "US",
            },
        },
    ],
)
def test_invalid_requests_return_422(
    api_client,
    payload,
):
    client, _, _ = api_client

    response = client.post(
        "/multi-agent/chat",
        json=payload,
    )

    assert response.status_code == 422


def test_knowledge_only_scope(api_client):
    client, observed, _ = api_client

    response = client.post(
        "/multi-agent/chat",
        json={
            "message": "Archer AX10 V1 US 是否支持 QoS？",
            "thread_id": "knowledge-test-001",
            "task_type": "knowledge_only",
            "public_product_scope": {
                "product_model": "Archer AX10",
                "hardware_version": "V1",
                "region": "US",
            },
        },
    )

    assert response.status_code == 200

    assert observed["public_product_scope"] == {
        "product_model": "Archer AX10",
        "hardware_version": "V1",
        "region": "US",
    }


def test_disabled_service_returns_503(api_client):
    client, _, fake_agent = api_client

    fake_agent.support_service = None

    response = client.post(
        "/multi-agent/chat",
        json={
            "message": "查询订单 DEMO-1001",
            "thread_id": "disabled-test-001",
            "task_type": "order_only",
        },
    )

    assert response.status_code == 503


def test_unpersisted_draft_is_rejected(api_client):
    client, observed, _ = api_client

    observed["draft_present"] = False

    response = client.post(
        "/multi-agent/chat",
        json={
            "message": (
                "订单 DEMO-1001 设备频繁掉线，"
                "请准备维修工单草稿。"
            ),
            "thread_id": "missing-draft-001",
            "task_type": "order_ticket",
        },
    )

    assert response.status_code == 503
    assert response.headers.get("X-Request-ID")
    assert "TKT-TEST123" not in response.text


def test_internal_error_is_sanitized(
    api_client,
    monkeypatch,
):
    client, _, _ = api_client

    class FailingSupervisor:
        def __init__(self, llm):
            self.llm = llm

        def run(self, **kwargs):
            raise RuntimeError(
                "secret-internal-error"
            )

    monkeypatch.setattr(
        http_router,
        "DynamicSupervisor",
        FailingSupervisor,
    )

    response = client.post(
        "/multi-agent/chat",
        json={
            "message": "查询订单 DEMO-1001",
            "thread_id": "error-test-001",
            "task_type": "order_only",
        },
    )

    assert response.status_code == 503
    assert "secret-internal-error" not in response.text
    assert response.headers.get("X-Request-ID")


# --- LLM intent routing -----------------------------------------------------

from app.multi_agent.intent_router import RouteDecision


def stub_router(monkeypatch, decision, observed):
    class StubRouter:
        def __init__(self, llm):
            observed["router_llm"] = llm

        def route(self, message, *, public_product_scope=None):
            observed["router_message"] = message
            observed["router_scope"] = public_product_scope
            return decision

    monkeypatch.setattr(http_router, "IntentRouter", StubRouter)


def test_omitted_task_type_is_routed_by_llm(api_client, monkeypatch):
    client, observed, fake_agent = api_client
    stub_router(
        monkeypatch,
        RouteDecision(
            intent="order_ticket",
            status="routed",
            reason="用户要求报修",
            order_id="DEMO-1001",
        ),
        observed,
    )

    response = client.post(
        "/multi-agent/chat",
        json={
            "message": "DEMO-1001 设备无法开机，帮我申请维修",
            "thread_id": "route-001",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "order_ticket"
    assert body["routing_source"] == "llm"
    assert body["routing"]["reason"] == "用户要求报修"
    assert observed["router_llm"] is fake_agent.llm
    assert observed["run"]["task_type"] == "order_ticket"
    assert observed["public_product_scope"] is None


def test_clarification_stops_before_any_agent(api_client, monkeypatch):
    client, observed, _ = api_client
    stub_router(
        monkeypatch,
        RouteDecision(
            intent="order_ticket",
            status="needs_clarification",
            reason="缺少订单号",
            clarification="请提供一个有效的订单号。",
        ),
        observed,
    )

    response = client.post(
        "/multi-agent/chat",
        json={"message": "路由器坏了能退吗？", "thread_id": "route-002"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_clarification"
    assert body["task_type"] is None
    assert body["summary"] == "请提供一个有效的订单号。"
    assert body["routing"]["intent"] == "order_ticket"
    assert body["tool_trace"] == []
    assert "run" not in observed
    assert "trusted_context" not in observed


def test_routed_knowledge_uses_validated_scope(api_client, monkeypatch):
    client, observed, _ = api_client
    scope = {
        "product_model": "Archer AX10",
        "hardware_version": "V1",
        "region": "US",
    }
    stub_router(
        monkeypatch,
        RouteDecision(
            intent="knowledge_only",
            status="routed",
            reason="技术问题",
            product_scope=scope,
        ),
        observed,
    )

    response = client.post(
        "/multi-agent/chat",
        json={
            "message": "Archer AX10 V1 US 怎么开访客网络？",
            "thread_id": "route-003",
        },
    )

    assert response.status_code == 200
    assert observed["run"]["task_type"] == "knowledge_only"
    assert observed["public_product_scope"] == scope


def test_client_task_type_skips_router(api_client, monkeypatch):
    client, observed, _ = api_client

    class ExplodingRouter:
        def __init__(self, llm):
            raise AssertionError("router must not run")

    monkeypatch.setattr(http_router, "IntentRouter", ExplodingRouter)

    response = client.post(
        "/multi-agent/chat",
        json={
            "message": "DEMO-1001 设备无法开机，帮我申请维修",
            "thread_id": "route-004",
            "task_type": "order_ticket",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["routing_source"] == "client"
    assert body["routing"] is None


def test_incomplete_scope_without_task_type_is_422(api_client):
    client, _, _ = api_client

    response = client.post(
        "/multi-agent/chat",
        json={
            "message": "怎么开访客网络？",
            "thread_id": "route-005",
            "public_product_scope": {"product_model": "Archer AX10"},
        },
    )

    assert response.status_code == 422


def test_router_failure_returns_503(api_client, monkeypatch):
    client, _, _ = api_client

    class FailingRouter:
        def __init__(self, llm):
            pass

        def route(self, message, *, public_product_scope=None):
            raise RuntimeError("model down")

    monkeypatch.setattr(http_router, "IntentRouter", FailingRouter)

    response = client.post(
        "/multi-agent/chat",
        json={"message": "DEMO-1001 掉线", "thread_id": "route-006"},
    )

    assert response.status_code == 503
    assert "model down" not in response.text
