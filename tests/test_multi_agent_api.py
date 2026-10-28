
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
