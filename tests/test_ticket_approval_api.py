"""Isolated FastAPI approval route tests; real demo SQLite, fake authenticated actor."""
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.support_store import SupportStore
from app.support_service import SupportService
from app.multi_agent.ticket_approval_api import (
    ActorContext,
    create_ticket_approval_router,
)


@pytest.fixture
def api_env(tmp_path):
    store = SupportStore(tmp_path / "support.db", settings.demo_orders_path)
    service = SupportService(store, settings.support_rules_path)
    actor = ActorContext(
        customer_id="demo-customer-001",
        thread_id=f"approval-test-{uuid4().hex[:12]}",
    )
    current = {"actor": actor}

    def test_actor():
        # Only for tests. Production must return an authenticated actor.
        return current["actor"]

    app = FastAPI()
    app.include_router(
        create_ticket_approval_router(service, actor_dependency=test_actor)
    )
    return TestClient(app), store, service, actor, current


def prepare_draft(service, actor):
    result = service.prepare_ticket(
        "DEMO-1001",
        "repair",
        "设备频繁掉线",
        customer_id=actor.customer_id,
        thread_id=actor.thread_id,
        user_messages=["设备频繁掉线"],
    )
    assert result["status"] == "draft_ready"
    return result["ticket"]["ticket_id"]


def start_approval(client, ticket_id):
    response = client.post("/ticket-approvals", json={"ticket_id": ticket_id})
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "pending"
    return response.json()["approval_id"]


def test_approve_and_repeated_decision(api_env):
    client, store, service, actor, _ = api_env
    ticket_id = prepare_draft(service, actor)
    approval_id = start_approval(client, ticket_id)
    assert store.get_ticket(ticket_id, actor.customer_id)["status"] == "draft"

    response = client.post(
        f"/ticket-approvals/{approval_id}/decision", json={"approved": True}
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "submitted"
    assert store.get_ticket(ticket_id, actor.customer_id)["status"] == "submitted"
    assert sum(
        event["event"] == "submitted"
        for event in store.events(ticket_id, actor.customer_id)
    ) == 1
    assert client.post(
        f"/ticket-approvals/{approval_id}/decision", json={"approved": True}
    ).status_code == 409


def test_rejection_keeps_draft(api_env):
    client, store, service, actor, _ = api_env
    ticket_id = prepare_draft(service, actor)
    approval_id = start_approval(client, ticket_id)
    response = client.post(
        f"/ticket-approvals/{approval_id}/decision", json={"approved": False}
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "rejected"
    assert store.get_ticket(ticket_id, actor.customer_id)["status"] == "draft"
    assert not any(
        event["event"] == "submitted"
        for event in store.events(ticket_id, actor.customer_id)
    )


def test_cross_customer_and_cross_thread_denied(api_env):
    client, _, service, actor, current = api_env
    ticket_id = prepare_draft(service, actor)
    approval_id = start_approval(client, ticket_id)

    for other in (
        ActorContext("another-customer", actor.thread_id),
        ActorContext(actor.customer_id, "different-thread"),
    ):
        current["actor"] = other
        assert client.get(f"/ticket-approvals/{approval_id}").status_code == 404
        assert client.post(
            f"/ticket-approvals/{approval_id}/decision", json={"approved": True}
        ).status_code == 404

    current["actor"] = actor
    assert client.get(f"/ticket-approvals/{approval_id}").status_code == 200


def test_missing_and_submitted_ticket_rejected(api_env):
    client, _, service, actor, _ = api_env
    assert client.post(
        "/ticket-approvals", json={"ticket_id": "TKT-DOES-NOT-EXIST"}
    ).status_code == 404

    ticket_id = prepare_draft(service, actor)
    approval_id = start_approval(client, ticket_id)
    assert client.post(
        f"/ticket-approvals/{approval_id}/decision", json={"approved": True}
    ).status_code == 200
    assert client.post(
        "/ticket-approvals", json={"ticket_id": ticket_id}
    ).status_code == 409
