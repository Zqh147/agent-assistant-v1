"""Ticket approvals survive restarts and half-finished decisions.

A "restart" builds a brand-new router, store and checkpointer on the same
SQLite file, exactly what a new server process would do.
"""

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.types import Command

from app.config import settings
from app.multi_agent.approval_store import ApprovalStore, open_approval_checkpointer
from app.multi_agent.ticket_approval_api import ActorContext, create_ticket_approval_router
from app.multi_agent.ticket_approval_graph import build_ticket_approval_graph
from app.support_service import SupportService
from app.support_store import SupportStore


@pytest.fixture
def env(tmp_path):
    store = SupportStore(tmp_path / "support.db", settings.demo_orders_path)
    service = SupportService(store, settings.support_rules_path)
    actor = ActorContext(customer_id="demo-customer-001", thread_id=f"dur-{uuid4().hex[:8]}")
    db_path = tmp_path / "approvals.db"

    def boot(current_actor=actor):
        """Start a fresh 'process' against the same approval database."""
        app = FastAPI()
        app.include_router(create_ticket_approval_router(
            service,
            actor_dependency=lambda: current_actor,
            checkpointer=open_approval_checkpointer(db_path),
            approval_store=ApprovalStore(db_path),
        ))
        return TestClient(app)

    ticket = service.prepare_ticket(
        "DEMO-1001", "repair", "设备频繁掉线",
        customer_id=actor.customer_id,
        thread_id=actor.thread_id,
        user_messages=["设备频繁掉线"],
    )["ticket"]
    return boot, service, store, actor, ticket["ticket_id"], db_path


def start(client, ticket_id):
    response = client.post("/ticket-approvals", json={"ticket_id": ticket_id})
    assert response.status_code == 201, response.text
    return response.json()["approval_id"]


def submitted_events(store, ticket_id, customer_id):
    return [e for e in store.events(ticket_id, customer_id) if e["event"] == "submitted"]


def test_pending_approval_survives_restart(env):
    boot, _, store, actor, ticket_id, _ = env
    approval_id = start(boot(), ticket_id)

    restarted = boot()

    status = restarted.get(f"/ticket-approvals/{approval_id}").json()
    assert status["status"] == "pending"

    decision = restarted.post(
        f"/ticket-approvals/{approval_id}/decision", json={"approved": True}
    )
    assert decision.status_code == 200
    assert decision.json()["status"] == "submitted"
    assert store.get_ticket(ticket_id, actor.customer_id)["status"] == "submitted"
    assert len(submitted_events(store, ticket_id, actor.customer_id)) == 1


def test_crash_before_resume_returns_to_pending(env):
    """Claimed, then the process died before the graph was resumed."""
    boot, _, store, actor, ticket_id, db_path = env
    approval_id = start(boot(), ticket_id)
    assert ApprovalStore(db_path).claim(approval_id)

    restarted = boot()

    assert restarted.get(f"/ticket-approvals/{approval_id}").json()["status"] == "pending"
    decision = restarted.post(
        f"/ticket-approvals/{approval_id}/decision", json={"approved": True}
    )
    assert decision.json()["status"] == "submitted"
    assert len(submitted_events(store, ticket_id, actor.customer_id)) == 1


def test_crash_after_submit_is_recorded_not_repeated(env):
    """The graph submitted the ticket, then the process died before saving the outcome."""
    boot, service, store, actor, ticket_id, db_path = env
    approval_id = start(boot(), ticket_id)

    sessions = ApprovalStore(db_path)
    session = sessions.get(approval_id)
    assert sessions.claim(approval_id)
    graph = build_ticket_approval_graph(service, checkpointer=open_approval_checkpointer(db_path))
    graph.invoke(
        Command(resume={"approved": True, "ticket_id": ticket_id}),
        config={"configurable": {"thread_id": session["checkpoint_thread_id"]}},
    )

    restarted = boot()

    status = restarted.get(f"/ticket-approvals/{approval_id}").json()
    assert status == {
        "approval_id": approval_id,
        "ticket_id": ticket_id,
        "status": "completed",
        "workflow_status": "submitted",
    }
    again = restarted.post(f"/ticket-approvals/{approval_id}/decision", json={"approved": True})
    assert again.status_code == 409
    assert len(submitted_events(store, ticket_id, actor.customer_id)) == 1


def test_rejection_survives_restart(env):
    boot, _, store, actor, ticket_id, _ = env
    approval_id = start(boot(), ticket_id)

    decision = boot().post(f"/ticket-approvals/{approval_id}/decision", json={"approved": False})

    assert decision.json()["status"] == "rejected"
    assert boot().get(f"/ticket-approvals/{approval_id}").json()["workflow_status"] == "rejected"
    assert store.get_ticket(ticket_id, actor.customer_id)["status"] == "draft"


def test_other_actor_cannot_see_approval_after_restart(env):
    boot, _, _, actor, ticket_id, _ = env
    approval_id = start(boot(), ticket_id)

    stranger = ActorContext(customer_id=actor.customer_id, thread_id="another-thread")
    response = boot(stranger).get(f"/ticket-approvals/{approval_id}")

    assert response.status_code == 404


def test_only_one_claim_wins(tmp_path):
    first = ApprovalStore(tmp_path / "a.db")
    second = ApprovalStore(tmp_path / "a.db")  # a second process
    first.create(
        "ap-1", customer_id="c", thread_id="t", ticket_id="TKT-1", checkpoint_thread_id="cp-1"
    )

    assert first.claim("ap-1") is True
    assert second.claim("ap-1") is False
