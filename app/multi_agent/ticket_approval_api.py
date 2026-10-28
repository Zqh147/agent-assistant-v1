"""Demo-only FastAPI router for cross-request LangGraph ticket approval.

Security contract: actor_dependency MUST return a server-authenticated ActorContext.
The browser never selects customer_id, business thread_id, or checkpoint thread_id.

Durability: pass a file-backed ``approval_store`` and ``checkpointer`` (see
``approval_store.py``) and a pending approval survives a process restart.
Without them, both default to in-memory storage for tests.
"""
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from langgraph.types import Command

from .approval_store import ApprovalStore
from .ticket_approval_graph import build_ticket_approval_graph


FINAL_STATUSES = frozenset({
    "submitted", "already_submitted", "rejected",
    "not_found", "revalidation_required", "invalid_ticket_status",
})


@dataclass(frozen=True)
class ActorContext:
    customer_id: str
    thread_id: str


class StartApprovalRequest(BaseModel):
    ticket_id: str


class DecisionRequest(BaseModel):
    approved: bool


def create_ticket_approval_router(
    service,
    *,
    actor_dependency: Callable,
    checkpointer=None,
    approval_store: ApprovalStore | None = None,
) -> APIRouter:
    """Build a router. Mount once on a single long-lived FastAPI app instance."""
    graph = build_ticket_approval_graph(service, checkpointer=checkpointer)
    store = approval_store or ApprovalStore()
    router = APIRouter(prefix="/ticket-approvals", tags=["ticket-approvals"])

    def valid_actor(actor: ActorContext) -> ActorContext:
        if (
            not isinstance(actor, ActorContext)
            or not isinstance(actor.customer_id, str)
            or not actor.customer_id.strip()
            or not isinstance(actor.thread_id, str)
            or not actor.thread_id.strip()
        ):
            raise HTTPException(status_code=401, detail="Valid authenticated actor is required")
        return actor

    def config_for(session: dict) -> dict:
        return {"configurable": {"thread_id": session["checkpoint_thread_id"]}}

    def reconcile(session: dict) -> dict:
        """Repair a session left in 'resolving' by a crash or failed request.

        The graph checkpoint is the source of truth: if the graph is still
        paused at the approval interrupt, the decision never took effect and
        the session goes back to pending. Otherwise the graph finished and its
        recorded workflow_status is the outcome.
        """
        if session["phase"] != "resolving":
            return session
        snapshot = graph.get_state(config_for(session))
        if snapshot.next:
            store.release(session["approval_id"])
        else:
            status = (snapshot.values or {}).get("workflow_status")
            store.complete(
                session["approval_id"],
                status if status in FINAL_STATUSES else "unknown",
            )
        return store.get(session["approval_id"])

    def get_session(approval_id: str, actor: ActorContext) -> dict:
        session = store.get(approval_id)
        if (
            session is None
            or session["customer_id"] != actor.customer_id
            or session["thread_id"] != actor.thread_id
        ):
            raise HTTPException(status_code=404, detail="Approval not found")
        return reconcile(session)

    @router.post("", status_code=201)
    def start_approval(
        request: StartApprovalRequest,
        actor: ActorContext = Depends(actor_dependency),
    ):
        actor = valid_actor(actor)
        try:
            found = service.get_ticket(request.ticket_id, customer_id=actor.customer_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid ticket ID") from None
        ticket = found.get("ticket") if found.get("status") == "found" else None
        if not isinstance(ticket, dict) or ticket.get("thread_id") != actor.thread_id:
            raise HTTPException(status_code=404, detail="Ticket not found")
        if ticket.get("status") != "draft":
            raise HTTPException(status_code=409, detail="Ticket is not a draft")

        approval_id = uuid4().hex
        checkpoint_thread_id = "approval-" + uuid4().hex
        initial = {
            "ticket_id": ticket["ticket_id"],
            "customer_id": actor.customer_id,
            "thread_id": actor.thread_id,
        }
        result = graph.invoke(
            initial,
            config={"configurable": {"thread_id": checkpoint_thread_id}},
        )
        interrupts = result.get("__interrupt__", ())
        if not interrupts:
            raise HTTPException(status_code=409, detail="Draft is no longer eligible for approval")
        payload = interrupts[0].value
        if not isinstance(payload, dict) or payload.get("ticket_id") != ticket["ticket_id"]:
            raise HTTPException(status_code=500, detail="Invalid approval checkpoint")

        store.create(
            approval_id,
            customer_id=actor.customer_id,
            thread_id=actor.thread_id,
            ticket_id=ticket["ticket_id"],
            checkpoint_thread_id=checkpoint_thread_id,
        )
        return {"approval_id": approval_id, "status": "pending", "approval": payload}

    @router.get("/{approval_id}")
    def get_approval(
        approval_id: str,
        actor: ActorContext = Depends(actor_dependency),
    ):
        session = get_session(approval_id, valid_actor(actor))
        return {
            "approval_id": approval_id,
            "ticket_id": session["ticket_id"],
            "status": session["phase"],
            "workflow_status": session["final_status"],
        }

    @router.post("/{approval_id}/decision")
    def decide_approval(
        approval_id: str,
        request: DecisionRequest,
        actor: ActorContext = Depends(actor_dependency),
    ):
        session = get_session(approval_id, valid_actor(actor))
        # Atomic pending -> resolving: a second concurrent decision loses here.
        if session["phase"] != "pending" or not store.claim(approval_id):
            raise HTTPException(status_code=409, detail="Approval already resolved")

        try:
            # Server chooses ticket_id from the stored session, never from request body.
            result = graph.invoke(
                Command(resume={"approved": request.approved, "ticket_id": session["ticket_id"]}),
                config=config_for(session),
            )
        except Exception:
            reconcile(store.get(approval_id))
            raise HTTPException(status_code=503, detail="Approval could not be applied; retry") from None

        status = result.get("workflow_status")
        if status not in FINAL_STATUSES:
            reconcile(store.get(approval_id))
            raise HTTPException(status_code=500, detail="Unexpected approval state")
        store.complete(approval_id, status)
        return {
            "approval_id": approval_id,
            "ticket_id": session["ticket_id"],
            "status": status,
        }

    return router
