"""Demo-only FastAPI router for cross-request LangGraph ticket approval.

Security contract: actor_dependency MUST return a server-authenticated ActorContext.
The browser never selects customer_id, business thread_id, or checkpoint thread_id.
InMemorySaver and this registry are single-process only: not production durable.
"""
from dataclasses import dataclass
from threading import Lock
from typing import Callable
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from langgraph.types import Command

from .ticket_approval_graph import build_ticket_approval_graph


@dataclass(frozen=True)
class ActorContext:
    customer_id: str
    thread_id: str


class StartApprovalRequest(BaseModel):
    ticket_id: str


class DecisionRequest(BaseModel):
    approved: bool


class ApprovalSession:
    def __init__(self, actor: ActorContext, ticket_id: str, checkpoint_thread_id: str):
        self.actor = actor
        self.ticket_id = ticket_id
        self.checkpoint_thread_id = checkpoint_thread_id
        self.phase = "pending"
        self.final_status = None
        self.lock = Lock()


def create_ticket_approval_router(
    service,
    *,
    actor_dependency: Callable,
    checkpointer=None,
) -> APIRouter:
    """Build a router. Mount once on a single long-lived FastAPI app instance."""
    graph = build_ticket_approval_graph(service, checkpointer=checkpointer)
    router = APIRouter(prefix="/ticket-approvals", tags=["ticket-approvals"])
    sessions: dict[str, ApprovalSession] = {}
    registry_lock = Lock()

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

    def get_session(approval_id: str, actor: ActorContext) -> ApprovalSession:
        with registry_lock:
            session = sessions.get(approval_id)
        if session is None or session.actor != actor:
            raise HTTPException(status_code=404, detail="Approval not found")
        return session

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
        config = {"configurable": {"thread_id": checkpoint_thread_id}}
        initial = {
            "ticket_id": ticket["ticket_id"],
            "customer_id": actor.customer_id,
            "thread_id": actor.thread_id,
        }
        result = graph.invoke(initial, config=config)
        interrupts = result.get("__interrupt__", ())
        if not interrupts:
            raise HTTPException(status_code=409, detail="Draft is no longer eligible for approval")
        payload = interrupts[0].value
        if not isinstance(payload, dict) or payload.get("ticket_id") != ticket["ticket_id"]:
            raise HTTPException(status_code=500, detail="Invalid approval checkpoint")

        session = ApprovalSession(actor, ticket["ticket_id"], checkpoint_thread_id)
        with registry_lock:
            sessions[approval_id] = session
        return {"approval_id": approval_id, "status": "pending", "approval": payload}

    @router.get("/{approval_id}")
    def get_approval(
        approval_id: str,
        actor: ActorContext = Depends(actor_dependency),
    ):
        session = get_session(approval_id, valid_actor(actor))
        with session.lock:
            return {
                "approval_id": approval_id,
                "ticket_id": session.ticket_id,
                "status": session.phase,
                "workflow_status": session.final_status,
            }

    @router.post("/{approval_id}/decision")
    def decide_approval(
        approval_id: str,
        request: DecisionRequest,
        actor: ActorContext = Depends(actor_dependency),
    ):
        session = get_session(approval_id, valid_actor(actor))
        with session.lock:
            if session.phase != "pending":
                raise HTTPException(status_code=409, detail="Approval already resolved")
            config = {"configurable": {"thread_id": session.checkpoint_thread_id}}
            # Server chooses ticket_id from the stored session, never from request body.
            result = graph.invoke(
                Command(resume={"approved": request.approved, "ticket_id": session.ticket_id}),
                config=config,
            )
            status = result.get("workflow_status")
            if status not in {
                "submitted", "already_submitted", "rejected",
                "not_found", "revalidation_required", "invalid_ticket_status",
            }:
                raise HTTPException(status_code=500, detail="Unexpected approval state")
            session.phase = "completed"
            session.final_status = status
            return {
                "approval_id": approval_id,
                "ticket_id": session.ticket_id,
                "status": status,
            }

    return router
