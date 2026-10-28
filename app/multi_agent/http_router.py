
"""HTTP entrypoint for the demonstration multi-agent workflow."""

import json
import logging
import time
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.support_tools import SupportTools

from .dynamic_supervisor import DynamicSupervisor
from .intent_router import IntentRouter
from .knowledge_agent import KnowledgeAgent
from .order_agent import OrderAgent
from .ticket_agent import TicketAgent
from .supervisor_tools import SupervisorTools


logger = logging.getLogger("uvicorn.error")

TaskType = Literal[
    "order_only",
    "knowledge_only",
    "order_knowledge",
    "order_ticket",
    "order_knowledge_ticket",
]


class MultiAgentChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    # Omit task_type to let the LLM intent router decide.
    task_type: TaskType | None = None
    public_product_scope: dict[str, str] | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_scope(self):
        scope = self.public_product_scope
        required = {
            "product_model",
            "hardware_version",
            "region",
        }

        complete = (
            isinstance(scope, dict)
            and set(scope) == required
            and all(
                isinstance(value, str) and value.strip()
                for value in scope.values()
            )
        )

        if self.task_type == "knowledge_only":
            if not complete:
                raise ValueError(
                    "knowledge_only requires a complete "
                    "public_product_scope"
                )

        elif self.task_type is None:
            # Routed by the LLM: the scope is optional and only used
            # if the router chooses knowledge_only.
            if scope is not None and not complete:
                raise ValueError(
                    "public_product_scope must be complete"
                )

        elif scope is not None:
            raise ValueError(
                "public_product_scope is only allowed "
                "for knowledge_only"
            )

        if complete:
            self.public_product_scope = {
                key: value.strip()
                for key, value in scope.items()
            }

        return self


class MultiAgentChatResponse(BaseModel):
    request_id: str
    thread_id: str
    task_type: TaskType | None
    routing_source: Literal["client", "llm"]
    routing: dict[str, Any] | None = None
    status: str
    summary: str
    ticket_id: str | None = None
    tool_trace: list[dict[str, Any]] = Field(
        default_factory=list
    )
    evidence: list[dict[str, Any]] = Field(
        default_factory=list
    )
    error: str | None = None


def emit_event(level: int, **fields: Any) -> None:
    """Emit structured metadata without user message or credentials."""
    logger.log(
        level,
        json.dumps(
            fields,
            ensure_ascii=False,
            default=str,
        ),
    )


def verify_pending_ticket(
    *,
    service,
    ticket_id: str | None,
    customer_id: str,
    thread_id: str,
) -> None:
    """Verify a claimed draft against the business database."""

    if not isinstance(ticket_id, str) or not ticket_id:
        raise RuntimeError("missing_pending_ticket_id")

    ticket = service.store.get_ticket(
        ticket_id,
        customer_id,
    )

    if (
        not isinstance(ticket, dict)
        or ticket.get("ticket_id") != ticket_id
        or ticket.get("thread_id") != thread_id
        or ticket.get("customer_id") != customer_id
        or ticket.get("status") != "draft"
    ):
        raise RuntimeError("pending_ticket_state_mismatch")


def create_multi_agent_router(agent) -> APIRouter:
    router = APIRouter(
        prefix="/multi-agent",
        tags=["multi-agent"],
    )

    @router.post(
        "/chat",
        response_model=MultiAgentChatResponse,
    )
    def multi_agent_chat(
        request: MultiAgentChatRequest,
        response: Response,
    ) -> MultiAgentChatResponse:

        started = time.perf_counter()
        request_id = uuid4().hex

        response.headers["X-Request-ID"] = request_id

        if (
            agent.settings.support_mode != "demo"
            or agent.support_service is None
        ):
            emit_event(
                logging.WARNING,
                event="request_rejected",
                request_id=request_id,
                reason="demo_service_disabled",
            )
            raise HTTPException(
                status_code=503,
                detail="Multi-agent demo service is disabled",
                headers={"X-Request-ID": request_id},
            )

        # Demo identity comes exclusively from server settings.
        customer_id = agent.settings.demo_customer_id

        try:
            llm = agent.llm
            service = agent.support_service

            task_type = request.task_type
            public_scope = request.public_product_scope
            routing_source = "client"
            routing = None

            if task_type is None:
                routing_source = "llm"
                decision = IntentRouter(llm).route(
                    request.message,
                    public_product_scope=public_scope,
                )
                routing = {
                    "intent": decision.intent,
                    "status": decision.status,
                    "reason": decision.reason,
                    "order_id": decision.order_id,
                    "warnings": decision.warnings,
                }

                if decision.task_type is None:
                    elapsed_ms = round(
                        (time.perf_counter() - started) * 1000,
                        2,
                    )
                    emit_event(
                        logging.INFO,
                        event="request_needs_clarification",
                        request_id=request_id,
                        routing_source=routing_source,
                        intent=decision.intent,
                        duration_ms=elapsed_ms,
                        warnings=decision.warnings,
                    )
                    return MultiAgentChatResponse(
                        request_id=request_id,
                        thread_id=request.thread_id,
                        task_type=None,
                        routing_source=routing_source,
                        routing=routing,
                        status="needs_clarification",
                        summary=decision.clarification,
                    )

                task_type = decision.task_type
                public_scope = (
                    decision.product_scope
                    if task_type == "knowledge_only"
                    else None
                )

            order_agent = OrderAgent(
                llm=llm,
                support_tools=SupportTools(service),
            )

            knowledge_agent = KnowledgeAgent(llm=llm)

            ticket_agent = TicketAgent(
                llm=llm,
                support_tools=SupportTools(service),
            )

            supervisor = DynamicSupervisor(llm=llm)

            tools = SupervisorTools(
                order_agent=order_agent,
                knowledge_agent=knowledge_agent,
                ticket_agent=ticket_agent,
                trusted_context={
                    "customer_id": customer_id,
                    "thread_id": request.thread_id,
                    "user_messages": [request.message],
                },
                public_product_scope=(
                    public_scope
                    if task_type == "knowledge_only"
                    else None
                ),
            )

            result = supervisor.run(
                user_question=request.message,
                tools=tools,
                task_type=task_type,
                max_steps=8,
            )

            data = (
                result.data
                if isinstance(result.data, dict)
                else {}
            )

            ticket_id = data.get("ticket_id")

            # Never trust a claimed draft without a DB read.
            if result.status == "pending_confirmation":
                verify_pending_ticket(
                    service=service,
                    ticket_id=ticket_id,
                    customer_id=customer_id,
                    thread_id=request.thread_id,
                )

            trace = data.get("tool_trace", [])
            evidence = result.evidence or []

            if not isinstance(trace, list):
                raise TypeError("invalid_tool_trace")

            if not isinstance(evidence, list):
                raise TypeError("invalid_evidence")

            elapsed_ms = round(
                (time.perf_counter() - started) * 1000,
                2,
            )

            emit_event(
                logging.INFO,
                event="request_finished",
                request_id=request_id,
                task_type=task_type,
                routing_source=routing_source,
                status=result.status,
                duration_ms=elapsed_ms,
                tool_count=len(trace),
                evidence_count=len(evidence),
            )

            return MultiAgentChatResponse(
                request_id=request_id,
                thread_id=request.thread_id,
                task_type=task_type,
                routing_source=routing_source,
                routing=routing,
                status=result.status,
                summary=result.summary,
                ticket_id=ticket_id,
                tool_trace=trace,
                evidence=evidence,
                error=result.error,
            )

        except Exception as exc:
            elapsed_ms = round(
                (time.perf_counter() - started) * 1000,
                2,
            )

            emit_event(
                logging.ERROR,
                event="request_failed",
                request_id=request_id,
                task_type=request.task_type,
                routing_source=(
                    "client" if request.task_type else "llm"
                ),
                duration_ms=elapsed_ms,
                error_type=type(exc).__name__,
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Multi-agent execution failed. "
                    "Check server logs and verify ticket state."
                ),
                headers={"X-Request-ID": request_id},
            ) from None

    return router
