import os
import hmac

from fastapi import Header

from .multi_agent.ticket_approval_api import (
    ActorContext,
    create_ticket_approval_router,
)

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, field_validator

from .graph import AgentGraph
from .usage import track_usage

agent = AgentGraph()

class UTF8JSONResponse(JSONResponse):
    # Explicit charset also supports legacy Windows PowerShell clients.
    media_type = "application/json; charset=utf-8"


app = FastAPI(title="网络设备售后咨询与工单处理 Agent", version="2.0.0", default_response_class=UTF8JSONResponse)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="default", min_length=1, max_length=100)
    customer_id: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("message", "thread_id", "customer_id")
    @classmethod
    def nonblank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value.strip() if value is not None else None


@app.exception_handler(HTTPException)
async def http_error(request, exc):
    return UTF8JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    return UTF8JSONResponse({"detail": jsonable_encoder(exc.errors())}, status_code=422)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "online": agent.llm.online, "rag_backend": agent.settings.rag_backend,
            "support_mode": agent.settings.support_mode, "tool_transport": agent.settings.tool_transport}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        with track_usage() as usage:
            result = agent.invoke(request.message, request.thread_id, customer_id=request.customer_id)
        return {**result, "usage": usage.to_dict()}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=503, detail="Agent 执行失败；请保留会话号并检查本机日志，勿假定工单已经提交。")


@app.get("/threads")
def threads() -> list[dict[str, str]]:
    return agent.session_store.list_threads()

@app.get("/tools")
def tools() -> list[str]:
    return sorted(agent.tools.keys())


@app.get("/demo/orders")
def demo_orders():
    if agent.support_service is None:
        raise HTTPException(status_code=404, detail="演示售后模块未启用")
    return {"data_kind": "fictional_demo_orders", "customer_id": agent.settings.demo_customer_id,
            "orders": agent.support_service.store.list_orders(agent.settings.demo_customer_id)}



# 仅用于本地开发演示的 HITL 审批接口。
# 默认不启用；生产环境必须替换为真实身份认证和会话管理。
if os.getenv("ENABLE_DEMO_TICKET_APPROVALS") == "1":

    approval_token = os.getenv("APPROVAL_DEMO_TOKEN", "")
    approval_thread_id = os.getenv("APPROVAL_DEMO_THREAD_ID", "")
    approval_customer_id = agent.settings.demo_customer_id

    if not approval_token or len(approval_token) < 32:
        raise RuntimeError(
            "APPROVAL_DEMO_TOKEN must contain at least 32 characters"
        )

    if not approval_thread_id.strip():
        raise RuntimeError(
            "APPROVAL_DEMO_THREAD_ID is required"
        )

    if agent.support_service is None:
        raise RuntimeError(
            "SupportService must be enabled for ticket approval"
        )

    def get_demo_approval_actor(
        x_approval_token: str | None = Header(
            default=None,
            alias="X-Approval-Token",
        ),
    ) -> ActorContext:
        if (
            x_approval_token is None
            or not hmac.compare_digest(
                x_approval_token,
                approval_token,
            )
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid approval credentials",
            )

        return ActorContext(
            customer_id=approval_customer_id,
            thread_id=approval_thread_id,
        )

    app.include_router(
        create_ticket_approval_router(
            service=agent.support_service,
            actor_dependency=get_demo_approval_actor,
        )
    )

from .multi_agent.http_router import create_multi_agent_router

app.include_router(create_multi_agent_router(agent))