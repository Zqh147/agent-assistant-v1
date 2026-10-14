from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, field_validator

from .graph import AgentGraph

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
        return agent.invoke(request.message, request.thread_id, customer_id=request.customer_id)
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
