from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .graph import AgentGraph

agent = AgentGraph()

app = FastAPI(title="Agent Assistant V1.0", version="1.0.0")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="default", min_length=1, max_length=100)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    return agent.invoke(request.message, request.thread_id)


@app.get("/threads")
def threads() -> list[dict[str, str]]:
    return agent.session_store.list_threads()

@app.get("/tools")
def tools() -> list[str]:
    return sorted(agent.tools.keys())
