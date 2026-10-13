from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from functools import partial
from typing import Any, TypedDict, Annotated

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import settings
from .llm_client import LLMClient
from .messages import latest_tool_calls
from .rag_tool import rag_tool_schema, search_knowledge_base
from .tools import TOOL_SCHEMAS, serialize_result, tool_map
from .storage import SQLiteSessionStore
from .planner import build_plan, schedule_tools, settle_tool, finish_plan
from .mcp_registry import MCPRegistry
from .mcp_client import StdioMCPClient


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    step_count: int
    tool_trace: list[dict[str, Any]]
    errors: list[str]
    plan: list[dict[str, Any]]


class AgentGraph:
    def __init__(self, llm: LLMClient | None = None, session_store: SQLiteSessionStore | None = None, runtime_settings=None) -> None:
        self.settings = runtime_settings or settings
        self.llm = llm or LLMClient(runtime_settings=self.settings)
        self.session_store = session_store or SQLiteSessionStore(self.settings.session_db)
        knowledge_tool = partial(search_knowledge_base, backend=self.settings.rag_backend,
                                 service_url=self.settings.rag_service_url,
                                 request_timeout=self.settings.rag_timeout_seconds)
        if self.settings.tool_transport == "mcp":
            registry = MCPRegistry(StdioMCPClient(timeout=self.settings.tool_timeout_seconds))
            registry.discover()
            if "search_knowledge_base" in registry.callables:
                raise ValueError("MCP tool collides with local search_knowledge_base")
            self.tools = {**registry.callables, "search_knowledge_base": knowledge_tool}
            self.schemas = registry.schemas + [rag_tool_schema()]
            self.sources = {name: "mcp" for name in registry.callables}
        elif self.settings.tool_transport == "local":
            self.tools = tool_map({"search_knowledge_base": knowledge_tool})
            self.schemas = TOOL_SCHEMAS + [rag_tool_schema()]
            self.sources = {name: "local" for name in self.tools}
        else:
            raise ValueError("TOOL_TRANSPORT must be local or mcp")
        self.sources["search_knowledge_base"] = "local"
        builder = StateGraph(AgentState)
        builder.add_node("plan", self.plan_node)
        builder.add_node("llm", self.llm_node)
        builder.add_node("tools", self.tool_node)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "llm")
        builder.add_conditional_edges("llm", self.route_after_llm, {"tools": "tools", "end": END})
        builder.add_conditional_edges("tools", self.route_after_tools, {"llm": "llm", "end": END})
        self.graph = builder.compile(checkpointer=InMemorySaver())

    def plan_node(self, state: AgentState) -> dict[str, Any]:
        return {"plan": build_plan(str(state["messages"][-1].content))}

    def llm_node(self, state: AgentState) -> dict[str, Any]:
        if state.get("step_count", 0) >= self.settings.max_steps:
            return {"messages": [AIMessage(content="任务达到最大执行步数，已安全停止。")], "errors": state.get("errors", []) + ["max_steps_exceeded"], "plan": finish_plan(state["plan"], stopped=True)}
        response = self.llm.invoke(state.get("messages", []), self.schemas)
        calls = latest_tool_calls(response)
        plan = schedule_tools(state["plan"], calls) if calls else finish_plan(state["plan"])
        return {"messages": [response], "step_count": state.get("step_count", 0) + 1, "plan": plan}

    def route_after_llm(self, state: AgentState) -> str:
        return "tools" if latest_tool_calls(state.get("messages", [])[-1]) else "end"

    def tool_node(self, state: AgentState) -> dict[str, Any]:
        last = state.get("messages", [])[-1]
        outputs: list[ToolMessage] = []
        traces = list(state.get("tool_trace", []))
        errors = list(state.get("errors", []))
        plan = state.get("plan", [])
        for call in latest_tool_calls(last):
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            call_id = call.get("id", name)
            started = time.perf_counter()
            status = "error"
            content = ""
            attempts = 0
            for attempts in range(1, 3):
                try:
                    fn = self.tools.get(name)
                    if fn is None:
                        raise ValueError(f"unknown tool: {name}")
                    pool = ThreadPoolExecutor(max_workers=1)
                    try:
                        future = pool.submit(fn, **args)
                        result = future.result(timeout=self.settings.tool_timeout_seconds)
                    finally:
                        pool.shutdown(wait=False, cancel_futures=True)
                    content = serialize_result(result)
                    status = "success"
                    break
                except FutureTimeout:
                    content = json.dumps({"error": "tool timeout", "tool": name}, ensure_ascii=False)
                    errors.append(f"{name}: timeout (attempt {attempts})")
                    # A timed-out thread may still run; do not duplicate its work.
                    break
                except Exception as exc:
                    content = json.dumps({"error": str(exc), "tool": name}, ensure_ascii=False)
                    errors.append(f"{name}: {exc} (attempt {attempts})")
                    if not isinstance(exc, ConnectionError):
                        break
                    if attempts < 2:
                        time.sleep(0.1)
            traces.append({"tool": name, "source": self.sources.get(name, "unknown"), "call_id": call_id, "status": status, "attempts": attempts, "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
            if name == "search_knowledge_base":
                traces[-1]["retrieval_backend"] = self.settings.rag_backend
            plan = settle_tool(plan, call_id, status)
            outputs.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
        return {"messages": outputs, "tool_trace": traces, "errors": errors, "plan": plan}

    def route_after_tools(self, state: AgentState) -> str:
        return "llm"

    def invoke(self, message: str, thread_id: str = "default") -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        # Bootstrap from SQLite only when no in-process checkpoint exists.
        previous = self.session_store.load(thread_id) if not snapshot.values else None
        history = []
        if previous:
            for item in previous["messages"][-12:]:
                if item.get("role") == "user":
                    history.append(HumanMessage(content=item.get("content", "")))
                elif item.get("role") == "assistant" and item.get("content"):
                    history.append(AIMessage(content=item.get("content", "")))
        result = self.graph.invoke({"messages": history + [HumanMessage(content=message)], "step_count": 0, "tool_trace": [], "errors": []}, config=config)
        last = result["messages"][-1]
        answer = getattr(last, "content", str(last))
        persisted = []
        for item in result["messages"]:
            item_type = getattr(item, "type", "")
            # Persist complete user/answer turns, not assistant tool-call stubs.
            if item_type == "human" or (item_type == "ai" and not latest_tool_calls(item)):
                persisted.append({"role": "user" if item_type == "human" else "assistant", "content": getattr(item, "content", "")})
        self.session_store.save(thread_id, persisted[-24:], result.get("tool_trace", []), result.get("errors", []))
        errors = result.get("errors", [])
        status = "stopped" if "max_steps_exceeded" in errors else ("completed_with_errors" if errors else "completed")
        return {"answer": answer, "thread_id": thread_id, "tool_trace": result.get("tool_trace", []), "errors": errors, "message_count": len(persisted), "plan": result["plan"], "status": status}

