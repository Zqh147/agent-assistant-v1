from __future__ import annotations

import json
import re
import time
from threading import RLock
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from functools import partial
from typing import Any, TypedDict, Annotated

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
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
from .support_store import SupportStore
from .support_service import SupportService
from .support_tools import SupportTools, SUPPORT_SCHEMAS, SUPPORT_NAMES


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    step_count: int
    tool_trace: list[dict[str, Any]]
    errors: list[str]
    plan: list[dict[str, Any]]
    business_context: dict[str, Any]
    customer_id: str
    conversation_id: str
    confirmed_ticket_id: str | None


class AgentGraph:
    def __init__(self, llm: LLMClient | None = None, session_store: SQLiteSessionStore | None = None, runtime_settings=None, support_service=None) -> None:
        self.settings = runtime_settings or settings
        self._invoke_lock = RLock()
        if self.settings.support_mode not in {"demo", "off"}:
            raise ValueError("SUPPORT_MODE must be demo or off")
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
        self.support_service = None
        self.support_tools = None
        if self.settings.support_mode == "demo":
            self.support_service = support_service or SupportService(
                SupportStore(self.settings.support_db, self.settings.demo_orders_path),
                self.settings.support_rules_path)
            self.support_tools = SupportTools(self.support_service)
            if set(self.tools) & SUPPORT_NAMES:
                raise ValueError("business tools collide with existing registry")
            self.tools.update(self.support_tools.registry())
            self.schemas += SUPPORT_SCHEMAS
            self.sources.update({name: "local" for name in SUPPORT_NAMES})
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
        messages = list(state.get("messages", []))
        if self.support_service is not None:
            context = {**state.get("business_context", {}), "customer_id": state["customer_id"]}
            messages.insert(0, SystemMessage(content="服务端业务上下文：" + json.dumps(context, ensure_ascii=False)))
        response = self.llm.invoke(messages, self.schemas)
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
        business_context = deepcopy(state.get("business_context", {}))
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
                    if name in SUPPORT_NAMES and self.support_tools is not None:
                        # SQLite transactions execute synchronously. A timed-out
                        # background write must never be blindly retried.
                        result = self.support_tools.call(name, args, context={
                            "customer_id": state["customer_id"],
                            "thread_id": state["conversation_id"],
                            "confirmed_ticket_id": state.get("confirmed_ticket_id"),
                            "user_messages": [message.content for message in state.get("messages", [])
                                              if isinstance(message, HumanMessage) and isinstance(message.content, str)]})
                        business_context = self.update_business_context(name, args, result, business_context, state["customer_id"])
                        self.session_store.save_business_context(state["conversation_id"], business_context)
                    else:
                        actual_args = dict(args)
                        if name == "search_knowledge_base" and self.settings.rag_backend == "project1":
                            for key, value in business_context.get("product_scope", {}).items():
                                supplied = actual_args.get(key)
                                if supplied is not None and re.sub(r"\s+", "", str(supplied)).upper() != re.sub(r"\s+", "", value).upper():
                                    raise ValueError("检索参数与已查询订单的设备范围不一致，请核对订单或设备信息。")
                                actual_args[key] = value
                        pool = ThreadPoolExecutor(max_workers=1)
                        try:
                            future = pool.submit(fn, **actual_args)
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
                    if name in SUPPORT_NAMES or not isinstance(exc, ConnectionError):
                        break
                    if attempts < 2:
                        time.sleep(0.1)
            traces.append({"tool": name, "source": self.sources.get(name, "unknown"), "call_id": call_id, "status": status, "attempts": attempts, "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
            if name == "search_knowledge_base":
                traces[-1]["retrieval_backend"] = self.settings.rag_backend
                if status == "success" and isinstance(result, dict):
                    traces[-1]["retrieval_status"] = result.get("status")
                    traces[-1]["evidence_count"] = len(result.get("results", []))
                    traces[-1]["resolved_scope"] = result.get("resolved_scope", {})
            if name in SUPPORT_NAMES and status == "success":
                traces[-1]["business_status"] = result.get("status")
                traces[-1]["arguments"] = args
                if result.get("ticket"):
                    traces[-1]["ticket_id"] = result["ticket"]["ticket_id"]
            plan = settle_tool(plan, call_id, status)
            outputs.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
        return {"messages": outputs, "tool_trace": traces, "errors": errors, "plan": plan, "business_context": business_context}

    def route_after_tools(self, state: AgentState) -> str:
        return "llm"

    def update_business_context(self, name, args, result, context, customer_id):
        if name == "lookup_order":
            if result.get("status") == "found":
                if context.get("order_id") != result["order"]["order_id"]:
                    for key in ("pending_ticket_id", "latest_ticket_id", "service_type"):
                        context.pop(key, None)
                context.update(order_id=result["order"]["order_id"], product_scope=result["product_scope"])
            else:
                for key in ("order_id", "product_scope", "pending_ticket_id", "latest_ticket_id"):
                    context.pop(key, None)
        ticket = result.get("ticket")
        if ticket:
            found = self.support_service.lookup_order(ticket["order_id"], customer_id=customer_id)
            context.update(order_id=ticket["order_id"], service_type=ticket["service_type"],
                           latest_ticket_id=ticket["ticket_id"], product_scope=found["product_scope"])
            if ticket["status"] == "draft":
                context["pending_ticket_id"] = ticket["ticket_id"]
            else:
                context.pop("pending_ticket_id", None)
        if name == "search_after_sales_rules":
            context["service_type"] = args["service_type"]
        return context

    def invoke(self, message: str, thread_id: str = "default", customer_id=None) -> dict[str, Any]:
        with self._invoke_lock:
            return self._invoke(message, thread_id, customer_id)

    def _invoke(self, message, thread_id, customer_id):
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        # Bootstrap from SQLite only when no in-process checkpoint exists.
        previous = self.session_store.load(thread_id) if (not snapshot.values or self.support_service is not None) else None
        business_context = deepcopy((previous or {}).get("business_context", {}))
        actor = customer_id or self.settings.demo_customer_id
        confirmation = None
        if self.support_service is not None:
            actor = self.support_service.identifier(actor, "customer_id")
            if business_context.get("customer_id", actor) != actor:
                raise ValueError("当前会话已绑定另一演示客户，请使用新的 thread_id。")
            business_context["customer_id"] = actor
            confirmation = confirmed_ticket(message, business_context)
            self.session_store.save_business_context(thread_id, business_context)
        history = []
        if previous and not snapshot.values:
            for item in previous["messages"][-12:]:
                if item.get("role") == "user":
                    history.append(HumanMessage(content=item.get("content", "")))
                elif item.get("role") == "assistant" and item.get("content"):
                    history.append(AIMessage(content=item.get("content", "")))
        result = self.graph.invoke({"messages": history + [HumanMessage(content=message)], "step_count": 0, "tool_trace": [], "errors": [],
            "business_context": business_context, "customer_id": actor,
            "conversation_id": thread_id, "confirmed_ticket_id": confirmation}, config=config)
        last = result["messages"][-1]
        answer = getattr(last, "content", str(last))
        persisted = []
        for item in result["messages"]:
            item_type = getattr(item, "type", "")
            # Persist complete user/answer turns, not assistant tool-call stubs.
            if item_type == "human" or (item_type == "ai" and not latest_tool_calls(item)):
                persisted.append({"role": "user" if item_type == "human" else "assistant", "content": getattr(item, "content", "")})
        self.session_store.save(thread_id, persisted[-24:], result.get("tool_trace", []), result.get("errors", []), business_context=result.get("business_context", {}))
        errors = result.get("errors", [])
        status = "stopped" if "max_steps_exceeded" in errors else ("completed_with_errors" if errors else "completed")
        return {"answer": answer, "thread_id": thread_id, "tool_trace": result.get("tool_trace", []), "errors": errors, "message_count": len(persisted), "plan": result["plan"], "status": status, "business_context": result.get("business_context", {}), "business_mode": self.settings.support_mode, "online": bool(getattr(self.llm, "online", False))}



def confirmed_ticket(message, context):
    """Only a literal current user confirmation authorizes the pending ticket.

    The model cannot pass a boolean or a customer ID to bypass this boundary.
    """
    text = message.strip().rstrip("。！!")
    match = re.fullmatch(r"确认提交(?:工单)?(?:\s+(TKT-[A-Za-z0-9]+))?", text)
    if not match:
        return None
    return match.group(1).upper() if match.group(1) else context.get("pending_ticket_id", context.get("latest_ticket_id"))
