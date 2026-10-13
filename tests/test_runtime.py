from dataclasses import replace
from threading import Event
from time import perf_counter
from types import SimpleNamespace

from app import graph as module
from app.graph import AgentGraph
from app.mcp_registry import MCPRegistry
from app.tools import calculate


def test_zero_operand_for_non_division():
    assert calculate(10, 0, "add")["result"] == 10
    assert calculate(10, 0, "multiply")["result"] == 0


def test_transient_connection_retries_once():
    system = AgentGraph()
    calls = []
    def flaky(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ConnectionError("temporary disconnect")
        return {"result": 1081}
    system.tools["calculate"] = flaky
    result = system.invoke("23 * 47", "retry")
    assert result["tool_trace"][0]["attempts"] == 2
    assert result["tool_trace"][0]["status"] == "success"


def test_parameter_error_does_not_retry():
    system = AgentGraph()
    def invalid(**kwargs):
        raise ValueError("bad argument")
    system.tools["calculate"] = invalid
    result = system.invoke("23 * 47", "invalid")
    assert result["tool_trace"][0]["attempts"] == 1
    assert result["plan"][1]["status"] == "failed"


def test_timeout_returns_without_waiting_for_worker(monkeypatch):
    monkeypatch.setattr(module, "settings", replace(module.settings, tool_timeout_seconds=0.02))
    release = Event()
    system = AgentGraph()
    system.tools["calculate"] = lambda **kwargs: release.wait(2)
    start = perf_counter()
    try:
        result = system.invoke("23 * 47", "timeout")
        assert perf_counter() - start < 1
        assert result["tool_trace"][0]["attempts"] == 1
        assert result["tool_trace"][0]["status"] == "error"
    finally:
        release.set()


def test_dynamic_registry_accepts_new_server_tool():
    class Server:
        def list_tools(self):
            return [SimpleNamespace(name="echo", description="Echo", inputSchema={"type": "object"})]
        def call_tool(self, name, args):
            return {"name": name, "args": args}
    registry = MCPRegistry(Server())
    registry.discover()
    assert registry.schemas[0]["function"]["name"] == "echo"
    assert registry.callables["echo"](text="hello") == {"name": "echo", "args": {"text": "hello"}}


def test_real_stdio_agent_and_server_error(monkeypatch):
    from app.mcp_client import MCPToolError, StdioMCPClient
    import pytest
    monkeypatch.setattr(module, "settings", replace(module.settings, tool_transport="mcp"))
    system = AgentGraph()
    result = system.invoke("23 * 47", "mcp-real")
    assert "1081" in result["answer"]
    assert result["tool_trace"][0]["source"] == "mcp"
    assert result["tool_trace"][0]["status"] == "success"
    with pytest.raises(MCPToolError, match="cannot divide by zero"):
        StdioMCPClient().call_tool("calculate", {"a": 10, "b": 0, "operation": "divide"})
