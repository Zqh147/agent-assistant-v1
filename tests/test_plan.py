from dataclasses import replace

from langchain_core.messages import AIMessage

from app import graph as graph_module
from app.graph import AgentGraph


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)

    def invoke(self, messages, tools):
        return next(self.responses)


def calculate_call(call_id, a, b, operation):
    return AIMessage(content="", tool_calls=[{
        "id": call_id, "name": "calculate", "type": "tool_call",
        "args": {"a": a, "b": b, "operation": operation},
    }])


def test_plan_tracks_two_sequential_tool_steps():
    llm = ScriptedLLM(calculate_call("first", 23, 47, "multiply"),
                      calculate_call("second", 1081, 2, "multiply"),
                      AIMessage(content="2162"))
    result = AgentGraph(llm=llm).invoke("计算23乘47再乘2", "multi-step")
    steps = [step for step in result["plan"] if step.get("tool")]
    assert [s["call_id"] for s in steps] == ["first", "second"]
    assert all(s["status"] == "completed" for s in steps)
    assert [s["call_id"] for s in steps] == [t["call_id"] for t in result["tool_trace"]]
    assert result["plan"][-1]["status"] == "completed"


def test_failed_tool_is_not_marked_completed():
    llm = ScriptedLLM(calculate_call("divide-zero", 1, 0, "divide"),
                      AIMessage(content="除零失败"))
    result = AgentGraph(llm=llm).invoke("计算1除0", "failed")
    assert result["plan"][1]["status"] == "failed"
    assert result["tool_trace"][0]["status"] == "error"
    assert result["status"] == "completed_with_errors"


def test_step_limit_skips_final_answer(monkeypatch):
    monkeypatch.setattr(graph_module, "settings", replace(graph_module.settings, max_steps=1))
    result = AgentGraph(llm=ScriptedLLM(calculate_call("limit", 2, 3, "add"))).invoke("计算", "limit")
    assert result["status"] == "stopped"
    assert result["plan"][-1]["status"] == "skipped"
    assert "max_steps_exceeded" in result["errors"]


def test_direct_answer_has_no_tool_steps():
    result = AgentGraph(llm=ScriptedLLM(AIMessage(content="你好"))).invoke("你好", "direct-plan")
    assert len(result["plan"]) == 2
    assert not any(step.get("tool") for step in result["plan"])
