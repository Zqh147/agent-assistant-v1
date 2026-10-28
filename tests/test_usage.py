"""Per-request LLM call and token accounting."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import HumanMessage

from app import graph
from app.llm_client import LLMClient
from app.usage import UsageMeter, record_llm_call, track_usage


def test_meter_sums_reported_usage():
    meter = UsageMeter()
    meter.add(SimpleNamespace(prompt_tokens=100, completion_tokens=20))
    meter.add(SimpleNamespace(prompt_tokens=50, completion_tokens=5))

    assert meter.to_dict() == {
        "llm_calls": 2,
        "prompt_tokens": 150,
        "completion_tokens": 25,
        "tokens_complete": True,
        "total_tokens": 175,
    }


def test_missing_usage_is_flagged_not_guessed():
    meter = UsageMeter()
    meter.add(None)

    assert meter.llm_calls == 1
    assert meter.total_tokens == 0
    assert meter.tokens_complete is False


def test_record_outside_request_is_ignored():
    record_llm_call(SimpleNamespace(prompt_tokens=1, completion_tokens=1))


def test_meters_do_not_leak_between_requests():
    with track_usage() as first:
        record_llm_call(None)
    with track_usage() as second:
        pass

    assert first.llm_calls == 1
    assert second.llm_calls == 0


def test_online_client_reports_provider_usage():
    client = LLMClient()
    create = Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=321, completion_tokens=12),
    ))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with track_usage() as usage:
        client.invoke([HumanMessage(content="hi")], [])

    assert usage.llm_calls == 1
    assert usage.prompt_tokens == 321
    assert usage.completion_tokens == 12


def test_single_agent_graph_calls_are_counted(tmp_path, monkeypatch):
    """LangGraph may run nodes on worker threads; the meter must still see them."""
    runtime = replace(
        graph.settings,
        session_db=tmp_path / "sessions.db",
        support_mode="off",
        tool_transport="local",
        rag_backend="demo",
    )
    agent = graph.AgentGraph(runtime_settings=runtime)

    with track_usage() as usage:
        result = agent.invoke("请计算 23 * 47", thread_id="usage-test")

    assert result["tool_trace"][0]["tool"] == "calculate"
    # One call that requests the tool, one that writes the answer.
    assert usage.llm_calls == 2
