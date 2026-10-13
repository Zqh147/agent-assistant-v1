import json
from dataclasses import replace

import pytest

from app import evaluation
from app.evaluation_cases import CASES


def test_offline_isolates_key_database_and_writes_report(tmp_path, monkeypatch):
    user_db = tmp_path / "user.db"
    user_db.write_bytes(b"existing user database")
    monkeypatch.setattr(evaluation, "settings", replace(evaluation.settings, api_key="dummy-online-key", session_db=user_db))
    from app.llm_client import LLMClient
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline evaluation must not construct the API client")
    monkeypatch.setattr(LLMClient, "__init__", forbidden)
    target = tmp_path / "report.json"
    report = evaluation.evaluate(output=target)
    assert report["summary"]["passed"] == 8
    assert user_db.read_bytes() == b"existing user database"
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 8
    assert "dummy-online-key" not in target.read_text(encoding="utf-8")
    assert saved["model"] == "scripted_fixture"


def test_correct_answer_without_required_tool_fails():
    result = {"answer": "1081", "tool_trace": [], "plan": [], "errors": [], "status": "completed"}
    checks = evaluation.assess(CASES[0], result, "local")
    assert checks["answer"]
    assert not checks["tool_sequence"]
    assert not checks["arguments"]


def test_online_requires_key(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "settings", replace(evaluation.settings, api_key=""))
    with pytest.raises(ValueError, match="requires DEEPSEEK_API_KEY"):
        evaluation.evaluate(mode="online", output=tmp_path / "report.json")


def test_exception_is_reported_and_cli_returns_failure(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("sensitive-provider-body")
    monkeypatch.setattr(evaluation.AgentGraph, "invoke", broken)
    target = tmp_path / "failed.json"
    monkeypatch.setattr("sys.argv", ["evaluation", "--case", "multiply", "--output", str(target)])
    assert evaluation.main() == 1
    result = json.loads(target.read_text(encoding="utf-8"))["results"][0]
    assert result["exception_type"] == "RuntimeError"
    assert result["failures"] == ["execution_exception"]
    assert "sensitive-provider-body" not in target.read_text(encoding="utf-8")


def test_offline_repeated_runs_are_isolated(tmp_path):
    first = evaluation.evaluate(case_ids=["session_restore"], output=tmp_path / "one.json")
    second = evaluation.evaluate(case_ids=["session_restore"], output=tmp_path / "two.json")
    assert first["run_id"] != second["run_id"]
    assert first["results"][0]["result"]["message_count"] == 4
    assert second["results"][0]["result"]["message_count"] == 4
