from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

from langchain_core.messages import AIMessage

from .config import settings, ROOT
from .evaluation_cases import CASES
from .graph import AgentGraph
from .storage import SQLiteSessionStore


class FixtureModel:
    """Known tool decisions exercise execution, not offline model intelligence."""
    def __init__(self, calls, restore_previous=False):
        self.calls = list(calls)
        self.index = 0
        self.restore_previous = restore_previous

    def invoke(self, messages, tools):
        if self.index < len(self.calls):
            name, arguments = self.calls[self.index]
            arguments = dict(arguments)
            if self.restore_previous and self.index == 0:
                # Read the prior answer actually restored from SQLite.
                answers = [m.content for m in messages if m.type == "ai" and m.content]
                if not answers:
                    raise AssertionError("No assistant history restored")
                prior = json.loads(answers[-1])
                arguments["a"] = prior["result"]
            self.index += 1
            return AIMessage(content="", tool_calls=[{
                "id": f"fixture-{uuid4().hex}", "name": name,
                "args": arguments, "type": "tool_call",
            }])
        if messages[-1].type == "tool":
            text = str(messages[-1].content)
            payload = json.loads(text)
            # MCP may return text content blocks rather than structuredContent.
            if isinstance(payload, dict) and isinstance(payload.get("content"), list):
                text = "\n".join(block.get("text", "") for block in payload["content"])
            return AIMessage(content=text)
        return AIMessage(content="你好")


def assess(case, result, transport):
    trace = result.get("tool_trace", [])
    expected_names = [name for name, _ in case.calls]
    tool_steps = [step for step in result.get("plan", []) if step.get("tool")]
    expected_status = "error" if case.expect_error else "success"
    expected_step = "failed" if case.expect_error else "completed"
    arguments_ok = len(tool_steps) == len(case.calls)
    for step, (name, arguments) in zip(tool_steps, case.calls):
        actual = step.get("arguments", {})
        if name == "calculate":
            arguments_ok = arguments_ok and all(actual.get(k) == v for k, v in arguments.items())
        elif name == "search_knowledge_base":
            arguments_ok = arguments_ok and bool(actual.get("question"))
    checks = {
        "answer": bool(re.search(case.answer_pattern, str(result.get("answer", "")), re.I)),
        "tool_sequence": [t.get("tool") for t in trace] == expected_names,
        "arguments": bool(arguments_ok),
        "tool_status": all(t.get("status") == expected_status for t in trace),
        "attempts": all(t.get("attempts") == 1 for t in trace),
        "source": all(t.get("source") == ("local" if t.get("tool") == "search_knowledge_base" else transport) for t in trace),
        "error_record": bool(result.get("errors")) == case.expect_error,
        "run_status": result.get("status") == ("completed_with_errors" if case.expect_error else "completed"),
        "plan_trace_match": len(tool_steps) == len(trace) and all(
            s.get("call_id") == t.get("call_id") and s.get("tool") == t.get("tool") and s.get("status") == expected_step
            for s, t in zip(tool_steps, trace)),
    }
    return checks


def evaluate(mode="offline", transport="local", output=None, case_ids=None):
    if mode not in {"offline", "online"} or transport not in {"local", "mcp"}:
        raise ValueError("Invalid evaluation mode or transport")
    if mode == "online" and not settings.api_key:
        raise ValueError("Online evaluation requires DEEPSEEK_API_KEY; no offline fallback is allowed")
    selected = [c for c in CASES if case_ids is None or c.id in case_ids]
    if not selected or (case_ids and set(case_ids) - {c.id for c in CASES}):
        raise ValueError("Unknown or empty case selection")
    run_id = uuid4().hex[:12]
    started = time.perf_counter()
    records = []
    with TemporaryDirectory(prefix="agent-eval-") as temp:
        cfg = replace(settings, api_key="" if mode == "offline" else settings.api_key,
                      rag_backend="demo" if mode == "offline" else settings.rag_backend,
                      tool_transport=transport, session_db=Path(temp) / "sessions.db", support_mode="off")
        store = SQLiteSessionStore(cfg.session_db)
        for case in selected:
            tick = time.perf_counter()
            record = {"id": case.id, "question": case.question,
                      "expected": {"tools": [name for name, _ in case.calls],
                                   "arguments": [args for _, args in case.calls],
                                   "answer_pattern": case.answer_pattern, "tool_error": case.expect_error},
                      "setup_result": None, "result": None, "checks": {}, "failures": []}
            try:
                thread_id = f"{run_id}-{case.id}"
                if case.setup_question:
                    setup_llm = FixtureModel(case.setup_calls) if mode == "offline" else None
                    initial = AgentGraph(llm=setup_llm, session_store=store, runtime_settings=cfg)
                    record["setup_result"] = initial.invoke(case.setup_question, thread_id)
                    del initial
                llm = FixtureModel(case.calls, restore_previous=bool(case.setup_question)) if mode == "offline" else None
                # A new graph creates a new in-memory saver, exercising DB restore.
                system = AgentGraph(llm=llm, session_store=store, runtime_settings=cfg)
                result = system.invoke(case.question, thread_id)
                record["result"] = result
                record["checks"] = assess(case, result, transport)
                if case.setup_question:
                    setup_case = replace(case, calls=case.setup_calls, answer_pattern=r"(?<!\d)1081(?!\d)")
                    record["checks"]["setup"] = all(assess(setup_case, record["setup_result"], transport).values())
                    saved = store.load(thread_id)["messages"]
                    record["checks"]["history_restored"] = any(m["role"] == "user" and m["content"] == case.setup_question for m in saved)
                    record["checks"]["message_count"] = result["message_count"] == 4
                else:
                    record["checks"]["message_count"] = result["message_count"] == 2
                record["failures"] = [key for key, passed in record["checks"].items() if not passed]
            except Exception as exc:
                # Do not dump HTTP headers, credentials or arbitrary provider bodies.
                record["exception_type"] = type(exc).__name__
                record["failures"] = ["execution_exception"]
            record["passed"] = not record["failures"]
            record["latency_ms"] = round((time.perf_counter() - tick) * 1000, 2)
            records.append(record)
    passed = sum(r["passed"] for r in records)
    tool_count = sum(len((r["result"] or {}).get("tool_trace", [])) for r in records)
    report = {"schema_version": 1, "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
              "mode": mode, "transport": transport, "model": settings.model if mode == "online" else "scripted_fixture",
              "rag_backend": cfg.rag_backend,
              "scope": "Finite acceptance set; offline fixtures validate execution, not model quality. Restore case uses a new graph instance, not an OS restart.",
              "summary": {"cases": len(records), "passed": passed, "failed": len(records)-passed,
                          "pass_rate": passed / len(records), "avg_tool_calls": tool_count / len(records),
                          "total_latency_ms": round((time.perf_counter()-started)*1000, 2)},
              "results": records}
    target = Path(output) if output else ROOT / "reports" / f"evaluation-{mode}-{transport}-{run_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(target.resolve())
    return report


def main():
    parser = argparse.ArgumentParser(description="Agent acceptance evaluation")
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--transport", choices=["local", "mcp"], default="local")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", choices=[c.id for c in CASES], dest="case_ids")
    args = parser.parse_args()
    try:
        report = evaluate(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    for result in report["results"]:
        print(f"{'PASS' if result['passed'] else 'FAIL'} {result['id']}: {', '.join(result['failures']) or 'ok'}")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"Report: {report['report_path']}")
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
