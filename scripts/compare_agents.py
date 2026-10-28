"""Compare the single-agent and multi-agent endpoints on the same cases.

Each case is sent unchanged to /chat (single agent) and /multi-agent/chat
(multi agent, routed by the LLM intent router). Outcomes are judged from
tool traces and the ticket database, never from the answer wording.

Start the API (and the RAG service) first, then from the repository root:

    python -m scripts.compare_agents --repeats 2
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "data" / "eval" / "e2e_cases.json"
ENDPOINTS = {
    "single": "/chat",
    "multi": "/multi-agent/chat",
}


# --- judging (pure functions, unit tested) ---------------------------------


def order_found(endpoint: str, body: dict, order_id: str | None) -> bool:
    trace = body.get("tool_trace") or []
    if endpoint == "single":
        return any(
            item.get("tool") == "lookup_order"
            and item.get("status") == "success"
            and item.get("business_status") == "found"
            and (
                order_id is None
                or str((item.get("arguments") or {}).get("order_id", "")).upper() == order_id
            )
            for item in trace
        )
    return any(
        item.get("tool") == "call_order_agent" and item.get("status") == "success"
        for item in trace
    )


def evidence_found(endpoint: str, body: dict) -> bool:
    if endpoint == "single":
        return any(
            item.get("tool") == "search_knowledge_base"
            and item.get("status") == "success"
            and item.get("retrieval_status") == "ok"
            and (item.get("evidence_count") or 0) > 0
            for item in body.get("tool_trace") or []
        )
    return bool(body.get("evidence"))


def judge(endpoint: str, case: dict, body: dict, tickets: list[dict]) -> dict:
    """Return named checks for one response; every check must hold to pass."""

    expect = case["expect"]
    checks: dict[str, bool] = {}

    if expect.get("order_lookup") is not None:
        checks["order_lookup"] = (
            order_found(endpoint, body, case.get("order")) == expect["order_lookup"]
        )

    if expect.get("evidence") is not None:
        checks["evidence"] = evidence_found(endpoint, body) == expect["evidence"]

    drafts = [t for t in tickets if t["status"] == "draft"]
    submitted = [t for t in tickets if t["status"] == "submitted"]

    if expect["ticket"] == "draft":
        checks["ticket"] = (
            len(tickets) == 1
            and len(drafts) == 1
            and drafts[0]["order_id"] == case.get("order")
        )
    elif expect["ticket"] == "none":
        checks["ticket"] = not tickets
    else:  # optional
        checks["ticket"] = len(tickets) <= 1

    checks["no_submission"] = not submitted

    return {
        "checks": checks,
        "passed": all(checks.values()),
        "unexpected_write": expect["ticket"] == "none" and bool(tickets),
        "unapproved_submission": bool(submitted),
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 1) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 1)


def summarize(records: list[dict]) -> dict:
    """records: one per (case, repeat) for a single endpoint."""

    ok = [r for r in records if r["http_status"] == 200]
    failed_checks: dict[str, int] = {}
    for record in records:
        for name, passed in record.get("checks", {}).items():
            if not passed:
                failed_checks[name] = failed_checks.get(name, 0) + 1

    by_kind: dict[str, dict] = {}
    for record in records:
        bucket = by_kind.setdefault(record["kind"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(record["passed"])

    usage = [r["usage"] for r in ok if r.get("usage")]
    return {
        "runs": len(records),
        "success": {
            "passed": sum(r["passed"] for r in records),
            "total": len(records),
            "rate": round(sum(r["passed"] for r in records) / len(records), 4) if records else None,
        },
        "by_kind": by_kind,
        "failed_checks": failed_checks,
        "unexpected_writes": sum(r.get("unexpected_write", False) for r in records),
        "unapproved_submissions": sum(r.get("unapproved_submission", False) for r in records),
        "http_errors": len(records) - len(ok),
        "llm_calls_mean": _mean([u["llm_calls"] for u in usage]),
        "total_tokens_mean": _mean([u["total_tokens"] for u in usage]),
        "tokens_complete": all(u.get("tokens_complete") for u in usage) if usage else None,
        "latency_ms_mean": _mean([r["latency_ms"] for r in ok]),
        "latency_ms_p95": _p95([r["latency_ms"] for r in ok]),
    }


# --- live run ----------------------------------------------------------------


def tickets_for_thread(db_path: Path, thread_id: str) -> list[dict]:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT ticket_id, order_id, service_type, status FROM tickets WHERE thread_id=?",
            (thread_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def run_case(client, endpoint: str, case: dict, db_path: Path) -> dict:
    thread_id = f"cmp-{endpoint}-{uuid4().hex[:12]}"
    payload = {"message": case["message"], "thread_id": thread_id}

    record: dict[str, Any] = {
        "id": case["id"],
        "kind": case["kind"],
        "endpoint": endpoint,
        "thread_id": thread_id,
        "passed": False,
        "checks": {},
    }

    started = time.perf_counter()
    try:
        response = client.post(ENDPOINTS[endpoint], json=payload)
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        record["http_status"] = response.status_code
        body = response.json() if response.status_code == 200 else {}
    except Exception as exc:
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        record["http_status"] = None
        record["error"] = type(exc).__name__
        body = {}

    tickets = tickets_for_thread(db_path, thread_id)
    record.update(judge(endpoint, case, body, tickets))
    if record["http_status"] != 200:
        record["passed"] = False
        record["checks"]["http_status"] = False

    record["usage"] = body.get("usage")
    record["status"] = body.get("status")
    record["routing"] = body.get("routing")
    record["tools"] = [item.get("tool") for item in body.get("tool_trace") or []]
    record["tickets"] = tickets
    record["answer"] = body.get("answer") or body.get("summary")
    return record


def print_table(summaries: dict[str, dict]) -> None:
    rows = [
        ("task success", lambda s: f"{s['success']['passed']}/{s['success']['total']} ({s['success']['rate']})"),
        ("unexpected writes", lambda s: str(s["unexpected_writes"])),
        ("unapproved submits", lambda s: str(s["unapproved_submissions"])),
        ("http errors", lambda s: str(s["http_errors"])),
        ("LLM calls / req", lambda s: str(s["llm_calls_mean"])),
        ("tokens / req", lambda s: str(s["total_tokens_mean"])),
        ("latency mean ms", lambda s: str(s["latency_ms_mean"])),
        ("latency p95 ms", lambda s: str(s["latency_ms_p95"])),
    ]
    print(f"\n{'':20s}{'single':>22s}{'multi':>22s}")
    for label, fmt in rows:
        print(f"{label:20s}{fmt(summaries['single']):>22s}{fmt(summaries['multi']):>22s}")
    for name in ("single", "multi"):
        print(f"\n{name} failed checks: {summaries[name]['failed_checks'] or 'none'}")
        print(f"{name} by kind: {summaries[name]['by_kind']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    import httpx

    from app.config import settings

    data = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = data["cases"]

    records: dict[str, list[dict]] = {"single": [], "multi": []}
    with httpx.Client(base_url=args.base_url, timeout=240.0) as client:
        health = client.get("/health").json()
        if not (
            health.get("online") is True
            and health.get("support_mode") == "demo"
            and health.get("rag_backend") == "project1"
        ):
            raise SystemExit(
                "API must run online with SUPPORT_MODE=demo and RAG_BACKEND=project1"
            )

        for repeat in range(1, args.repeats + 1):
            for case in cases:
                # Alternate the order so neither endpoint always goes first.
                order = ("single", "multi") if (repeat + int(case["id"][-2:])) % 2 else ("multi", "single")
                for endpoint in order:
                    record = run_case(client, endpoint, case, settings.support_db)
                    record["repeat"] = repeat
                    records[endpoint].append(record)
                    mark = "PASS" if record["passed"] else "FAIL " + str(
                        [k for k, v in record["checks"].items() if not v]
                    )
                    print(f"[{repeat}] {case['id']} {endpoint:6s} {record['latency_ms']:>8.0f} ms  {mark}")

    summaries = {name: summarize(items) for name, items in records.items()}
    print_table(summaries)

    report = {
        "evaluation_type": "single_vs_multi_agent",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "cases_version": data.get("version"),
        "repeats": args.repeats,
        "summary": summaries,
        "records": records,
        "notes": [
            "Same messages, same server, same model; multi-agent requests are routed by the LLM router.",
            "Judged from tool traces and the ticket database, not answer wording.",
            "Tickets are local demo drafts; no submission is ever expected.",
            "Small hand-written case set; not a sample of real traffic.",
        ],
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    path = report_dir / (
        "compare_agents_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:6]
        + ".json"
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nReport:", path)


if __name__ == "__main__":
    main()
