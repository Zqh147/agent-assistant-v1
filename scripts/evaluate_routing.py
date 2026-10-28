"""Evaluate the LLM intent router on hand-labeled messages.

Only the router runs (one LLM call per message); no sub-agent, RAG or
database is touched. Run from the repository root:

    python -m scripts.evaluate_routing --split dev --repeats 3
    python -m scripts.evaluate_routing --split test --repeats 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"
SPLITS = {
    "dev": EVAL_DIR / "routing_dev.json",
    "test": EVAL_DIR / "routing_test.json",
}
LABELS = (
    "order_only",
    "knowledge_only",
    "order_knowledge",
    "order_ticket",
    "order_knowledge_ticket",
    "unclear",
)


def cases_digest(path: Path) -> str:
    """SHA-256 of the case file, ignoring CRLF/LF differences between checkouts."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def rate(passed: int, total: int) -> dict:
    return {
        "passed": passed,
        "total": total,
        "rate": round(passed / total, 4) if total else None,
    }


def summarize(cases: list[dict], runs: dict[str, list[dict]]) -> dict:
    """Compute routing metrics.

    runs maps case id -> one record per repeat:
    {"intent": str, "status": str, "latency_ms": float, "error": str | None}
    An errored run counts as wrong.
    """

    all_runs = [(case, run) for case in cases for run in runs[case["id"]]]

    intent_ok = sum(run["intent"] == case["intent"] for case, run in all_runs)
    status_ok = sum(run["status"] == case["status"] for case, run in all_runs)
    both_ok = sum(
        run["intent"] == case["intent"] and run["status"] == case["status"]
        for case, run in all_runs
    )

    confusion = {
        expected: {predicted: 0 for predicted in (*LABELS, "error")}
        for expected in LABELS
    }
    for case, run in all_runs:
        predicted = run["intent"] if run["intent"] in LABELS else "error"
        confusion[case["intent"]][predicted] += 1

    per_label = {}
    for label in LABELS:
        label_runs = [(c, r) for c, r in all_runs if c["intent"] == label]
        per_label[label] = rate(
            sum(r["intent"] == label for _, r in label_runs),
            len(label_runs),
        )

    stable = 0
    unstable_cases = []
    for case in cases:
        outcomes = {(run["intent"], run["status"]) for run in runs[case["id"]]}
        if len(outcomes) == 1:
            stable += 1
        else:
            unstable_cases.append(case["id"])

    failures = []
    for case in cases:
        wrong = [
            run for run in runs[case["id"]]
            if run["intent"] != case["intent"] or run["status"] != case["status"]
        ]
        if wrong:
            first = wrong[0]
            failures.append({
                "id": case["id"],
                "message": case["message"],
                "expected": f"{case['intent']} / {case['status']}",
                "got": f"{first['intent']} / {first['status']}",
                "wrong_runs": len(wrong),
                "reason": first.get("reason"),
                "warnings": first.get("warnings", []),
                "error": first.get("error"),
            })

    latencies = [run["latency_ms"] for _, run in all_runs if run.get("error") is None]

    return {
        "case_count": len(cases),
        "run_count": len(all_runs),
        "intent_accuracy": rate(intent_ok, len(all_runs)),
        "status_accuracy": rate(status_ok, len(all_runs)),
        "exact_accuracy": rate(both_ok, len(all_runs)),
        "consistency": rate(stable, len(cases)),
        "per_label_recall": per_label,
        "confusion_matrix": confusion,
        "errors": sum(run.get("error") is not None for _, run in all_runs),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else None,
            "p95": (
                round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 1)
                if latencies else None
            ),
        },
        "unstable_cases": unstable_cases,
        "failures": failures,
    }


def print_report(metrics: dict) -> None:
    print("\n========== ROUTING METRICS ==========")
    for key in ("intent_accuracy", "status_accuracy", "exact_accuracy", "consistency"):
        value = metrics[key]
        print(f"{key:18s} {value['passed']}/{value['total']}  ({value['rate']})")
    print(f"{'errors':18s} {metrics['errors']}")
    print(f"{'latency_ms':18s} mean={metrics['latency_ms']['mean']} p95={metrics['latency_ms']['p95']}")

    short = {
        "order_only": "OO",
        "knowledge_only": "KO",
        "order_knowledge": "OK",
        "order_ticket": "OT",
        "order_knowledge_ticket": "OKT",
        "unclear": "UN",
        "error": "ERR",
    }
    columns = list(short)
    print("\nconfusion (rows = expected, columns = predicted)")
    print("      " + "".join(f"{short[c]:>5s}" for c in columns))
    for expected, row in metrics["confusion_matrix"].items():
        print(f"{short[expected]:>5s} " + "".join(f"{row[c]:>5d}" for c in columns))

    if metrics["failures"]:
        print("\nfailures")
        for item in metrics["failures"]:
            print(
                f"- {item['id']}: expected {item['expected']}, got {item['got']} "
                f"({item['wrong_runs']} runs) | {item['message']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), default="dev")
    parser.add_argument("--cases", type=Path, help="Custom case file; overrides --split.")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    from app.config import settings
    from app.llm_client import LLMClient
    from app.multi_agent.intent_router import IntentRouter

    llm = LLMClient()
    if not llm.online:
        raise SystemExit("A real model is required: set DEEPSEEK_API_KEY in .env")

    cases_path = args.cases or SPLITS[args.split]
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    print(f"Cases: {cases_path.name} ({data.get('split', 'custom')}), repeats={args.repeats}")
    cases = data["cases"]
    router = IntentRouter(llm)

    runs: dict[str, list[dict]] = {}
    for index, case in enumerate(cases, start=1):
        runs[case["id"]] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            try:
                decision = router.route(case["message"])
                record = {
                    "intent": decision.intent,
                    "status": decision.status,
                    "model_intent": decision.model_intent,
                    "reason": decision.reason,
                    "warnings": decision.warnings,
                    "error": None,
                }
            except Exception as exc:  # count as a wrong answer, keep going
                record = {
                    "intent": "error",
                    "status": "error",
                    "warnings": [],
                    "error": type(exc).__name__,
                }
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            runs[case["id"]].append(record)

        marks = "".join(
            "." if r["intent"] == case["intent"] and r["status"] == case["status"] else "x"
            for r in runs[case["id"]]
        )
        print(f"[{index:02d}/{len(cases)}] {case['id']:7s} {marks}")

    metrics = summarize(cases, runs)
    print_report(metrics)

    report = {
        "evaluation_type": "intent_routing",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "cases_version": data.get("version"),
        "cases_split": data.get("split"),
        "cases_sha256": cases_digest(cases_path),
        "repeats": args.repeats,
        "metrics": metrics,
        "runs": runs,
        "notes": [
            "Only the router runs; sub-agents are not executed.",
            "intent = model judgment after validation; status = routed or needs_clarification.",
            "An errored run counts as wrong.",
            "Hand-written cases; not a sample of real user traffic.",
        ],
    }

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    path = report_dir / (
        f"routing_eval_{data.get('split', 'custom')}_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:6]
        + ".json"
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nReport:", path)


if __name__ == "__main__":
    main()
