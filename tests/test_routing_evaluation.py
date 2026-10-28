"""Metric calculation for the routing evaluation script (no LLM)."""

from scripts.evaluate_routing import rate, summarize


CASES = [
    {"id": "a", "message": "m1", "intent": "order_only", "status": "routed"},
    {"id": "b", "message": "m2", "intent": "order_ticket", "status": "needs_clarification"},
    {"id": "c", "message": "m3", "intent": "unclear", "status": "needs_clarification"},
]


def run(intent, status, latency=100.0, error=None):
    return {"intent": intent, "status": status, "latency_ms": latency, "error": error}


def test_rate_handles_zero():
    assert rate(0, 0)["rate"] is None
    assert rate(1, 4) == {"passed": 1, "total": 4, "rate": 0.25}


def test_summarize_counts_every_run():
    runs = {
        "a": [run("order_only", "routed"), run("order_only", "routed")],
        "b": [run("order_ticket", "needs_clarification"), run("order_ticket", "routed")],
        "c": [run("order_only", "routed"), run("error", "error", error="TimeoutError")],
    }

    metrics = summarize(CASES, runs)

    assert metrics["run_count"] == 6
    assert metrics["intent_accuracy"] == {"passed": 4, "total": 6, "rate": 0.6667}
    assert metrics["status_accuracy"] == {"passed": 3, "total": 6, "rate": 0.5}
    assert metrics["exact_accuracy"]["passed"] == 3
    assert metrics["consistency"] == {"passed": 1, "total": 3, "rate": 0.3333}
    assert metrics["unstable_cases"] == ["b", "c"]
    assert metrics["errors"] == 1

    confusion = metrics["confusion_matrix"]
    assert confusion["order_only"]["order_only"] == 2
    assert confusion["unclear"]["order_only"] == 1
    assert confusion["unclear"]["error"] == 1

    assert metrics["per_label_recall"]["order_ticket"]["rate"] == 1.0
    assert metrics["per_label_recall"]["unclear"]["rate"] == 0.0
    assert {item["id"] for item in metrics["failures"]} == {"b", "c"}


def test_errored_runs_are_excluded_from_latency():
    runs = {
        "a": [run("order_only", "routed", latency=100.0)],
        "b": [run("error", "error", latency=9000.0, error="TimeoutError")],
        "c": [run("unclear", "needs_clarification", latency=300.0)],
    }

    metrics = summarize(CASES, runs)

    assert metrics["latency_ms"]["mean"] == 200.0
