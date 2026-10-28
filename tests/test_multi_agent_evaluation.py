from scripts.evaluate_multi_agent import fraction, summarize


def make_case(
    *,
    passed=True,
    route_ok=True,
    knowledge=False,
    evidence_ok=False,
    ticket=False,
    ticket_ok=False,
):
    return {
        "passed": passed,
        "route_ok": route_ok,
        "knowledge": knowledge,
        "evidence_ok": evidence_ok,
        "ticket": ticket,
        "ticket_ok": ticket_ok,
        "tool_good": 2,
        "tool_total": 2,
        "http_status": 200,
        "duration_ms": 1000.0,
    }


def test_fraction_calculation():
    assert fraction(2, 4) == {
        "passed": 2,
        "total": 4,
        "rate": 0.5,
    }


def test_fraction_zero_denominator():
    assert fraction(0, 0)["rate"] is None


def test_metrics_calculation():
    positive = [
        make_case(),
        make_case(
            knowledge=True,
            evidence_ok=True,
        ),
        make_case(
            ticket=True,
            ticket_ok=True,
        ),
    ]

    negative = [
        {"passed": True},
        {"passed": False},
    ]

    metrics = summarize(positive, negative)

    assert metrics["task_success_rate"]["rate"] == 1.0
    assert metrics["routing_accuracy"]["rate"] == 1.0
    assert metrics["tool_execution_success_rate"]["rate"] == 1.0
    assert metrics["evidence_coverage"]["rate"] == 1.0
    assert metrics["ticket_integrity_rate"]["rate"] == 1.0

    assert metrics["guardrail_pass_rate"] == {
        "passed": 2,
        "total": 3,
        "rate": 0.6667,
    }

    assert metrics["average_http_latency_ms"] == 1000.0