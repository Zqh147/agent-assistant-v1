"""Judging logic of the single vs multi-agent comparison (no LLM, no HTTP)."""

import json
from pathlib import Path

import pytest

from app.config import settings
from app.support_store import SupportStore
from scripts.compare_agents import judge, summarize, tickets_for_thread


CASES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "e2e_cases.json"


def case(**expect):
    return {
        "id": "e-99",
        "kind": "normal",
        "message": "m",
        "order": "DEMO-1001",
        "expect": {"order_lookup": None, "evidence": None, "ticket": "none", **expect},
    }


SINGLE_ORDER_AND_EVIDENCE = {
    "tool_trace": [
        {
            "tool": "lookup_order",
            "status": "success",
            "business_status": "found",
            "arguments": {"order_id": "demo-1001"},
        },
        {
            "tool": "search_knowledge_base",
            "status": "success",
            "retrieval_status": "ok",
            "evidence_count": 3,
        },
    ]
}

MULTI_ORDER_AND_EVIDENCE = {
    "tool_trace": [
        {"tool": "call_order_agent", "status": "success"},
        {"tool": "call_knowledge_agent", "status": "success"},
    ],
    "evidence": [{"chunk_id": 0}],
}


@pytest.mark.parametrize("endpoint,body", [
    ("single", SINGLE_ORDER_AND_EVIDENCE),
    ("multi", MULTI_ORDER_AND_EVIDENCE),
])
def test_order_and_evidence_detected(endpoint, body):
    result = judge(endpoint, case(order_lookup=True, evidence=True), body, [])

    assert result["passed"] is True
    assert result["checks"] == {
        "order_lookup": True,
        "evidence": True,
        "ticket": True,
        "no_submission": True,
    }


def test_single_lookup_of_other_order_does_not_count():
    body = {"tool_trace": [{
        "tool": "lookup_order",
        "status": "success",
        "business_status": "found",
        "arguments": {"order_id": "DEMO-1002"},
    }]}

    result = judge("single", case(order_lookup=True), body, [])

    assert result["checks"]["order_lookup"] is False


def test_not_found_lookup_satisfies_no_lookup_expectation():
    body = {"tool_trace": [{
        "tool": "lookup_order",
        "status": "success",
        "business_status": "not_found",
        "arguments": {"order_id": "DEMO-1004"},
    }]}

    result = judge("single", {**case(order_lookup=False), "order": "DEMO-1004"}, body, [])

    assert result["passed"] is True


def test_empty_rag_result_is_not_evidence():
    body = {"tool_trace": [{
        "tool": "search_knowledge_base",
        "status": "success",
        "retrieval_status": "needs_clarification",
        "evidence_count": 0,
    }]}

    assert judge("single", case(evidence=False), body, [])["passed"] is True


def test_expected_draft_must_match_order():
    good = [{"ticket_id": "T1", "order_id": "DEMO-1001", "service_type": "repair", "status": "draft"}]
    wrong = [{"ticket_id": "T1", "order_id": "DEMO-1002", "service_type": "repair", "status": "draft"}]

    assert judge("multi", case(ticket="draft"), {}, good)["passed"] is True
    assert judge("multi", case(ticket="draft"), {}, wrong)["checks"]["ticket"] is False


def test_unexpected_write_is_flagged():
    tickets = [{"ticket_id": "T1", "order_id": "DEMO-1001", "service_type": "repair", "status": "draft"}]

    result = judge("single", case(ticket="none"), {}, tickets)

    assert result["passed"] is False
    assert result["unexpected_write"] is True


def test_submission_always_fails_even_when_ticket_optional():
    tickets = [{"ticket_id": "T1", "order_id": "DEMO-1001", "service_type": "repair", "status": "submitted"}]

    result = judge("multi", case(ticket="optional"), {}, tickets)

    assert result["checks"]["ticket"] is True
    assert result["checks"]["no_submission"] is False
    assert result["unapproved_submission"] is True


def test_summary_metrics():
    records = [
        {"kind": "normal", "passed": True, "checks": {"ticket": True}, "http_status": 200,
         "latency_ms": 1000.0, "usage": {"llm_calls": 2, "total_tokens": 1000, "tokens_complete": True}},
        {"kind": "adversarial", "passed": False, "checks": {"ticket": False}, "http_status": 200,
         "latency_ms": 3000.0, "unexpected_write": True,
         "usage": {"llm_calls": 4, "total_tokens": 3000, "tokens_complete": True}},
        {"kind": "normal", "passed": False, "checks": {"http_status": False}, "http_status": 503,
         "latency_ms": 9000.0, "usage": None},
    ]

    summary = summarize(records)

    assert summary["success"] == {"passed": 1, "total": 3, "rate": 0.3333}
    assert summary["by_kind"] == {
        "normal": {"passed": 1, "total": 2},
        "adversarial": {"passed": 0, "total": 1},
    }
    assert summary["failed_checks"] == {"ticket": 1, "http_status": 1}
    assert summary["unexpected_writes"] == 1
    assert summary["http_errors"] == 1
    assert summary["llm_calls_mean"] == 3.0
    assert summary["total_tokens_mean"] == 2000.0
    assert summary["latency_ms_mean"] == 2000.0


def test_tickets_for_thread_reads_only_that_thread(tmp_path):
    store = SupportStore(tmp_path / "support.db", settings.demo_orders_path)
    store.draft("demo-customer-001", "thread-a", "DEMO-1001", "repair", "掉线", "v1")
    store.draft("demo-customer-001", "thread-b", "DEMO-1002", "return", "不想要了", "v1")

    tickets = tickets_for_thread(tmp_path / "support.db", "thread-a")

    assert [(t["order_id"], t["status"]) for t in tickets] == [("DEMO-1001", "draft")]


def test_e2e_case_file_is_well_formed():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]

    assert len({c["id"] for c in cases}) == len(cases)
    for item in cases:
        assert item["kind"] in {"normal", "boundary", "adversarial"}
        assert item["expect"]["ticket"] in {"draft", "none", "optional", "if_eligible"}
        assert item["expect"]["order_lookup"] in {True, False, None}
        assert item["expect"]["evidence"] in {True, False, None}
        if item["expect"]["ticket"] in {"draft", "if_eligible"}:
            assert item["order"]
        if item["expect"]["ticket"] == "if_eligible":
            assert item["service_type"] in {"repair", "return"}


DRAFT = [{"ticket_id": "T1", "order_id": "DEMO-1001", "service_type": "return", "status": "draft"}]


@pytest.mark.parametrize("eligible,tickets,passed", [
    (True, DRAFT, True),
    (True, [], False),
    (False, [], True),
    (False, DRAFT, False),
])
def test_if_eligible_follows_business_rules(eligible, tickets, passed):
    item = {**case(ticket="if_eligible"), "service_type": "return"}

    result = judge("multi", item, {}, tickets, eligible=eligible)

    assert result["passed"] is passed
    assert result["expected_ticket"] == ("draft" if eligible else "none")


def test_if_eligible_without_eligibility_is_an_error():
    with pytest.raises(ValueError):
        judge("multi", case(ticket="if_eligible"), {}, [])
