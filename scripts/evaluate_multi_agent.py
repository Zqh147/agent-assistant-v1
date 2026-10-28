"""Real HTTP evaluation of the multi-agent endpoint."""

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.config import settings
from app.support_store import SupportStore


BASE_URL = "http://127.0.0.1:8000"
EXPECTED_ISSUE = "设备频繁掉线"

SCOPE = {
    "product_model": "Archer AX10",
    "hardware_version": "V1",
    "region": "US",
}

CASES = [
    {
        "name": "order_only",
        "message": "请查询订单 DEMO-1001 的设备型号、硬件版本和地区。",
        "tools": ["call_order_agent"],
        "status": "success",
        "knowledge": False,
        "ticket": False,
    },
    {
        "name": "knowledge_only",
        "message": "Archer AX10 V1 US 在 AP 模式下是否支持 QoS？",
        "tools": ["call_knowledge_agent"],
        "status": "success",
        "knowledge": True,
        "ticket": False,
        "public_product_scope": SCOPE,
    },
    {
        "name": "order_knowledge",
        "message": (
            "请查询订单 DEMO-1001 的实际设备，"
            "并根据该设备的技术资料解释 AP 模式下的 QoS 限制。"
        ),
        "tools": [
            "call_order_agent",
            "call_knowledge_agent",
        ],
        "status": "success",
        "knowledge": True,
        "ticket": False,
    },
    {
        "name": "order_ticket",
        "message": (
            "我有订单 DEMO-1001，设备频繁掉线。"
            "请为该订单准备维修工单草稿。"
            "申请类型为 repair，故障描述请逐字摘录"
            "“设备频繁掉线”。请勿正式提交。"
        ),
        "tools": [
            "call_order_agent",
            "call_ticket_agent",
        ],
        "status": "pending_confirmation",
        "knowledge": False,
        "ticket": True,
    },
    {
        "name": "order_knowledge_ticket",
        "message": (
            "订单 DEMO-1001 的设备频繁掉线。"
            "请先查询该设备的掉线排查资料，"
            "再为该订单准备维修工单草稿。"
            "申请类型为 repair，故障描述请逐字摘录"
            "“设备频繁掉线”。请勿正式提交。"
        ),
        "tools": [
            "call_order_agent",
            "call_knowledge_agent",
            "call_ticket_agent",
        ],
        "status": "pending_confirmation",
        "knowledge": True,
        "ticket": True,
    },
]


def fraction(passed, total):
    return {
        "passed": passed,
        "total": total,
        "rate": round(passed / total, 4) if total else None,
    }


def summarize(positive, negative):
    knowledge = [r for r in positive if r["knowledge"]]
    tickets = [r for r in positive if r["ticket"]]

    tool_total = sum(r["tool_total"] for r in positive)
    tool_good = sum(r["tool_good"] for r in positive)

    guardrail_total = len(negative) + len(tickets)
    guardrail_good = (
        sum(r["passed"] for r in negative)
        + sum(r["ticket_ok"] for r in tickets)
    )

    durations = [
        r["duration_ms"]
        for r in positive
        if r["http_status"] == 200
    ]

    return {
        "task_success_rate": fraction(
            sum(r["passed"] for r in positive),
            len(positive),
        ),
        "routing_accuracy": fraction(
            sum(r["route_ok"] for r in positive),
            len(positive),
        ),
        "tool_execution_success_rate": fraction(
            tool_good,
            tool_total,
        ),
        "evidence_coverage": fraction(
            sum(r["evidence_ok"] for r in knowledge),
            len(knowledge),
        ),
        "ticket_integrity_rate": fraction(
            sum(r["ticket_ok"] for r in tickets),
            len(tickets),
        ),
        "guardrail_pass_rate": fraction(
            guardrail_good,
            guardrail_total,
        ),
        "average_http_latency_ms": (
            round(statistics.mean(durations), 2)
            if durations else None
        ),
    }


def evaluate_case(client, case, store):
    thread_id = "multi-agent-eval-" + uuid4().hex

    payload = {
        "message": case["message"],
        "thread_id": thread_id,
        "task_type": case["name"],
    }

    if "public_product_scope" in case:
        payload["public_product_scope"] = case[
            "public_product_scope"
        ]

    record = {
        "name": case["name"],
        "knowledge": case["knowledge"],
        "ticket": case["ticket"],
        "passed": False,
        "route_ok": False,
        "evidence_ok": False,
        "ticket_ok": False,
        "tool_good": 0,
        "tool_total": 0,
        "http_status": None,
        "duration_ms": None,
        "ticket_id": None,
        "failed_checks": [],
    }

    started = time.perf_counter()

    try:
        response = client.post(
            "/multi-agent/chat",
            json=payload,
        )

        record["duration_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )
        record["http_status"] = response.status_code

        body = response.json()

        if response.status_code != 200:
            record["failed_checks"].append("http_status")
            return record

        trace = body.get("tool_trace", [])
        evidence = body.get("evidence", [])

        if not isinstance(trace, list):
            trace = []

        if not isinstance(evidence, list):
            evidence = []

        tool_names = [
            item.get("tool")
            for item in trace
            if isinstance(item, dict)
        ]

        record["route_ok"] = (
            tool_names == case["tools"]
        )

        record["tool_total"] = len(trace)
        record["tool_good"] = sum(
            isinstance(item, dict)
            and (
                item.get("status") == "success"
                or (
                    item.get("tool") == "call_ticket_agent"
                    and item.get("status")
                    == "pending_confirmation"
                )
            )
            for item in trace
        )

        record["evidence_ok"] = (
            bool(evidence) if case["knowledge"] else False
        )

        record["ticket_id"] = body.get("ticket_id")

        if case["ticket"]:
            ticket_id = record["ticket_id"]

            ticket = (
                store.get_ticket(
                    ticket_id,
                    settings.demo_customer_id,
                )
                if isinstance(ticket_id, str)
                else None
            )

            events = (
                store.events(
                    ticket_id,
                    settings.demo_customer_id,
                )
                if ticket is not None
                else []
            )

            event_names = [
                event["event"] for event in events
            ]

            record["ticket_ok"] = (
                ticket is not None
                and ticket["status"] == "draft"
                and ticket["order_id"] == "DEMO-1001"
                and ticket["thread_id"] == thread_id
                and ticket["service_type"] == "repair"
                and ticket["issue_summary"] == EXPECTED_ISSUE
                and event_names.count("draft_created") == 1
                and event_names.count("submitted") == 0
            )

        checks = {
            "http_status": response.status_code == 200,
            "business_status": (
                body.get("status") == case["status"]
            ),
            "response_error": body.get("error") is None,
            "request_id": (
                bool(body.get("request_id"))
                and body.get("request_id")
                == response.headers.get("X-Request-ID")
            ),
            "route": record["route_ok"],
            "tool_execution": (
                record["tool_total"] == len(case["tools"])
                and record["tool_good"]
                == record["tool_total"]
            ),
        }

        if case["knowledge"]:
            checks["evidence"] = record["evidence_ok"]

        if case["ticket"]:
            checks["ticket_integrity"] = record["ticket_ok"]

        record["failed_checks"] = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        record["passed"] = not record["failed_checks"]

    except Exception as exc:
        record["duration_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )
        record["failed_checks"].append(
            type(exc).__name__
        )

    return record


def evaluate_invalid_requests(client):
    base = {
        "message": "查询订单 DEMO-1001",
        "thread_id": "multi-agent-invalid-" + uuid4().hex,
        "task_type": "order_only",
    }

    cases = [
        (
            "customer_id_injection",
            {**base, "customer_id": "another-customer"},
        ),
        (
            "invalid_task_type",
            {**base, "task_type": "invalid_task"},
        ),
        (
            "missing_knowledge_scope",
            {**base, "task_type": "knowledge_only"},
        ),
    ]

    results = []

    for name, payload in cases:
        try:
            response = client.post(
                "/multi-agent/chat",
                json=payload,
            )

            passed = response.status_code == 422
            status = response.status_code

        except Exception:
            passed = False
            status = None

        results.append({
            "name": name,
            "passed": passed,
            "http_status": status,
        })

    return results


def main():
    if settings.support_mode != "demo":
        raise SystemExit("SUPPORT_MODE must be demo")

    if settings.rag_backend != "project1":
        raise SystemExit("RAG_BACKEND must be project1")

    store = SupportStore(
        settings.support_db,
        settings.demo_orders_path,
    )

    with httpx.Client(
        base_url=BASE_URL,
        timeout=180.0,
    ) as client:

        health = client.get("/health")
        health.raise_for_status()

        health_data = health.json()

        if (
            health_data.get("status") != "ok"
            or health_data.get("support_mode") != "demo"
            or health_data.get("rag_backend") != "project1"
            or health_data.get("online") is not True
        ):
            raise SystemExit(
                "HTTP service is not in online demo mode"
            )

        positive = []

        for case in CASES:
            print("Evaluating:", case["name"])

            record = evaluate_case(
                client,
                case,
                store,
            )

            positive.append(record)

            print(
                "PASS" if record["passed"] else "FAIL",
                record["failed_checks"],
            )

        negative = evaluate_invalid_requests(client)

    metrics = summarize(positive, negative)

    report = {
        "evaluation_type": "multi_agent_real_http",
        "executed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "environment": {
            "support_mode": "demo",
            "rag_backend": "project1",
            "llm_online": True,
        },
        "metrics": metrics,
        "positive_cases": positive,
        "negative_cases": negative,
        "notes": [
            "Positive cases use real HTTP and online LLM.",
            "Tickets are local demonstration drafts.",
            "No approval or ticket submission is executed.",
            "Evidence coverage is not factual accuracy.",
            "Routing measures execution under supplied task_type.",
            "Five positive cases are a small fixed evaluation set.",
        ],
    }

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    filename = (
        "multi_agent_eval_"
        + datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_"
        + uuid4().hex[:6]
        + ".json"
    )

    path = report_dir / filename

    path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== MULTI-AGENT METRICS ==========")

    for name, value in metrics.items():
        print(name, ":", value)

    print("\nReport:", path.resolve())

    all_passed = (
        all(r["passed"] for r in positive)
        and all(r["passed"] for r in negative)
    )

    print(
        "\nMulti-agent evaluation:",
        "PASS" if all_passed else "FAIL",
    )

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()