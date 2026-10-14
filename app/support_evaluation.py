"""Finite demo workflows; offline checks are not online LLM quality scores."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .config import settings, ROOT
from .graph import AgentGraph


CASES = [
    ("missing_order", "路由器无法开机，我想申请维修", "missing"),
    ("order_lookup", "查询订单 DEMO-1001 的状态和设备信息", "lookup"),
    ("repair_flow", "查询我的订单 DEMO-1001，设备无法开机。请查演示维修规则，判断资格，符合时准备工单草稿供我核对。", "flow"),
    ("return_flow", "查询我的订单 DEMO-1001，设备无法开机。请查演示退货规则，判断资格，符合时准备工单草稿供我核对。", "flow"),
    ("expired_return", "订单 DEMO-1002 的设备无法开机，我想申请退货，请查规则并判断资格。", "ineligible"),
    ("expired_repair", "订单 DEMO-1003 的设备无法开机，我想申请维修，请查规则并判断资格。", "ineligible"),
    ("unknown_order", "订单 DEMO-9999 的设备无法开机，我想申请维修。", "not_found"),
    ("other_customer_order", "查询订单 DEMO-1004，我想申请维修。", "not_found"),
    ("order_technical", "查询订单 DEMO-1001，然后检索 Wi-Fi 掉线的排查资料并注明来源。", "technical"),
]


def fixture_rag(question, **kwargs):
    scope = {name: kwargs[name] for name in ("product_model", "hardware_version", "region")}
    return {"question": question, "backend": "project1", "status": "ok", "has_results": True,
            "resolved_scope": scope, "results": [{"content": "离线执行夹具：先核对有线连接是否稳定。", "source": "offline_fixture", "chunk_id": 0, "evidence_id": "offline-wireless"}]}


def evaluate(mode="offline", live_rag=False, output=None):
    if mode == "online" and not settings.api_key:
        raise ValueError("online evaluation requires DEEPSEEK_API_KEY; no offline fallback")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = Path(output) if output else ROOT / "reports" / f"support-{mode}-{stamp}.json"
    if target.exists():
        raise ValueError("output exists; preserve earlier runs and choose another filename")
    records = []
    with TemporaryDirectory(prefix="support-evaluation-") as temp:
        for case_id, question, kind in CASES:
            location = Path(temp) / case_id
            location.mkdir()
            cfg = replace(settings, api_key=settings.api_key if mode == "online" else "", support_mode="demo", rag_backend="project1",
                          tool_transport="local", session_db=location / "sessions.db", support_db=location / "support.db", max_steps=10)
            thread = "evaluation-" + uuid4().hex
            rows = []
            record = {"id": case_id, "question": question, "turns": rows, "checks": {}}
            try:
                def graph():
                    agent = AgentGraph(runtime_settings=cfg)
                    if not (live_rag or mode == "online"):
                        agent.tools["search_knowledge_base"] = fixture_rag
                    return agent
                agent = graph()
                result = agent.invoke(question, thread)
                rows.append(result)
                service = agent.support_service
                trace = result["tool_trace"]
                checks = record["checks"]
                checks["completed"] = result["status"] == "completed" and not result["errors"]
                checks["mode"] = result["online"] == (mode == "online")
                if kind == "missing":
                    checks["asks_order"] = "订单" in result["answer"]
                    checks["no_ticket_action"] = not any(t["tool"] in {"prepare_ticket", "submit_ticket"} for t in trace)
                elif kind == "lookup":
                    checks["actual_lookup"] = any(t["tool"] == "lookup_order" and t.get("business_status") == "found" for t in trace)
                    checks["scope"] = result["business_context"].get("product_scope") == {"product_model":"Archer AX10","hardware_version":"V1","region":"US"}
                elif kind == "flow":
                    ticket_id = result["business_context"].get("pending_ticket_id")
                    checks["business_tools"] = {"lookup_order", "search_after_sales_rules", "assess_after_sales", "prepare_ticket"} <= {t["tool"] for t in trace}
                    checks["draft_before_confirmation"] = bool(ticket_id) and service.store.get_ticket(ticket_id, cfg.demo_customer_id)["status"] == "draft"
                    checks["issue_description_from_user"] = bool(ticket_id) and service.store.get_ticket(ticket_id, cfg.demo_customer_id)["issue_summary"] in question
                    if ticket_id:
                        # New graph instances exercise SQLite business/history restore.
                        for message in ("确认提交", "查询工单进度", "确认提交"):
                            rows.append(graph().invoke(message, thread))
                        ticket = service.store.get_ticket(ticket_id, cfg.demo_customer_id)
                        checks["submitted_record"] = ticket["status"] == "submitted"
                        checks["submit_tool"] = any(t["tool"] == "submit_ticket" and t.get("business_status") in {"submitted", "already_submitted"} for t in rows[1]["tool_trace"])
                        checks["query_tool"] = any(t["tool"] == "get_ticket" for t in rows[2]["tool_trace"])
                        checks["idempotency"] = [e["event"] for e in service.store.events(ticket_id, cfg.demo_customer_id)].count("submitted") == 1
                        record["saved_ticket"] = ticket
                    else:
                        checks["submitted_record"] = False
                elif kind in {"ineligible", "not_found"}:
                    expected = "ineligible" if kind == "ineligible" else "not_found"
                    checks["business_result"] = any(t.get("business_status") == expected for t in trace)
                    with service.store.connect() as db:
                        checks["no_ticket_written"] = db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0
                elif kind == "technical":
                    rag = [t for t in trace if t["tool"] == "search_knowledge_base"]
                    checks["project1_adapter"] = bool(rag) and all(t.get("retrieval_backend") == "project1" and t["status"] == "success" for t in rag)
                    checks["order_scope"] = bool(rag) and rag[-1].get("resolved_scope") == {"product_model":"Archer AX10","hardware_version":"V1","region":"US"}
                    checks["has_evidence"] = bool(rag) and rag[-1].get("evidence_count", 0) > 0
                checks["all_turns_completed"] = all(r["status"] == "completed" and not r["errors"] for r in rows)
            except Exception as exc:
                record["exception_type"] = type(exc).__name__
                record["checks"]["execution"] = False
            record["passed"] = bool(record["checks"]) and all(record["checks"].values())
            record["failures"] = [name for name, value in record["checks"].items() if not value]
            records.append(record)
            print(f"{'PASS' if record['passed'] else 'FAIL'} {case_id}: {', '.join(record['failures']) or 'ok'}", flush=True)
    report = {"evaluation_type": "finite demonstration workflow acceptance", "executed_at": datetime.now(timezone.utc).isoformat(),
              "mode": mode, "model": settings.model if mode == "online" else "deterministic_offline_rules",
              "rag_evidence": "live_project1" if live_rag or mode == "online" else "execution_fixture",
              "policy_kind": "fictional_demo_policy", "isolated_databases": True,
              "limitations": "Offline rules validate execution, not model intelligence. New graph instances validate SQLite restore, not an OS restart. Online checks validate these finite workflows, not semantic answer accuracy or real refund/repair completion.",
              "summary": {"cases": len(records), "passed": sum(r["passed"] for r in records), "failed": sum(not r["passed"] for r in records)}, "results": records}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Report:", target.resolve())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--live-rag", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
