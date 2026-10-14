"""Real HTTP service workflow check; does not relabel offline routing as online."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4
import httpx
from .config import ROOT


def check(url="http://127.0.0.1:8000", require_online=False, output=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = Path(output) if output else ROOT / "reports" / f"support-http-{stamp}.json"
    if target.exists():
        raise ValueError("output already exists")
    checks = {name: False for name in ("health", "project1_backend", "business_tools_registered", "utf8_response", "missing_order_clarification", "draft_prepared", "no_early_submit", "confirmed_submission", "ticket_query", "all_turns_completed")}
    turns, state = [], {}
    failure = None
    thread = "support-http-" + uuid4().hex
    try:
        with httpx.Client(base_url=url, timeout=180, trust_env=False) as client:
            health = client.get("/health")
            health.raise_for_status()
            state = health.json()
            checks["health"] = state.get("status") == "ok" and state.get("support_mode") == "demo"
            checks["project1_backend"] = state.get("rag_backend") == "project1"
            if require_online and not state.get("online"):
                raise ValueError("service is offline; enable its model client before online acceptance")
            tools = client.get("/tools").json()
            checks["business_tools_registered"] = {"lookup_order", "search_after_sales_rules", "assess_after_sales", "prepare_ticket", "submit_ticket", "get_ticket", "search_knowledge_base"} <= set(tools)
            def chat(message):
                response = client.post("/chat", json={"message": message, "thread_id": thread})
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    turns.append({"message": message, "http_status": response.status_code, "error": "HTTPStatusError"})
                    raise
                valid_charset = "charset=utf-8" in response.headers.get("content-type", "")
                checks["utf8_response"] = valid_charset if not turns else checks["utf8_response"] and valid_charset
                result = response.json()
                turns.append({"message": message, "result": result})
                return result
            missing = chat("路由器无法开机，我想申请维修，请问需要什么订单信息？")
            checks["missing_order_clarification"] = "订单" in missing["answer"] and not any(t["tool"] in {"prepare_ticket", "submit_ticket"} for t in missing["tool_trace"])
            draft = chat("我的示例订单是 DEMO-1001，设备无法开机。请查询订单和演示维修规则，判断资格；符合时准备工单草稿，先展示给我核对，不要提交。")
            ticket = draft.get("business_context", {}).get("pending_ticket_id")
            checks["draft_prepared"] = bool(ticket) and any(t["tool"] == "prepare_ticket" and t.get("business_status") == "draft_ready" for t in draft["tool_trace"])
            checks["no_early_submit"] = not any(t.get("business_status") == "submitted" for t in draft["tool_trace"])
            if ticket:
                confirmed = chat("确认提交")
                checks["confirmed_submission"] = any(t["tool"] == "submit_ticket" and t.get("business_status") in {"submitted", "already_submitted"} for t in confirmed["tool_trace"])
                query = chat("查询工单进度")
                checks["ticket_query"] = any(t["tool"] == "get_ticket" and t.get("business_status") == "found" for t in query["tool_trace"])
            else:
                checks["confirmed_submission"] = checks["ticket_query"] = False
            checks["all_turns_completed"] = all(t["result"]["status"] == "completed" and not t["result"]["errors"] for t in turns)
    except Exception as exc:
        # Preserve the thread and partial turns without provider response bodies.
        failure = type(exc).__name__
        checks["all_turns_completed"] = False
    report = {"evaluation_type": "real HTTP finite demonstration workflow", "executed_at": datetime.now(timezone.utc).isoformat(),
              "service_url": url, "online": state.get("online"), "model_decisions": "online_llm" if state.get("online") else "offline_rules",
              "thread_id": thread, "exception_type": failure, "checks": checks, "turns": turns,
              "summary": {"checks": len(checks), "passed": sum(checks.values())},
              "limitations": "Creates a demonstration draft/submitted ticket in the service database. Not a real vendor submission, refund, or repair. No OS restart is performed by this checker."}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, passed in checks.items():
        print(name, passed)
    print("Report:", target.resolve())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--require-online", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(**vars(args))
    print(f"{report['summary']['passed']}/{report['summary']['checks']} live workflow checks passed")
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
