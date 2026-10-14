"""Deterministic demo fallback. This is not a model capability evaluation."""
import json
import re
from uuid import uuid4
from langchain_core.messages import AIMessage


def tool(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "offline-support-" + uuid4().hex, "type": "tool_call"}])


def route_demo(messages):
    humans = [item for item in messages if item.type == "human"]
    if not humans:
        return None
    text = str(humans[-1].content)
    context = {}
    for item in messages:
        if item.type == "system" and str(item.content).startswith("服务端业务上下文："):
            context = json.loads(str(item.content).split("：", 1)[1])
    order_match = re.search(r"DEMO-\d+", text, re.I)
    ticket_match = re.search(r"TKT-[A-Za-z0-9]+", text, re.I)
    order_id = order_match.group().upper() if order_match else context.get("order_id")
    ticket_id = ticket_match.group().upper() if ticket_match else context.get("pending_ticket_id", context.get("latest_ticket_id"))
    wants_application = any(word in text for word in ("申请", "工单", "退货", "维修"))
    wants_query = any(word in text for word in ("查询工单", "工单进度", "查一下工单", "处理进度"))
    kind = "return" if "退货" in text else ("repair" if "维修" in text else context.get("service_type"))
    if messages[-1].type == "tool":
        name = getattr(messages[-1], "name", "")
        if name not in {"lookup_order", "search_after_sales_rules", "assess_after_sales"}:
            return None
        payload = json.loads(messages[-1].content)
        if name == "lookup_order" and payload["status"] == "found":
            if wants_application and not wants_query and kind:
                return tool("search_after_sales_rules", service_type=kind)
            if any(word in text.lower() for word in ("wi-fi", "wifi", "ap 模式", "掉线", "密码")):
                # Keep order lookup instructions out of the product search query.
                match = re.search(r"(?:然后|再|接着|并)\s*(?:检索|查阅|查询产品知识)\s*[:：]?\s*(.+)", text)
                question = match.group(1) if match else text
                question = re.sub(r"[，,]?\s*并注明来源[。！!]?\s*$", "", question).strip()
                return tool("search_knowledge_base", question=question, top_k=3)
        if name == "search_after_sales_rules" and wants_application and order_id and kind:
            return tool("assess_after_sales", order_id=order_id, service_type=kind)
        if name == "assess_after_sales" and payload.get("eligible") and wants_application and kind:
            return tool("prepare_ticket", order_id=order_id, service_type=kind, issue_summary=text[:1000])
        return AIMessage(content="离线演示工具结果：" + messages[-1].content)
    if re.fullmatch(r"确认提交(?:工单)?(?:\s+TKT-[A-Za-z0-9]+)?[。！!]?", text):
        return tool("submit_ticket", ticket_id=ticket_id) if ticket_id else AIMessage(content="请先准备并核对演示工单草稿。")
    if wants_query:
        return tool("get_ticket", ticket_id=ticket_id) if ticket_id else AIMessage(content="请提供需要查询的演示工单号。")
    if "规则" in text and kind and not order_id:
        return tool("search_after_sales_rules", service_type=kind)
    if wants_application or "订单" in text:
        if not order_id:
            return AIMessage(content="请提供示例订单号，例如 DEMO-1001；申请时请说明维修或退货及问题描述。")
        if wants_application and not kind:
            return AIMessage(content="请说明需要维修还是退货申请，并提供问题描述。")
        return tool("lookup_order", order_id=order_id)
    return None
