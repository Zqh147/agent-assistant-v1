"""Business facts and eligibility come from records/rules, never model guesses."""
from datetime import date
import json
from pathlib import Path
import re

DEMO_NOTICE = "本地演示订单、演示规则及演示工单；提交后等待人工审核，不表示已经退款、维修或向厂商提交。"
TICKET_CAPABILITY_NOTICE = "当前演示版不支持编辑或取消已保存工单；同一会话、订单及申请类型再次准备会复用原记录。描述有误时先不要确认提交。"


class SupportService:
    def __init__(self, store, rules_path):
        self.store = store
        self.policy = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        if self.policy.get("policy_kind") != "fictional_demo_policy":
            raise ValueError("business rules must be labelled demonstration policy")
        for kind in ("repair", "return"):
            rule = self.policy["rules"][kind]
            if type(rule["max_days_since_delivery"]) is not int or rule["max_days_since_delivery"] < 0:
                raise ValueError("invalid policy window")

    @staticmethod
    def identifier(value, name):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
            raise ValueError(f"invalid {name}")
        return value.upper() if name in {"order_id", "ticket_id"} else value

    def base(self, status, **fields):
        return {"status": status, "data_kind": "local_demo", "notice": DEMO_NOTICE, **fields}

    def lookup_order(self, order_id, *, customer_id, **context):
        order_id = self.identifier(order_id, "order_id")
        order = self.store.lookup_order(order_id, customer_id)
        if order is None:
            return self.base("not_found", message="未找到该客户的示例订单，请核对订单号。")
        order["status_label"] = {"delivered": "已交付", "shipped": "已发货"}.get(order["status"], "未知状态")
        order["delivered_at_label"] = "交付日期"
        return self.base("found", order=order, product_scope={key: order[key] for key in ("product_model", "hardware_version", "region")})

    def search_after_sales_rules(self, service_type, **context):
        if service_type not in self.policy["rules"]:
            raise ValueError("service_type must be repair or return")
        return self.base("found", policy_version=self.policy["policy_version"], policy_kind=self.policy["policy_kind"],
                         rule=self.policy["rules"][service_type], policy_notice=self.policy["notice"])

    def assess_after_sales(self, order_id, service_type, *, customer_id, **context):
        rules = self.search_after_sales_rules(service_type)
        found = self.lookup_order(order_id, customer_id=customer_id)
        if found["status"] != "found":
            return self.base("not_found", eligible=False, reasons=[found["message"]])
        order, rule = found["order"], rules["rule"]
        today = self.store.today()
        days = (today - date.fromisoformat(order["delivered_at"])).days if order["delivered_at"] else None
        reasons = []
        if order["status"] != rule["required_order_status"]:
            reasons.append("演示规则要求订单已交付。")
        if days is None or days < 0:
            reasons.append("交付日期缺失或不适用，不能确认资格。")
        elif days > rule["max_days_since_delivery"]:
            reasons.append(f"已交付 {days} 天，超过演示规则的 {rule['max_days_since_delivery']} 天窗口。")
        eligible = not reasons
        return self.base("eligible" if eligible else "ineligible", eligible=eligible, order_id=order["order_id"],
                         service_type=service_type, days_since_delivery=days, reference_date=today.isoformat(),
                         policy_version=self.policy["policy_version"], rule_id=rule["rule_id"],
                         reasons=reasons or ["符合演示申请窗口，可准备工单草稿；最终处理待人工审核。"])

    def prepare_ticket(self, order_id, service_type, issue_summary, *, customer_id, thread_id, user_messages=None, **context):
        if not isinstance(issue_summary, str) or not 1 <= len(issue_summary.strip()) <= 1000:
            raise ValueError("issue_summary must contain 1 to 1000 characters")
        if user_messages is not None and not any(
                isinstance(text, str) and issue_summary.strip() in text for text in user_messages):
            return self.base("needs_clarification", required_fields=["issue_summary"],
                             message="故障描述必须逐字摘录当前会话中用户已提供的原话；本轮未保存工单。请去掉补写或改写的现象，细节不足时先追问用户。")
        assessment = self.assess_after_sales(order_id, service_type, customer_id=customer_id)
        if not assessment["eligible"]:
            return self.base("blocked", assessment=assessment, message="当前不能准备演示工单，请核对订单与规则。")
        ticket, created = self.store.draft(customer_id, thread_id, self.identifier(order_id, "order_id"), service_type,
                                          issue_summary.strip(), self.policy["policy_version"])
        return self.base("draft_ready" if ticket["status"] == "draft" else "already_submitted", ticket=ticket,
                         created=created, assessment=assessment,
                         ticket_edit_supported=False, ticket_cancel_supported=False,
                         capability_notice=TICKET_CAPABILITY_NOTICE, timestamp_timezone="UTC",
                         confirmation_prompt="请核对订单、申请类型和故障描述；确认后回复：确认提交。" if ticket["status"] == "draft" else "该申请已提交，无需重复提交。")

    def submit_ticket(self, ticket_id, *, customer_id, thread_id, confirmed_ticket_id=None, **context):
        ticket_id = self.identifier(ticket_id, "ticket_id")
        if ticket_id != confirmed_ticket_id:
            return self.base("confirmation_required", message="请先核对草稿，再明确回复“确认提交”；本轮没有提交工单。")
        ticket = self.store.get_ticket(ticket_id, customer_id)
        if ticket is None or ticket["thread_id"] != thread_id:
            return self.base("not_found", message="未找到当前客户和会话中的演示工单。")
        if ticket["status"] != "submitted":
            assessment = self.assess_after_sales(ticket["order_id"], ticket["service_type"], customer_id=customer_id)
            if not assessment["eligible"] or ticket["policy_version"] != self.policy["policy_version"]:
                return self.base("revalidation_required", assessment=assessment, message="订单资格或规则版本已变化，请重新准备并确认草稿。")
        saved, changed = self.store.submit(ticket_id, customer_id, thread_id)
        if saved is None:
            return self.base("not_found", message="未找到当前会话工单，本轮未提交。")
        return self.base("submitted" if changed else "already_submitted", ticket=saved, changed=changed,
                         ticket_edit_supported=False, ticket_cancel_supported=False,
                         capability_notice=TICKET_CAPABILITY_NOTICE, timestamp_timezone="UTC",
                         message="演示工单已保存为 submitted，等待人工审核。")

    def get_ticket(self, ticket_id, *, customer_id, **context):
        ticket_id = self.identifier(ticket_id, "ticket_id")
        ticket = self.store.get_ticket(ticket_id, customer_id)
        if ticket is None:
            return self.base("not_found", message="未找到该客户的演示工单。")
        return self.base("found", ticket=ticket, events=self.store.events(ticket_id, customer_id),
                         ticket_edit_supported=False, ticket_cancel_supported=False,
                         capability_notice=TICKET_CAPABILITY_NOTICE, timestamp_timezone="UTC")
