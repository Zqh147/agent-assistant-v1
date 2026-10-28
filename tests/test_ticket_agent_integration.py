
import json
from datetime import date
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from app.multi_agent.ticket_agent import TicketAgent
from app.support_service import SupportService
from app.support_store import SupportStore
from app.support_tools import SupportTools


CUSTOMER = "demo-customer-001"
THREAD = "ticket-integration-001"
ORDER = "DEMO-1001"
ISSUE = "设备频繁掉线"


@pytest.fixture
def business_env(tmp_path):
    """创建隔离的真实 SQLite 业务环境。"""

    orders_path = tmp_path / "orders.json"
    rules_path = tmp_path / "rules.json"
    db_path = tmp_path / "support_test.db"

    orders = {
        "data_kind": "fictional_demo_orders",
        "orders": [
            {
                "order_id": ORDER,
                "customer_id": CUSTOMER,
                "product_model": "Archer AX10",
                "hardware_version": "V1",
                "region": "US",
                "status": "delivered",
                "delivered_days_ago": 2,
            }
        ],
    }

    rules = {
        "policy_kind": "fictional_demo_policy",
        "policy_version": "test-v1",
        "notice": "仅用于自动化测试的演示规则",
        "rules": {
            "repair": {
                "rule_id": "TEST-REPAIR",
                "required_order_status": "delivered",
                "max_days_since_delivery": 365,
            },
            "return": {
                "rule_id": "TEST-RETURN",
                "required_order_status": "delivered",
                "max_days_since_delivery": 14,
            },
        },
    }

    orders_path.write_text(
        json.dumps(orders, ensure_ascii=False),
        encoding="utf-8",
    )

    rules_path.write_text(
        json.dumps(rules, ensure_ascii=False),
        encoding="utf-8",
    )

    store = SupportStore(
        db_path,
        orders_path,
        today=lambda: date(2026, 10, 27),
    )

    service = SupportService(store, rules_path)

    return store, service


def run_ticket_tool(service, name, arguments, context):
    """Mock LLM 决策，但使用真实业务服务。"""

    llm = Mock()

    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": arguments,
                    "id": "call-test-001",
                }
            ],
        ),
        AIMessage(content="工具执行完成"),
    ]

    agent = TicketAgent(
        llm=llm,
        support_tools=SupportTools(service),
    )

    return agent.run(
        "处理当前工单请求",
        trusted_context=context,
    )


def test_real_ticket_lifecycle_and_idempotency(business_env):
    """验证草稿、确认、正式提交和重复提交。"""

    store, service = business_env

    prepare_args = {
        "order_id": ORDER,
        "service_type": "repair",
        "issue_summary": ISSUE,
    }

    prepare_context = {
        "customer_id": CUSTOMER,
        "thread_id": THREAD,
        "user_messages": [ISSUE],
    }

    # 1. 首次创建草稿
    first = run_ticket_tool(
        service,
        "prepare_ticket",
        prepare_args,
        prepare_context,
    )

    assert first.status == "pending_confirmation"

    ticket_id = first.data["tool_results"][0]["result"][
        "ticket"
    ]["ticket_id"]

    assert store.get_ticket(ticket_id, CUSTOMER)["status"] == "draft"

    # 2. 重复创建同一草稿
    second = run_ticket_tool(
        service,
        "prepare_ticket",
        prepare_args,
        prepare_context,
    )

    second_id = second.data["tool_results"][0]["result"][
        "ticket"
    ]["ticket_id"]

    assert second_id == ticket_id

    # 3. 未确认时尝试提交
    submit_args = {"ticket_id": ticket_id}

    unconfirmed = run_ticket_tool(
        service,
        "submit_ticket",
        submit_args,
        {
            "customer_id": CUSTOMER,
            "thread_id": THREAD,
        },
    )

    assert unconfirmed.status == "pending_confirmation"
    assert unconfirmed.data["business_status"] == "confirmation_required"
    assert store.get_ticket(ticket_id, CUSTOMER)["status"] == "draft"

    # 4. 模拟服务端已经验证用户确认
    confirmed_context = {
        "customer_id": CUSTOMER,
        "thread_id": THREAD,
        "confirmed_ticket_id": ticket_id,
    }

    submitted = run_ticket_tool(
        service,
        "submit_ticket",
        submit_args,
        confirmed_context,
    )

    assert submitted.status == "success"
    assert submitted.data["business_status"] == "submitted"
    assert store.get_ticket(ticket_id, CUSTOMER)["status"] == "submitted"

    # 5. 再次提交，应该返回原工单
    repeated = run_ticket_tool(
        service,
        "submit_ticket",
        submit_args,
        confirmed_context,
    )

    assert repeated.status == "success"
    assert repeated.data["business_status"] == "already_submitted"

    # 6. 数据库中只能有一次 submitted 事件
    events = store.events(ticket_id, CUSTOMER)
    event_names = [item["event"] for item in events]

    assert event_names.count("draft_created") == 1
    assert event_names.count("submitted") == 1


def test_unauthorized_ticket_submission(business_env):
    """验证其他客户无法提交已有工单。"""

    store, service = business_env

    draft = run_ticket_tool(
        service,
        "prepare_ticket",
        {
            "order_id": ORDER,
            "service_type": "repair",
            "issue_summary": ISSUE,
        },
        {
            "customer_id": CUSTOMER,
            "thread_id": THREAD,
            "user_messages": [ISSUE],
        },
    )

    ticket_id = draft.data["tool_results"][0][
        "result"
    ]["ticket"]["ticket_id"]

    result = run_ticket_tool(
        service,
        "submit_ticket",
        {"ticket_id": ticket_id},
        {
            "customer_id": "another-customer",
            "thread_id": THREAD,
            "confirmed_ticket_id": ticket_id,
        },
    )

    assert result.status == "no_evidence"
    assert result.data["business_status"] == "not_found"
    assert store.get_ticket(ticket_id, CUSTOMER)["status"] == "draft"


def test_issue_summary_must_match_user_message(business_env):
    """模型不能自行补写用户没有提供的故障现象。"""

    store, service = business_env

    result = run_ticket_tool(
        service,
        "prepare_ticket",
        {
            "order_id": ORDER,
            "service_type": "repair",
            "issue_summary": "设备无法开机",
        },
        {
            "customer_id": CUSTOMER,
            "thread_id": THREAD,
            "user_messages": ["设备频繁掉线"],
        },
    )

    assert result.status == "needs_clarification"
    assert result.data["business_status"] == "needs_clarification"

    with store.connect() as db:
        count = db.execute(
            "SELECT COUNT(*) FROM tickets"
        ).fetchone()[0]

    assert count == 0



def test_prepare_ticket_rejects_wrong_verified_order(business_env):
    """Ticket Agent 不能改写 Supervisor 已验证的订单号。"""

    store, service = business_env

    result = run_ticket_tool(
        service,
        "prepare_ticket",
        {
            "order_id": "DEMO-1002",
            "service_type": "repair",
            "issue_summary": ISSUE,
        },
        {
            "customer_id": CUSTOMER,
            "thread_id": THREAD,
            "user_messages": [ISSUE],
            "expected_order_id": ORDER,
        },
    )

    assert result.status == "error"
    assert "does not match verified order" in result.error

    with store.connect() as db:
        count = db.execute(
            "SELECT COUNT(*) FROM tickets"
        ).fetchone()[0]

    assert count == 0


def test_prepare_ticket_accepts_verified_order(business_env):
    """订单匹配时，允许继续执行草稿准备。"""

    store, service = business_env

    result = run_ticket_tool(
        service,
        "prepare_ticket",
        {
            "order_id": ORDER,
            "service_type": "repair",
            "issue_summary": ISSUE,
        },
        {
            "customer_id": CUSTOMER,
            "thread_id": THREAD,
            "user_messages": [ISSUE],
            "expected_order_id": ORDER,
        },
    )

    assert result.status == "pending_confirmation"
    ticket = result.data["tool_results"][0]["result"]["ticket"]
    assert ticket["order_id"] == ORDER
    assert ticket["status"] == "draft"
    assert store.get_ticket(ticket["ticket_id"], CUSTOMER)["status"] == "draft"
