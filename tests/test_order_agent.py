
from unittest.mock import Mock

import pytest

from app.multi_agent.order_agent import OrderAgent
from app.support_tools import SupportTools


def make_agent():
    service = Mock()
    service.lookup_order.return_value = {"status": "found"}

    agent = OrderAgent(
        llm=Mock(),
        support_tools=SupportTools(service),
    )

    return agent, service


def test_only_three_tools_visible():
    agent, _ = make_agent()

    names = {
        item["function"]["name"]
        for item in agent.schemas
    }

    assert names == {
        "lookup_order",
        "search_after_sales_rules",
        "assess_after_sales",
    }


def test_allowed_tool_execution():
    agent, service = make_agent()

    result = agent.execute_tool(
        "lookup_order",
        {"order_id": "DEMO-1001"},
        trusted_context={"customer_id": "demo-customer-001"},
    )

    assert result["status"] == "found"

    service.lookup_order.assert_called_once_with(
        order_id="DEMO-1001",
        customer_id="demo-customer-001",
    )


def test_forbidden_tool():
    agent, service = make_agent()

    with pytest.raises(PermissionError):
        agent.execute_tool(
            "submit_ticket",
            {"ticket_id": "TKT-001"},
            trusted_context={"customer_id": "demo-customer-001"},
        )

    service.assert_not_called()


def test_missing_customer_context():
    agent, service = make_agent()

    with pytest.raises(ValueError):
        agent.execute_tool(
            "lookup_order",
            {"order_id": "DEMO-1001"},
            trusted_context={},
        )

    service.assert_not_called()


def test_model_cannot_inject_customer_id():
    agent, service = make_agent()

    with pytest.raises(ValueError):
        agent.execute_tool(
            "lookup_order",
            {
                "order_id": "DEMO-1001",
                "customer_id": "another-customer",
            },
            trusted_context={"customer_id": "demo-customer-001"},
        )

    service.assert_not_called()


def test_order_agent_tool_loop():
    from langchain_core.messages import AIMessage, ToolMessage

    agent, service = make_agent()

    agent.llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "lookup_order",
                "args": {"order_id": "DEMO-1001"},
                "id": "call_001",
            }],
        ),
        AIMessage(content="订单查询完成。"),
    ]

    result = agent.run(
        "查询订单 DEMO-1001",
        trusted_context={
            "customer_id": "demo-customer-001"
        },
    )

    assert result.status == "success"
    assert result.summary == "订单查询已完成，尚未执行售后资格评估。"

    assert len(result.data["tool_results"]) == 1
    assert result.data["tool_results"][0]["tool"] == "lookup_order"

    assert agent.llm.invoke.call_count == 2

    second_messages = agent.llm.invoke.call_args_list[1].args[0]

    assert any(
        isinstance(message, ToolMessage)
        and message.tool_call_id == "call_001"
        for message in second_messages
    )


def test_order_agent_multi_tool_loop():
    from langchain_core.messages import AIMessage

    agent, service = make_agent()

    service.search_after_sales_rules.return_value = {
        "status": "found"
    }
    service.assess_after_sales.return_value = {
        "status": "ineligible"
    }

    agent.llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "lookup_order",
                "args": {"order_id": "DEMO-1003"},
                "id": "call_001",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "search_after_sales_rules",
                "args": {"service_type": "repair"},
                "id": "call_002",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "assess_after_sales",
                "args": {
                    "order_id": "DEMO-1003",
                    "service_type": "repair",
                },
                "id": "call_003",
            }],
        ),
        AIMessage(content="该订单不符合维修申请条件。"),
    ]

    result = agent.run(
        "查询 DEMO-1003 的维修资格",
        trusted_context={
            "customer_id": "demo-customer-001"
        },
    )

    assert result.status == "success"
    assert agent.llm.invoke.call_count == 4
    assert len(result.data["tool_results"]) == 3

    assert [
        item["tool"]
        for item in result.data["tool_results"]
    ] == [
        "lookup_order",
        "search_after_sales_rules",
        "assess_after_sales",
    ]

    assert (
        result.data["tool_results"][2]["result"]["status"]
        == "ineligible"
    )


def test_order_agent_rejects_forbidden_tool():
    from langchain_core.messages import AIMessage

    agent, service = make_agent()

    agent.llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "submit_ticket",
            "args": {"ticket_id": "TKT-001"},
            "id": "call_forbidden",
        }],
    )

    result = agent.run(
        "提交工单",
        trusted_context={
            "customer_id": "demo-customer-001"
        },
    )

    assert result.status == "error"
    assert "cannot execute tool" in result.error

    service.assert_not_called()


def test_order_agent_max_steps():
    from langchain_core.messages import AIMessage

    agent, service = make_agent()

    agent.llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "lookup_order",
                "args": {"order_id": "DEMO-1001"},
                "id": "call_001",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "lookup_order",
                "args": {"order_id": "DEMO-1001"},
                "id": "call_002",
            }],
        ),
    ]

    result = agent.run(
        "查询订单",
        trusted_context={
            "customer_id": "demo-customer-001"
        },
        max_steps=2,
    )

    assert result.status == "error"
    assert result.error == "max_steps_exceeded"

    assert agent.llm.invoke.call_count == 2
    assert len(result.data["tool_results"]) == 2


def test_no_tool_cannot_claim_eligibility():
    from langchain_core.messages import AIMessage

    agent, service = make_agent()

    agent.llm.invoke.return_value = AIMessage(
        content="你的订单符合维修资格。"
    )

    result = agent.run(
        "查询 DEMO-1003 的维修资格",
        trusted_context={
            "customer_id": "demo-customer-001"
        },
    )

    assert result.status == "needs_clarification"
    assert result.data["tool_results"] == []
    service.assert_not_called()


def test_business_result_overrides_llm_claim():
    from langchain_core.messages import AIMessage

    agent, service = make_agent()

    service.assess_after_sales.return_value = {
        "status": "ineligible"
    }

    agent.llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "assess_after_sales",
                "args": {
                    "order_id": "DEMO-1003",
                    "service_type": "repair",
                },
                "id": "call_assess",
            }],
        ),
        AIMessage(
            content="该订单符合维修资格。"
        ),
    ]

    result = agent.run(
        "查询 DEMO-1003 的维修资格",
        trusted_context={
            "customer_id": "demo-customer-001"
        },
    )

    assert result.status == "success"
    assert result.data["eligibility"] == "ineligible"
    assert "不符合申请条件" in result.summary
    service.assess_after_sales.assert_called_once()

