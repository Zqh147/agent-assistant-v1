
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from app.multi_agent.ticket_agent import TicketAgent
from app.support_tools import SupportTools


CUSTOMER = "demo-customer-001"
THREAD = "thread-001"
TICKET = "TKT-TEST001"


def make_tool_call(name, arguments):
    """模拟 LLM 返回一次 Tool Call。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": arguments,
                "id": "call-001",
            }
        ],
    )


def make_agent(responses):
    """创建使用 Mock LLM 和 Mock Service 的 Ticket Agent。"""
    llm = Mock()
    llm.invoke.side_effect = responses

    service = Mock()
    agent = TicketAgent(
        llm=llm,
        support_tools=SupportTools(service),
    )

    return agent, service, llm


def test_prepare_ticket_pending_confirmation():
    """成功准备草稿，但不能直接当作提交成功。"""
    agent, service, llm = make_agent([
        make_tool_call(
            "prepare_ticket",
            {
                "order_id": "DEMO-1001",
                "service_type": "repair",
                "issue_summary": "设备频繁掉线",
            },
        ),
    ])

    service.prepare_ticket.return_value = {
        "status": "draft_ready",
        "ticket": {"ticket_id": TICKET, "status": "draft"},
    }

    result = agent.run(
        "订单 DEMO-1001，设备频繁掉线，申请维修",
        trusted_context={
            "customer_id": CUSTOMER,
            "thread_id": THREAD,
            "user_messages": ["设备频繁掉线"],
        },
    )

    assert result.status == "pending_confirmation"
    assert result.data["business_status"] == "draft_ready"
    assert llm.invoke.call_count == 1

    service.prepare_ticket.assert_called_once_with(
        order_id="DEMO-1001",
        service_type="repair",
        issue_summary="设备频繁掉线",
        customer_id=CUSTOMER,
        thread_id=THREAD,
        user_messages=["设备频繁掉线"],
    )


def test_submit_without_confirmation():
    """模拟 Service 拒绝未经确认的提交。"""
    agent, service, _ = make_agent([
        make_tool_call(
            "submit_ticket",
            {"ticket_id": TICKET},
        ),
    ])

    service.submit_ticket.return_value = {
        "status": "confirmation_required",
        "message": "请先确认工单草稿。",
    }

    result = agent.run(
        "提交维修工单",
        trusted_context={
            "customer_id": CUSTOMER,
            "thread_id": THREAD,
        },
    )

    assert result.status == "pending_confirmation"
    assert result.data["business_status"] == "confirmation_required"

    service.submit_ticket.assert_called_once_with(
        ticket_id=TICKET,
        customer_id=CUSTOMER,
        thread_id=THREAD,
        confirmed_ticket_id=None,
    )


def test_submit_with_confirmation():
    """模拟已确认后的成功提交。"""
    agent, service, _ = make_agent([
        make_tool_call(
            "submit_ticket",
            {"ticket_id": TICKET},
        ),
    ])

    service.submit_ticket.return_value = {
        "status": "submitted",
        "ticket": {"ticket_id": TICKET, "status": "submitted"},
    }

    result = agent.run(
        "确认提交维修工单",
        trusted_context={
            "customer_id": CUSTOMER,
            "thread_id": THREAD,
            "confirmed_ticket_id": TICKET,
        },
    )

    assert result.status == "success"
    assert result.data["business_status"] == "submitted"

    service.submit_ticket.assert_called_once_with(
        ticket_id=TICKET,
        customer_id=CUSTOMER,
        thread_id=THREAD,
        confirmed_ticket_id=TICKET,
    )


def test_get_ticket_agent_loop():
    """工具返回后，LLM 再执行一次决策。"""
    agent, service, llm = make_agent([
        make_tool_call(
            "get_ticket",
            {"ticket_id": TICKET},
        ),
        AIMessage(content="查询结束"),
    ])

    service.get_ticket.return_value = {
        "status": "found",
        "ticket": {"ticket_id": TICKET, "status": "draft"},
    }

    result = agent.run(
        "查询工单状态",
        trusted_context={"customer_id": CUSTOMER},
    )

    assert result.status == "success"
    assert llm.invoke.call_count == 2
    assert result.data["business_status"] == "found"
    service.get_ticket.assert_called_once()


def test_forbidden_order_tool_is_rejected_and_llm_can_correct():
    """拒绝订单工具调用，并将允许的工具反馈给 LLM。"""
    agent, service, llm = make_agent([
        make_tool_call(
            "lookup_order",
            {"order_id": "DEMO-1001"},
        ),
        AIMessage(content="没有执行工单操作。"),
    ])

    result = agent.run(
        "准备维修工单",
        trusted_context={"customer_id": CUSTOMER},
    )

    assert result.status == "needs_clarification"
    assert llm.invoke.call_count == 2
    feedback = llm.invoke.call_args_list[1].args[0][-2]
    assert '"status": "tool_not_allowed"' in feedback.content
    service.lookup_order.assert_not_called()


def test_missing_user_messages():
    """缺少真实用户消息时，拒绝创建草稿。"""
    agent, service, _ = make_agent([])

    try:
        agent.execute_tool(
            "prepare_ticket",
            {
                "order_id": "DEMO-1001",
                "service_type": "repair",
                "issue_summary": "设备频繁掉线",
            },
            trusted_context={
                "customer_id": CUSTOMER,
                "thread_id": THREAD,
            },
        )
    except ValueError as exc:
        assert "user_messages" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    service.prepare_ticket.assert_not_called()


def test_llm_answer_without_tool_not_success():
    """没有真实工具结果，不能声称业务成功。"""
    agent, service, _ = make_agent([
        AIMessage(content="工单已提交成功"),
    ])

    result = agent.run(
        "帮我提交工单",
        trusted_context={"customer_id": CUSTOMER},
    )

    assert result.status != "success"
    assert result.status == "needs_clarification"
    service.submit_ticket.assert_not_called()


def test_recover_from_unauthorized_tool_call():
    """LLM 首次选择非法工具，收到拒绝反馈后改为准备草稿。"""

    agent, service, llm = make_agent([
        make_tool_call(
            "search_knowledge_base",
            {
                "question": "设备频繁掉线如何排查",
                "order_id": "DEMO-1001",
            },
        ),
        make_tool_call(
            "prepare_ticket",
            {
                "order_id": "DEMO-1001",
                "service_type": "repair",
                "issue_summary": "设备频繁掉线",
            },
        ),
    ])

    service.prepare_ticket.return_value = {
        "status": "draft_ready",
        "ticket": {
            "ticket_id": TICKET,
            "order_id": "DEMO-1001",
            "status": "draft",
        },
    }

    result = agent.run(
        "为 DEMO-1001 准备维修工单草稿",
        trusted_context={
            "customer_id": CUSTOMER,
            "thread_id": THREAD,
            "user_messages": ["设备频繁掉线"],
            "expected_order_id": "DEMO-1001",
        },
    )

    assert result.status == "pending_confirmation"
    assert result.data["business_status"] == "draft_ready"
    assert llm.invoke.call_count == 2

    service.prepare_ticket.assert_called_once()
    service.search_knowledge_base.assert_not_called()
    service.submit_ticket.assert_not_called()

    # 第二轮 LLM 必须能够看到上一轮的工具拒绝反馈
    second_call_messages = llm.invoke.call_args_list[1].args[0]

    assert any(
        message.type == "tool"
        and "tool_not_allowed" in str(message.content)
        for message in second_call_messages
    )


def test_repeated_unauthorized_tools_stop_at_max_steps():
    """反复请求越权工具时，必须终止且不能执行写操作。"""

    agent, service, llm = make_agent([
        make_tool_call(
            "search_knowledge_base",
            {"question": "查询设备故障"},
        ),
        make_tool_call(
            "lookup_order",
            {"order_id": "DEMO-1001"},
        ),
        make_tool_call(
            "search_knowledge_base",
            {"question": "再次查询"},
        ),
    ])

    result = agent.run(
        "为订单 DEMO-1001 准备维修工单",
        trusted_context={
            "customer_id": CUSTOMER,
            "thread_id": THREAD,
            "user_messages": ["设备频繁掉线"],
            "expected_order_id": "DEMO-1001",
        },
        max_steps=3,
    )

    assert result.status == "error"
    assert result.error == "max_steps_exceeded"
    assert llm.invoke.call_count == 3

    service.prepare_ticket.assert_not_called()
    service.submit_ticket.assert_not_called()
    service.search_knowledge_base.assert_not_called()
    service.lookup_order.assert_not_called()
