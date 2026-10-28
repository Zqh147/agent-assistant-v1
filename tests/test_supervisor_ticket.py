
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from app.multi_agent.contracts import AgentResult
from app.multi_agent.dynamic_supervisor import DynamicSupervisor
from app.multi_agent.supervisor_tools import SupervisorTools


ORDER_ID = "DEMO-1001"
CUSTOMER_ID = "demo-customer-001"
THREAD_ID = "thread-001"

QUESTION = (
    "订单 DEMO-1001 的设备频繁掉线，"
    "请帮我准备维修工单。"
)

SCOPE = {
    "product_model": "Archer AX10",
    "hardware_version": "V1",
    "region": "US",
}


def make_call(name, arguments, call_id):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": arguments,
                "id": call_id,
            }
        ],
    )


def make_environment(responses):
    supervisor_llm = Mock()
    supervisor_llm.invoke.side_effect = responses

    order_agent = Mock()
    knowledge_agent = Mock()
    ticket_agent = Mock()

    order_agent.run.return_value = AgentResult(
        agent="order_agent",
        status="success",
        summary="订单验证成功。",
        data={
            "tool_results": [
                {
                    "tool": "lookup_order",
                    "result": {
                        "status": "found",
                        "order": {
                            "order_id": ORDER_ID,
                        },
                        "product_scope": dict(SCOPE),
                    },
                },
            ],
        },
    )

    knowledge_agent.run.return_value = AgentResult(
        agent="knowledge_agent",
        status="success",
        summary="已找到路由器掉线排查资料。",
        evidence=[
            {
                "source": "test-manual",
                "content": "Wi-Fi 故障排查说明",
            }
        ],
    )

    ticket_agent.run.return_value = AgentResult(
        agent="ticket_agent",
        status="pending_confirmation",
        summary="工单草稿已创建。",
        data={
            "tool_results": [
                {
                    "tool": "prepare_ticket",
                    "result": {
                        "status": "draft_ready",
                        "ticket": {
                            "ticket_id": "TKT-TEST001",
                            "order_id": ORDER_ID,
                            "customer_id": CUSTOMER_ID,
                            "thread_id": THREAD_ID,
                            "status": "draft",
                        },
                    },
                }
            ]
        },
    )

    tools = SupervisorTools(
        order_agent=order_agent,
        knowledge_agent=knowledge_agent,
        ticket_agent=ticket_agent,
        trusted_context={
            "customer_id": CUSTOMER_ID,
            "thread_id": THREAD_ID,
            "user_messages": [QUESTION],
        },
    )

    supervisor = DynamicSupervisor(
        llm=supervisor_llm,
    )

    return (
        supervisor,
        tools,
        supervisor_llm,
        order_agent,
        knowledge_agent,
        ticket_agent,
    )


def test_order_then_ticket():
    """验证 Order -> Ticket 草稿创建流程。"""

    env = make_environment([
        make_call(
            "call_order_agent",
            {"order_id": ORDER_ID},
            "call-order",
        ),
        make_call(
            "call_ticket_agent",
            {"order_id": ORDER_ID},
            "call-ticket",
        ),
    ])

    supervisor, tools, llm, order, knowledge, ticket = env

    result = supervisor.run(
        user_question=QUESTION,
        tools=tools,
        task_type="order_ticket",
    )

    assert result.status == "pending_confirmation"
    assert result.data["ticket_id"] == "TKT-TEST001"

    assert [
        item["tool"]
        for item in result.data["tool_trace"]
    ] == [
        "call_order_agent",
        "call_ticket_agent",
    ]

    order.run.assert_called_once()
    ticket.run.assert_called_once()
    knowledge.run.assert_not_called()

    # 草稿创建后不应继续请求 LLM
    assert llm.invoke.call_count == 2


def test_order_knowledge_ticket():
    """验证三个 Agent 按业务依赖关系执行。"""

    env = make_environment([
        make_call(
            "call_order_agent",
            {"order_id": ORDER_ID},
            "call-order",
        ),
        make_call(
            "call_knowledge_agent",
            {"question": "设备频繁掉线如何排查？"},
            "call-knowledge",
        ),
        make_call(
            "call_ticket_agent",
            {"order_id": ORDER_ID},
            "call-ticket",
        ),
    ])

    supervisor, tools, llm, order, knowledge, ticket = env

    result = supervisor.run(
        user_question=QUESTION,
        tools=tools,
        task_type="order_knowledge_ticket",
    )

    assert result.status == "pending_confirmation"

    assert [
        item["tool"]
        for item in result.data["tool_trace"]
    ] == [
        "call_order_agent",
        "call_knowledge_agent",
        "call_ticket_agent",
    ]

    assert len(result.evidence) == 1
    assert llm.invoke.call_count == 3

    order.run.assert_called_once()
    knowledge.run.assert_called_once()
    ticket.run.assert_called_once()


def test_ticket_before_order_is_rejected():
    """不能在订单验证之前执行 Ticket Agent。"""

    env = make_environment([
        make_call(
            "call_ticket_agent",
            {"order_id": ORDER_ID},
            "call-ticket",
        ),
    ])

    supervisor, tools, _, order, _, ticket = env

    result = supervisor.run(
        user_question=QUESTION,
        tools=tools,
        task_type="order_ticket",
    )

    assert result.status == "parameter_conflict"
    assert result.error == "order_verification_required"

    order.run.assert_not_called()
    ticket.run.assert_not_called()


def test_ticket_order_mismatch_is_rejected():
    """LLM 不能替换用户指定的工单订单号。"""

    env = make_environment([
        make_call(
            "call_order_agent",
            {"order_id": ORDER_ID},
            "call-order",
        ),
        make_call(
            "call_ticket_agent",
            {"order_id": "DEMO-9999"},
            "call-ticket",
        ),
    ])

    supervisor, tools, _, order, _, ticket = env

    result = supervisor.run(
        user_question=QUESTION,
        tools=tools,
        task_type="order_ticket",
    )

    assert result.status == "parameter_conflict"
    assert result.error == "ticket_order_id_mismatch"

    order.run.assert_called_once()
    ticket.run.assert_not_called()


def test_ticket_not_allowed_in_order_only():
    """不能在纯订单任务中执行工单工具。"""

    env = make_environment([
        make_call(
            "call_ticket_agent",
            {"order_id": ORDER_ID},
            "call-ticket",
        ),
    ])

    supervisor, tools, _, _, _, ticket = env

    result = supervisor.run(
        user_question=QUESTION,
        tools=tools,
        task_type="order_only",
    )

    assert result.status == "error"
    assert result.error == "tool_not_allowed_for_task"

    ticket.run.assert_not_called()


def test_three_agent_flow_cannot_finish_early():
    """三 Agent 任务不能只完成 Order 和 Knowledge。"""

    env = make_environment([
        make_call(
            "call_order_agent",
            {"order_id": ORDER_ID},
            "call-order",
        ),
        make_call(
            "call_knowledge_agent",
            {"question": "如何排查 Wi-Fi 掉线？"},
            "call-knowledge",
        ),
        AIMessage(content="全部任务已经完成。"),
    ])

    supervisor, tools, llm, _, knowledge, ticket = env

    result = supervisor.run(
        user_question=QUESTION,
        tools=tools,
        task_type="order_knowledge_ticket",
    )

    assert result.status != "success"

    knowledge.run.assert_called_once()
    ticket.run.assert_not_called()

    assert llm.invoke.call_count == 3
