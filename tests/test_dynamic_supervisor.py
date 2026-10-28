from unittest.mock import Mock

from langchain_core.messages import AIMessage

from app.multi_agent.contracts import AgentResult
from app.multi_agent.dynamic_supervisor import DynamicSupervisor
from app.multi_agent.supervisor_tools import SupervisorTools


def make_tools():
    return SupervisorTools(
        order_agent=Mock(),
        knowledge_agent=Mock(),
        trusted_context={
            "customer_id": "demo-customer-001"
        },
    )


def test_llm_cannot_answer_without_required_tools():
    """模型直接回答，不能绕过必要的业务查询。"""

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content="订单已查询，AP模式不支持QoS。"
    )

    tools = make_tools()

    result = DynamicSupervisor(llm).run(
        "查询订单DEMO-1001，并解释AP模式的QoS限制",
        tools,
    )

    assert result.status == "no_evidence"
    tools.order_agent.run.assert_not_called()
    tools.knowledge_agent.run.assert_not_called()


def test_knowledge_requires_verified_order():
    """订单关联检索必须先验证订单。"""

    llm = Mock()

    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_001",
                    "name": "call_knowledge_agent",
                    "args": {
                        "question": "AP模式是否支持QoS？"
                    },
                }
            ],
        ),
        AIMessage(content="任务结束"),
    ]

    tools = make_tools()

    result = DynamicSupervisor(llm).run(
        "查询订单DEMO-1001，并解释AP模式的QoS限制",
        tools,
    )

    assert result.status == "needs_clarification"
    tools.knowledge_agent.run.assert_not_called()


def test_mismatched_order_result_is_rejected():
    """业务工具返回的订单号不一致时拒绝使用设备范围。"""

    llm = Mock()

    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_002",
                    "name": "call_order_agent",
                    "args": {
                        "order_id": "DEMO-1001"
                    },
                }
            ],
        ),
        AIMessage(content="任务结束"),
    ]

    tools = make_tools()

    tools.order_agent.run.return_value = AgentResult(
        agent="order_agent",
        status="success",
        summary="订单查询完成",
        data={
            "tool_results": [
                {
                    "tool": "lookup_order",
                    "result": {
                        "status": "found",
                        "order": {
                            "order_id": "DEMO-1002"
                        },
                        "product_scope": {
                            "product_model": "Archer AX10",
                            "hardware_version": "V1",
                            "region": "US",
                        },
                    },
                }
            ]
        },
    )

    result = DynamicSupervisor(llm).run(
        "查询订单DEMO-1001，并解释AP模式的QoS限制",
        tools,
    )

    assert result.status == "no_evidence"
    assert tools.verified_product_scope is None
    tools.knowledge_agent.run.assert_not_called()


def test_llm_wrong_order_id_is_rejected():
    """LLM 不能把用户请求的订单替换为其他订单。"""

    llm = Mock()

    llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_wrong_order",
                "name": "call_order_agent",
                "args": {
                    "order_id": "DEMO-1002"
                },
            }
        ],
    )

    tools = make_tools()

    result = DynamicSupervisor(llm).run(
        "查询订单DEMO-1001，并解释AP模式的QoS限制",
        tools,
    )

    assert result.status == "parameter_conflict"
    assert result.error == "order_id_mismatch"

    tools.order_agent.run.assert_not_called()
    tools.knowledge_agent.run.assert_not_called()


def test_multiple_order_ids_need_clarification():
    """当前单订单工作流不能擅自选择多个订单中的一个。"""

    llm = Mock()
    tools = make_tools()

    result = DynamicSupervisor(llm).run(
        "请查询DEMO-1001和DEMO-1002的设备功能",
        tools,
    )

    assert result.status == "needs_clarification"
    llm.invoke.assert_not_called()
    tools.order_agent.run.assert_not_called()



def test_llm_wrong_order_id_is_rejected():
    """LLM 不能把用户请求的订单替换为其他订单。"""

    llm = Mock()

    llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_wrong_order",
                "name": "call_order_agent",
                "args": {
                    "order_id": "DEMO-1002"
                },
            }
        ],
    )

    tools = make_tools()

    result = DynamicSupervisor(llm).run(
        "查询订单DEMO-1001，并解释AP模式的QoS限制",
        tools,
    )

    assert result.status == "parameter_conflict"
    assert result.error == "order_id_mismatch"

    tools.order_agent.run.assert_not_called()
    tools.knowledge_agent.run.assert_not_called()


def test_multiple_order_ids_need_clarification():
    """当前单订单工作流不能擅自选择多个订单中的一个。"""

    llm = Mock()
    tools = make_tools()

    result = DynamicSupervisor(llm).run(
        "请查询DEMO-1001和DEMO-1002的设备功能",
        tools,
    )

    assert result.status == "needs_clarification"
    llm.invoke.assert_not_called()
    tools.order_agent.run.assert_not_called()



def test_order_only_task_succeeds():
    """纯订单任务只需 Order Agent 成功。"""

    llm = Mock()

    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_order_001",
                    "name": "call_order_agent",
                    "args": {
                        "order_id": "DEMO-1001",
                    },
                }
            ],
        ),
        AIMessage(content="订单查询完成。"),
    ]

    tools = make_tools()

    tools.order_agent.run.return_value = AgentResult(
        agent="order_agent",
        status="success",
        summary="订单查询成功",
        data={
            "tool_results": [
                {
                    "tool": "lookup_order",
                    "result": {
                        "status": "found",
                        "order": {
                            "order_id": "DEMO-1001",
                        },
                        "product_scope": {
                            "product_model": "Archer AX10",
                            "hardware_version": "V1",
                            "region": "US",
                        },
                    },
                }
            ]
        },
    )

    result = DynamicSupervisor(llm).run(
        "查询订单 DEMO-1001 的设备型号",
        tools,
        task_type="order_only",
    )

    assert result.status == "success"
    assert result.data["order_id"] == "DEMO-1001"
    assert "Archer AX10" in result.summary

    assert len(result.data["tool_trace"]) == 1
    assert result.data["tool_trace"][0]["tool"] == "call_order_agent"

    tools.order_agent.run.assert_called_once()
    tools.knowledge_agent.run.assert_not_called()


def test_order_only_rejects_knowledge_agent():
    """纯订单任务禁止执行不必要的知识检索。"""

    llm = Mock()

    llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_knowledge_001",
                "name": "call_knowledge_agent",
                "args": {
                    "question": "AP 模式是否支持 QoS？",
                },
            }
        ],
    )

    tools = make_tools()

    result = DynamicSupervisor(llm).run(
        "查询订单 DEMO-1001 的设备型号",
        tools,
        task_type="order_only",
    )

    assert result.status == "error"
    assert result.error == "tool_not_allowed_for_task"

    tools.order_agent.run.assert_not_called()
    tools.knowledge_agent.run.assert_not_called()


def test_public_knowledge_scope_is_passed_to_agent():
    """公开知识模式能正确传递指定设备范围。"""

    knowledge_agent = Mock()

    knowledge_agent.run.return_value = AgentResult(
        agent="knowledge_agent",
        status="success",
        summary="模拟知识检索完成。",
        evidence=[{"source": "mock_document"}],
    )

    tools = SupervisorTools(
        order_agent=Mock(),
        knowledge_agent=knowledge_agent,
        trusted_context={
            "customer_id": "demo-customer-001"
        },
        public_product_scope={
            "product_model": "Archer AX10",
            "hardware_version": "V1",
            "region": "US",
        },
    )

    result = tools.call_knowledge_agent(
        "如何配置AP模式？",
        allow_public=True,
    )

    assert result["status"] == "success"
    assert result["order_id"] is None
    assert result["scope_source"] == "public_input"

    task = knowledge_agent.run.call_args.args[0]

    assert task.product_scope == {
        "product_model": "Archer AX10",
        "hardware_version": "V1",
        "region": "US",
    }


def test_public_scope_cannot_bypass_order_mode():
    """仅配置公开设备范围不能绕过订单关联模式。"""

    knowledge_agent = Mock()

    tools = SupervisorTools(
        order_agent=Mock(),
        knowledge_agent=knowledge_agent,
        trusted_context={
            "customer_id": "demo-customer-001"
        },
        public_product_scope={
            "product_model": "Archer AX10",
            "hardware_version": "V1",
            "region": "US",
        },
    )

    result = tools.call_knowledge_agent(
        "如何配置AP模式？"
    )

    assert result["status"] == "needs_clarification"
    knowledge_agent.run.assert_not_called()


def make_public_tools():
    knowledge_agent = Mock()

    knowledge_agent.run.return_value = AgentResult(
        agent="knowledge_agent",
        status="success",
        summary="AP 模式配置方法已检索完成。",
        evidence=[{"source": "mock_document"}],
    )

    return SupervisorTools(
        order_agent=Mock(),
        knowledge_agent=knowledge_agent,
        trusted_context={
            "customer_id": "demo-customer-001"
        },
        public_product_scope={
            "product_model": "Archer AX10",
            "hardware_version": "V1",
            "region": "US",
        },
    )


def test_knowledge_only_task_succeeds():
    llm = Mock()

    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_k001",
                    "name": "call_knowledge_agent",
                    "args": {
                        "question": "如何配置 AP 模式？"
                    },
                }
            ],
        ),
        AIMessage(content="查询完成。"),
    ]

    tools = make_public_tools()

    result = DynamicSupervisor(llm).run(
        "Archer AX10 V1 US 如何配置 AP 模式？",
        tools,
        task_type="knowledge_only",
    )

    assert result.status == "success"
    assert result.data["scope_source"] == "public_input"
    assert len(result.evidence) == 1

    assert len(result.data["tool_trace"]) == 1
    assert result.data["tool_trace"][0]["tool"] == "call_knowledge_agent"

    tools.order_agent.run.assert_not_called()
    tools.knowledge_agent.run.assert_called_once()

    task = tools.knowledge_agent.run.call_args.args[0]
    assert task.product_scope["product_model"] == "Archer AX10"

    schemas = llm.invoke.call_args_list[0].args[1]
    assert [
        s["function"]["name"] for s in schemas
    ] == ["call_knowledge_agent"]


def test_knowledge_only_rejects_order_agent():
    llm = Mock()

    llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_o001",
                "name": "call_order_agent",
                "args": {
                    "order_id": "DEMO-1001"
                },
            }
        ],
    )

    tools = make_public_tools()

    result = DynamicSupervisor(llm).run(
        "Archer AX10 V1 US 如何配置 AP 模式？",
        tools,
        task_type="knowledge_only",
    )

    assert result.status == "error"
    assert result.error == "tool_not_allowed_for_task"

    tools.order_agent.run.assert_not_called()
    tools.knowledge_agent.run.assert_not_called()


def test_knowledge_only_requires_public_scope():
    llm = Mock()

    tools = SupervisorTools(
        order_agent=Mock(),
        knowledge_agent=Mock(),
        trusted_context={
            "customer_id": "demo-customer-001"
        },
    )

    result = DynamicSupervisor(llm).run(
        "如何配置 AP 模式？",
        tools,
        task_type="knowledge_only",
    )

    assert result.status == "needs_clarification"
    llm.invoke.assert_not_called()
    tools.knowledge_agent.run.assert_not_called()

