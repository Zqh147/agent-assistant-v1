import pytest

from app.multi_agent.contracts import AgentResult, AgentTask
from app.multi_agent.supervisor import extract_order_scope
from unittest.mock import Mock


def make_order_result(order_id="DEMO-1001"):
    return AgentResult(
        agent="order_agent",
        status="success",
        summary="订单查询完成",
        data={
            "tool_results": [
                {
                    "tool": "lookup_order",
                    "result": {
                        "status": "found",
                        "order": {"order_id": order_id},
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


def test_valid_order_scope():
    result = make_order_result()

    scope = extract_order_scope(result, "DEMO-1001")

    assert scope == {
        "product_model": "Archer AX10",
        "hardware_version": "V1",
        "region": "US",
    }


def test_wrong_order_is_rejected():
    result = make_order_result("DEMO-1002")

    with pytest.raises(ValueError):
        extract_order_scope(result, "DEMO-1001")


def test_no_verified_order_is_rejected():
    result = AgentResult(
        agent="order_agent",
        status="success",
        summary="没有查询订单",
        data={"tool_results": []},
    )

    with pytest.raises(ValueError):
        extract_order_scope(result, "DEMO-1001")


from unittest.mock import Mock

from app.multi_agent.supervisor import Supervisor


def test_supervisor_order_then_knowledge():
    # 创建两个模拟 Agent
    order_agent = Mock()
    knowledge_agent = Mock()

    # 模拟 Order Agent 查询到真实订单信息
    order_agent.run.return_value = make_order_result()

    # 模拟 Knowledge Agent 返回技术检索结果
    knowledge_agent.run.return_value = AgentResult(
        agent="knowledge_agent",
        status="success",
        summary="已检索到适用的 Wi-Fi 排障资料。",
        evidence=[],
    )

    supervisor = Supervisor(
        order_agent=order_agent,
        knowledge_agent=knowledge_agent,
    )

    # 执行任务
    result = supervisor.run(
        order_id="DEMO-1001",
        technical_question="如何排查 Wi-Fi 频繁掉线？",
        trusted_context={
            "customer_id": "demo-customer-001"
        },
    )

    # 检查最终结果
    assert result.status == "success"

    # 检查两个子 Agent 都被调用
    order_agent.run.assert_called_once()
    knowledge_agent.run.assert_called_once()

    # 检查传递给 Knowledge Agent 的参数
    knowledge_task = knowledge_agent.run.call_args.args[0]

    assert isinstance(knowledge_task, AgentTask)

    assert knowledge_task.task == "如何排查 Wi-Fi 频繁掉线？"

    assert knowledge_task.product_scope == {
        "product_model": "Archer AX10",
        "hardware_version": "V1",
        "region": "US",
    }


def test_order_failure_stops_knowledge():
    order_agent = Mock()
    knowledge_agent = Mock()

    order_agent.run.return_value = AgentResult(
        agent="order_agent",
        status="error",
        summary="订单查询失败",
        error="database_unavailable",
    )

    supervisor = Supervisor(order_agent, knowledge_agent)

    result = supervisor.run(
        "DEMO-1001",
        "如何排查 Wi-Fi 掉线？",
        trusted_context={"customer_id": "demo-customer-001"},
    )

    assert result.status == "error"
    knowledge_agent.run.assert_not_called()


def test_wrong_order_stops_knowledge():
    order_agent = Mock()
    knowledge_agent = Mock()

    order_agent.run.return_value = make_order_result(
        "DEMO-1002"
    )

    supervisor = Supervisor(order_agent, knowledge_agent)

    result = supervisor.run(
        "DEMO-1001",
        "如何排查 Wi-Fi 掉线？",
        trusted_context={"customer_id": "demo-customer-001"},
    )

    assert result.status == "no_evidence"
    knowledge_agent.run.assert_not_called()


def test_order_exception_is_handled():
    order_agent = Mock()
    knowledge_agent = Mock()

    order_agent.run.side_effect = RuntimeError(
        "Database unavailable"
    )

    supervisor = Supervisor(order_agent, knowledge_agent)

    result = supervisor.run(
        "DEMO-1001",
        "如何排查 Wi-Fi 掉线？",
        trusted_context={"customer_id": "demo-customer-001"},
    )

    assert result.status == "error"
    assert "order_agent_failed" in result.error
    knowledge_agent.run.assert_not_called()


def test_knowledge_exception_is_handled():
    order_agent = Mock()
    knowledge_agent = Mock()

    order_agent.run.return_value = make_order_result()
    knowledge_agent.run.side_effect = RuntimeError(
        "RAG service unavailable"
    )

    supervisor = Supervisor(order_agent, knowledge_agent)

    result = supervisor.run(
        "DEMO-1001",
        "如何排查 Wi-Fi 掉线？",
        trusted_context={"customer_id": "demo-customer-001"},
    )

    assert result.status == "error"
    assert "knowledge_agent_failed" in result.error

    order_agent.run.assert_called_once()
    knowledge_agent.run.assert_called_once()
