from unittest.mock import Mock
from langchain_core.messages import AIMessage, ToolMessage
import json
from unittest.mock import Mock
from app.multi_agent.contracts import AgentTask
from app.multi_agent.knowledge_agent import (
    KnowledgeAgent,
    KNOWLEDGE_PROMPT,
)


def test_knowledge_agent_decision():
    fake_llm = Mock()
    fake_llm.invoke.return_value = AIMessage(content="测试响应")

    agent = KnowledgeAgent(llm=fake_llm)

    task = AgentTask(
        task="Archer AX20 为什么经常掉线？",
        product_scope={
            "product_model": "Archer AX20",
            "hardware_version": "V1",
            "region": "US",
        },
    )

    response = agent.decide(task)

    assert response.content == "测试响应"

    args, kwargs = fake_llm.invoke.call_args

    # 验证只暴露了 RAG 工具
    tool_names = [
        item["function"]["name"]
        for item in args[1]
    ]
    assert tool_names == ["search_knowledge_base"]

    # 验证使用独立 Prompt
    assert kwargs["system_prompt"] == KNOWLEDGE_PROMPT

    # 验证任务及设备字段正确传递
    request = json.loads(args[0][0].content)
    assert request["technical_task"] == task.task
    assert request["device_scope"] == task.product_scope



def test_knowledge_agent_full_loop():


    fake_llm = Mock()
    fake_llm.invoke.side_effect = [
        # 第一轮：LLM 请求调用检索工具
        AIMessage(
            content="",
            tool_calls=[{
                "name": "search_knowledge_base",
                "args": {
                    "question": "AX20 经常掉线怎么办？",
                    "top_k": 3,
                },
                "id": "call-rag-001",
            }],
        ),

        # 第二轮：LLM 已经拿到工具结果
        AIMessage(content="根据检索到的设备资料，已整理相关建议。"),
    ]

    fake_retrieve = Mock(return_value={
        "status": "ok",
        "backend": "project1",
        "results": [
            {
                "content": "设备掉线排障参考资料",
                "source": "manual",
                "evidence_id": "test-evidence-001",
                "chunk_id": 0,
            }
        ],
    })

    agent = KnowledgeAgent(
        llm=fake_llm,
        retrieve=fake_retrieve,
    )

    task = AgentTask(
        task="查询 AX20 频繁掉线的处理方法",
        product_scope={
            "product_model": "Archer AX20",
            "hardware_version": "V1",
            "region": "US",
        },
    )

    result = agent.run(task)

    # 模型进行了两轮决策
    assert fake_llm.invoke.call_count == 2

    # 检索工具被实际执行了一次
    fake_retrieve.assert_called_once()

    # 设备范围从任务上下文传入
    actual_args = fake_retrieve.call_args.kwargs
    assert actual_args["product_model"] == "Archer AX20"
    assert actual_args["hardware_version"] == "V1"
    assert actual_args["region"] == "US"

    # 第二轮 LLM 收到了 ToolMessage
    second_messages = fake_llm.invoke.call_args_list[1].args[0]
    assert isinstance(second_messages[-2], ToolMessage)
    assert second_messages[-2].tool_call_id == "call-rag-001"

    # Knowledge Agent 返回统一结果
    assert result.status == "success"
    assert result.agent == "knowledge_agent"
    assert len(result.evidence) == 1


def test_knowledge_agent_rejects_unauthorized_tool():
    from unittest.mock import Mock
    from langchain_core.messages import AIMessage

    fake_llm = Mock()
    fake_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "submit_ticket",
            "args": {"ticket_id": "TKT-001"},
            "id": "call-unauthorized",
        }],
    )

    fake_retrieve = Mock()

    agent = KnowledgeAgent(
        llm=fake_llm,
        retrieve=fake_retrieve,
    )

    task = AgentTask(
        task="查询 AX20 的技术资料",
        product_scope={
            "product_model": "Archer AX20",
            "hardware_version": "V1",
            "region": "US",
        },
    )

    result = agent.run(task)

    assert result.status == "error"
    assert "unauthorized tool" in result.error

    # 确认没有执行任何 RAG 检索
    fake_retrieve.assert_not_called()


def test_knowledge_agent_rejects_scope_conflict():
    from unittest.mock import Mock
    from langchain_core.messages import AIMessage

    fake_llm = Mock()
    fake_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_knowledge_base",
            "args": {
                "question": "如何修改 Wi-Fi 密码？",
                "product_model": "Archer AX10",
            },
            "id": "call-conflict",
        }],
    )

    fake_retrieve = Mock()

    agent = KnowledgeAgent(
        llm=fake_llm,
        retrieve=fake_retrieve,
    )

    task = AgentTask(
        task="查询 AX20 修改 Wi-Fi 密码的方法",
        product_scope={
            "product_model": "Archer AX20",
            "hardware_version": "V1",
            "region": "US",
        },
    )

    result = agent.run(task)

    assert result.status == "parameter_conflict"

    # 参数冲突时，不能执行检索
    fake_retrieve.assert_not_called()


def test_knowledge_agent_handles_rag_failure():
    from unittest.mock import Mock
    from langchain_core.messages import AIMessage

    fake_llm = Mock()
    fake_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_knowledge_base",
            "args": {
                "question": "AX20 经常掉线怎么办？"
            },
            "id": "call-error",
        }],
    )

    fake_retrieve = Mock(
        side_effect=ConnectionError(
            "RAG service unavailable"
        )
    )

    agent = KnowledgeAgent(
        llm=fake_llm,
        retrieve=fake_retrieve,
    )

    task = AgentTask(
        task="查询 AX20 频繁掉线的解决方法",
        product_scope={
            "product_model": "Archer AX20",
            "hardware_version": "V1",
            "region": "US",
        },
    )

    result = agent.run(task)

    assert result.status == "error"
    assert "ConnectionError" in result.error

    fake_retrieve.assert_called_once()


def test_knowledge_agent_no_evidence():
    fake_llm = Mock()

    fake_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "search_knowledge_base",
                "args": {
                    "question": "查询设备技术资料"
                },
                "id": "call-no-evidence",
            }],
        ),
        AIMessage(content="没有找到相关证据。"),
    ]

    fake_retrieve = Mock(return_value={
        "status": "no_evidence",
        "backend": "project1",
        "results": [],
        "has_results": False,
        "message": "本次没有找到符合条件的技术证据。",
    })

    agent = KnowledgeAgent(
        llm=fake_llm,
        retrieve=fake_retrieve,
    )

    task = AgentTask(
        task="查询设备技术资料",
        product_scope={
            "product_model": "Archer AX20",
            "hardware_version": "V1",
            "region": "US",
        },
    )

    result = agent.run(task)

    assert result.status == "no_evidence"
    assert result.evidence == []
    assert "没有找到" in result.summary

    fake_retrieve.assert_called_once()
    assert fake_llm.invoke.call_count == 2


def test_knowledge_agent_max_steps():
    fake_llm = Mock()

    fake_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_knowledge_base",
            "args": {
                "question": "继续查询设备信息"
            },
            "id": "call-loop",
        }],
    )

    fake_retrieve = Mock(return_value={
        "status": "ok",
        "backend": "project1",
        "results": [{
            "content": "模拟设备资料",
            "source": "manual",
            "chunk_id": 0,
        }],
    })

    agent = KnowledgeAgent(
        llm=fake_llm,
        retrieve=fake_retrieve,
    )

    task = AgentTask(
        task="查询路由器技术问题",
        product_scope={
            "product_model": "Archer AX20",
            "hardware_version": "V1",
            "region": "US",
        },
    )

    result = agent.run(task, max_steps=1)

    assert result.status == "error"
    assert result.error == "max_steps_exceeded"

    # 不允许无限调用模型
    assert fake_llm.invoke.call_count == 1

    # 当前版本允许最后一轮执行工具，
    # 但不会再次调用模型
    fake_retrieve.assert_called_once()
