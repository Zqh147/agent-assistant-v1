
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import HumanMessage

from app.llm_client import (
    LLMClient,
    SYSTEM_PROMPT,
    SUPPORT_PROMPT,
)


def test_independent_agent_prompt():
    # 模拟 LLM 的响应，不调用真实 API
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="测试回答",
                    tool_calls=[],
                )
            )
        ]
    )

    fake_create = Mock(return_value=fake_response)

    llm = LLMClient(
        runtime_settings=SimpleNamespace(
            api_key=None,
            support_mode="demo",
            model="deepseek-flash",
        )
    )

    llm._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=fake_create
            )
        )
    )

    messages = [HumanMessage(content="测试问题")]

    # 测试一：独立 Agent 的 Prompt
    custom_prompt = "你是 Knowledge Agent，只负责知识检索。"

    llm.invoke(
        messages,
        [],
        system_prompt=custom_prompt,
    )

    payload = fake_create.call_args.kwargs["messages"]

    assert payload[0]["content"] == custom_prompt

    # 测试二：旧版调用仍使用原来的 Prompt
    llm.invoke(messages, [])

    payload = fake_create.call_args.kwargs["messages"]

    assert payload[0]["content"] == SYSTEM_PROMPT + SUPPORT_PROMPT
