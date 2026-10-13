import json
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.llm_client import LLMClient


def test_online_tool_call_round_trip():
    client = LLMClient()
    create = Mock(side_effect=[
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, tool_calls=[SimpleNamespace(id="calc-1", function=SimpleNamespace(
                name="calculate", arguments='{"a":23,"b":47,"operation":"multiply"}'
            ))]
        ))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="1081", tool_calls=[]))]),
    ])
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    human = HumanMessage(content="23 * 47")
    response = client.invoke([human], [])
    assert isinstance(response, AIMessage)
    assert response.tool_calls[0]["type"] == "tool_call"
    answer = client.invoke([human, response, ToolMessage(content="1081", tool_call_id="calc-1")], [])
    payload = create.call_args.kwargs["messages"]
    sent_call = payload[2]["tool_calls"][0]
    assert sent_call["type"] == "function"
    assert sent_call["id"] == payload[3]["tool_call_id"]
    assert json.loads(sent_call["function"]["arguments"]) == {"a": 23, "b": 47, "operation": "multiply"}
    assert answer.content == "1081"
