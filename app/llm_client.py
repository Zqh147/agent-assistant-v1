from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage

from .config import settings

SYSTEM_PROMPT = """你是一个可靠的任务执行 Agent。
数值运算必须调用 calculate，包括简单加减乘除；不要自行心算代替工具。
用户明确要求使用工具时，必须调用对应工具；需要时先从历史对话确定参数。
查询当前时间时调用时间工具，需要知识库证据时调用检索工具。
收到本次所需的工具结果后，结合结果回答用户，无需重复计算。
如果工具失败，要明确说明失败原因。不要编造知识库中不存在的事实。"""


class LLMClient:
    def __init__(self, runtime_settings=None) -> None:
        self.settings = runtime_settings or settings
        self._client = None
        if self.settings.api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.settings.api_key, base_url=self.settings.base_url)

    @property
    def online(self) -> bool:
        return self._client is not None

    def invoke(self, messages: list[Any], tools: list[dict[str, Any]]) -> AIMessage:
        if self._client is None:
            return self._offline(messages)
        payload: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for message in messages:
            role = {"human": "user", "ai": "assistant", "tool": "tool"}.get(getattr(message, "type", "user"), "user")
            item: dict[str, Any] = {"role": role, "content": getattr(message, "content", "")}
            if getattr(message, "tool_calls", None):
                # LangChain stores args as a dict; the API needs function.arguments
                # as a JSON string and type='function'. Preserve the call ID.
                item["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["args"], ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ]
            if role == "tool":
                item["tool_call_id"] = getattr(message, "tool_call_id", "")
            payload.append(item)
        response = self._client.chat.completions.create(model=self.settings.model, messages=payload, tools=tools, temperature=0)
        choice = response.choices[0].message
        calls = []
        for call in choice.tool_calls or []:
            calls.append({"name": call.function.name, "args": json.loads(call.function.arguments or "{}"), "id": call.id, "type": "tool_call"})
        return AIMessage(content=choice.content or "", tool_calls=calls)

    def _offline(self, messages: list[Any]) -> AIMessage:
        last = messages[-1]
        text = str(getattr(last, "content", ""))
        if getattr(last, "type", "") == "tool":
            return AIMessage(content=f"工具已经返回结果：{last.content}")
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:\*|×|乘以)\s*(-?\d+(?:\.\d+)?)", text)
        if match:
            return AIMessage(content="", tool_calls=[{"name": "calculate", "args": {"a": float(match.group(1)), "b": float(match.group(2)), "operation": "multiply"}, "id": "offline-calc", "type": "tool_call"}])
        if any(key in text for key in ("几点", "时间", "现在")):
            return AIMessage(content="", tool_calls=[{"name": "get_current_time", "args": {}, "id": "offline-time", "type": "tool_call"}])
        if any(key in text.lower() for key in ("rag", "向量", "embedding", "agent", "mcp", "langgraph")):
            return AIMessage(content="", tool_calls=[{"name": "search_knowledge_base", "args": {"question": text, "top_k": 3}, "id": "offline-rag", "type": "tool_call"}])
        return AIMessage(content="这是一个不需要调用外部工具的直接回答：" + text)
