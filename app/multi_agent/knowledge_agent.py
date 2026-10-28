from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from app.llm_client import LLMClient
from app.rag_tool import (
    rag_tool_schema,
    search_knowledge_base,
)

from .contracts import AgentTask, AgentResult


KNOWLEDGE_PROMPT = """
你是 Knowledge Agent，专门负责网络设备技术知识检索。

你的职责：
1. 根据技术问题判断是否需要检索知识库。
2. 需要技术资料时，调用 search_knowledge_base。
3. 设备型号、硬件版本、销售地区缺失时，不得自行猜测。
4. 技术事实与操作建议必须依据适用的检索证据。
5. 无证据或检索失败时，明确说明限制。
6. 每轮最多请求一次 search_knowledge_base。如果需要多个检索问题，应分轮调用，不能在同一轮生成多个 Tool Call。

你不负责订单查询、售后资格判断或工单操作。
"""


class KnowledgeAgent:
    def __init__(
        self,
        llm: LLMClient | None = None,
        retrieve: Callable[..., dict[str, Any]] | None = None,
    ):
        self.llm = llm if llm is not None else LLMClient()
        self.schemas = [rag_tool_schema()]

        # 默认使用真实 RAG 工具，测试时可以注入 Mock
        self.retrieve = (
            retrieve if retrieve is not None
            else search_knowledge_base
        )

    def decide(self, task: AgentTask) -> AIMessage:
        """执行第一轮 LLM 决策，暂不执行工具。"""

        request = {
            "technical_task": task.task,
            "device_scope": task.product_scope,
        }

        messages = [
            HumanMessage(
                content=json.dumps(request, ensure_ascii=False)
            )
        ]

        response = self.llm.invoke(
            messages,
            self.schemas,
            system_prompt=KNOWLEDGE_PROMPT,
        )

        return response
    
    def run(self, task: AgentTask, max_steps: int = 4) -> AgentResult:
        """执行 Knowledge Agent 的完整检索循环。"""

        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        messages = [
            HumanMessage(
                content=json.dumps(
                    {
                        "technical_task": task.task,
                        "device_scope": task.product_scope,
                    },
                    ensure_ascii=False,
                )
            )
        ]

        retrieval_status = None
        retrieval_message = ""
        evidence = []

        allowed_args = {
            "question", "top_k",
            "product_model", "hardware_version", "region",
        }
        scope_fields = (
            "product_model", "hardware_version", "region"
        )

        for _ in range(max_steps):
            response = self.llm.invoke(
                messages,
                self.schemas,
                system_prompt=KNOWLEDGE_PROMPT,
            )
            messages.append(response)

            # LLM 不再调用工具，说明本轮决策结束
            if not response.tool_calls:
                if retrieval_status == "ok":
                    return AgentResult(
                        agent="knowledge_agent",
                        status="success",
                        summary=response.content or "检索已完成。",
                        evidence=evidence,
                    )

                if retrieval_status in {
                    "needs_clarification",
                    "parameter_conflict",
                    "unsupported_product",
                    "no_evidence",
                }:
                    return AgentResult(
                        agent="knowledge_agent",
                        status=retrieval_status,
                        summary=retrieval_message or "本次没有可用证据。",
                    )

                return AgentResult(
                    agent="knowledge_agent",
                    status=(
                        "needs_clarification"
                        if len(task.product_scope) < 3
                        else "no_evidence"
                    ),
                    summary="尚未获得经过检索验证的技术证据。",
                )

            # 第一版每轮仅允许一个 Tool Call
            
            # 模型一次请求多个工具时，要求重新决策
            if len(response.tool_calls) > 1:

                for call in response.tool_calls:
                    call_id = call.get("id")
                    name = call.get("name")

                    if not isinstance(call_id, str) or not call_id.strip():
                        return AgentResult(
                            agent="knowledge_agent",
                            status="error",
                            summary="工具调用缺少有效 ID。",
                        )

                    messages.append(
                        ToolMessage(
                            content=json.dumps(
                                {
                                    "status": "error",
                                    "message": (
                                        "本轮未执行检索。"
                                        "每轮只能请求一次 "
                                        "search_knowledge_base，"
                                        "请重新选择一个检索问题。"
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                            tool_call_id=call_id,
                            name=(
                                name
                                if isinstance(name, str)
                                else "invalid_tool"
                            ),
                        )
                    )

                continue


            call = response.tool_calls[0]
            name = call.get("name")
            args = call.get("args", {})

            # 执行层再次检查工具权限
            if name != "search_knowledge_base":
                return AgentResult(
                    agent="knowledge_agent",
                    status="error",
                    summary="子 Agent 请求了未授权工具。",
                    error=f"unauthorized tool: {name}",
                )

            if not isinstance(args, dict) or set(args) - allowed_args:
                return AgentResult(
                    agent="knowledge_agent",
                    status="error",
                    summary="检索工具参数不合法。",
                )

            args = dict(args)

            # 不允许 LLM 擅自增加或修改设备范围
            for field in scope_fields:
                expected = task.product_scope.get(field)
                supplied = args.get(field)

                if expected is None:
                    args.pop(field, None)
                else:
                    if not isinstance(expected, str):
                        return AgentResult(
                            agent="knowledge_agent",
                            status="error",
                            summary="设备范围参数类型不合法。",
                        )
                    if supplied is not None and (
                        not isinstance(supplied, str)
                        or supplied.strip().casefold()
                        != expected.strip().casefold()
                    ):
                        return AgentResult(
                            agent="knowledge_agent",
                            status="parameter_conflict",
                            summary="检索参数与任务设备范围不一致。",
                        )
                    args[field] = expected

            try:
                result = self.retrieve(**args)

                if not isinstance(result, dict):
                    raise ValueError("RAG result must be a dict")

                retrieval_status = result.get("status")
                if retrieval_status is None:
                    retrieval_status = (
                        "ok" if result.get("has_results")
                        else "no_evidence"
                    )

                valid_statuses = {
                    "ok", "needs_clarification",
                    "parameter_conflict",
                    "unsupported_product", "no_evidence",
                }
                if retrieval_status not in valid_statuses:
                    raise ValueError("Unknown RAG retrieval status")

                evidence = (
                    result.get("results", [])
                    if retrieval_status == "ok"
                    else []
                )
                if not isinstance(evidence, list):
                    raise ValueError("Invalid RAG evidence")

                if retrieval_status == "ok" and not evidence:
                    retrieval_status = "no_evidence"

                retrieval_message = result.get("message", "")

            except Exception as exc:
                return AgentResult(
                    agent="knowledge_agent",
                    status="error",
                    summary="知识库检索执行失败。",
                    error=f"{type(exc).__name__}: {exc}",
                )

            # 把真实工具结果返回给 LLM
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=call["id"],
                    name=name,
                )
            )

        return AgentResult(
            agent="knowledge_agent",
            status="error",
            summary="Knowledge Agent 达到最大模型调用步数。",
            error="max_steps_exceeded",
        )

