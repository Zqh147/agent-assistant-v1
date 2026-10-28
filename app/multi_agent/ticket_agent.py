from copy import deepcopy
from typing import Any, Mapping
from ..support_tools import SUPPORT_SCHEMAS, SupportTools

import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..messages import latest_tool_calls
from .contracts import AgentResult



# Ticket Agent 允许调用的业务工具
TICKET_ALLOWED_TOOLS = frozenset({
    "prepare_ticket",
    "submit_ticket",
    "get_ticket",
})


class TicketAgent:
    def __init__(self, llm, support_tools: SupportTools):

        # 1. 依赖注入
        self.llm = llm
        self.support_tools = support_tools

        # 2. 只向 LLM 提供 Ticket Agent 允许的工具
        self.schemas = [
            deepcopy(schema)
            for schema in SUPPORT_SCHEMAS
            if schema["function"]["name"] in TICKET_ALLOWED_TOOLS
        ]

        # 3. 提取实际注册的工具名称
        schema_names = [
            schema["function"]["name"]
            for schema in self.schemas
        ]

        # 4. 校验工具是否完整且没有重复注册
        if (
            len(schema_names) != len(TICKET_ALLOWED_TOOLS)
            or set(schema_names) != TICKET_ALLOWED_TOOLS
        ):
            raise ValueError(
                "Ticket Agent tool schemas are incomplete or duplicated"
            )

        
    def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        trusted_context: Mapping[str, Any],
    ) -> Any:

        # 1. 执行层工具白名单
        if name not in TICKET_ALLOWED_TOOLS:
            raise PermissionError(
                f"Ticket Agent cannot execute tool: {name}"
            )

        # 2. 验证可信上下文
        if not isinstance(trusted_context, Mapping):
            raise ValueError(
                "trusted_context must be a mapping"
            )

        customer_id = trusted_context.get("customer_id")

        if not isinstance(customer_id, str) or not customer_id.strip():
            raise ValueError(
                "trusted customer_id is required"
            )

        # 3. 只提取经过允许的服务端上下文字段
        context = {
            "customer_id": customer_id,
        }

        # 创建草稿和提交工单必须绑定会话
        if name in {"prepare_ticket", "submit_ticket"}:

            thread_id = trusted_context.get("thread_id")

            if not isinstance(thread_id, str) or not thread_id.strip():
                raise ValueError(
                    "trusted thread_id is required"
                )

            context["thread_id"] = thread_id

        # 4. 创建草稿时必须提供真实用户消息
        if name == "prepare_ticket":

            user_messages = trusted_context.get("user_messages")

            if (
                not isinstance(user_messages, (list, tuple))
                or not user_messages
                or not all(
                    isinstance(message, str) and message.strip()
                    for message in user_messages
                )
            ):
                raise ValueError(
                    "trusted user_messages are required"
                )

            context["user_messages"] = list(user_messages)

        # 5. 提交工单时，确认信息只能来自可信上下文
        if name == "submit_ticket":

            confirmed_ticket_id = trusted_context.get(
                "confirmed_ticket_id"
            )

            if (
                confirmed_ticket_id is not None
                and (
                    not isinstance(confirmed_ticket_id, str)
                    or not confirmed_ticket_id.strip()
                )
            ):
                raise ValueError(
                    "invalid confirmed_ticket_id"
                )

            context["confirmed_ticket_id"] = confirmed_ticket_id

        # Supervisor 已验证订单时，禁止 Ticket Agent 改写订单号
        if name == "prepare_ticket":

            expected_order_id = trusted_context.get(
                "expected_order_id"
            )

            if expected_order_id is not None:

                if (
                    not isinstance(expected_order_id, str)
                    or not expected_order_id.strip()
                ):
                    raise ValueError(
                        "invalid expected_order_id"
                    )

                supplied_order_id = (
                    arguments.get("order_id")
                    if isinstance(arguments, dict)
                    else None
                )

                if (
                    not isinstance(supplied_order_id, str)
                    or supplied_order_id.strip().upper()
                    != expected_order_id.strip().upper()
                ):
                    raise ValueError(
                        "ticket order_id does not match verified order"
                    )

        # 6. 交给已有工具层进行参数校验与业务分发
        return self.support_tools.call(
            name,
            arguments,
            context=context,
        )

    
    def run(
        self,
        task: str,
        *,
        trusted_context: dict,
        max_steps: int = 6,
    ) -> AgentResult:

        if not isinstance(task, str) or not task.strip():
            return AgentResult(
                agent="ticket_agent",
                status="needs_clarification",
                summary="请提供需要处理的工单问题。",
            )

        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        # 1. 初始化 LLM 消息
        messages = [
            SystemMessage(
                content=(
                    "你是售后工单 Ticket Agent。"
                    "只能准备工单草稿、提交工单和查询工单。"
                    "只能使用提供的三个工单工具。"
                    "故障描述必须来自用户提供的原话。"
                    "准备草稿不等于正式提交。"
                    "提交必须经过服务端确认和业务校验。"
                    "不能编造工单编号、工单状态或处理结果。"
                    "每轮最多请求一个工具。"
                )
            ),
            HumanMessage(content=task.strip()),
        ]

        tool_results = []

        # 2. Agent Loop
        for step in range(max_steps):

            try:
                response = self.llm.invoke(
                    messages,
                    self.schemas,
                )

                if not isinstance(response, AIMessage):
                    raise TypeError("LLM must return AIMessage")

            except Exception as exc:
                return AgentResult(
                    agent="ticket_agent",
                    status="error",
                    summary="Ticket Agent 模型调用失败。",
                    data={"tool_results": tool_results},
                    error=str(exc),
                )

            messages.append(response)
            calls = latest_tool_calls(response)

            # 3. LLM 不再调用工具
            if not calls:

                if not tool_results:
                    return AgentResult(
                        agent="ticket_agent",
                        status="needs_clarification",
                        summary="尚未执行工单操作，请明确工单需求。",
                        data={"tool_results": tool_results},
                    )

                last = tool_results[-1]
                result = last["result"]

                if (
                    last["tool"] == "get_ticket"
                    and result.get("status") == "found"
                ):
                    ticket = result.get("ticket", {})

                    if (
                        isinstance(ticket, dict)
                        and ticket.get("ticket_id")
                        and ticket.get("status")
                    ):
                        return AgentResult(
                            agent="ticket_agent",
                            status="success",
                            summary=(
                                f"已查询演示工单 {ticket['ticket_id']}，"
                                f"当前状态：{ticket['status']}。"
                            ),
                            data={
                                "tool_results": tool_results,
                                "business_status": "found",
                            },
                        )

                return AgentResult(
                    agent="ticket_agent",
                    status="no_evidence",
                    summary="未取得可确认的工单业务结果。",
                    data={"tool_results": tool_results},
                )

            # 4. 每轮最多执行一个工具
            if len(calls) != 1:
                return AgentResult(
                    agent="ticket_agent",
                    status="error",
                    summary="每轮只能调用一个工单工具。",
                    data={"tool_results": tool_results},
                    error="multiple_tool_calls",
                )

            call = calls[0]
            name = call.get("name")
            arguments = call.get("args", {})
            call_id = call.get("id")

            # 未授权工具不执行，但允许 LLM 在有限轮次内纠正
            if name not in TICKET_ALLOWED_TOOLS:

                if not isinstance(call_id, str) or not call_id.strip():
                    return AgentResult(
                        agent="ticket_agent",
                        status="error",
                        summary="非法工具调用缺少有效的 tool_call_id。",
                        data={"tool_results": tool_results},
                        error="invalid_tool_call_id",
                    )

                allowed_names = ", ".join(
                    sorted(TICKET_ALLOWED_TOOLS)
                )

                # 反馈给 LLM，但不调用任何业务服务
                messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "tool_not_allowed",
                                "message": (
                                    f"工具 {name} 不属于 Ticket Agent。"
                                    f"允许的工具只有：{allowed_names}。"
                                    "本次请求尚未执行任何业务操作。"
                                    "如果当前任务是准备工单草稿，"
                                    "请重新选择 prepare_ticket。"
                                ),
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=call_id,
                        name=(
                            name
                            if isinstance(name, str) and name
                            else "invalid_tool"
                        ),
                    )
                )

                continue

            # 5. 执行工具
            try:
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError("missing tool_call_id")

                result = self.execute_tool(
                    name,
                    arguments,
                    trusted_context=trusted_context,
                )

                if (
                    not isinstance(result, dict)
                    or not isinstance(result.get("status"), str)
                ):
                    raise ValueError("invalid tool result")

            except Exception as exc:
                return AgentResult(
                    agent="ticket_agent",
                    status="error",
                    summary="工单工具执行失败。",
                    data={"tool_results": tool_results},
                    error=f"{name}: {exc}",
                )

            tool_results.append({
                "tool": name,
                "result": result,
            })

            # 6. 记录 ToolMessage
            messages.append(
                ToolMessage(
                    content=json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    ),
                    tool_call_id=call_id,
                    name=name,
                )
            )

            business_status = result["status"]

            data = {
                "tool_results": tool_results,
                "business_status": business_status,
            }

            # 7. 草稿已准备，必须等待确认
            if name == "prepare_ticket" and business_status == "draft_ready":
                return AgentResult(
                    agent="ticket_agent",
                    status="pending_confirmation",
                    summary="工单草稿已准备，请核对内容后明确确认提交。",
                    data=data,
                )

            # 8. 正式提交成功或已经提交
            if (
                name in {"prepare_ticket", "submit_ticket"}
                and business_status == "already_submitted"
            ) or (
                name == "submit_ticket"
                and business_status == "submitted"
            ):
                return AgentResult(
                    agent="ticket_agent",
                    status="success",
                    summary="演示工单已经提交，等待人工审核。",
                    data=data,
                )

            # 9. 尚未获得有效确认
            if business_status == "confirmation_required":
                return AgentResult(
                    agent="ticket_agent",
                    status="pending_confirmation",
                    summary=result.get(
                        "message",
                        "请先确认工单草稿。",
                    ),
                    data=data,
                )

            # 10. 业务需要澄清或重新校验
            if business_status in {
                "needs_clarification",
                "revalidation_required",
            }:
                return AgentResult(
                    agent="ticket_agent",
                    status="needs_clarification",
                    summary=result.get(
                        "message",
                        "请补充或重新确认工单信息。",
                    ),
                    data=data,
                )

            # 11. 业务被拒绝或记录不存在
            if business_status in {"blocked", "not_found"}:
                return AgentResult(
                    agent="ticket_agent",
                    status="no_evidence",
                    summary=result.get(
                        "message",
                        "当前无法完成工单操作。",
                    ),
                    data=data,
                )

            # 12. 查询成功后，将结果交回 LLM 继续处理
            if name == "get_ticket" and business_status == "found":
                continue

            return AgentResult(
                agent="ticket_agent",
                status="error",
                summary="工单工具返回了未预期的业务状态。",
                data=data,
                error="unexpected_business_status",
            )

        return AgentResult(
            agent="ticket_agent",
            status="error",
            summary="达到最大模型调用次数，任务未完成。",
            data={"tool_results": tool_results},
            error="max_steps_exceeded",
        )
