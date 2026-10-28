import json

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..messages import latest_tool_calls
from .contracts import AgentResult
from copy import deepcopy
from typing import Any, Mapping

from ..support_tools import SUPPORT_SCHEMAS, SupportTools


ORDER_ALLOWED_TOOLS = frozenset({
    "lookup_order",
    "search_after_sales_rules",
    "assess_after_sales",
})


class OrderAgent:
    def __init__(self, llm, support_tools: SupportTools):

        self.llm = llm
        self.support_tools = support_tools

        self.schemas = [
            deepcopy(schema)
            for schema in SUPPORT_SCHEMAS
            if schema["function"]["name"] in ORDER_ALLOWED_TOOLS
        ]

        schema_names = {
            schema["function"]["name"]
            for schema in self.schemas
        }

        if schema_names != ORDER_ALLOWED_TOOLS:
            raise ValueError("Order Agent tool schemas are incomplete")

    def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        trusted_context: Mapping[str, Any],
    ) -> Any:

        # 第一层：执行端白名单
        if name not in ORDER_ALLOWED_TOOLS:
            raise PermissionError(
                f"Order Agent cannot execute tool: {name}"
            )

        # 第二层：要求上游提供可信身份上下文
        customer_id = trusted_context.get("customer_id")

        if not isinstance(customer_id, str) or not customer_id.strip():
            raise ValueError("trusted customer_id is required")

        # 第三层：交给现有工具层做参数校验与业务分发
        return self.support_tools.call(
            name,
            arguments,
            context=dict(trusted_context),
        )
    
    def run(
        self,
        task: str,
        *,
        trusted_context: dict,
        max_steps: int = 6,
    ) -> AgentResult:

        if not task.strip():
            return AgentResult(
                agent="order_agent",
                status="needs_clarification",
                summary="请提供需要处理的订单问题。",
            )

        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        messages = [
            SystemMessage(
                content=(
                    "你是订单与售后资格查询 Agent。"
                    "只能查询订单、售后规则和申请资格。"
                    "不能创建或提交工单。"
                    "不能编造订单信息或资格结果。"
                    "缺少订单号时应询问用户。"
                    "涉及资格判断时必须依据工具结果。"
                )
            ),
            HumanMessage(content=task),
        ]

        tool_results = []

        for step in range(max_steps):

            try:
                response = self.llm.invoke(
                    messages,
                    self.schemas,
                )
            except Exception as exc:
                return AgentResult(
                    agent="order_agent",
                    status="error",
                    summary="模型调用失败。",
                    data={"tool_results": tool_results},
                    error=str(exc),
                )

            messages.append(response)

            calls = latest_tool_calls(response)


            if not calls:
                return self._finish_from_tools(tool_results)


            for call in calls:
                name = call["name"]
                arguments = call.get("args", {})
                call_id = call.get("id")

                try:
                    if not call_id:
                        raise ValueError("missing tool_call_id")

                    result = self.execute_tool(
                        name,
                        arguments,
                        trusted_context=trusted_context,
                    )

                except Exception as exc:
                    return AgentResult(
                        agent="order_agent",
                        status="error",
                        summary="订单工具执行失败。",
                        data={"tool_results": tool_results},
                        error=f"{name}: {exc}",
                    )

                tool_results.append({
                    "tool": name,
                    "result": result,
                })

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

        return AgentResult(
            agent="order_agent",
            status="error",
            summary="达到最大模型调用次数，任务未完成。",
            data={"tool_results": tool_results},
            error="max_steps_exceeded",
        )

    
    def _finish_from_tools(self, tool_results: list[dict]) -> AgentResult:

        data = {"tool_results": tool_results}

        # 情况一：模型没有执行任何工具
        if not tool_results:
            return AgentResult(
                agent="order_agent",
                status="needs_clarification",
                summary="尚未执行订单或售后查询，请提供订单号或明确要查询的售后规则。",
                data=data,
            )

        assessment = None
        order_status = None
        rules_status = None

        # 检查实际执行过的业务工具
        for item in tool_results:
            name = item["tool"]
            result = item["result"]

            if not isinstance(result, dict):
                continue

            if name == "assess_after_sales":
                assessment = result

            elif name == "lookup_order":
                order_status = result.get("status")

            elif name == "search_after_sales_rules":
                rules_status = result.get("status")

        # 情况二：确实执行了售后资格评估
        if assessment is not None:
            eligibility = assessment.get("status")
            data["eligibility"] = eligibility

            if eligibility == "eligible":
                summary = "售后资格评估结果：符合申请条件（演示规则）。"

            elif eligibility == "ineligible":
                summary = "售后资格评估结果：不符合申请条件（演示规则）。"

            else:
                return AgentResult(
                    agent="order_agent",
                    status="no_evidence",
                    summary="尚无法确认售后资格，请核对订单及业务结果。",
                    data=data,
                )

            return AgentResult(
                agent="order_agent",
                status="success",
                summary=summary,
                data=data,
            )

        # 情况三：没有执行售后资格评估
        if order_status == "not_found":
            return AgentResult(
                agent="order_agent",
                status="no_evidence",
                summary="未找到可访问的订单，请核对订单号。",
                data=data,
            )

        if order_status == "found":
            summary = "订单查询已完成，尚未执行售后资格评估。"

        elif rules_status == "found":
            summary = "已查询演示售后规则，尚未评估具体订单资格。"

        else:
            return AgentResult(
                agent="order_agent",
                status="no_evidence",
                summary="没有获得足够的有效业务结果。",
                data=data,
            )

        return AgentResult(
            agent="order_agent",
            status="success",
            summary=summary,
            data=data,
        )

    


