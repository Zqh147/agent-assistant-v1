from collections.abc import Mapping
from .supervisor import extract_order_scope
from .contracts import AgentTask, AgentResult
from copy import deepcopy
from typing import Any


SUPERVISOR_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "call_order_agent",
            "description": (
                "调用 Order Agent 查询用户订单信息、"
                "验证订单访问权限，并获取订单关联的设备信息。"
                "涉及订单归属的查询应优先使用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "需要查询的订单编号",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_knowledge_agent",
            "description": (
                "调用 Knowledge Agent 检索设备功能、"
                "配置方法和故障排查知识。"
                "如果技术问题依赖用户的订单信息，"
                "必须先完成订单验证。"
                "可信设备范围由 Python 执行层提供。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "需要查询的设备技术问题",
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    
    {
        "type": "function",
        "function": {
            "name": "call_ticket_agent",
            "description": (
                "为已验证归属的订单调用 Ticket Agent 准备售后工单草稿。"
                "只准备草稿，不正式提交。"
                "必须先调用 call_order_agent 验证订单。"
                "故障描述必须来自用户真实消息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "已通过订单验证的订单编号",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
]


def get_supervisor_schemas() -> list[dict[str, Any]]:
    return deepcopy(SUPERVISOR_TOOL_SCHEMAS)


class SupervisorTools:

    
    
    def __init__(
        self,
        order_agent,
        knowledge_agent,
        *,
        trusted_context: Mapping[str, Any],
        public_product_scope: Mapping[str, str] | None = None,
        ticket_agent=None,
    ):


        if not isinstance(trusted_context, Mapping):
            raise ValueError("trusted_context must be a mapping")

        customer_id = trusted_context.get("customer_id")

        if not isinstance(customer_id, str) or not customer_id.strip():
            raise ValueError("trusted customer_id is required")

        self.order_agent = order_agent
        self.knowledge_agent = knowledge_agent
        self.ticket_agent = ticket_agent
        self.trusted_context = dict(trusted_context)

        # 保存经过真实订单工具验证的设备范围
        self.verified_order_id = None
        self.verified_product_scope = None

        # 独立公开知识查询使用的设备范围
        self.public_product_scope = None

        if public_product_scope is not None:
            required_fields = {
                "product_model",
                "hardware_version",
                "region",
            }

            if (
                not isinstance(public_product_scope, Mapping)
                or set(public_product_scope) != required_fields
                or not all(
                    isinstance(public_product_scope[key], str)
                    and public_product_scope[key].strip()
                    for key in required_fields
                )
            ):
                raise ValueError(
                    "public_product_scope must contain "
                    "valid model, version and region"
                )

            self.public_product_scope = {
                key: public_product_scope[key].strip()
                for key in required_fields
            }

    
    def call_order_agent(self, order_id: str) -> dict[str, Any]:


        # 新一次订单验证不能沿用上一次验证结果
        self.verified_order_id = None
        self.verified_product_scope = None

        # 1. 校验 LLM 提供的订单号
        if not isinstance(order_id, str) or not order_id.strip():
            return {
                "status": "needs_clarification",
                "summary": "请提供有效的订单编号。",
            }

        order_id = order_id.strip()

        # 2. 调用已有的 Order Agent
        try:
            order_result = self.order_agent.run(
                f"请查询订单 {order_id} 的订单状态和设备信息。",
                trusted_context=dict(self.trusted_context),
            )
        except Exception:
            return {
                "status": "error",
                "summary": "订单子任务执行异常。",
            }

        # 3. 检查子 Agent 的执行状态
        if order_result.status != "success":
            return {
                "status": order_result.status,
                "summary": order_result.summary,
            }

        # 4. 从真实工具结果中验证设备范围
        try:
            scope = extract_order_scope(
                order_result,
                expected_order_id=order_id,
            )
        except ValueError:
            return {
                "status": "no_evidence",
                "summary": "未获得与指定订单匹配的完整设备信息。",
            }

        # 5. 保存可信的订单和设备范围
        self.verified_order_id = order_id
        self.verified_product_scope = dict(scope)

        # 6. 返回供 Supervisor 使用的结构化结果
        return {
            "status": "success",
            "summary": "订单验证成功，已获得设备信息。",
            "order_id": order_id,
            "product_scope": dict(scope),
        }
    
    def call_knowledge_agent(
        self,
        question: str,
        *,
        allow_public: bool = False,
    ) -> dict[str, Any]:

        # 1. 校验技术问题
        if not isinstance(question, str) or not question.strip():
            return {
                "status": "needs_clarification",
                "summary": "请提供需要查询的技术问题。",
            }

        # 2. 订单关联查询使用已验证设备范围；公开查询仅使用显式配置范围
        if (
            self.verified_order_id is not None
            and self.verified_product_scope is not None
        ):
            scope = dict(self.verified_product_scope)
            order_id = self.verified_order_id
            scope_source = "verified_order"

        elif allow_public and self.public_product_scope is not None:
            scope = dict(self.public_product_scope)
            order_id = None
            scope_source = "public_input"

        else:
            return {
                "status": "needs_clarification",
                "summary": "缺少可用的设备范围，请先提供或验证设备信息。",
            }

        # 3. 使用可信设备范围创建 AgentTask
        task = AgentTask(
            task=question.strip(),
            product_scope=scope,
        )


        # 4. 调用已有 Knowledge Agent
        try:
            knowledge_result = self.knowledge_agent.run(task)
        except Exception:
            return {
                "status": "error",
                "summary": "知识检索子任务执行异常。",
            }

        # 5. 检查子 Agent 返回值
        if not isinstance(knowledge_result, AgentResult):
            return {
                "status": "error",
                "summary": "Knowledge Agent 返回了不合法的结果。",
            }

        # 6. 转换为 Supervisor 可处理的 Tool Result
        return {
            "status": knowledge_result.status,
            "summary": knowledge_result.summary,
            "order_id": order_id,
            "product_scope": scope,
            "scope_source": scope_source,
            "evidence": (
                knowledge_result.evidence[:3]
                if knowledge_result.status == "success"
                else []
            ),
        }

    
    def call_ticket_agent(self, order_id: str) -> dict[str, Any]:
        """为已验证订单准备工单草稿，不允许直接提交。"""

        # 1. 检查 Ticket Agent 是否已经注入
        if self.ticket_agent is None:
            return {
                "status": "error",
                "summary": "Ticket Agent 尚未配置。",
            }

        # 2. 校验订单参数
        if not isinstance(order_id, str) or not order_id.strip():
            return {
                "status": "needs_clarification",
                "summary": "请提供有效的订单编号。",
            }

        order_id = order_id.strip().upper()

        # 3. 必须使用本轮已经验证的订单
        if (
            self.verified_order_id != order_id
            or self.verified_product_scope is None
        ):
            return {
                "status": "parameter_conflict",
                "summary": "订单尚未通过验证，或订单号与已验证订单不一致。",
            }

        # 4. 获取服务端可信会话信息
        thread_id = self.trusted_context.get("thread_id")
        user_messages = self.trusted_context.get("user_messages")

        if not isinstance(thread_id, str) or not thread_id.strip():
            return {
                "status": "needs_clarification",
                "summary": "缺少可信会话 ID。",
            }

        if (
            not isinstance(user_messages, (list, tuple))
            or not user_messages
            or not all(
                isinstance(message, str) and message.strip()
                for message in user_messages
            )
        ):
            return {
                "status": "needs_clarification",
                "summary": "缺少可信用户原始消息。",
            }

        # 5. 组装受限的可信上下文
        context = {
            "customer_id": self.trusted_context["customer_id"],
            "thread_id": thread_id,
            "user_messages": list(user_messages),
            "confirmed_ticket_id": None,
            "expected_order_id": order_id,
        }

        # 6. 调用 Ticket Agent
        try:
            ticket_task = (
                "你正在执行一个已经由 Supervisor 分配的工单子任务。\n"
                f"当前唯一目标：为已验证订单 {order_id} "
                "准备售后工单草稿。\n"
                "订单归属验证已经完成，不需要重新查询订单。\n"
                "如果之前已经进行了知识检索，不需要再次排障。\n"
                "只能使用 prepare_ticket 创建草稿。\n"
                "不能调用 search_knowledge_base 或其他 Agent 的工具。\n"
                "不能调用 submit_ticket 正式提交工单。\n"
                "service_type 必须从用户原始请求中确定，"
                "只能是 repair 或 return。\n"
                "issue_summary 必须逐字摘录用户实际提供的故障描述，"
                "不得改写或补充不存在的现象。\n\n"
                "原始用户请求（仅用于提取工单业务参数）：\n"
                f"{user_messages[-1]}"
            )

            ticket_result = self.ticket_agent.run(
                task=ticket_task,
                trusted_context=context,
            )
        except Exception:
            return {
                "status": "error",
                "summary": "Ticket Agent 执行异常。",
            }

        # 7. 验证返回的数据类型和来源
        if (
            not isinstance(ticket_result, AgentResult)
            or ticket_result.agent != "ticket_agent"
        ):
            return {
                "status": "error",
                "summary": "Ticket Agent 返回结果不合法。",
            }

        tool_results = ticket_result.data.get("tool_results", [])

        # 8. 只接受真实 prepare_ticket 的草稿结果
        if ticket_result.status == "pending_confirmation":
            for item in tool_results:
                if not isinstance(item, dict):
                    continue

                if item.get("tool") != "prepare_ticket":
                    continue

                result = item.get("result", {})

                if not isinstance(result, dict):
                    continue

                if result.get("status") != "draft_ready":
                    continue

                ticket = result.get("ticket", {})

                if not isinstance(ticket, dict):
                    continue

                if (
                    ticket.get("order_id") == order_id
                    and ticket.get("customer_id")
                    == self.trusted_context["customer_id"]
                    and ticket.get("thread_id") == thread_id
                    and ticket.get("status") == "draft"
                    and isinstance(ticket.get("ticket_id"), str)
                    and ticket["ticket_id"]
                ):
                    return {
                        "status": "pending_confirmation",
                        "summary": "工单草稿已准备，等待用户确认。",
                        "order_id": order_id,
                        "ticket_id": ticket["ticket_id"],
                        "ticket": ticket,
                        "business_status": "draft_ready",
                    }

        # 9. 保留明确的失败或澄清结果
        if ticket_result.status in {
            "needs_clarification",
            "no_evidence",
            "error",
        }:
            return {
                "status": ticket_result.status,
                "summary": ticket_result.summary,
            }

        return {
            "status": "error",
            "summary": "没有获得与当前订单匹配的有效工单草稿。",
        }
