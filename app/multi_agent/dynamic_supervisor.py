import json
import re
from typing import Any

from typing import Literal


from .contracts import AgentResult

from langchain_core.messages import ToolMessage

from .supervisor_tools import SupervisorTools

from langchain_core.messages import AIMessage, HumanMessage

from .supervisor_tools import get_supervisor_schemas



SUPERVISOR_PROMPT = """
你是网络设备售后系统的 Dynamic Supervisor。

职责：
1. 根据当前任务类型，选择合适的子 Agent 工具。
2. Order Agent 负责订单查询、归属验证和设备信息。
3. Knowledge Agent 负责设备功能、配置方法和故障排查。
4. Ticket Agent 负责售后工单草稿准备。
5. 任何订单关联任务都必须先调用 Order Agent。
6. order_knowledge_ticket 任务必须在 Order Agent 和
   Knowledge Agent 完成后才能调用 Ticket Agent。
7. Ticket Agent 只能为已验证的订单准备草稿。
8. 工单草稿创建成功后，立即结束本轮调度，
   等待用户确认，不得自动提交。
9. 每轮最多请求一个子 Agent 工具。
10. 不得编造订单状态、设备信息、知识证据或工单结果。
11. 只能调用当前允许使用的工具。

真正的权限校验、调用顺序和业务状态判断由 Python 程序负责。
"""




ORDER_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])(?:DEMO-\d+|A\d{4,})(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)


def extract_requested_order_id(question: str) -> str | None:
    """从当前演示场景的用户问题中提取唯一订单号。"""

    matches = {
        item.upper()
        for item in ORDER_ID_PATTERN.findall(question)
    }

    if len(matches) != 1:
        return None

    return next(iter(matches))


class DynamicSupervisor:

    def __init__(self, llm):
        self.llm = llm
        self.schemas = get_supervisor_schemas()

    def decide(self, user_question: str) -> AIMessage:
        """执行一次 Supervisor LLM 决策，不执行子 Agent。"""

        if not isinstance(user_question, str) or not user_question.strip():
            raise ValueError("user_question must not be blank")

        messages = [
            HumanMessage(content=user_question.strip())
        ]

        response = self.llm.invoke(
            messages,
            self.schemas,
            system_prompt=SUPERVISOR_PROMPT,
        )

        if not isinstance(response, AIMessage):
            raise TypeError("LLM must return an AIMessage")

        return response
    
    
    def execute_tool_call(
        self,
        tool_call: dict[str, Any],
        tools: SupervisorTools,
        *,
        allow_public_knowledge: bool = False,
    ) -> ToolMessage:

        """校验并执行 Supervisor LLM 请求的子 Agent 工具。"""

        if not isinstance(tool_call, dict):
            raise ValueError("tool_call must be a dict")

        name = tool_call.get("name")
        arguments = tool_call.get("args", {})
        call_id = tool_call.get("id")

        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("missing tool_call_id")

        # 1. 参数必须是字典
        if not isinstance(arguments, dict):
            result = {
                "status": "error",
                "summary": "工具参数格式不合法。",
            }

        # 2. 调用 Order Agent
        elif name == "call_order_agent":

            if set(arguments) != {"order_id"}:
                result = {
                    "status": "error",
                    "summary": "订单工具参数不合法。",
                }
            else:
                try:
                    result = tools.call_order_agent(
                        arguments["order_id"]
                    )
                except Exception:
                    result = {
                        "status": "error",
                        "summary": "订单 Agent 工具调用失败。",
                    }

        # 3. 调用 Knowledge Agent
        elif name == "call_knowledge_agent":

            if set(arguments) != {"question"}:
                result = {
                    "status": "error",
                    "summary": "知识检索工具参数不合法。",
                }
            else:
                try:
                    if allow_public_knowledge:
                        result = tools.call_knowledge_agent(
                            arguments["question"],
                            allow_public=True,
                        )
                    else:
                        result = tools.call_knowledge_agent(
                            arguments["question"]
                        )
                except Exception:
                    result = {
                        "status": "error",
                        "summary": "知识检索 Agent 工具调用失败。",
                    }

        # 调用 Ticket Agent
        elif name == "call_ticket_agent":

            if set(arguments) != {"order_id"}:
                result = {
                    "status": "error",
                    "summary": "工单工具参数不合法。",
                }
            else:
                try:
                    result = tools.call_ticket_agent(
                        arguments["order_id"]
                    )
                except Exception:
                    result = {
                        "status": "error",
                        "summary": "Ticket Agent 工具调用失败。",
                    }

        # 4. 拒绝未知工具
        else:
            result = {
                "status": "error",
                "summary": "Supervisor 请求了未注册的工具。",
            }

        # 5. 将执行结果转换成 ToolMessage
        return ToolMessage(
            content=json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            ),
            tool_call_id=call_id,
            name=name if isinstance(name, str) else "invalid_tool",
        )

    
    
    def run(
        self,
        user_question: str,
        tools: SupervisorTools,
        *,
        max_steps: int = 6,
        
        
        task_type: Literal[
            "order_only",
            "order_knowledge",
            "knowledge_only",
            "order_ticket",
            "order_knowledge_ticket",
        ] = "order_knowledge",


    ) -> AgentResult:

        """执行订单关联技术问题的动态多 Agent 调度。"""

        if not isinstance(user_question, str) or not user_question.strip():
            return AgentResult(
                agent="supervisor",
                status="needs_clarification",
                summary="请提供需要处理的问题。",
            )

        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        
        
        
        if task_type not in {
            "order_only",
            "order_knowledge",
            "knowledge_only",
            "order_ticket",
            "order_knowledge_ticket",
        }:
            raise ValueError("unsupported task_type")



        # 每次请求重新验证订单，避免复用上次的可信状态
        tools.verified_order_id = None
        tools.verified_product_scope = None
        
        expected_order_id = None

        
        if task_type in {
            "order_only",
            "order_knowledge",
            "order_ticket",
            "order_knowledge_ticket",
        }:
            expected_order_id = extract_requested_order_id(
                user_question
            )

            if expected_order_id is None:
                return AgentResult(
                    agent="supervisor",
                    status="needs_clarification",
                    summary="请提供唯一且有效的演示订单编号。",
                )

        elif extract_requested_order_id(user_question) is not None:
            return AgentResult(
                agent="supervisor",
                status="needs_clarification",
                summary=(
                    "问题包含订单编号，请使用订单查询或"
                    "订单关联技术查询模式。"
                ),
            )

        if (
            task_type == "knowledge_only"
            and tools.public_product_scope is None
        ):
            return AgentResult(
                agent="supervisor",
                status="needs_clarification",
                summary="请提供完整的设备型号、硬件版本和地区。",
            )


        messages = [
            HumanMessage(content=user_question.strip())
        ]

        
        allowed_tools = {
            "order_only": {
                "call_order_agent",
            },
            "knowledge_only": {
                "call_knowledge_agent",
            },
            "order_knowledge": {
                "call_order_agent",
                "call_knowledge_agent",
            },
            "order_ticket": {
                "call_order_agent",
                "call_ticket_agent",
            },
            "order_knowledge_ticket": {
                "call_order_agent",
                "call_knowledge_agent",
                "call_ticket_agent",
            },
        }[task_type]


        active_schemas = [
            schema
            for schema in self.schemas
            if schema["function"]["name"] in allowed_tools
        ]

        tool_trace = []
        latest_knowledge = None
        knowledge_order_id = None
        knowledge_scope = None
        last_failure = None

        for step in range(max_steps):

            # 1. 请求 Supervisor LLM 进行决策
            try:
                response = self.llm.invoke(
                    messages,
                    active_schemas,
                    system_prompt=SUPERVISOR_PROMPT,
                )

                if not isinstance(response, AIMessage):
                    raise TypeError("LLM must return AIMessage")

            except Exception as exc:
                return AgentResult(
                    agent="supervisor",
                    status="error",
                    summary="Supervisor 模型调用失败。",
                    data={"tool_trace": tool_trace},
                    error=type(exc).__name__,
                )

            messages.append(response)
            calls = response.tool_calls

            # 2. 如果 LLM 不再请求工具，检查任务是否真正完成
            if not calls:
                
                if (
                    task_type == "knowledge_only"
                    and latest_knowledge is not None
                    and last_failure is None
                ):
                    return AgentResult(
                        agent="supervisor",
                        status="success",
                        summary=latest_knowledge["summary"],
                        evidence=latest_knowledge["evidence"],
                        data={
                            "product_scope": dict(
                                tools.public_product_scope
                            ),
                            "scope_source": "public_input",
                            "tool_trace": tool_trace,
                            "step_count": step + 1,
                        },
                    )

                # 纯订单任务：真实订单验证成功即可完成
                order_completed = (
                    tools.verified_order_id == expected_order_id
                    and isinstance(
                        tools.verified_product_scope,
                        dict,
                    )
                    and any(
                        item["tool"] == "call_order_agent"
                        and item["status"] == "success"
                        for item in tool_trace
                    )
                )

                if (
                    task_type == "order_only"
                    and order_completed
                    and last_failure is None
                ):
                    scope = dict(tools.verified_product_scope)

                    return AgentResult(
                        agent="supervisor",
                        status="success",
                        summary=(
                            f"订单 {expected_order_id} 已通过验证，"
                            f"关联设备为 "
                            f"{scope['product_model']} "
                            f"{scope['hardware_version']} "
                            f"（{scope['region']}）。"
                        ),
                        data={
                            "order_id": expected_order_id,
                            "product_scope": scope,
                            "tool_trace": tool_trace,
                            "step_count": step + 1,
                        },
                    )


                completed = (
                    latest_knowledge is not None
                    and tools.verified_order_id is not None
                    and tools.verified_product_scope is not None
                    and knowledge_order_id == tools.verified_order_id
                    and knowledge_scope == tools.verified_product_scope
                    and last_failure is None
                )

                if task_type == "order_knowledge" and completed:
                    summary = (
                        response.content
                        if isinstance(response.content, str)
                        and response.content.strip()
                        else latest_knowledge["summary"]
                    )

                    return AgentResult(
                        agent="supervisor",
                        status="success",
                        summary=summary,
                        evidence=latest_knowledge.get("evidence", []),
                        data={
                            "order_id": tools.verified_order_id,
                            "product_scope": dict(
                                tools.verified_product_scope
                            ),
                            "tool_trace": tool_trace,
                            "step_count": step + 1,
                        },
                    )

                # 工具已经失败，返回失败状态
                if last_failure is not None:
                    return AgentResult(
                        agent="supervisor",
                        status=last_failure["status"],
                        summary=last_failure["summary"],
                        data={"tool_trace": tool_trace},
                    )

                # 未完成必要的查询，不能相信 LLM 直接回答
                return AgentResult(
                    agent="supervisor",
                    status="no_evidence",
                    summary="尚未完成订单验证与技术检索，无法给出可靠结论。",
                    data={"tool_trace": tool_trace},
                )

            # 3. 第一版每轮只允许一个子 Agent Tool Call
            if len(calls) != 1:
                return AgentResult(
                    agent="supervisor",
                    status="error",
                    summary="每轮只能调用一个子 Agent。",
                    data={"tool_trace": tool_trace},
                    error="multiple_tool_calls",
                )

            call = calls[0]
            name = call.get("name")

            
            # 当前任务不允许的工具，直接拒绝执行
            if name not in allowed_tools:
                return AgentResult(
                    agent="supervisor",
                    status="error",
                    summary="当前任务不允许调用该子 Agent。",
                    data={"tool_trace": tool_trace},
                    error="tool_not_allowed_for_task",
                )


            
            if (
                task_type == "order_only"
                and name == "call_knowledge_agent"
            ):
                return AgentResult(
                    agent="supervisor",
                    status="error",
                    summary="纯订单任务不允许调用知识检索 Agent。",
                    data={"tool_trace": tool_trace},
                    error="tool_not_allowed_for_task",
                )

            if (
                task_type == "knowledge_only"
                and name == "call_order_agent"
            ):
                return AgentResult(
                    agent="supervisor",
                    status="error",
                    summary="公开知识查询不允许调用订单 Agent。",
                    data={"tool_trace": tool_trace},
                    error="tool_not_allowed_for_task",
                )

            
            # 防止 LLM 查询与用户请求不一致的订单
            if name == "call_order_agent":

                arguments = call.get("args", {})

                supplied_order_id = (
                    arguments.get("order_id")
                    if isinstance(arguments, dict)
                    else None
                )

                if (
                    not isinstance(supplied_order_id, str)
                    or supplied_order_id.strip().upper()
                    != expected_order_id
                ):
                    return AgentResult(
                        agent="supervisor",
                        status="parameter_conflict",
                        summary=(
                            "模型生成的订单号与用户请求不一致，"
                            "已阻止订单查询。"
                        ),
                        data={"tool_trace": tool_trace},
                        error="order_id_mismatch",
                    )

            # Ticket Agent 必须在正确的订单验证后执行
            if name == "call_ticket_agent":

                arguments = call.get("args", {})

                supplied_order_id = (
                    arguments.get("order_id")
                    if isinstance(arguments, dict)
                    else None
                )

                # 防止 LLM 将工单创建到其他订单
                if (
                    not isinstance(supplied_order_id, str)
                    or supplied_order_id.strip().upper()
                    != expected_order_id
                ):
                    return AgentResult(
                        agent="supervisor",
                        status="parameter_conflict",
                        summary="工单订单号与用户请求不一致。",
                        data={"tool_trace": tool_trace},
                        error="ticket_order_id_mismatch",
                    )

                # 必须已经完成真实订单验证
                if (
                    tools.verified_order_id != expected_order_id
                    or tools.verified_product_scope is None
                ):
                    return AgentResult(
                        agent="supervisor",
                        status="parameter_conflict",
                        summary="尚未完成订单验证，禁止创建工单。",
                        data={"tool_trace": tool_trace},
                        error="order_verification_required",
                    )

                # 三 Agent 模式必须先完成知识检索
                if task_type == "order_knowledge_ticket":
                    knowledge_completed = (
                        latest_knowledge is not None
                        and knowledge_order_id == expected_order_id
                        and knowledge_scope
                        == tools.verified_product_scope
                    )

                    if not knowledge_completed:
                        return AgentResult(
                            agent="supervisor",
                            status="no_evidence",
                            summary=(
                                "尚未完成当前订单的知识检索，"
                                "不能进入工单准备阶段。"
                            ),
                            data={"tool_trace": tool_trace},
                            error="knowledge_required_before_ticket",
                        )

            
            # 4. 新订单调用会使之前的知识检索结果失效
            if name == "call_order_agent":
                latest_knowledge = None
                knowledge_order_id = None
                knowledge_scope = None

            if name == "call_knowledge_agent":
                latest_knowledge = None
                knowledge_order_id = None
                knowledge_scope = None

            # 5. 通过 Python 执行工具
            try:
                tool_message = self.execute_tool_call(
                    call,
                    tools,
                    allow_public_knowledge=(
                        task_type == "knowledge_only"
                    ),
                )

                tool_result = json.loads(tool_message.content)

                if not isinstance(tool_result, dict):
                    raise ValueError("Invalid tool result")

                if not isinstance(tool_result.get("status"), str):
                    raise ValueError("Missing tool status")

            except Exception as exc:
                return AgentResult(
                    agent="supervisor",
                    status="error",
                    summary="Supervisor 工具执行或结果解析失败。",
                    data={"tool_trace": tool_trace},
                    error=type(exc).__name__,
                )

            # 6. 记录 ToolMessage，供下一轮 LLM 使用
            messages.append(tool_message)

            tool_trace.append({
                "step": step + 1,
                "tool": name,
                "status": tool_result["status"],
            })

            # 草稿准备成功后立即结束本轮
            if name == "call_ticket_agent":

                if tool_result["status"] == "pending_confirmation":

                    ticket_id = tool_result.get("ticket_id")
                    ticket = tool_result.get("ticket")

                    valid_ticket = (
                        isinstance(ticket_id, str)
                        and bool(ticket_id.strip())
                        and isinstance(ticket, dict)
                        and ticket.get("ticket_id") == ticket_id
                        and ticket.get("order_id")
                        == expected_order_id
                        and ticket.get("status") == "draft"
                        and ticket.get("customer_id")
                        == tools.trusted_context["customer_id"]
                        and ticket.get("thread_id")
                        == tools.trusted_context.get("thread_id")
                    )

                    if not valid_ticket:
                        return AgentResult(
                            agent="supervisor",
                            status="error",
                            summary="工单草稿结果校验失败。",
                            data={"tool_trace": tool_trace},
                            error="invalid_ticket_result",
                        )

                    return AgentResult(
                        agent="supervisor",
                        status="pending_confirmation",
                        summary=(
                            f"订单 {expected_order_id} 的工单草稿"
                            f"已创建，工单号：{ticket_id}。"
                            "请核对后确认是否正式提交。"
                        ),
                        evidence=(
                            latest_knowledge.get("evidence", [])
                            if latest_knowledge is not None
                            else []
                        ),
                        data={
                            "order_id": expected_order_id,
                            "ticket_id": ticket_id,
                            "ticket": ticket,
                            "product_scope": dict(
                                tools.verified_product_scope
                            ),
                            "tool_trace": tool_trace,
                            "step_count": step + 1,
                        },
                    )

            # 7. 检查是否成功完成知识检索
            valid_public_knowledge = (
                task_type == "knowledge_only"
                and tool_result.get("scope_source") == "public_input"
                and tool_result.get("product_scope")
                == tools.public_product_scope
                and isinstance(tool_result.get("evidence"), list)
                and bool(tool_result["evidence"])
            )

            valid_order_knowledge = (
                task_type in {
                    "order_knowledge",
                    "order_knowledge_ticket",
                }
                and tools.verified_order_id is not None
                and tools.verified_product_scope is not None
            )

            if (
                name == "call_knowledge_agent"
                and tool_result["status"] == "success"
                and (
                    valid_public_knowledge
                    or valid_order_knowledge
                )
            ):
                latest_knowledge = tool_result
                knowledge_order_id = tools.verified_order_id
                knowledge_scope = (
                    dict(tools.verified_product_scope)
                    if tools.verified_product_scope is not None
                    else None
                )
                last_failure = None

            elif tool_result["status"] != "success":
                valid_statuses = {
                    "needs_clarification",
                    "parameter_conflict",
                    "unsupported_product",
                    "no_evidence",
                    "error",
                }

                status = tool_result["status"]

                last_failure = {
                    "status": (
                        status if status in valid_statuses else "error"
                    ),
                    "summary": tool_result.get(
                        "summary",
                        "子 Agent 执行失败。",
                    ),
                }

            elif name == "call_knowledge_agent":
                last_failure = {
                    "status": "no_evidence",
                    "summary": (
                        "知识检索结果不满足当前任务的"
                        "设备范围或证据要求。"
                    ),
                }

            else:
                last_failure = None

        # 8. 达到最大 LLM 调用轮次
        return AgentResult(
            agent="supervisor",
            status="error",
            summary="Supervisor 达到最大调度轮次，任务未完成。",
            data={"tool_trace": tool_trace},
            error="max_steps_exceeded",
        )
