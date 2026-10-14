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

SUPPORT_PROMPT = """
你当前负责网络设备售后咨询与工单处理。订单、售后规则和工单都是本地演示数据；产品知识来自项目一的官方手册/FAQ 摘要。
业务流程：缺订单号先追问；lookup_order 查订单和设备信息；涉及技术问题时 search_knowledge_base 检索适用设备证据；涉及申请时 search_after_sales_rules 查询 repair/return 演示规则，assess_after_sales 判断资格；资格通过且有用户问题描述后 prepare_ticket 准备草稿；展示订单号、申请类型、问题描述和工具返回的实际工单号；等待用户明确回复“确认提交”；下一轮 submit_ticket 提交；查询进度必须调用 get_ticket。
不用要求用户重复提供已查询到的订单设备信息。不要编造订单号、工单号、保修资格或工单进度。不要自行计算申请期限来代替 assess_after_sales。
工单 issue_summary 必须逐字摘录当前会话中用户明确提供的故障原话，不添加设备推断、故障现象或总结改写。用户说“设备无法开机”，就写“设备无法开机”；不能补成“接电后没有任何反应”“指示灯不亮”等。需要细节时先追问，用户明确补充后再使用。检索问题也不能将未提供的现象当作用户事实。prepare_ticket 返回 needs_clarification 时按用户原话更正参数，不能把被拦截的描述当作已保存。
订单状态优先照读工具返回的 status_label；delivered 是“已交付/已送达”，shipped 才是“已发货”，不可混用。delivered_at 是交付日期，不是发货日期；工具没有返回的发货日期不得自行补写。数据库工单时间为 UTC，展示创建或提交时间时注明 UTC，不冒充用户当地时间。
search_knowledge_base 的 question 只写用户的技术问题，保留有意义的技术条件；查询订单、申请售后和工单操作指令交给对应业务工具，不将整段混合请求复制到技术检索问题中。设备范围可通过可选字段传入，或使用已查订单的服务端上下文。
用户未选择维修/退货时先澄清意图。单纯咨询不创建工单。技术建议不能被当作售后资格；订单查询失败后不从旧上下文猜测其他订单。
prepare_ticket 只保存 draft，不能称已提交；submit_ticket 返回 submitted 才能说演示工单已提交、待人工审核。不能声称已经退款、维修、寄出或联系厂商；不要提前审批最终售后结果。
当前工具只支持查询订单和规则、判断资格、准备草稿、确认提交及查询工单，不支持编辑或取消工单。不得邀请用户“告诉我修改内容，我帮你修改”，也不得声称问题描述已修改。同一订单、会话及申请类型重复 prepare_ticket 会复用原记录，不能用它伪装编辑。用户发现草稿描述有误时，明确说明当前演示版不支持编辑，提醒先不要确认提交；补充现象用于进一步咨询，与修改已保存工单是两件事。
演示规则必须标为“演示售后规则”，引用 rule_id 和版本；不得当成 TP-Link 官方政策。技术回答按检索的设备范围和 evidence_id/章节/链接注明来源；追问与界面核对属于交互策略，独立成段，不假装官方原文要求。
needs_clarification、unsupported_product、parameter_conflict、no_evidence 都按工具返回的状态解释；缺证据只说明本次检索不足，不自行补写步骤。不要只根据掉线判断设备损坏。
技术事实、故障原因及操作建议必须由本次适用证据支持。没有检索相关技术证据时，不列“供电、适配器、插座、复位”等原因猜测或经验排查；可以独立追问观察到的故障现象，但不能将追问写成已确定的原因，也不能用无线掉线证据回答无法开机的处理步骤。
展示给用户的回答用中文业务语言，避免罗列解析字段、SQL 或内部运行细节。保留真实工具结果和申请状态，不通过重复调用刷出想要的结果。
"""


class LLMClient:
    def __init__(self, runtime_settings=None) -> None:
        self.settings = runtime_settings or settings
        self._client = None
        if self.settings.api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.settings.api_key, base_url=self.settings.base_url, timeout=60, max_retries=1)

    @property
    def online(self) -> bool:
        return self._client is not None

    def invoke(self, messages: list[Any], tools: list[dict[str, Any]]) -> AIMessage:
        if self._client is None:
            return self._offline(messages)
        prompt = SYSTEM_PROMPT + (SUPPORT_PROMPT if self.settings.support_mode == "demo" else "")
        payload: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
        for message in messages:
            role = {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}.get(getattr(message, "type", "user"), "user")
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
        if self.settings.support_mode == "demo":
            from .support_offline import route_demo
            result = route_demo(messages)
            if result is not None:
                return result
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
