"""Schemas contain business inputs; identity/confirmation come from the server."""
from functools import partial


def schema(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}

ORDER = {"type": "string", "description": "用户提供的示例订单号，例如 DEMO-1001；不可编造。"}
TYPE = {"type": "string", "enum": ["repair", "return"], "description": "repair 为维修申请，return 为退货申请。"}
TICKET = {"type": "string", "description": "工具实际返回的工单号，例如 TKT-...；不可编造。"}
SUPPORT_SCHEMAS = [
    schema("lookup_order", "查询当前演示客户的订单状态和设备信息。订单号缺失先向用户询问。", {"order_id": ORDER}, ["order_id"]),
    schema("search_after_sales_rules", "检索带版本和来源标记的演示售后规则，不代表厂商正式政策。", {"service_type": TYPE}, ["service_type"]),
    schema("assess_after_sales", "按数据库日期和演示规则确定申请资格，禁止自行计算或猜测资格。", {"order_id": ORDER, "service_type": TYPE}, ["order_id", "service_type"]),
    schema("prepare_ticket", "资格通过后保存工单草稿并返回实际工单号；issue_summary 必须逐字摘录用户在当前会话中提供的故障原话，禁止补写现象或改写。此步骤尚未提交，向用户展示详情并请求明确确认。", {"order_id": ORDER, "service_type": TYPE, "issue_summary": {"type": "string", "minLength": 1, "maxLength": 1000}}, ["order_id", "service_type", "issue_summary"]),
    schema("submit_ticket", "用户本轮明确确认后提交当前会话草稿；不执行退款或实际维修。未确认会被服务器拒绝。", {"ticket_id": TICKET}, ["ticket_id"]),
    schema("get_ticket", "从 SQLite 查询当前演示客户的工单状态及事件；不要根据历史回答猜测进度。", {"ticket_id": TICKET}, ["ticket_id"]),
]
SUPPORT_NAMES = {item["function"]["name"] for item in SUPPORT_SCHEMAS}


class SupportTools:
    def __init__(self, service):
        self.service = service

    def call(self, name, arguments=None, *, context=None, **kwargs):
        if context is None:
            raise ValueError("business tools require server conversation context")
        arguments = kwargs if arguments is None else arguments
        definition = next(item["function"] for item in SUPPORT_SCHEMAS if item["function"]["name"] == name)
        parameters = definition["parameters"]
        if not isinstance(arguments, dict) or set(arguments) - set(parameters["properties"]) or set(parameters["required"]) - set(arguments):
            raise ValueError("invalid business tool arguments")
        for key, value in arguments.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid {key}")
        return getattr(self.service, name)(**arguments, **context)

    def registry(self):
        return {name: partial(self.call, name) for name in SUPPORT_NAMES}
