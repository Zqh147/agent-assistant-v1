from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Callable

ToolFn = Callable[..., Any]


def calculate(a: float, b: float, operation: str) -> dict[str, Any]:
    if operation not in {"add", "subtract", "multiply", "divide"}:
        raise ValueError("operation must be add, subtract, multiply, or divide")
    if operation == "divide" and b == 0:
        raise ValueError("cannot divide by zero")
    operations = {"add": lambda: a + b, "subtract": lambda: a - b,
                  "multiply": lambda: a * b, "divide": lambda: a / b}
    return {"operation": operation, "a": a, "b": b, "result": operations[operation]()}


def get_current_time() -> dict[str, str]:
    return {"iso_time": datetime.now().astimezone().isoformat()}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "calculate", "description": "Perform one basic arithmetic operation.", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}, "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]}}, "required": ["a", "b", "operation"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "Get the current local time.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


def tool_map(extra: dict[str, ToolFn] | None = None) -> dict[str, ToolFn]:
    result: dict[str, ToolFn] = {"calculate": calculate, "get_current_time": get_current_time}
    if extra:
        result.update(extra)
    return result


def serialize_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
