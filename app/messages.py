from typing import Any


def latest_tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None) or []
    return [dict(call) for call in calls]


def message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return {"type": getattr(message, "type", "unknown"), "content": str(getattr(message, "content", message))}
