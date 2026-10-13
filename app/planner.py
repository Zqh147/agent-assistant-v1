"""Incremental execution plan built from actual model tool decisions.

This tracks runtime steps; it does not decompose all subtasks upfront.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any


def build_plan(question: str) -> list[dict[str, Any]]:
    return [
        {"id": 1, "action": "接收用户问题", "status": "completed"},
        {"id": 2, "action": "生成最终回答", "status": "pending"},
    ]


def schedule_tools(plan: list[dict[str, Any]], calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated = deepcopy(plan)
    for call in calls:
        updated.insert(-1, {
            "action": f"执行工具：{call['name']}", "status": "pending",
            "tool": call["name"], "call_id": call["id"], "arguments": call.get("args", {}),
        })
    for index, step in enumerate(updated, 1):
        step["id"] = index
    return updated


def settle_tool(plan: list[dict[str, Any]], call_id: str, status: str) -> list[dict[str, Any]]:
    updated = deepcopy(plan)
    for step in updated:
        if step.get("call_id") == call_id and step["status"] == "pending":
            step["status"] = "completed" if status == "success" else "failed"
            break
    return updated


def finish_plan(plan: list[dict[str, Any]], stopped: bool = False) -> list[dict[str, Any]]:
    updated = deepcopy(plan)
    updated[-1]["status"] = "skipped" if stopped else "completed"
    return updated
