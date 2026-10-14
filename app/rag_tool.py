from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import httpx
from .config import settings


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
    return {word for word in words if word not in {"的", "是", "什么", "如何", "吗", "了"}}


def search_knowledge_base(question: str, knowledge_path: Path | None = None, top_k: int = 3,
                          backend: str | None = None, service_url: str | None = None,
                          request_timeout: float | None = None, product_model=None,
                          hardware_version=None, region=None) -> dict[str, Any]:
    selected = backend or settings.rag_backend
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 5:
        raise ValueError("top_k must be an integer between 1 and 5")
    if not question.strip():
        raise ValueError("question must not be blank")
    if selected == "project1":
        return retrieve_from_project1(question, top_k, service_url or settings.rag_service_url,
                                      request_timeout or settings.rag_timeout_seconds,
                                      product_model, hardware_version, region)
    if selected != "demo":
        raise ValueError("RAG_BACKEND must be demo or project1")
    path = knowledge_path or Path(__file__).resolve().parent.parent / "data" / "knowledge.md"
    if not path.exists():
        raise FileNotFoundError(f"knowledge base not found: {path}")
    raw = path.read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", raw) if chunk.strip() and not chunk.startswith("#")]
    query_tokens = _tokens(question)
    ranked = sorted(((len(query_tokens & _tokens(chunk)), chunk) for chunk in chunks), key=lambda item: item[0], reverse=True)
    results = [{"content": chunk, "score": score, "source": path.name, "chunk_id": index} for index, (score, chunk) in enumerate(ranked[:top_k]) if score > 0]
    return {"question": question, "backend": "demo", "results": results, "has_results": bool(results)}


def retrieve_from_project1(question: str, top_k: int, url: str, timeout: float,
                          product_model=None, hardware_version=None, region=None) -> dict[str, Any]:
    request_body = {"question": question, "top_k": top_k}
    request_body.update({name: value for name, value in {
        "product_model": product_model, "hardware_version": hardware_version, "region": region
    }.items() if value is not None})
    try:
        # Local retrieval should not accidentally go through machine-wide proxies.
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(url.rstrip("/") + "/retrieve", json=request_body)
    except httpx.ConnectError as exc:
        raise ConnectionError("Project One RAG service unavailable; start it on the configured URL") from exc
    except httpx.TimeoutException as exc:
        raise TimeoutError("Project One RAG retrieval timed out") from exc
    if response.status_code >= 400:
        raise ValueError(f"Project One RAG service returned HTTP {response.status_code}")
    payload = response.json()
    if payload.get("backend") != "project1" or not isinstance(payload.get("results"), list):
        raise ValueError("Invalid Project One RAG response")
    for item in payload["results"]:
        if not isinstance(item, dict) or not all(key in item for key in ("content", "source", "chunk_id")):
            raise ValueError("RAG evidence missing content or source metadata")
    return payload


def rag_tool_schema() -> dict[str, Any]:
    return {"type": "function", "function": {"name": "search_knowledge_base", "description": "检索产品手册与官方 FAQ。question 只包含技术问题和技术条件，不混入查询订单或工单操作指令。返回证据、来源和设备范围。已查询订单时服务端自动补齐设备信息；缺少型号、硬件版本或地区时按返回状态追问。", "parameters": {"type": "object", "properties": {"question": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 5}, "product_model": {"type": "string"}, "hardware_version": {"type": "string"}, "region": {"type": "string"}}, "required": ["question"], "additionalProperties": False}}}
