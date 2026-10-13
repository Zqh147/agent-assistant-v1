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
                          request_timeout: float | None = None) -> dict[str, Any]:
    selected = backend or settings.rag_backend
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 5:
        raise ValueError("top_k must be an integer between 1 and 5")
    if not question.strip():
        raise ValueError("question must not be blank")
    if selected == "project1":
        return retrieve_from_project1(question, top_k, service_url or settings.rag_service_url,
                                      request_timeout or settings.rag_timeout_seconds)
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


def retrieve_from_project1(question: str, top_k: int, url: str, timeout: float) -> dict[str, Any]:
    try:
        # Local retrieval should not accidentally go through machine-wide proxies.
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(url.rstrip("/") + "/retrieve", json={"question": question, "top_k": top_k})
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
    return {"type": "function", "function": {"name": "search_knowledge_base", "description": "Search the local knowledge base and return evidence with source and chunk IDs.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 5}}, "required": ["question"]}}}
