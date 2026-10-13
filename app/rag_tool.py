from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
    return {word for word in words if word not in {"的", "是", "什么", "如何", "吗", "了"}}


def search_knowledge_base(question: str, knowledge_path: Path | None = None, top_k: int = 3) -> dict[str, Any]:
    path = knowledge_path or Path(__file__).resolve().parent.parent / "data" / "knowledge.md"
    if not path.exists():
        raise FileNotFoundError(f"knowledge base not found: {path}")
    raw = path.read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", raw) if chunk.strip() and not chunk.startswith("#")]
    query_tokens = _tokens(question)
    ranked = sorted(((len(query_tokens & _tokens(chunk)), chunk) for chunk in chunks), key=lambda item: item[0], reverse=True)
    results = [{"content": chunk, "score": score, "source": path.name, "chunk_id": index} for index, (score, chunk) in enumerate(ranked[:top_k]) if score > 0]
    return {"question": question, "results": results, "has_results": bool(results)}


def rag_tool_schema() -> dict[str, Any]:
    return {"type": "function", "function": {"name": "search_knowledge_base", "description": "Search the local knowledge base and return evidence with source and chunk IDs.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 5}}, "required": ["question"]}}}
