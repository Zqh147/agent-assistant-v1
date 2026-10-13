from dataclasses import replace
import json

import httpx
import pytest

from app.config import settings
from app.graph import AgentGraph
from app.rag_tool import search_knowledge_base


PAYLOAD = {"question": "RAG", "backend": "project1", "pipeline": "vector+bm25+rrf+cross_encoder",
           "has_results": True, "results": [{"content": "RAG检索增强生成", "source": "manual.pdf",
           "page": 3, "chunk_id": 7, "rrf_score": 0.03, "reranker_score": 1.2}]}


def mock_client(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_project1_http_preserves_evidence(monkeypatch):
    def handler(request):
        assert request.url.path == "/retrieve"
        assert json.loads(request.content) == {"question": "RAG", "top_k": 3}
        return httpx.Response(200, json=PAYLOAD)
    mock_client(monkeypatch, handler)
    result = search_knowledge_base("RAG", backend="project1")
    assert result["backend"] == "project1"
    assert result["results"][0]["chunk_id"] == 7
    assert result["results"][0]["page"] == 3
    assert result["results"][0]["reranker_score"] == 1.2


def test_unavailable_service_does_not_fall_back_to_demo(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("offline", request=request)
    mock_client(monkeypatch, handler)
    with pytest.raises(ConnectionError, match="unavailable"):
        search_knowledge_base("RAG", backend="project1")


def test_bad_evidence_is_rejected(monkeypatch):
    mock_client(monkeypatch, lambda request: httpx.Response(200, json={"backend": "project1", "results": [{}]}))
    with pytest.raises(ValueError, match="missing"):
        search_knowledge_base("RAG", backend="project1")


def test_graph_records_project1_backend(monkeypatch, tmp_path):
    mock_client(monkeypatch, lambda request: httpx.Response(200, json=PAYLOAD))
    cfg = replace(settings, api_key="", tool_transport="local", rag_backend="project1", session_db=tmp_path / "sessions.db")
    result = AgentGraph(runtime_settings=cfg).invoke("什么是RAG", "linked-rag")
    assert result["tool_trace"][0]["retrieval_backend"] == "project1"
    assert result["tool_trace"][0]["status"] == "success"
    assert "manual.pdf" in result["answer"]
