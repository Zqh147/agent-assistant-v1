from app.tools import calculate
from app.rag_tool import search_knowledge_base


def test_calculate():
    assert calculate(23, 47, "multiply")["result"] == 1081


def test_division_by_zero():
    try:
        calculate(1, 0, "divide")
    except ValueError as exc:
        assert "zero" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rag_has_source():
    result = search_knowledge_base("什么是 RAG")
    assert result["has_results"] is True
    assert result["results"][0]["source"] == "knowledge.md"
