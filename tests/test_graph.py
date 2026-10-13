from app.graph import AgentGraph


def test_graph_calculation():
    result = AgentGraph().invoke("23 * 47 等于多少？", thread_id="test-calc")
    assert "1081" in result["answer"]
    assert result["tool_trace"][0]["tool"] == "calculate"


def test_graph_direct_answer():
    result = AgentGraph().invoke("你好", thread_id="test-direct")
    assert result["tool_trace"] == []
