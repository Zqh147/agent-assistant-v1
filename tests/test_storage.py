from app.graph import AgentGraph
from app.storage import SQLiteSessionStore
import sqlite3
import pytest


def test_session_survives_new_graph_instance(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    first = AgentGraph(session_store=store).invoke("23 * 47 等于多少？", "persistent")
    second = AgentGraph(session_store=store).invoke("你好，继续回答", "persistent")
    assert "1081" in first["answer"]
    assert second["message_count"] >= 4
    assert store.list_threads()[0]["thread_id"] == "persistent"


def test_same_process_history_is_not_duplicated(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    graph = AgentGraph(session_store=store)
    first = graph.invoke("23 * 47 等于多少？", "same-process")
    second = graph.invoke("你好", "same-process")
    assert first["message_count"] == 2
    assert second["message_count"] == 4
    saved = store.load("same-process")["messages"]
    assert [m["role"] for m in saved] == ["user", "assistant", "user", "assistant"]
    assert sum(m["content"] == "23 * 47 等于多少？" for m in saved) == 1


def test_all_store_operations_close_connections(tmp_path, monkeypatch):
    original = sqlite3.connect
    connections = []
    def tracked(*args, **kwargs):
        conn = original(*args, **kwargs)
        connections.append(conn)
        return conn
    monkeypatch.setattr(sqlite3, "connect", tracked)
    store = SQLiteSessionStore(tmp_path / "closed.db")
    store.save("demo", [{"role": "user", "content": "你好"}], [], [])
    assert store.load("demo")["messages"][0]["content"] == "你好"
    assert len(store.list_threads()) == 1
    assert len(connections) == 4
    # Retain references so garbage collection cannot mask connection leaks.
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")


def test_connection_rolls_back_and_closes_on_exception(tmp_path):
    store = SQLiteSessionStore(tmp_path / "rollback.db")
    with pytest.raises(RuntimeError, match="abort"):
        with store._connect() as conn:
            conn.execute("INSERT INTO sessions(thread_id,messages,tool_trace,errors) VALUES ('abort','[]','[]','[]')")
            raise RuntimeError("abort")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
    assert store.load("abort") is None
