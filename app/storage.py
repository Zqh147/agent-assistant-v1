from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class SQLiteSessionStore:
    """Small durable session store used beside LangGraph's runtime checkpoint."""

    def __init__(self, path: str | Path = "agent_sessions.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    thread_id TEXT PRIMARY KEY,
                    messages TEXT NOT NULL,
                    tool_trace TEXT NOT NULL,
                    errors TEXT NOT NULL,
                    business_context TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            if "business_context" not in columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN business_context TEXT NOT NULL DEFAULT '{}'")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            conn.row_factory = sqlite3.Row
            # The connection context commits/rolls back but does not close it.
            with conn:
                yield conn
        finally:
            conn.close()

    def load(self, thread_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE thread_id = ?", (thread_id,)).fetchone()
        if row is None:
            return None
        return {"messages": json.loads(row["messages"]), "tool_trace": json.loads(row["tool_trace"]), "errors": json.loads(row["errors"]), "business_context": json.loads(row["business_context"])}

    def save(self, thread_id: str, messages: list[dict[str, Any]], tool_trace: list[dict[str, Any]], errors: list[str], business_context=None) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO sessions(thread_id, messages, tool_trace, errors, updated_at)
                   VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(thread_id) DO UPDATE SET messages=excluded.messages,
                   tool_trace=excluded.tool_trace, errors=excluded.errors, updated_at=CURRENT_TIMESTAMP""",
                (thread_id, json.dumps(messages, ensure_ascii=False), json.dumps(tool_trace, ensure_ascii=False), json.dumps(errors, ensure_ascii=False)),
            )
            if business_context is not None:
                conn.execute("UPDATE sessions SET business_context=? WHERE thread_id=?", (json.dumps(business_context, ensure_ascii=False), thread_id))

    def save_business_context(self, thread_id, context):
        """Persist business progress even if a later model response fails."""
        with self._connect() as conn:
            conn.execute("""INSERT INTO sessions(thread_id,messages,tool_trace,errors,business_context)
                            VALUES (?,'[]','[]','[]',?) ON CONFLICT(thread_id) DO UPDATE
                            SET business_context=excluded.business_context,updated_at=CURRENT_TIMESTAMP""",
                         (thread_id, json.dumps(context, ensure_ascii=False)))

    def list_threads(self) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT thread_id, updated_at FROM sessions ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]
