"""SQLite persistence for ticket approvals.

Two things must survive a restart for a pending approval to be resumable:
the LangGraph checkpoint (paused at ``interrupt()``) and the approval
session that maps an approval ID to its actor, ticket and checkpoint thread.
Both live in the same SQLite file.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver


PHASES = ("pending", "resolving", "completed")


def open_approval_checkpointer(path: str | Path) -> SqliteSaver:
    """LangGraph checkpointer backed by SQLite; safe across FastAPI worker threads."""
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


class ApprovalStore:
    """Approval sessions with an atomic pending -> resolving -> completed lifecycle."""

    def __init__(self, path: str | Path = ":memory:"):
        self._conn = sqlite3.connect(
            str(path),
            check_same_thread=False,
            timeout=5,
            isolation_level=None,  # autocommit; each statement is atomic
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_sessions (
                    approval_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    ticket_id TEXT NOT NULL,
                    checkpoint_thread_id TEXT NOT NULL UNIQUE,
                    phase TEXT NOT NULL CHECK (phase IN ('pending', 'resolving', 'completed')),
                    final_status TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TEXT
                )
                """
            )

    def create(
        self,
        approval_id: str,
        *,
        customer_id: str,
        thread_id: str,
        ticket_id: str,
        checkpoint_thread_id: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO approval_sessions"
                "(approval_id, customer_id, thread_id, ticket_id, checkpoint_thread_id, phase)"
                " VALUES (?, ?, ?, ?, ?, 'pending')",
                (approval_id, customer_id, thread_id, ticket_id, checkpoint_thread_id),
            )

    def get(self, approval_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM approval_sessions WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        return dict(row) if row else None

    def claim(self, approval_id: str) -> bool:
        """Move pending -> resolving. Only one caller, in any process, can win."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE approval_sessions SET phase = 'resolving'"
                " WHERE approval_id = ? AND phase = 'pending'",
                (approval_id,),
            )
        return cursor.rowcount == 1

    def release(self, approval_id: str) -> None:
        """Return a claimed session to pending when its decision did not take effect."""
        with self._lock:
            self._conn.execute(
                "UPDATE approval_sessions SET phase = 'pending'"
                " WHERE approval_id = ? AND phase = 'resolving'",
                (approval_id,),
            )

    def complete(self, approval_id: str, final_status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE approval_sessions SET phase = 'completed', final_status = ?,"
                " resolved_at = CURRENT_TIMESTAMP"
                " WHERE approval_id = ? AND phase = 'resolving'",
                (final_status, approval_id),
            )
