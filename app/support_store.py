"""Local demonstration orders and tickets; all writes are SQLite transactions."""
from contextlib import contextmanager
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


class SupportStore:
    def __init__(self, path, orders_path, today=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.today = today or date.today
        source = json.loads(Path(orders_path).read_text(encoding="utf-8"))
        if source.get("data_kind") != "fictional_demo_orders":
            raise ValueError("only explicitly labelled demonstration orders are supported")
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,
                    product_model TEXT NOT NULL, hardware_version TEXT NOT NULL,
                    region TEXT NOT NULL, status TEXT NOT NULL, delivered_at TEXT
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY, draft_key TEXT UNIQUE NOT NULL,
                    customer_id TEXT NOT NULL, thread_id TEXT NOT NULL,
                    order_id TEXT NOT NULL REFERENCES orders(order_id),
                    service_type TEXT NOT NULL CHECK(service_type IN ('repair','return')),
                    issue_summary TEXT NOT NULL, policy_version TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('draft','submitted')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    submitted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS ticket_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                    event TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
            anchor = self.today()
            for order in source["orders"]:
                offset = order["delivered_days_ago"]
                delivered = (anchor - timedelta(days=offset)).isoformat() if offset is not None else None
                db.execute("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?)", (
                    order["order_id"], order["customer_id"], order["product_model"],
                    order["hardware_version"], order["region"], order["status"], delivered))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def lookup_order(self, order_id, customer_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM orders WHERE order_id=? AND customer_id=?", (order_id, customer_id)).fetchone()
        return dict(row) if row else None

    def list_orders(self, customer_id):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM orders WHERE customer_id=? ORDER BY order_id", (customer_id,)).fetchall()
        return [dict(row) for row in rows]

    def draft(self, customer_id, thread_id, order_id, service_type, issue_summary, policy_version):
        # Same conversation/order/type resolves to one application, regardless
        # of LLM paraphrases, retries, or repeated prepare requests.
        key = hashlib.sha256(json.dumps([customer_id, thread_id, order_id, service_type]).encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM tickets WHERE draft_key=?", (key,)).fetchone()
            if row:
                if row["status"] == "draft" and row["policy_version"] != policy_version:
                    db.execute("UPDATE tickets SET policy_version=?,issue_summary=? WHERE ticket_id=?", (policy_version, issue_summary, row["ticket_id"]))
                    db.execute("INSERT INTO ticket_events(ticket_id,event) VALUES (?,?)", (row["ticket_id"], "draft_revalidated"))
                    row = db.execute("SELECT * FROM tickets WHERE ticket_id=?", (row["ticket_id"],)).fetchone()
                return dict(row), False
            ticket_id = "TKT-" + uuid4().hex[:16].upper()
            db.execute("INSERT INTO tickets(ticket_id,draft_key,customer_id,thread_id,order_id,service_type,issue_summary,policy_version,status) VALUES (?,?,?,?,?,?,?,?,?)", (
                ticket_id, key, customer_id, thread_id, order_id, service_type, issue_summary, policy_version, "draft"))
            db.execute("INSERT INTO ticket_events(ticket_id,event) VALUES (?,?)", (ticket_id, "draft_created"))
            row = db.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
            return dict(row), True

    def get_ticket(self, ticket_id, customer_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM tickets WHERE ticket_id=? AND customer_id=?", (ticket_id, customer_id)).fetchone()
        return dict(row) if row else None

    def submit(self, ticket_id, customer_id, thread_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM tickets WHERE ticket_id=? AND customer_id=? AND thread_id=?", (ticket_id, customer_id, thread_id)).fetchone()
            if row is None:
                return None, False
            if row["status"] == "submitted":
                return dict(row), False
            db.execute("UPDATE tickets SET status='submitted',submitted_at=CURRENT_TIMESTAMP WHERE ticket_id=? AND status='draft'", (ticket_id,))
            db.execute("INSERT INTO ticket_events(ticket_id,event) VALUES (?,?)", (ticket_id, "submitted"))
            row = db.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
            return dict(row), True

    def events(self, ticket_id, customer_id):
        if self.get_ticket(ticket_id, customer_id) is None:
            return []
        with self.connect() as db:
            rows = db.execute("SELECT event,created_at FROM ticket_events WHERE ticket_id=? ORDER BY id", (ticket_id,)).fetchall()
        return [dict(row) for row in rows]
