from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    max_steps: int = int(os.getenv("MAX_STEPS", "6"))
    tool_timeout_seconds: float = float(os.getenv("TOOL_TIMEOUT_SECONDS", "15"))
    knowledge_path: Path = ROOT / os.getenv("KNOWLEDGE_PATH", "data/knowledge.md")
    session_db: Path = ROOT / os.getenv("SESSION_DB", "data/agent_sessions.db")
    tool_transport: str = os.getenv("TOOL_TRANSPORT", "local")
    rag_backend: str = os.getenv("RAG_BACKEND", "project1")
    rag_service_url: str = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8001")
    rag_timeout_seconds: float = float(os.getenv("RAG_TIMEOUT_SECONDS", "30"))
    support_mode: str = os.getenv("SUPPORT_MODE", "demo")
    support_db: Path = ROOT / os.getenv("SUPPORT_DB", "data/support_demo.db")
    demo_orders_path: Path = ROOT / "data/demo_orders.json"
    support_rules_path: Path = ROOT / "data/demo_after_sales_rules.json"
    demo_customer_id: str = os.getenv("DEMO_CUSTOMER_ID", "demo-customer-001")


settings = Settings()
