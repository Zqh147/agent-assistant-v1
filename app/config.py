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
    model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    max_steps: int = int(os.getenv("MAX_STEPS", "6"))
    tool_timeout_seconds: float = float(os.getenv("TOOL_TIMEOUT_SECONDS", "15"))
    knowledge_path: Path = ROOT / os.getenv("KNOWLEDGE_PATH", "data/knowledge.md")
    session_db: Path = ROOT / os.getenv("SESSION_DB", "data/agent_sessions.db")
    tool_transport: str = os.getenv("TOOL_TRANSPORT", "local")


settings = Settings()
