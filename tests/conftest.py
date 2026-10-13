"""Unit tests must not depend on API credentials or existing user sessions."""

from dataclasses import replace
import os

import pytest

# Set before importing app modules; unit tests never require a live model.
# python-dotenv does not override an already set environment variable.
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["TOOL_TRANSPORT"] = "local"
os.environ["RAG_BACKEND"] = "demo"

from app import graph, llm_client


@pytest.fixture(autouse=True)
def isolated_agent_environment(monkeypatch, tmp_path):
    # Force the deterministic offline client, even when a real .env exists.
    monkeypatch.setattr(llm_client, "settings", replace(llm_client.settings, api_key=""))
    # Each test gets a fresh database; never read/write the user's conversations.
    monkeypatch.setattr(graph, "settings", replace(graph.settings, session_db=tmp_path / "sessions.db"))
