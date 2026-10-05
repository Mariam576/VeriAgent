"""Shared test setup.

Tests never call a real LLM or download a model: they use the offline mock
LLM and the hash embedding provider. Integration tests additionally need a
PostgreSQL with pgvector and skip themselves when it is not reachable.
"""

from __future__ import annotations

import os

# Must be set before any ``app`` module reads the settings.
os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["EMBEDDING_DIM"] = "384"
os.environ["LLM_FALLBACK_PROVIDER"] = ""
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/veriagent")
os.environ.setdefault("UPLOAD_DIR", "uploads")

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()


@pytest.fixture
def settings():
    return get_settings()
