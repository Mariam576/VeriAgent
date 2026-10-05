"""Shared test setup.

Tests never call a real LLM or download a model: they use the offline mock
LLM and the hash embedding provider. Integration tests additionally need a
PostgreSQL with pgvector and skip themselves when it is not reachable.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

# Must be set before any ``app`` module reads the settings.
os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["LLM_FALLBACK_PROVIDER"] = ""
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="veriagent-test-uploads-")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/veriagent")

import httpx  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
# Every test run uses its own users, so runs never see each other's documents.
RUN_ID = uuid.uuid4().hex[:8]


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    if not (SAMPLES_DIR / "Contract.pdf").exists():
        from scripts.make_sample_docs import main as make_samples

        make_samples()
    return SAMPLES_DIR


@pytest_asyncio.fixture(scope="session")
async def database():
    """Create tables once per test session; skip integration tests without a database."""
    from app.database.session import check_db, dispose_engine, get_session_factory, init_db

    if not await check_db():
        pytest.skip("PostgreSQL is not reachable - integration tests skipped")
    await init_db()
    yield
    # Remove everything this test run created (cascades to documents, chunks, findings...).
    from app.database.models import User

    async with get_session_factory()() as session:
        await session.execute(delete(User).where(User.name.like(f"test-{RUN_ID}-%")))
        await session.commit()
    # Paused test investigations leave checkpoints behind; their rows are gone now, so sweep them.
    from app.agents.checkpoint import close_checkpointer, get_checkpointer, purge_stale_checkpoints

    await get_checkpointer()
    await purge_stale_checkpoints()
    await close_checkpointer()
    await dispose_engine()


@pytest_asyncio.fixture
async def client(database):
    """HTTP client that talks to the FastAPI app in-process (no server needed)."""
    from app.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


@pytest.fixture
def user_headers():
    """Headers for a brand-new user."""
    return {"X-User-Id": f"test-{RUN_ID}-{uuid.uuid4().hex[:8]}"}
