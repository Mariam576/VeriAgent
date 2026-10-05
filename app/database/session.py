"""Async database engine and session handling.

Concept - why async?
    Talking to PostgreSQL is I/O: the program mostly waits for the network.
    With ``async``/``await`` FastAPI can serve other requests while one request
    waits for the database, instead of blocking a whole thread.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

DB_STARTUP_ATTEMPTS = 15
DB_STARTUP_DELAY_SECONDS = 2.0


@lru_cache
def get_engine() -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(
        settings.async_database_url,
        pool_pre_ping=True,  # detect dropped connections before using them
        pool_size=5,
        max_overflow=10,
    )


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, always closed afterwards."""
    async with get_session_factory()() as session:
        yield session


async def init_db() -> None:
    """Enable pgvector and create tables that do not exist yet.

    Version 1 keeps this simple with ``create_all``. When the schema starts to
    change between releases, switch to Alembic migrations.
    """
    from app.database import models  # noqa: F401  (registers the tables on Base)
    from app.database.models import Base

    settings = get_settings()
    engine = get_engine()

    # The database container may still be starting when the API starts.
    # Wait for it a bounded number of times instead of crashing immediately.
    for attempt in range(1, DB_STARTUP_ATTEMPTS + 1):
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == DB_STARTUP_ATTEMPTS:
                raise RuntimeError(
                    "Could not connect to the database. Check DATABASE_URL and that PostgreSQL is running."
                ) from exc
            logger.warning("Database not ready yet (attempt %d/%d), retrying...", attempt, DB_STARTUP_ATTEMPTS)
            await asyncio.sleep(DB_STARTUP_DELAY_SECONDS)

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

        # The vector column has a fixed size. If someone changes EMBEDDING_DIM
        # after the table exists, inserts would fail with a confusing error,
        # so check it here and explain the fix.
        result = await conn.execute(
            text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding'"
            )
        )
        existing_dim = result.scalar_one_or_none()
        if existing_dim is not None and existing_dim > 0 and existing_dim != settings.EMBEDDING_DIM:
            raise RuntimeError(
                f"The database stores {existing_dim}-dimensional embeddings but "
                f"EMBEDDING_DIM is {settings.EMBEDDING_DIM}. Set EMBEDDING_DIM={existing_dim}, "
                "or delete the database volume and upload the documents again."
            )
    logger.info("Database ready (pgvector enabled, embedding dimension %s)", settings.EMBEDDING_DIM)


async def check_db() -> bool:
    """Used by /health. Returns False instead of raising."""
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 - health check must never crash
        logger.warning("Database health check failed: %s", exc)
        return False


async def dispose_engine() -> None:
    await get_engine().dispose()
