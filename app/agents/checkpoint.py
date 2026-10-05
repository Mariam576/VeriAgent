"""The LangGraph checkpointer: where a running or paused investigation is saved.

Concept - checkpointing:
    After every node, LangGraph writes the complete graph state to the
    checkpointer under a ``thread_id``. VeriAgent uses the investigation id as
    the thread id. Three things follow from that:

    * pause   - a node can stop the graph (human approval) and nothing is lost
    * resume  - a later request continues exactly where the graph stopped
    * recover - if the server stops mid-run, the investigation can be resumed
                from the last completed node instead of starting again

    The checkpoints are stored in PostgreSQL (tables ``checkpoints``,
    ``checkpoint_blobs``, ``checkpoint_writes``), so they survive a restart.
    An in-memory checkpointer would forget every paused investigation.

The checkpointer library talks to PostgreSQL through psycopg, so it has its
own small connection pool, separate from the SQLAlchemy engine.
"""

from __future__ import annotations

import asyncio
import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

logger = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None
_saver: AsyncPostgresSaver | None = None
_lock = asyncio.Lock()


def _connection_url() -> str:
    """psycopg wants a plain postgresql:// URL (no SQLAlchemy "+driver" part)."""
    url = get_settings().DATABASE_URL
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    return url


async def get_checkpointer() -> AsyncPostgresSaver:
    """Open the pool and create the checkpoint tables on first use."""
    global _pool, _saver
    if _saver is not None:
        return _saver
    async with _lock:
        if _saver is None:
            pool = AsyncConnectionPool(
                _connection_url(),
                min_size=1,
                max_size=5,
                open=False,
                # The checkpointer manages its own transactions and returns rows as dicts.
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            )
            await pool.open(wait=True, timeout=30)
            saver = AsyncPostgresSaver(pool)
            await saver.setup()  # creates the checkpoint tables if they do not exist
            _pool, _saver = pool, saver
            logger.info("Checkpointer ready (PostgreSQL)")
    return _saver


async def close_checkpointer() -> None:
    global _pool, _saver
    pool, _pool, _saver = _pool, None, None
    if pool is not None:
        await pool.close()
