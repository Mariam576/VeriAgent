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
import sys

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings
from app.database.urls import parse_database_url

logger = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None
_saver: AsyncPostgresSaver | None = None
_lock = asyncio.Lock()


def _connection_url() -> str:
    """psycopg wants a plain postgresql:// URL (no SQLAlchemy "+driver" part)."""
    return parse_database_url(get_settings().DATABASE_URL).psycopg_url


async def get_checkpointer() -> AsyncPostgresSaver:
    """Open the pool and create the checkpoint tables on first use."""
    global _pool, _saver
    if _saver is not None:
        return _saver
    if sys.platform == "win32" and type(asyncio.get_running_loop()).__name__ == "ProactorEventLoop":
        # psycopg (used by the checkpoint library) cannot work on this event loop. Say so
        # clearly instead of failing later with a message about sockets.
        raise RuntimeError(
            "Checkpointing cannot run on the default Windows event loop. "
            "Start VeriAgent with Docker (docker compose up), which runs it on Linux."
        )
    async with _lock:
        if _saver is None:
            target = parse_database_url(get_settings().DATABASE_URL)
            pool = AsyncConnectionPool(
                target.psycopg_url,
                min_size=1,
                max_size=min(5, get_settings().DB_POOL_SIZE),
                open=False,
                # A hosted database may close idle connections (scale to zero). Each connection is
                # tested when it is taken from the pool, and replaced if it is dead.
                check=AsyncConnectionPool.check_connection,
                max_idle=240,
                # The checkpointer manages its own transactions and returns rows as dicts.
                kwargs={"autocommit": True, "prepare_threshold": target.psycopg_prepare_threshold, "row_factory": dict_row},
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


CHECKPOINT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")


async def purge_stale_checkpoints() -> dict[str, int]:
    """Delete saved graph state that nobody can use any more. Runs once at startup.

    A checkpoint holds a copy of the text the agent read. It is deleted when its
    investigation finishes, but some are left behind: the investigation (or its
    user) was deleted, or the server stopped at the wrong moment. And an
    investigation that nobody decided or resumed for CHECKPOINT_TTL_DAYS is closed
    as failed, so its saved state does not wait in the database forever.
    """
    from sqlalchemy import text

    from app.database.session import get_engine

    settings = get_settings()
    expired = 0
    async with get_engine().begin() as conn:
        if settings.CHECKPOINT_TTL_DAYS > 0:
            result = await conn.execute(
                text(
                    "UPDATE investigations SET status = 'failed', completed_at = now(), "
                    "error = 'Closed automatically: it was left unfinished for too long.' "
                    "WHERE completed_at IS NULL AND created_at < now() - make_interval(days => :days)"
                ),
                {"days": settings.CHECKPOINT_TTL_DAYS},
            )
            expired = result.rowcount
        # Which threads may stay: unfinished investigations - and, if the owner asked to
        # keep checkpoints for debugging, every investigation that still exists.
        keep = "SELECT id::text FROM investigations"
        if not settings.KEEP_CHECKPOINTS:
            keep += " WHERE completed_at IS NULL AND status <> 'failed'"
        deleted = 0
        for table in CHECKPOINT_TABLES:  # fixed names, not user input
            result = await conn.execute(text(f"DELETE FROM {table} WHERE thread_id NOT IN ({keep})"))
            deleted += result.rowcount
    if expired or deleted:
        logger.info("Checkpoint clean-up: %d investigation(s) expired, %d stale checkpoint row(s) deleted", expired, deleted)
    return {"expired_investigations": expired, "deleted_rows": deleted}
