"""Upkeep that nobody asks for but every long-running service needs.

Runs once at startup and then every MAINTENANCE_INTERVAL_MINUTES:

* sign-ins that expired are deleted (their rows are useless and only grow),
* saved graph state that nobody can use any more is deleted, and
  investigations left unfinished for too long are closed,
* documents stuck in "processing" are marked failed: their background job was
  lost when the server restarted, and without this the user would wait forever.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import update

from app.agents.checkpoint import purge_stale_checkpoints
from app.config import get_settings
from app.database import repositories
from app.database.models import Document
from app.database.session import get_session_factory
from app.jobs import get_jobs

logger = logging.getLogger(__name__)

STUCK_MESSAGE = "Processing was interrupted (the server restarted). Upload the file again."


async def fail_stuck_documents(older_than_minutes: int, *, being_processed: set[uuid.UUID] | None = None) -> int:
    """Mark documents that will never leave "processing" as failed.

    ``being_processed`` are the documents whose job is alive in this process
    (waiting for its turn, or running): they are left alone however long they
    have been waiting, because a job that finishes writes its own result.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)
    statement = update(Document).where(Document.status == "processing", Document.uploaded_at <= cutoff)
    if being_processed:
        statement = statement.where(Document.id.not_in(being_processed))
    async with get_session_factory()() as session:
        result = await session.execute(statement.values(status="failed", error=STUCK_MESSAGE))
        await session.commit()
        return result.rowcount


def documents_with_a_job() -> set[uuid.UUID]:
    ids = set()
    for name in get_jobs().names():
        kind, _, value = name.partition(":")
        if kind == "document":
            try:
                ids.add(uuid.UUID(value))
            except ValueError:
                continue
    return ids


async def run_maintenance(*, at_startup: bool = False) -> dict[str, int]:
    """One round of upkeep. Returns what was cleaned, for the log and the tests.

    ``at_startup``: the process has just started, so no background job can be
    alive. Every document that still says "processing" lost its job in the
    restart, however recently it was uploaded, and is marked failed at once -
    otherwise its owner would watch "Reading..." until the next round.
    """
    settings = get_settings()
    async with get_session_factory()() as session:
        expired_sign_ins = await repositories.delete_expired_refresh_tokens(session)
    checkpoints = await purge_stale_checkpoints()
    if at_startup:
        stuck = await fail_stuck_documents(0)
    else:
        stuck = await fail_stuck_documents(settings.DOCUMENT_PROCESSING_TIMEOUT_MINUTES, being_processed=documents_with_a_job())
    done = {"expired_sign_ins": expired_sign_ins, "stuck_documents": stuck, **checkpoints}
    if any(done.values()):
        logger.info("maintenance %s", " ".join(f"{name}={count}" for name, count in done.items()))
    return done
