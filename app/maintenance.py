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
from datetime import datetime, timedelta, timezone

from sqlalchemy import update

from app.agents.checkpoint import purge_stale_checkpoints
from app.config import get_settings
from app.database import repositories
from app.database.models import Document
from app.database.session import get_session_factory

logger = logging.getLogger(__name__)

STUCK_MESSAGE = "Processing was interrupted (the server restarted). Upload the file again."


async def fail_stuck_documents(older_than_minutes: int) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)
    async with get_session_factory()() as session:
        result = await session.execute(
            update(Document)
            .where(Document.status == "processing", Document.uploaded_at < cutoff)
            .values(status="failed", error=STUCK_MESSAGE)
        )
        await session.commit()
        return result.rowcount


async def run_maintenance() -> dict[str, int]:
    """One round of upkeep. Returns what was cleaned, for the log and the tests."""
    settings = get_settings()
    async with get_session_factory()() as session:
        expired_sign_ins = await repositories.delete_expired_refresh_tokens(session)
    checkpoints = await purge_stale_checkpoints()
    stuck = await fail_stuck_documents(settings.DOCUMENT_PROCESSING_TIMEOUT_MINUTES)
    done = {"expired_sign_ins": expired_sign_ins, "stuck_documents": stuck, **checkpoints}
    if any(done.values()):
        logger.info("maintenance %s", " ".join(f"{name}={count}" for name, count in done.items()))
    return done
