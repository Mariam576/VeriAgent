"""Small, ordered schema upgrades for databases that already exist.

Why this file exists
    ``Base.metadata.create_all`` creates tables that are missing, but it never
    changes a table that is already there. When Version 2 adds a column or an
    index to a Version 1 table, a database created by Version 1 would stay
    behind. Each entry below brings such a database up to date.

Rules for adding an entry
    * Append only. Never edit or reorder an entry that has shipped.
    * Every statement must be safe to run twice (``IF NOT EXISTS``), because a
      brand-new database already has the change through ``create_all``.

Applied entries are recorded in the ``schema_migrations`` table. For a project
with many developers or zero-downtime releases, replace this with Alembic.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)

MIGRATIONS: list[tuple[str, list[str]]] = [
    (
        "v2_001_fulltext_index",
        [
            # Keyword search (hybrid retrieval) needs an index over the words of each chunk.
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_content_fts "
            "ON document_chunks USING gin (to_tsvector('english', content))",
        ],
    ),
    (
        "v2_002_investigation_approval",
        [
            "ALTER TABLE investigations ADD COLUMN IF NOT EXISTS approval_status VARCHAR(20) NOT NULL DEFAULT 'not_required'",
            "ALTER TABLE investigations ADD COLUMN IF NOT EXISTS approval_note TEXT",
            "ALTER TABLE investigations ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ",
        ],
    ),
    (
        "v2_003_investigation_error",
        ["ALTER TABLE investigations ADD COLUMN IF NOT EXISTS error TEXT"],
    ),
    (
        "v2_004_backfill_review_flag",
        [
            # Version 1 reports only had a yes/no "requires_human_review". Carry that over,
            # so an old report that needed a person still says so after the upgrade.
            "UPDATE investigations SET approval_status = 'not_requested' "
            "WHERE approval_status = 'not_required' AND report ->> 'requires_human_review' = 'true' "
            "AND report -> 'human_review' IS NULL",
        ],
    ),
]

# Any fixed number: it only has to be the same in every API process.
MIGRATION_LOCK_ID = 7_316_204


async def run_migrations(conn: AsyncConnection) -> list[str]:
    """Apply every migration that has not been recorded yet. Returns the names applied."""
    # If two API processes start at the same moment, the second waits here until the
    # first has finished (the lock is released when the surrounding transaction ends).
    await conn.execute(text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": MIGRATION_LOCK_ID})
    await conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "name VARCHAR(100) PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
    )
    done = set((await conn.execute(text("SELECT name FROM schema_migrations"))).scalars().all())
    applied = []
    for name, statements in MIGRATIONS:
        if name in done:
            continue
        for statement in statements:
            await conn.execute(text(statement))
        await conn.execute(text("INSERT INTO schema_migrations (name) VALUES (:name)"), {"name": name})
        applied.append(name)
    if applied:
        logger.info("Applied database migrations: %s", ", ".join(applied))
    return applied
