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
]


async def run_migrations(conn: AsyncConnection) -> list[str]:
    """Apply every migration that has not been recorded yet. Returns the names applied."""
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
