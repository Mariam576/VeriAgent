"""Reading and writing long-term memory in PostgreSQL.

The agent's nodes and the /memory API both go through ``MemoryStore``. Like
the retriever, the store opens its own short database sessions, so the graph
does not depend on a request's session, and tests can replace the whole
store with an in-memory one.

Every method takes the user id and filters by it: one user's memory is never
visible to another.
"""

from __future__ import annotations

import hashlib
import uuid
from functools import lru_cache
from typing import Any, Protocol

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database.models import SemanticMemory, UserMemory, utcnow
from app.embeddings.base import EmbeddingProvider

MAX_NOTES_LISTED = 200
MAX_NOTES_PER_USER = 500


def _uuid_or_none(value: str | None) -> uuid.UUID | None:
    try:
        return uuid.UUID(value) if value else None
    except ValueError:
        return None


def text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


class MemoryStore(Protocol):
    """What the agent needs from memory. Tests use an in-memory implementation."""

    async def get_preferences(self, user_id: str) -> dict[str, str]: ...

    async def set_preference(self, user_id: str, key: str, value: str) -> None: ...

    async def add_note(
        self, user_id: str, text: str, *, source: str, investigation_id: str | None = None
    ) -> str: ...

    async def search_notes(
        self, user_id: str, query: str, *, top_k: int, min_similarity: float, exclude_ids: list[str] | None = None
    ) -> list[dict[str, Any]]: ...


class PgMemoryStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], embedder: EmbeddingProvider) -> None:
        self._session_factory = session_factory
        self._embedder = embedder

    # ------------------------------------------------------------- profile
    async def get_preferences(self, user_id: str) -> dict[str, str]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(UserMemory.memory_key, UserMemory.memory_value).where(UserMemory.user_id == uuid.UUID(user_id))
            )
            return {key: value for key, value in result.all()}

    async def set_preference(self, user_id: str, key: str, value: str) -> None:
        # INSERT ... ON CONFLICT DO UPDATE ("upsert"): one statement, safe when two requests arrive at once.
        statement = pg_insert(UserMemory).values(
            user_id=uuid.UUID(user_id), memory_key=key, memory_value=value, updated_at=utcnow()
        )
        statement = statement.on_conflict_do_update(
            constraint="uq_user_memory_user_key",
            set_={"memory_value": value, "updated_at": utcnow()},
        )
        async with self._session_factory() as session:
            await session.execute(statement)
            await session.commit()

    async def delete_preference(self, user_id: str, key: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                delete(UserMemory).where(UserMemory.user_id == uuid.UUID(user_id), UserMemory.memory_key == key)
            )
            await session.commit()
            return result.rowcount > 0

    # ------------------------------------------------------------ semantic
    async def add_note(
        self, user_id: str, text: str, *, source: str, investigation_id: str | None = None
    ) -> str:
        """Store a note with its embedding. The same text is kept once; saving it again refreshes it."""
        vector = await self._embedder.aembed_text(text)
        values = dict(
            user_id=uuid.UUID(user_id), text=text, text_hash=text_hash(text), embedding=vector,
            embedding_model=self._embedder.model_id, source=source,
            investigation_id=_uuid_or_none(investigation_id), created_at=utcnow(),
        )
        statement = pg_insert(SemanticMemory).values(**values)
        statement = statement.on_conflict_do_update(
            constraint="uq_semantic_memory_user_text",
            set_={
                "embedding": vector, "embedding_model": self._embedder.model_id,
                "investigation_id": values["investigation_id"], "created_at": values["created_at"],
            },
        ).returning(SemanticMemory.id)
        async with self._session_factory() as session:
            note_id = (await session.execute(statement)).scalar_one()
            # Keep memory bounded: beyond the limit, the oldest notes of this user make room.
            oldest = (
                select(SemanticMemory.id)
                .where(SemanticMemory.user_id == uuid.UUID(user_id))
                .order_by(SemanticMemory.created_at.desc())
                .offset(MAX_NOTES_PER_USER)
            )
            await session.execute(delete(SemanticMemory).where(SemanticMemory.id.in_(oldest)))
            await session.commit()
        return str(note_id)

    async def search_notes(
        self, user_id: str, query: str, *, top_k: int, min_similarity: float, exclude_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """The notes closest in meaning to the query, most similar first."""
        vector = await self._embedder.aembed_text(query)
        distance = SemanticMemory.embedding.cosine_distance(vector)
        statement = (
            select(SemanticMemory, distance.label("distance"))
            .where(
                SemanticMemory.user_id == uuid.UUID(user_id),
                SemanticMemory.embedding_model == self._embedder.model_id,
            )
            .order_by(distance)
            .limit(top_k + len(exclude_ids or []))
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        excluded = set(exclude_ids or [])
        results = []
        for note, note_distance in rows:
            similarity = round(1.0 - float(note_distance), 4)
            if str(note.id) in excluded or similarity < min_similarity:
                continue
            results.append(
                {
                    "id": str(note.id), "text": note.text, "source": note.source, "similarity": similarity,
                    "created_at": note.created_at.isoformat(timespec="seconds"),
                    "investigation_id": str(note.investigation_id) if note.investigation_id else None,
                }
            )
        return results[:top_k]

    async def list_notes(self, user_id: str) -> list[SemanticMemory]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(SemanticMemory)
                .where(SemanticMemory.user_id == uuid.UUID(user_id))
                .order_by(SemanticMemory.created_at.desc())
                .limit(MAX_NOTES_LISTED)
            )
            return list(result.scalars().all())

    async def get_note(self, user_id: str, note_id: str) -> SemanticMemory | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(SemanticMemory).where(
                    SemanticMemory.id == uuid.UUID(note_id), SemanticMemory.user_id == uuid.UUID(user_id)
                )
            )
            return result.scalar_one_or_none()

    async def delete_note(self, user_id: str, note_id: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                delete(SemanticMemory).where(
                    SemanticMemory.id == uuid.UUID(note_id), SemanticMemory.user_id == uuid.UUID(user_id)
                )
            )
            await session.commit()
            return result.rowcount > 0

    async def forget_everything(self, user_id: str) -> dict[str, int]:
        async with self._session_factory() as session:
            preferences = await session.execute(delete(UserMemory).where(UserMemory.user_id == uuid.UUID(user_id)))
            notes = await session.execute(delete(SemanticMemory).where(SemanticMemory.user_id == uuid.UUID(user_id)))
            await session.commit()
            return {"preferences": preferences.rowcount, "notes": notes.rowcount}


@lru_cache
def get_memory_store() -> PgMemoryStore:
    from app.database.session import get_session_factory
    from app.embeddings.factory import get_embedding_provider

    return PgMemoryStore(get_session_factory(), get_embedding_provider())
