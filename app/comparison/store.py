"""Where document profiles come from.

A profile costs one LLM call, so it is built once and stored:

    needed for the first time  ->  read the document's text  ->  build  ->  save
    needed again               ->  load

``ProfileSource`` is the small interface the rest of the application uses.
``PgProfileStore`` implements it on PostgreSQL; the evaluation and the unit
tests use ``MemoryProfileSource``, which keeps everything in a Python dict.

Both also hand out the chunks of one page. A finding from a comparison quotes
a place in a document, and a citation in a report always points at a chunk
that exists (see app/rag/citations.py), so the comparison needs the chunk
that contains its quote.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.comparison.llm import build_profile
from app.comparison.profile import PROFILE_VERSION, PROFILED_TYPES, DocumentProfile
from app.database.models import Document, DocumentChunk, DocumentProfileRow
from app.llm.base import BaseLLM
from app.rag.loaders import PageText
from app.schemas.investigation import RetrievedChunk

logger = logging.getLogger(__name__)

MAX_OVERLAP = 400  # more than any configured chunk overlap


@dataclass
class ProfileBatch:
    profiles: list[DocumentProfile] = field(default_factory=list)
    llm_calls: int = 0  # how many profiles the model had to build just now
    used_rules: bool = False  # the model was unavailable for at least one of them


class ProfileSource(Protocol):
    async def profiles(self, user_id: str, document_ids: list[str]) -> ProfileBatch:
        """Profiles of the given documents, in the same order. Documents of other types are left out."""
        ...

    async def page_chunks(self, user_id: str, document_id: str, page: int) -> list[RetrievedChunk]:
        """The chunks of one page of one of the user's documents, in reading order."""
        ...


def merge_chunks(texts: list[str]) -> str:
    """Put the overlapping chunks of one page back together.

    Neighbouring chunks share their edges (see app/rag/chunking.py). Joining
    them as they are would repeat those words, and a repeated invoice line
    would be counted twice. So the longest text that ends one chunk and starts
    the next is written only once.
    """
    merged = ""
    for text in texts:
        if not merged:
            merged = text
            continue
        limit = min(len(merged), len(text), MAX_OVERLAP)
        overlap = next((size for size in range(limit, 0, -1) if merged.endswith(text[:size])), 0)
        merged += text[overlap:] if overlap else "\n" + text
    return merged


def pages_from_chunks(chunks: list[tuple[int, int, str]]) -> list[PageText]:
    """(page number, chunk index, text) rows -> the text of every page."""
    by_page: dict[int, list[str]] = {}
    for page, _, text in sorted(chunks):
        by_page.setdefault(page, []).append(text)
    return [PageText(page, merge_chunks(texts)) for page, texts in sorted(by_page.items())]


class PgProfileStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], llm: BaseLLM) -> None:
        self._session_factory = session_factory
        self._llm = llm

    async def profiles(self, user_id: str, document_ids: list[str]) -> ProfileBatch:
        try:
            wanted = [uuid.UUID(str(document_id)) for document_id in dict.fromkeys(document_ids)]
            owner = uuid.UUID(str(user_id))
        except ValueError:
            return ProfileBatch()
        if not wanted:
            return ProfileBatch()

        async with self._session_factory() as session:
            rows = (await session.execute(
                select(Document, DocumentProfileRow)
                .outerjoin(DocumentProfileRow, DocumentProfileRow.document_id == Document.id)
                .where(Document.id.in_(wanted), Document.user_id == owner, Document.status == "ready")  # ownership
            )).all()
            documents = {document.id: (document, stored) for document, stored in rows}
            ready: dict[uuid.UUID, DocumentProfile] = {}
            to_build: list[Document] = []
            for document_id in wanted:
                if document_id not in documents:
                    continue
                document, stored = documents[document_id]
                if document.document_type not in PROFILED_TYPES:
                    continue
                # A profile built by rules means the model was down at the time: try the model again.
                if stored is not None and stored.version == PROFILE_VERSION and stored.built_by == "llm" \
                        and stored.profile.get("document_type") == document.document_type:
                    ready[document_id] = DocumentProfile.model_validate(stored.profile)
                else:
                    to_build.append(document)

            pages: dict[uuid.UUID, list[PageText]] = {}
            if to_build:
                chunk_rows = (await session.execute(
                    select(DocumentChunk.document_id, DocumentChunk.page_number, DocumentChunk.chunk_index, DocumentChunk.content)
                    .where(DocumentChunk.document_id.in_([document.id for document in to_build]))
                )).all()
                grouped: dict[uuid.UUID, list[tuple[int, int, str]]] = {}
                for document_id, page, index, content in chunk_rows:
                    grouped.setdefault(document_id, []).append((page, index, content))
                pages = {document_id: pages_from_chunks(chunks) for document_id, chunks in grouped.items()}
            identities = [(document.id, document.filename, document.document_type) for document in to_build]

        batch = ProfileBatch()
        if identities:
            # One model call per document, all at the same time (the LLM client limits its own rate).
            built = await asyncio.gather(*(
                build_profile(
                    self._llm, document_id=str(document_id), filename=filename, document_type=document_type,
                    pages=pages.get(document_id, []),
                )
                for document_id, filename, document_type in identities
            ))
            async with self._session_factory() as session:
                for (document_id, _, _), (profile, by_model) in zip(identities, built):
                    ready[document_id] = profile
                    batch.llm_calls += 1 if by_model else 0
                    batch.used_rules = batch.used_rules or not by_model
                    values = dict(version=PROFILE_VERSION, built_by=profile.built_by, profile=profile.model_dump(mode="json"))
                    statement = insert(DocumentProfileRow).values(document_id=document_id, **values)
                    # The document may have been deleted while the model was working: then there is nothing to save.
                    exists = await session.scalar(select(Document.id).where(Document.id == document_id))
                    if exists is not None:
                        await session.execute(statement.on_conflict_do_update(index_elements=["document_id"], set_=values))
                await session.commit()

        batch.profiles = [ready[document_id] for document_id in wanted if document_id in ready]
        return batch

    async def page_chunks(self, user_id: str, document_id: str, page: int) -> list[RetrievedChunk]:
        try:
            owner, wanted = uuid.UUID(str(user_id)), uuid.UUID(str(document_id))
        except ValueError:
            return []
        async with self._session_factory() as session:
            rows = (await session.execute(
                select(DocumentChunk, Document)
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(Document.id == wanted, Document.user_id == owner, DocumentChunk.page_number == page)
                .order_by(DocumentChunk.chunk_index)
            )).all()
        return [
            RetrievedChunk(
                chunk_id=str(chunk.id), document_id=str(document.id), filename=document.filename,
                document_type=document.document_type, page=chunk.page_number, chunk_index=chunk.chunk_index,
                text=chunk.content,
            )
            for chunk, document in rows
        ]


class MemoryProfileSource:
    """Profiles for documents that live in a list of chunks (evaluation and unit tests)."""

    def __init__(self, chunks: list[RetrievedChunk], llm: BaseLLM) -> None:
        self._chunks = chunks
        self._llm = llm
        self._cache: dict[str, DocumentProfile] = {}

    async def profiles(self, user_id: str, document_ids: list[str]) -> ProfileBatch:
        batch = ProfileBatch()
        for document_id in dict.fromkeys(document_ids):
            own = [chunk for chunk in self._chunks if chunk.document_id == document_id]
            if not own or own[0].document_type not in PROFILED_TYPES:
                continue
            if document_id not in self._cache:
                pages = pages_from_chunks([(chunk.page, chunk.chunk_index, chunk.text) for chunk in own])
                profile, by_model = await build_profile(
                    self._llm, document_id=document_id, filename=own[0].filename,
                    document_type=own[0].document_type, pages=pages,
                )
                batch.llm_calls += 1 if by_model else 0
                batch.used_rules = batch.used_rules or not by_model
                if by_model:
                    self._cache[document_id] = profile
                batch.profiles.append(profile)
            else:
                batch.profiles.append(self._cache[document_id])
        return batch

    async def page_chunks(self, user_id: str, document_id: str, page: int) -> list[RetrievedChunk]:
        own = [chunk for chunk in self._chunks if chunk.document_id == document_id and chunk.page == page]
        return sorted(own, key=lambda chunk: chunk.chunk_index)


_store: PgProfileStore | None = None


def get_profile_store() -> PgProfileStore:
    """The profile store of this process (it holds no data itself, only how to reach the database and the model)."""
    global _store
    if _store is None:
        from app.database.session import get_session_factory
        from app.llm.factory import get_llm

        _store = PgProfileStore(get_session_factory(), get_llm())
    return _store


def reset_profile_store() -> None:
    global _store
    _store = None
