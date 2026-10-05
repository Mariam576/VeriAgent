"""Semantic search over document chunks stored in pgvector.

Concept - RAG (retrieval-augmented generation):
    Instead of pasting whole PDFs into the prompt, we (1) embed the question,
    (2) ask the database for the chunks whose vectors are closest to it and
    (3) give only those chunks to the LLM. The model answers from evidence
    that we selected and can cite, and the prompt stays small.

Concept - metadata filtering:
    The vector search is combined with ordinary SQL conditions (which user,
    which document, which page ...). The user condition is always applied, so
    retrieval can never return another user's documents.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database.models import Document, DocumentChunk
from app.embeddings.base import EmbeddingProvider
from app.rag.reranker import rerank
from app.schemas.investigation import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchFilters:
    document_ids: list[str] = field(default_factory=list)
    document_type: str | None = None
    page: int | None = None
    year: int | None = None


class Retriever(Protocol):
    """Anything with this method can serve the agent (tests use an in-memory one)."""

    async def search(
        self, user_id: str, query: str, *, top_k: int, rerank_top_k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]: ...


class PgVectorRetriever:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], embedder: EmbeddingProvider) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._pgvector_version: tuple[int, ...] | None = None

    async def _prepare_exact_filtered_search(self, session: AsyncSession) -> None:
        """Make sure filters never cause missing results.

        An HNSW index returns its nearest neighbours first and applies the
        WHERE conditions afterwards. If the nearest chunks belong to other
        users, a plain index scan could return too few rows, or none.
        pgvector 0.8+ solves this with "iterative scan" (keep scanning until
        enough rows pass the filter). On older versions we tell the planner
        not to use the index for this query, which gives an exact search.
        """
        if self._pgvector_version is None:
            version = await session.scalar(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
            self._pgvector_version = tuple(int(part) for part in (version or "0").split(".") if part.isdigit())
        if self._pgvector_version >= (0, 8):
            await session.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
        else:
            await session.execute(text("SET LOCAL enable_indexscan = off"))

    async def search(
        self, user_id: str, query: str, *, top_k: int, rerank_top_k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        filters = filters or SearchFilters()
        # 1. Query embedding
        query_vector = await self._embedder.aembed_text(query)

        # 2. Metadata filtering + 3. vector retrieval, in one SQL statement.
        #    "<=>" (cosine_distance) is pgvector's cosine distance operator:
        #    0 = same direction, 2 = opposite. All values are bound parameters.
        distance = DocumentChunk.embedding.cosine_distance(query_vector)
        statement = (
            select(DocumentChunk, Document, distance.label("distance"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                Document.user_id == uuid.UUID(str(user_id)),  # ownership: always applied
                Document.status == "ready",
                Document.embedding_model == self._embedder.model_id,
            )
        )
        if filters.document_ids:
            statement = statement.where(Document.id.in_([uuid.UUID(str(d)) for d in filters.document_ids]))
        if filters.document_type:
            statement = statement.where(Document.document_type == filters.document_type)
        if filters.page is not None:
            statement = statement.where(DocumentChunk.page_number == filters.page)
        if filters.year is not None:
            statement = statement.where(DocumentChunk.meta["year"].as_integer() == filters.year)
        statement = statement.order_by(distance).limit(top_k)

        # Each search opens its own session, so several searches can run at
        # the same time with asyncio.gather (one session must not be shared
        # between concurrent tasks).
        async with self._session_factory() as session:
            async with session.begin():
                await self._prepare_exact_filtered_search(session)
                rows = (await session.execute(statement)).all()

        # 4. Top-K candidates
        candidates = [
            RetrievedChunk(
                chunk_id=str(chunk.id),
                document_id=str(document.id),
                filename=document.filename,
                document_type=document.document_type,
                page=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.content,
                score=round(1.0 - float(row_distance), 6),
            )
            for chunk, document, row_distance in rows
        ]
        # 5. Rerank and keep the best few.
        best = rerank(query, candidates, rerank_top_k)
        logger.info(
            "retrieval query=%r candidates=%d kept=%d chunk_ids=%s",
            query[:80], len(candidates), len(best), [chunk.chunk_id[:8] for chunk in best],
        )
        return best
