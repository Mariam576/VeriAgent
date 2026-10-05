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

The full pipeline for one query (spec section 14):

    query embedding -> metadata filters -> vector search  --+
                                         keyword search  ---+-> combine (RRF) -> rerank -> best evidence
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import Select, bindparam, func, literal_column, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database.models import Document, DocumentChunk
from app.embeddings.base import EmbeddingProvider
from app.rag.hybrid import keyword_terms, reciprocal_rank_fusion, to_websearch_query
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
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: EmbeddingProvider,
        *,
        hybrid: bool = True,
        keyword_top_k: int = 8,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._hybrid = hybrid
        self._keyword_top_k = keyword_top_k
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

    def _filtered(self, statement: Select, user_id: str, filters: SearchFilters) -> Select:
        """Conditions shared by the vector and the keyword search. All values are bound parameters."""
        statement = statement.join(Document, Document.id == DocumentChunk.document_id).where(
            Document.user_id == uuid.UUID(str(user_id)),  # ownership: always applied
            Document.status == "ready",
            Document.embedding_model == self._embedder.model_id,
        )
        if filters.document_ids:
            statement = statement.where(Document.id.in_([uuid.UUID(str(d)) for d in filters.document_ids]))
        if filters.document_type:
            statement = statement.where(Document.document_type == filters.document_type)
        if filters.page is not None:
            statement = statement.where(DocumentChunk.page_number == filters.page)
        if filters.year is not None:
            statement = statement.where(DocumentChunk.meta["year"].as_integer() == filters.year)
        return statement

    @staticmethod
    def _to_chunk(chunk: DocumentChunk, document: Document, distance: float, keyword_rank: float = 0.0) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=str(chunk.id),
            document_id=str(document.id),
            filename=document.filename,
            document_type=document.document_type,
            page=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.content,
            score=round(1.0 - float(distance), 6),
            keyword_score=round(float(keyword_rank), 6),
        )

    async def _vector_search(
        self, user_id: str, query_vector: list[float], top_k: int, filters: SearchFilters
    ) -> list[RetrievedChunk]:
        # "<=>" (cosine_distance) is pgvector's cosine distance operator:
        # 0 = same direction, 2 = opposite.
        distance = DocumentChunk.embedding.cosine_distance(query_vector)
        statement = self._filtered(select(DocumentChunk, Document, distance.label("distance")), user_id, filters)
        statement = statement.order_by(distance).limit(top_k)
        # Each search opens its own session, so several searches can run at
        # the same time with asyncio.gather (one session must not be shared
        # between concurrent tasks).
        async with self._session_factory() as session:
            async with session.begin():
                await self._prepare_exact_filtered_search(session)
                rows = (await session.execute(statement)).all()
        return [self._to_chunk(chunk, document, row_distance) for chunk, document, row_distance in rows]

    async def _keyword_search(
        self, user_id: str, query: str, query_vector: list[float], top_k: int, filters: SearchFilters
    ) -> list[RetrievedChunk]:
        """Full-text search. Exact identifiers (INV-102, 7.3) rank above ordinary words."""
        words, identifiers = keyword_terms(query)
        if not words and not identifiers:
            return []

        # Must be written exactly like the index expression so PostgreSQL can use the GIN index.
        document_words = func.to_tsvector(literal_column("'english'"), DocumentChunk.content)

        def tsquery(name: str) -> Any:
            return func.websearch_to_tsquery(literal_column("'english'"), bindparam(name))

        any_term = tsquery("any_term")
        rank = func.ts_rank_cd(document_words, any_term)
        parameters = {"any_term": to_websearch_query(words + identifiers)}
        if identifiers:
            # A match on an exact code weighs ten times a match on an ordinary word.
            exact_code = tsquery("exact_code")
            rank = rank + func.ts_rank_cd(document_words, exact_code) * 10.0
            parameters["exact_code"] = to_websearch_query(identifiers)

        distance = DocumentChunk.embedding.cosine_distance(query_vector)
        statement = self._filtered(
            select(DocumentChunk, Document, distance.label("distance"), rank.label("rank")), user_id, filters
        )
        statement = statement.where(document_words.op("@@")(any_term)).order_by(rank.desc()).limit(top_k)

        async with self._session_factory() as session:
            rows = (await session.execute(statement, parameters)).all()
        return [
            self._to_chunk(chunk, document, row_distance, row_rank) for chunk, document, row_distance, row_rank in rows
        ]

    async def search(
        self, user_id: str, query: str, *, top_k: int, rerank_top_k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        filters = filters or SearchFilters()
        # 1. Query embedding
        query_vector = await self._embedder.aembed_text(query)

        # 2. Metadata filtering + 3. vector retrieval and keyword retrieval, at the same time.
        if self._hybrid:
            vector_hits, keyword_hits = await asyncio.gather(
                self._vector_search(user_id, query_vector, top_k, filters),
                self._keyword_search(user_id, query, query_vector, self._keyword_top_k, filters),
            )
        else:
            vector_hits = await self._vector_search(user_id, query_vector, top_k, filters)
            keyword_hits = []

        # 4. Combine the candidates (reciprocal rank fusion), 5. rerank and keep the best few.
        candidates = reciprocal_rank_fusion(vector_hits, keyword_hits, limit=top_k)
        best = rerank(query, candidates, rerank_top_k)
        logger.info(
            "retrieval query=%r vector=%d keyword=%d candidates=%d kept=%d chunk_ids=%s",
            query[:80], len(vector_hits), len(keyword_hits), len(candidates), len(best),
            [chunk.chunk_id[:8] for chunk in best],
        )
        return best
