"""Test doubles for the agent: an in-memory retriever and scripted LLMs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.agents.graph import build_investigation_graph, recursion_limit
from app.agents.heuristics import create_mock_llm
from app.agents.nodes import AgentDeps
from app.agents.state import initial_state
from app.config import Settings, get_settings
from app.embeddings.hash_embeddings import HashEmbeddingProvider
from app.llm.base import BaseLLM, LLMUnavailableError, ToolCallRequest
from app.rag.chunking import chunk_pages
from app.rag.loaders import guess_document_type, load_pages
from app.rag.reranker import rerank
from app.rag.retriever import SearchFilters
from app.schemas.investigation import RetrievedChunk


class InMemoryRetriever:
    """Same interface as PgVectorRetriever, but everything lives in a Python list."""

    def __init__(self) -> None:
        self._embedder = HashEmbeddingProvider(256)
        self._chunks: list[tuple[RetrievedChunk, list[float]]] = []
        self.queries: list[str] = []
        self.fail = False

    def add_text(self, filename: str, pages: list[str], document_type: str | None = None) -> None:
        from app.rag.loaders import PageText

        page_objects = [PageText(page_number=i, text=text) for i, text in enumerate(pages, start=1)]
        doc_type = document_type or guess_document_type(filename, pages[0] if pages else "")
        for chunk in chunk_pages(page_objects, 800, 120):
            record = RetrievedChunk(
                chunk_id=f"{filename}#{chunk.chunk_index}", document_id=f"doc:{filename}", filename=filename,
                document_type=doc_type, page=chunk.page_number, chunk_index=chunk.chunk_index, text=chunk.text,
            )
            self._chunks.append((record, self._embedder.embed_text(chunk.text)))

    def add_file(self, path: Path, document_type: str | None = None) -> None:
        pages = load_pages(path.read_bytes(), path.suffix.lower())
        self.add_text(path.name, [page.text for page in pages], document_type)

    async def search(
        self, user_id: str, query: str, *, top_k: int, rerank_top_k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        self.queries.append(query)
        if self.fail:
            raise ConnectionError("vector store is down")
        query_vector = self._embedder.embed_text(query)
        scored = []
        for chunk, vector in self._chunks:
            if filters and filters.document_ids and chunk.document_id not in filters.document_ids:
                continue
            score = sum(a * b for a, b in zip(query_vector, vector))
            scored.append(chunk.model_copy(update={"score": round(score, 6)}))
        scored.sort(key=lambda c: c.score, reverse=True)
        return rerank(query, scored[:top_k], rerank_top_k)


class ScriptedLLM(BaseLLM):
    """An LLM whose answers are fixed in advance, per schema.

    ``answers`` maps a schema class to an object, a list of objects (used in
    order, the last one repeats), a function ``(system, user) -> object`` or an
    exception to raise. Schemas that are not listed are answered by the normal
    rule-based mock.
    """

    provider = "scripted"
    model = "test"

    def __init__(self, answers: dict[type[BaseModel], Any] | None = None, tool_calls: Any = None) -> None:
        super().__init__(max_retries=0, backoff_seconds=0.0)
        self.answers = answers or {}
        self.tool_calls = tool_calls
        self.default = create_mock_llm()
        self.calls: list[str] = []
        self.prompts: dict[str, list[tuple[str, str]]] = {}

    async def _structured(self, schema, system, user):
        self.calls.append(schema.__name__)
        self.prompts.setdefault(schema.__name__, []).append((system, user))
        if schema not in self.answers:
            return await self.default._structured(schema, system, user)
        answer = self.answers[schema]
        if isinstance(answer, list):
            answer = answer.pop(0) if len(answer) > 1 else answer[0]
        if callable(answer) and not isinstance(answer, BaseModel):
            answer = answer(system, user)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    async def _choose_tools(self, system, user, tools):
        self.calls.append("choose_tools")
        self.prompts.setdefault("choose_tools", []).append((system, user))
        if self.tool_calls is None:
            return await self.default._choose_tools(system, user, tools)
        if isinstance(self.tool_calls, BaseException):
            raise self.tool_calls
        return [ToolCallRequest(**call) if isinstance(call, dict) else call for call in self.tool_calls]


class DownLLM(BaseLLM):
    """A configured provider that is currently unreachable."""

    provider = "down"
    model = "test"

    def __init__(self) -> None:
        super().__init__(max_retries=0, backoff_seconds=0.0)

    async def _structured(self, schema, system, user):
        raise LLMUnavailableError("provider is down")

    async def _choose_tools(self, system, user, tools):
        raise LLMUnavailableError("provider is down")


async def run_graph(
    question: str,
    retriever: InMemoryRetriever,
    llm: BaseLLM | None = None,
    settings: Settings | None = None,
    history: list[dict[str, str]] | None = None,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    graph = build_investigation_graph(AgentDeps(llm=llm or create_mock_llm(), retriever=retriever, settings=settings))
    state = initial_state(
        user_id="user-1", investigation_id="inv-1", question=question, history=history, document_ids=document_ids
    )
    return await graph.ainvoke(state, config={"recursion_limit": recursion_limit(settings.MAX_AGENT_ITERATIONS)})


def sample_library(samples_dir: Path, *names: str) -> InMemoryRetriever:
    retriever = InMemoryRetriever()
    for name in names or ("Contract.pdf", "Invoice_102.pdf", "Procurement_Policy.pdf"):
        retriever.add_file(samples_dir / name)
    return retriever
