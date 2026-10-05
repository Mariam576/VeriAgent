"""Document sets for evaluation cases, and an in-memory retriever to search them.

A corpus is a handful of small documents. Some come from the sample PDFs in
``samples/``, the others are written inline so that every case is
self-contained and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.embeddings.base import EmbeddingProvider
from app.embeddings.hash_embeddings import HashEmbeddingProvider
from app.rag.chunking import chunk_pages
from app.rag.loaders import PageText, guess_document_type, load_pages
from app.rag.reranker import rerank
from app.rag.retriever import SearchFilters
from app.schemas.investigation import RetrievedChunk

SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"


@dataclass
class CorpusDocument:
    filename: str
    pages: list[str]
    document_type: str | None = None
    source_path: Path | None = None  # set for the sample files, which exist on disk

    def as_upload(self) -> bytes:
        """The bytes to upload: the original file if there is one, else the pages as a .txt
        (a form feed is the plain-text convention for "new page")."""
        if self.source_path is not None:
            return self.source_path.read_bytes()
        return "\f".join(self.pages).encode("utf-8")


@dataclass
class Corpus:
    name: str
    documents: list[CorpusDocument] = field(default_factory=list)

    def page_text(self, filename: str, page: int) -> str | None:
        for document in self.documents:
            if document.filename == filename and 1 <= page <= len(document.pages):
                return document.pages[page - 1]
        return None


def sample_document(relative_path: str) -> CorpusDocument:
    """One of the files in ``samples/``, read page by page with the normal loaders."""
    path = SAMPLES_DIR / relative_path
    if not path.exists():
        from scripts.make_sample_docs import main as make_samples

        make_samples()
    pages = load_pages(path.read_bytes(), path.suffix.lower())
    return CorpusDocument(filename=path.name, pages=[page.text for page in pages], source_path=path)


class InMemoryRetriever:
    """Same interface as PgVectorRetriever, but everything lives in a Python list.

    It embeds with the given provider (the offline hash embeddings by
    default), ranks by cosine similarity and applies the same reranker as the
    real retriever. There is no keyword search here; hybrid search is part of
    the PostgreSQL retriever (use ``run_eval --backend postgres`` to evaluate it).
    """

    def __init__(self, embedder: EmbeddingProvider | None = None, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self._embedder = embedder or HashEmbeddingProvider(256)
        self._chunk_size, self._chunk_overlap = chunk_size, chunk_overlap
        self._chunks: list[tuple[RetrievedChunk, list[float]]] = []
        self.queries: list[str] = []
        self.fail = False

    def add_text(self, filename: str, pages: list[str], document_type: str | None = None) -> None:
        page_objects = [PageText(page_number=i, text=text) for i, text in enumerate(pages, start=1)]
        doc_type = document_type or guess_document_type(filename, pages[0] if pages else "")
        chunks = chunk_pages(page_objects, self._chunk_size, self._chunk_overlap)
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks]) if chunks else []
        for chunk, vector in zip(chunks, vectors):
            record = RetrievedChunk(
                chunk_id=f"{filename}#{chunk.chunk_index}", document_id=f"doc:{filename}", filename=filename,
                document_type=doc_type, page=chunk.page_number, chunk_index=chunk.chunk_index, text=chunk.text,
            )
            self._chunks.append((record, vector))

    def add_file(self, path: Path, document_type: str | None = None) -> None:
        pages = load_pages(path.read_bytes(), path.suffix.lower())
        self.add_text(path.name, [page.text for page in pages], document_type)

    def add_corpus(self, corpus: Corpus) -> "InMemoryRetriever":
        for document in corpus.documents:
            self.add_text(document.filename, document.pages, document.document_type)
        return self

    async def search(
        self, user_id: str, query: str, *, top_k: int, rerank_top_k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        self.queries.append(query)
        if self.fail:
            raise ConnectionError("vector store is down")
        query_vector = await self._embedder.aembed_text(query)
        scored = []
        for chunk, vector in self._chunks:
            if filters and filters.document_ids and chunk.document_id not in filters.document_ids:
                continue
            score = sum(a * b for a, b in zip(query_vector, vector))
            scored.append(chunk.model_copy(update={"score": round(score, 6)}))
        scored.sort(key=lambda c: c.score, reverse=True)
        return rerank(query, scored[:top_k], rerank_top_k)
