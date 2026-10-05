"""Second-stage ranking of retrieved chunks.

Concept - why rerank?
    Vector search is fast but approximate: it retrieves chunks that are
    "about the same thing" as the query. Retrieving a few more than needed
    (RAG_TOP_K = 8) and then re-scoring them more carefully lets us pass only
    the best ones (RERANK_TOP_K = 4) to the LLM. Fewer, better chunks mean a
    shorter prompt, lower cost and fewer distractions for the model.

Version 1 scoring (simple and free, as the spec suggests):
    score = 70% vector similarity + 30% share of query words found in the chunk
    Exact identifiers such as "INV-102" or "Clause 7.3" count double, because
    embeddings are weak at matching exact codes.

    The selection also makes sure that every document that produced a
    candidate keeps at least one chunk when there is room. A question like
    "compare the invoice with the contract" needs both sides.

A dedicated cross-encoder reranking model can replace ``_score`` later
without touching any other file.
"""

from __future__ import annotations

import re

from app.schemas.investigation import RetrievedChunk

_WORD = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were what "
    "which who will with does do did any there these those between across about tell me please".split()
)


def _terms(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS and len(word) > 1]


def _is_identifier(term: str) -> bool:
    """Codes like inv-102, po-4421, 7.3 or contract-2026-004."""
    return any(char.isdigit() for char in term) and not term.isdigit()


def lexical_overlap(query: str, text: str) -> float:
    """Share of the query's meaningful words that occur in the text (0..1)."""
    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    text_terms = set(_terms(text))
    total = 0.0
    found = 0.0
    for term in query_terms:
        weight = 2.0 if _is_identifier(term) else 1.0
        total += weight
        if term in text_terms:
            found += weight
    return found / total


def _score(query: str, chunk: RetrievedChunk) -> float:
    return 0.7 * max(chunk.score, 0.0) + 0.3 * lexical_overlap(query, chunk.text)


def rerank(query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    """Return the best ``top_n`` candidates, best first."""
    if top_n <= 0 or not candidates:
        return []

    scored = [chunk.model_copy(update={"rerank_score": round(_score(query, chunk), 6)}) for chunk in candidates]
    scored.sort(key=lambda chunk: chunk.rerank_score, reverse=True)

    # First the best chunk of each document, then fill up by score.
    selected: list[RetrievedChunk] = []
    seen_documents: set[str] = set()
    for chunk in scored:
        if chunk.document_id not in seen_documents and len(selected) < top_n:
            selected.append(chunk)
            seen_documents.add(chunk.document_id)
    chosen_ids = {chunk.chunk_id for chunk in selected}
    for chunk in scored:
        if len(selected) >= top_n:
            break
        if chunk.chunk_id not in chosen_ids:
            selected.append(chunk)
            chosen_ids.add(chunk.chunk_id)

    selected.sort(key=lambda chunk: chunk.rerank_score, reverse=True)
    return selected
