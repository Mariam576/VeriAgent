"""Hybrid search helpers: keyword queries and rank fusion.

Concept - why hybrid search?
    Embeddings capture meaning, which makes them good at "payment terms" vs
    "when the invoice is due", and poor at exact codes: to an embedding model
    INV-129 and INV-192 look almost the same. Keyword search is the opposite.
    Running both and merging the results gives the strengths of each.

Concept - reciprocal rank fusion (RRF)
    The two searches produce scores on different scales (cosine similarity vs
    text rank), so the scores cannot simply be added. RRF ignores the scores
    and uses only the positions: a chunk gets 1 / (60 + rank) from each list
    it appears in, and the contributions are summed. A chunk that both
    searches rank highly rises to the top.
"""

from __future__ import annotations

import re

from app.schemas.investigation import RetrievedChunk

RRF_K = 60  # the constant from the original RRF paper; dampens the effect of rank 1 vs rank 2

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were what "
    "which who will with does do did any there these those between across about tell me please should "
    "whether can could would not no if then than into out up".split()
)
MAX_KEYWORD_TERMS = 12


def is_identifier(term: str) -> bool:
    """Codes such as INV-102, PO-4421, 7.3 or Contract-2026-004: digits plus something else."""
    return any(char.isdigit() for char in term) and not term.isdigit()


def keyword_terms(query: str) -> tuple[list[str], list[str]]:
    """Split a question into (ordinary words, exact identifiers) worth searching for."""
    words: list[str] = []
    identifiers: list[str] = []
    for token in _TOKEN.findall(query):
        lowered = token.lower()
        if is_identifier(lowered):
            if lowered not in identifiers:
                identifiers.append(lowered)
        elif lowered not in _STOPWORDS and len(lowered) > 2 and lowered not in words:
            words.append(lowered)
    return words[:MAX_KEYWORD_TERMS], identifiers[:MAX_KEYWORD_TERMS]


def to_websearch_query(terms: list[str]) -> str:
    """Build the text PostgreSQL's ``websearch_to_tsquery`` expects: any of the terms.

    Each term is quoted, so an identifier like inv-102 is searched as the exact
    phrase "inv-102" and not as two unrelated words. Terms contain only
    letters, digits and - . / (see ``_TOKEN``), and the string is still sent
    as a bound parameter, never pasted into SQL.
    """
    return " or ".join(f'"{term}"' for term in terms)


def reciprocal_rank_fusion(
    vector_hits: list[RetrievedChunk], keyword_hits: list[RetrievedChunk], limit: int
) -> list[RetrievedChunk]:
    """Merge two ranked lists into one, best first."""
    merged: dict[str, RetrievedChunk] = {}
    fused: dict[str, float] = {}
    sources: dict[str, list[str]] = {}

    for source, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
        for rank, chunk in enumerate(hits, start=1):
            previous = merged.get(chunk.chunk_id)
            if previous is None:
                merged[chunk.chunk_id] = chunk
            else:
                # Keep the best of each score; the chunk text and source are identical.
                merged[chunk.chunk_id] = previous.model_copy(
                    update={
                        "score": max(previous.score, chunk.score),
                        "keyword_score": max(previous.keyword_score, chunk.keyword_score),
                    }
                )
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            sources.setdefault(chunk.chunk_id, []).append(source)

    ranked = sorted(merged.values(), key=lambda chunk: fused[chunk.chunk_id], reverse=True)
    return [
        chunk.model_copy(update={"fused_score": round(fused[chunk.chunk_id], 6), "matched_by": sources[chunk.chunk_id]})
        for chunk in ranked[:limit]
    ]
