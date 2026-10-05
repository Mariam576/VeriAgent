"""Citations and grounding checks.

Concept - how hallucinated citations are prevented
    1. The model never writes a file name or page number. It only points at a
       chunk reference such as "C2" that we gave it.
    2. Code looks up "C2" and copies filename, page and ids from the database
       record. An unknown reference is thrown away.
    3. The quote (and any number) the model attributes to the chunk must really
       appear in that chunk's text. If not, the fact is dropped.

    A citation in the final report is therefore always a real place in a real
    uploaded document.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.schemas.investigation import EvidenceItem, RetrievedChunk

_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


class Citation(BaseModel):
    document: str
    page: int
    document_id: str
    chunk_ids: list[str] = []


def format_citation(filename: str, page: int) -> str:
    return f"[{filename}, page {page}]"


def normalize(text: str) -> str:
    """Lower-case and collapse whitespace so line breaks in a PDF do not matter."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def numbers_in_text(text: str) -> list[float]:
    """All numbers written in the text: '£40,000' -> 40000.0, '30 days' -> 30.0."""
    values = []
    for match in _NUMBER.findall(text):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def is_grounded(quote: str, value_number: float | None, chunk_text: str) -> bool:
    """True when the chunk really supports the fact.

    - A fact with a number is grounded only if that number is written in the chunk.
    - A fact without a number is grounded only if its quote appears in the chunk.
    """
    if value_number is not None:
        return any(abs(value - value_number) < 0.005 for value in numbers_in_text(chunk_text))
    cleaned_quote = normalize(quote).strip(" .\"'")
    return len(cleaned_quote) >= 8 and cleaned_quote in normalize(chunk_text)


def citations_from_evidence(evidence: list[EvidenceItem]) -> list[Citation]:
    """Unique (document, page) sources, in the order they were first used."""
    citations: dict[tuple[str, int], Citation] = {}
    for item in evidence:
        slot = (item.document_id, item.page)
        if slot not in citations:
            citations[slot] = Citation(document=item.filename, page=item.page, document_id=item.document_id)
        if item.chunk_id not in citations[slot].chunk_ids:
            citations[slot].chunk_ids.append(item.chunk_id)
    return list(citations.values())


def unsupported_citations(citations: list[Citation], retrieved: list[RetrievedChunk]) -> list[Citation]:
    """Citations that do not match any retrieved chunk. The target is always an empty list."""
    known = {(chunk.document_id, chunk.page, chunk.chunk_id) for chunk in retrieved}
    bad = []
    for citation in citations:
        if not citation.chunk_ids or any(
            (citation.document_id, citation.page, chunk_id) not in known for chunk_id in citation.chunk_ids
        ):
            bad.append(citation)
    return bad
