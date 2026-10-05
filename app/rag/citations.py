"""Citations and grounding checks.

Concept - how hallucinated citations are prevented
    1. The model never writes a file name or page number. It only points at a
       chunk reference such as "C2" that we gave it.
    2. Code looks up "C2" and copies filename, page and ids from the database
       record. An unknown reference is thrown away.
    3. The quote the model attributes to the chunk must really appear in that
       chunk's text, and a numeric fact's number must be written inside that
       quote. If not, the fact is dropped.

    A citation in the final report is therefore always a real place in a real
    uploaded document.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.numbers import find_number_tokens, number_appears, parse_number
from app.schemas.investigation import EvidenceItem, RetrievedChunk

MIN_QUOTE_CHARS = 8  # a shorter text quote proves nothing ("pay", "the")
MIN_NUMERIC_QUOTE_CHARS = 2  # a numeric quote may be as short as "£40,000"


class Citation(BaseModel):
    document: str
    page: int
    document_id: str
    chunk_ids: list[str] = []


def format_citation(filename: str, page: int) -> str:
    return f"[{filename}, page {page}]"


def normalize(text: str) -> str:
    """Make text comparable: same quotes and dashes, one space, lower case."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐-―−]", "-", text)  # every kind of dash becomes "-"
    return re.sub(r"\s+", " ", text).strip().lower()


def _text_variants(text: str) -> list[str]:
    """The chunk as a reader might retype it.

    PDF extraction breaks lines in the middle of sentences and sometimes in
    the middle of words ("pay-\\nment"). A model quoting the passage usually
    repairs that, so the quote is compared against each plausible repair.
    """
    joined_words = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # "pay-\nment"  -> "payment"
    kept_hyphen = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1-\2", text)  # "INV-\n102"   -> "INV-102"
    return list(dict.fromkeys(normalize(variant) for variant in (text, joined_words, kept_hyphen)))


def quote_in_text(quote: str, text: str, min_chars: int = MIN_QUOTE_CHARS) -> bool:
    """Does the quote appear in the text, allowing for line breaks and "..." gaps?"""
    fragments = [
        normalize(fragment).strip(" .,;:\"'")
        for fragment in re.split(r"\.{3,}|…|\[\.\.\.\]", quote)
    ]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments or sum(len(fragment) for fragment in fragments) < min_chars:
        return False

    for variant in _text_variants(text):
        position = 0
        for fragment in fragments:  # fragments must appear in the same order
            found = variant.find(fragment, position)
            if found == -1:
                break
            position = found + len(fragment)
        else:
            return True
    return False


def is_grounded(quote: str, value_number: float | None, chunk_text: str) -> bool:
    """True when the chunk really supports the fact.

    - The quote must appear in the chunk.
    - If the fact has a number, that number must be written inside the quote.
      (A number that merely occurs somewhere else in the chunk is not enough:
      "30" in "30 days" must not support "approved by 30 directors".)
    """
    if value_number is None:
        return quote_in_text(quote, chunk_text)
    return quote_in_text(quote, chunk_text, MIN_NUMERIC_QUOTE_CHARS) and number_appears(value_number, quote)


_FILENAME = re.compile(r"\S+\.(?:pdf|docx|txt)\b", re.IGNORECASE)


def numbers_not_in_source(claim: str, source_text: str) -> list[str]:
    """Numbers written in a claim that do not occur anywhere in its source text.

    The quote check proves that the quoted words exist. It does not prove that
    the claim says the same thing as the quote: a model could quote a real
    sentence and attach an invented date or amount to it. A fact is something
    a document states, so every number in it must be in that document. (A
    number the agent worked out itself is a calculation and comes from a tool.)

    File names are ignored ("Invoice_102.pdf states ..."), and a number counts
    as present when its digits match a number in the source, however it is
    punctuated there.
    """
    source_digits = {re.sub(r"\D", "", token) for token, _, _ in find_number_tokens(source_text)}
    missing = []
    for token, _, _ in find_number_tokens(_FILENAME.sub(" ", claim)):
        digits = re.sub(r"\D", "", token)
        if not digits or digits in source_digits:
            continue
        if any(number_appears(value, source_text) for value in parse_number(token)):
            continue
        missing.append(token)
    return missing


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
