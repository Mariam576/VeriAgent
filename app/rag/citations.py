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

from app.numbers import find_number_tokens, number_appears, number_stated, parse_number
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


def _unify(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[‐-―−]", "-", text)  # every kind of dash becomes "-"


def normalize(text: str) -> str:
    """Make text comparable: same quotes and dashes, one space, lower case."""
    return re.sub(r"\s+", " ", _unify(text)).strip().lower()


# "Total contract value ........ £40,000": the dots that lead the eye across a price list or a
# table of contents. They are layout, not text, and must not be mistaken for a "..." in a quote.
_LEADER = re.compile(r"(?:\.[ \t]?){4,}|…{2,}|_{4,}")


def _flatten(text: str) -> tuple[str, frozenset[int]]:
    """``normalize(text)``, plus the positions of the spaces that were line breaks."""
    pieces: list[str] = []
    breaks: set[int] = set()
    position = 0
    for piece in re.split(r"(\s+)", _unify(_LEADER.sub(" ", text)).lower().strip()):
        if not piece:
            continue
        if piece.isspace():
            if "\n" in piece:
                breaks.add(position)
            piece = " "
        pieces.append(piece)
        position += len(piece)
    return "".join(pieces), frozenset(breaks)


def _text_variants(text: str) -> list[tuple[str, frozenset[int]]]:
    """The chunk as a reader might retype it.

    PDF extraction breaks lines in the middle of sentences and sometimes in
    the middle of words ("pay-\\nment"). A model quoting the passage usually
    repairs that, so the quote is compared against each plausible repair.
    """
    joined_words = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # "pay-\nment"  -> "payment"
    kept_hyphen = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1-\2", text)  # "INV-\n102"   -> "INV-102"
    variants: dict[str, frozenset[int]] = {}
    for variant in (text, joined_words, kept_hyphen):
        flat, breaks = _flatten(variant)
        variants.setdefault(flat, breaks)
    return list(variants.items())


MAX_GAP_CHARS = 120


def _whole(text: str, start: int, end: int) -> bool:
    """Does text[start:end] begin and end where a word or a number begins and ends?

    "£58" is written in "£58,000" and "40,000" in "£140,000" - as characters.
    As values they are not there. A quote that has to state a value may not be
    cut out of the middle of a longer number or word.
    """
    first, last = text[start], text[end - 1]
    if first.isalnum() and start > 0 and text[start - 1].isalnum():
        return False
    if first.isdigit() and re.search(r"\d[.,]$", text[max(start - 2, 0):start]):
        return False
    if last.isalnum() and end < len(text) and text[end].isalnum():
        return False
    if last.isdigit() and re.match(r"[.,]\d", text[end:end + 2]):
        return False
    return True


def _occurrences(text: str, fragment: str, start: int, whole: bool):
    position = text.find(fragment, start)
    while position != -1:
        if not whole or _whole(text, position, position + len(fragment)):
            yield position
        position = text.find(fragment, position + 1)


def _may_skip(skipped: str, before: str, after: str) -> bool:
    """May a "..." in a quote that states a value stand for these words?"""
    if len(skipped) > MAX_GAP_CHARS:
        return False
    cells = skipped.strip()
    if (before.endswith("|") or cells.startswith("|")) and (after.startswith("|") or cells.endswith("|")):
        return True  # whole cells of one table row: "Lift maintenance | ... | £1,000"
    rest = re.sub(r"\.{2,}", " ", skipped)  # dots that lead across a line are not the end of a sentence
    return not re.search(r"\d|[.!?;] ", rest)


def quote_in_text(quote: str, text: str, min_chars: int = MIN_QUOTE_CHARS, *, strict_gaps: bool = False) -> bool:
    """Does the quote appear in the text, allowing for line breaks and "..." gaps?

    ``strict_gaps``: for quotes that have to state a value. Then

    * every piece of the quote must begin and end at the edge of a word or
      number ("£58" is not a quote from "£58,000"), and
    * a "..." may only skip a few words inside one sentence on one line, and
      never a number - unless it skips whole cells of one table row.

    Otherwise a quote could be stitched together from a label and a figure that
    stand in different sentences ("Total amount due: ... £58,000" on a page that
    says "Total amount due: £40,000" and mentions a limit of £58,000 further down).
    """
    fragments = [
        normalize(fragment).strip(" .,;:\"'")
        for fragment in re.split(r"\.{3,}|…|\[\.\.\.\]", _LEADER.sub(" ", quote))
    ]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments or sum(len(fragment) for fragment in fragments) < min_chars:
        return False

    for variant, breaks in _text_variants(text):
        for start in _occurrences(variant, fragments[0], 0, strict_gaps):
            position = start + len(fragments[0])
            for index, fragment in enumerate(fragments[1:], start=1):  # fragments must appear in the same order
                found = next(_occurrences(variant, fragment, position, strict_gaps), -1)
                if found == -1:
                    break
                if strict_gaps:
                    crosses_a_line = any(position <= line_break < found for line_break in breaks)
                    if crosses_a_line or not _may_skip(variant[position:found], fragments[index - 1], fragment):
                        break
                position = found + len(fragment)
            else:
                return True
            if not strict_gaps:
                break  # as before: only the first place the quote starts is tried
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
    return quote_in_text(quote, chunk_text, MIN_NUMERIC_QUOTE_CHARS, strict_gaps=True) and number_stated(value_number, quote)


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
