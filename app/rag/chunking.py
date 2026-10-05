"""Split page text into overlapping chunks.

Concept - why chunk?
    An embedding of a whole 40-page contract is a blurry average of everything
    in it. An embedding of one paragraph is sharp: a search for "payment terms"
    lands on the paragraph about payment terms. Small chunks also keep the
    prompt short, which matters on a free API tier.

Concept - recursive chunking:
    Try to split on paragraphs first; if a piece is still too long split on
    lines, then sentences, then words. Text is cut at natural boundaries
    whenever possible.

Concept - overlap:
    Neighbouring chunks share some text, so a sentence that straddles a cut is
    still complete in at least one chunk.

Every page is chunked on its own, so a chunk always belongs to exactly one page.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loaders import PageText


@dataclass(frozen=True)
class Chunk:
    page_number: int
    chunk_index: int  # position within the whole document, starting at 0
    text: str


def chunk_pages(pages: list[PageText], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(page_number=page.page_number, chunk_index=len(chunks), text=piece))
    return chunks
