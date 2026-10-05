"""Turn an uploaded file into text, one entry per page.

Page numbers are kept from the very first step. They travel with every chunk
all the way to the final report, which is what makes a citation like
"[Contract.pdf, page 4]" possible.

Only text that a reader can see is extracted. A file can also carry text
nobody sees: white on white, covered by a box, formatted as "hidden". Such
text must not become evidence, and it is the easiest place to hide
instructions for an AI system. What was left out is reported by the forensics
module (app/forensics).
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass

from app.errors import DocumentProcessingError


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based, as a person would count
    text: str


def clean_text(text: str) -> str:
    """Normalise extracted text without changing its meaning."""
    text = unicodedata.normalize("NFKC", text)  # e.g. the "ﬁ" ligature becomes "fi"
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(data: bytes) -> list[PageText]:
    import pymupdf

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError("The PDF could not be opened. It may be damaged.") from exc
    try:
        if document.needs_pass:
            raise DocumentProcessingError("The PDF is password protected. Remove the password and upload it again.")
        from app.forensics.pdf import read_pdf, remember_inspection

        # One pass gives the visible text and what the forensic report needs (app/forensics/pdf.py).
        texts, found, metadata = read_pdf(document, data)
        remember_inspection(data, found, metadata)
        return [PageText(page_number=index, text=clean_text(text)) for index, text in enumerate(texts, start=1)]
    except DocumentProcessingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError("The PDF could not be read. It may be damaged.") from exc
    finally:
        document.close()


def _load_docx(data: bytes) -> list[PageText]:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError("The Word document could not be opened. It may be damaged.") from exc

    from app.forensics.docx import read_docx

    # Headers, the body with its tables and text boxes in the order of the document, footers:
    # only what Word shows a reader (app/forensics/docx.py).
    try:
        lines = read_docx(document).lines
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError("The Word document could not be read. It may be damaged.") from exc
    # A .docx file has no fixed pages (they depend on the printer and fonts),
    # so the whole document is reported as page 1.
    return [PageText(page_number=1, text=clean_text("\n".join(lines)))]


def _load_txt(data: bytes) -> list[PageText]:
    # Windows tools write several encodings; try them from most to least specific.
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16",)
    else:
        encodings = ("utf-8-sig", "cp1252", "latin-1")
    for encoding in encodings:
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocumentProcessingError("The text file could not be decoded. Save it as UTF-8 and upload it again.")
    # A form-feed character is the plain-text convention for "new page".
    return [
        PageText(page_number=index, text=clean_text(part))
        for index, part in enumerate(text.split("\f"), start=1)
    ]


def load_pages(data: bytes, extension: str) -> list[PageText]:
    """Extract text page by page. Raises ``DocumentProcessingError`` for bad files."""
    loaders = {".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt}
    loader = loaders.get(extension.lower())
    if loader is None:
        raise DocumentProcessingError(f"No loader for '{extension}' files.")

    pages = loader(data)
    if not any(page.text for page in pages):
        raise DocumentProcessingError(
            "No text could be extracted. If this is a scanned document it needs OCR, "
            "which is not enabled in this version."
        )
    return pages


_TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("purchase_order", ("purchase order", "po")),
    ("invoice", ("invoice", "inv")),
    ("contract", ("contract", "agreement")),
    ("policy", ("policy", "procedure", "guideline")),
    ("quotation", ("quotation", "quote", "estimate")),
    ("receipt", ("receipt",)),
    ("financial_statement", ("financial statement", "balance sheet", "income statement", "annual report")),
    ("report", ("report", "audit", "valuation")),
]


def _has_word(text: str, keyword: str) -> bool:
    """True when ``keyword`` occurs as a whole word ("po" in "po 4421", not in "expo")."""
    return re.search(rf"(?<![a-z]){re.escape(keyword)}s?(?![a-z])", text) is not None


def guess_document_type(filename: str, first_page_text: str) -> str:
    """Best-effort label used for metadata filtering. The user can override it on upload."""
    name = re.sub(r"[_\-.]+", " ", filename.lower())
    for doc_type, keywords in _TYPE_KEYWORDS:
        if any(_has_word(name, keyword) for keyword in keywords):
            return doc_type
    # Otherwise look at the title only (the first line). Looking deeper would
    # label a cover note that merely mentions an invoice as an invoice.
    title = next((line for line in first_page_text.splitlines() if line.strip()), "")[:120].lower()
    for doc_type, keywords in _TYPE_KEYWORDS:
        if any(_has_word(title, keyword) for keyword in keywords if len(keyword) > 4):
            return doc_type
    return "other"


def guess_year(text: str) -> int | None:
    """Most frequent 4-digit year in the text, used as filterable metadata."""
    years = re.findall(r"\b(19[89]\d|20\d{2})\b", text)
    if not years:
        return None
    return int(max(set(years), key=years.count))
