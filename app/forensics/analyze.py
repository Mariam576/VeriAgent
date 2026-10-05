"""Run the forensic checks that apply to one uploaded file."""

from __future__ import annotations

import logging

from app.forensics.model import ForensicReport, Indicator, build_report, page_list
from app.forensics.text import conflicting_fields
from app.rag.loaders import PageText

logger = logging.getLogger(__name__)

PLAIN_TEXT_NOTE = "A plain text file has no layout, fonts or file history to inspect. Only its content was checked."
FAILED_NOTE = "The file could not be inspected for signs of editing."


def analyze_document(data: bytes, extension: str, document_type: str, pages: list[PageText]) -> ForensicReport:
    """Inspect one file. Never raises: a file that cannot be inspected gets a report that says so.

    ``pages`` is the text the application read from the file (hidden text already left out).
    """
    indicators: list[Indicator] = []
    metadata: dict = {}
    note = None
    try:
        if extension == ".pdf":
            from app.forensics.pdf import inspect_pdf

            indicators, metadata = inspect_pdf(data)
        elif extension == ".docx":
            from app.forensics.docx import inspect_docx

            indicators, metadata = inspect_docx(data)
        else:
            note = PLAIN_TEXT_NOTE
        indicators += conflicting_fields(pages, document_type)
    except Exception as exc:  # noqa: BLE001 - forensics is an extra; a strange file must still be usable
        logger.warning("forensic analysis failed: %s: %s", type(exc).__name__, exc)
        return ForensicReport(analysed=False, note=FAILED_NOTE, caveats=[FAILED_NOTE])
    return build_report(indicators, metadata, note, reading_caveats(metadata))


def reading_caveats(metadata: dict) -> list[str]:
    """What the reader of an investigation should know about how a file's text was obtained."""
    caveats = []
    if metadata.get("scanned_pages"):
        pages = page_list(metadata["scanned_pages"])
        caveats.append(
            f"{pages[0].upper()}{pages[1:]} {'is a scan' if pages.startswith('page ') else 'are scans'}. The text was taken from the "
            "recognised-text layer stored in the file and could not be compared with the picture of the page."
        )
    if metadata.get("pages_not_checked"):
        pages = page_list(metadata["pages_not_checked"])
        caveats.append(f"{pages[0].upper()}{pages[1:]} could not be examined for hidden text.")
    if metadata.get("pages_checked_by_drawing_order_only"):
        pages = page_list(metadata["pages_checked_by_drawing_order_only"])
        caveats.append(f"{pages[0].upper()}{pages[1:]} could only be examined in a simpler way, which misses some ways of hiding text.")
    return caveats
