"""Run the forensic checks that apply to one uploaded file."""

from __future__ import annotations

import logging

from app.forensics.model import ForensicReport, Indicator, build_report
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
        return ForensicReport(analysed=False, note=FAILED_NOTE)
    return build_report(indicators, metadata, note)
