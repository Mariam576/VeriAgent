"""Hidden text, tracked changes and file history of Word documents.

A .docx file is a zip archive of XML files. ``word/document.xml`` holds the
text as paragraphs (``w:p``) made of runs (``w:r``); a run is a piece of text
with one formatting. Three things in there matter for forensics:

    hidden runs       formatting "hidden" (``w:vanish``), white text, or text smaller than 2 pt.
                      Word does not show them, a text extractor would.
    tracked changes   insertions (``w:ins``) and deletions (``w:del``) that nobody accepted
                      or rejected. The file then holds two versions of the text.
    core properties   author, last editor, creation and modification time (``docProps/core.xml``).

``visible_paragraph_text`` is what the application reads: the text Word shows
with all changes accepted, without hidden runs.
"""

from __future__ import annotations

import io
from datetime import timedelta
from typing import Any

from app.forensics.model import Indicator, indicator
from app.security.guardrails import detect_prompt_injection

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MIN_VISIBLE_HALF_POINTS = 4  # w:sz counts half points: 4 = 2 pt
_OFF = ("0", "false", "off")


def _flag(properties: Any, name: str) -> bool:
    """Is an on/off formatting property set? (<w:vanish/> = on, <w:vanish w:val="0"/> = off)"""
    element = properties.find(f"{W}{name}") if properties is not None else None
    return element is not None and (element.get(f"{W}val") or "true").lower() not in _OFF


def hidden_reason(run: Any) -> str | None:
    """Why a run is not shown to a reader, or None if it is shown."""
    properties = run.find(f"{W}rPr")
    if properties is None:
        return None
    if _flag(properties, "vanish"):
        return "formatted as hidden text"
    colour = properties.find(f"{W}color")
    shading = properties.find(f"{W}shd")
    highlighted = properties.find(f"{W}highlight") is not None or (
        shading is not None and (shading.get(f"{W}fill") or "auto").upper() not in ("AUTO", "FFFFFF")
    )
    if colour is not None and (colour.get(f"{W}val") or "").upper() == "FFFFFF" and not highlighted:
        return "white text"
    size = properties.find(f"{W}sz")
    try:
        if size is not None and int(size.get(f"{W}val") or 0) and int(size.get(f"{W}val")) < MIN_VISIBLE_HALF_POINTS:
            return "smaller than 2 points"
    except ValueError:
        pass
    return None


def _run_text(run: Any) -> str:
    parts = []
    for child in run:
        if child.tag == f"{W}t":
            parts.append(child.text or "")
        elif child.tag == f"{W}tab":
            parts.append("\t")
        elif child.tag in (f"{W}br", f"{W}cr"):
            parts.append("\n")
        elif child.tag == f"{W}noBreakHyphen":
            parts.append("-")
    return "".join(parts)


def visible_paragraph_text(paragraph_element: Any) -> str:
    """The text of one paragraph as Word shows it once all tracked changes are accepted.

    Runs inside ``w:ins`` (inserted text) and hyperlinks are included; deleted
    text is stored as ``w:delText`` and is therefore never collected; hidden
    runs are skipped.
    """
    return "".join(_run_text(run) for run in paragraph_element.iter(f"{W}r") if hidden_reason(run) is None)


def inspect_docx(data: bytes) -> tuple[list[Indicator], dict[str, Any]]:
    import docx

    document = docx.Document(io.BytesIO(data))
    body = document.element.body
    found: list[Indicator] = []

    # ---- hidden runs
    hidden: dict[str, list[str]] = {}
    for run in body.iter(f"{W}r"):
        reason = hidden_reason(run)
        text = _run_text(run).strip()
        if reason and text:
            hidden.setdefault(reason, []).append(text)
    for reason, texts in hidden.items():
        text = " ".join(texts)
        kind = "hidden_instruction" if detect_prompt_injection(text) else "hidden_text"
        extra = " and reads like instructions to an AI system" if kind == "hidden_instruction" else ""
        found.append(indicator(
            kind,
            f"The document contains text that a reader does not see ({reason}){extra}. It was not read as part of the document.",
            region=text,
        ))

    # ---- tracked changes
    inserted = [element for element in body.iter(f"{W}ins") if element.find(f".//{W}r") is not None]
    deleted = [element for element in body.iter(f"{W}del") if element.find(f".//{W}delText") is not None]
    if inserted or deleted:
        removed = " ".join("".join(node.text or "" for node in element.iter(f"{W}delText")) for element in deleted).strip()
        added = " ".join("".join(_run_text(run) for run in element.iter(f"{W}r")) for element in inserted).strip()
        region = "; ".join(part for part in (f"deleted: {removed}" if removed else "", f"inserted: {added}" if added else "") if part)
        found.append(indicator(
            "tracked_changes",
            f"The document contains tracked changes that were never accepted or rejected ({len(inserted)} insertion(s), "
            f"{len(deleted)} deletion(s)). It was read with the changes applied.",
            region=region,
        ))

    # ---- file history
    properties = document.core_properties
    metadata: dict[str, Any] = {}
    for name in ("author", "last_modified_by", "title"):
        value = getattr(properties, name, None)
        if value:
            metadata[name] = str(value)[:200]
    created, modified = properties.created, properties.modified
    if created:
        metadata["created"] = created.isoformat()
    if modified:
        metadata["modified"] = modified.isoformat()
    if properties.revision:
        metadata["revision"] = properties.revision
    if created and modified:
        if modified < created - timedelta(minutes=1):
            found.append(indicator("metadata_inconsistent", "The file says it was last changed before it was created."))
        elif modified - created > timedelta(minutes=2):
            found.append(indicator(
                "modified_after_creation",
                f"The file was changed after it was created (created {created.day} {created:%B %Y}, "
                f"last changed {modified.day} {modified:%B %Y}).",
            ))
    author, editor = (properties.author or "").strip(), (properties.last_modified_by or "").strip()
    if author and editor and author.lower() != editor.lower():
        found.append(indicator(
            "different_last_editor", "The person who last saved the file is not the person named as its author.",
            region=f"author: {author}; last saved by: {editor}",
        ))
    return found, metadata
