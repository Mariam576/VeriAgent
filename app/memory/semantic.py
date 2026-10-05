"""Semantic long-term memory: what an investigation leaves behind (spec section 26).

Concept - semantic memory:
    Some things worth remembering have no fixed name: "the March invoice from
    the HVAC vendor was 45% over the contract". Such a note is stored as text
    together with its embedding, and found again by meaning: a later question
    like "have we seen similar vendor issues before?" is embedded too, and the
    closest notes are returned.

What is stored
    One short note per finished investigation that found something, written
    by code from the structured report - not by a model, so the note cannot
    contain anything the report did not say. Small talk and simple look-ups
    leave no note.

How it is used
    Recalled notes are shown in the report under "From Earlier
    Investigations". They are context for the reader, never evidence: they
    are not cited as sources and they are not given to the language model,
    so an old note cannot change a new risk assessment.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.memory.store import MemoryStore
from app.observability import tracing
from app.schemas.response import InvestigationReport
from app.security.guardrails import redact_secrets

logger = logging.getLogger(__name__)

MAX_ITEMS_PER_KIND = 3
MAX_TEXT_CHARS = 600


def investigation_note(report: InvestigationReport) -> str | None:
    """The note to remember for a finished investigation, or None when it is not worth keeping."""
    if report.status in ("not_applicable", "insufficient_evidence"):
        return None
    if report.human_review.status == "pending":
        return None  # not finished: the decision is part of what is worth remembering
    contradictions = [item.description for item in report.contradictions][:MAX_ITEMS_PER_KIND]
    missing = list(report.missing_evidence)[:MAX_ITEMS_PER_KIND]
    if not contradictions and not missing and report.status == "no_issues_found" and not report.calculations:
        return None  # a plain look-up with nothing notable

    documents = list(dict.fromkeys(source.document for source in report.sources))[:4]
    parts = [f"Earlier investigation of {', '.join(documents) if documents else 'the uploaded documents'}"]
    parts[0] += f" ({report.status_label.lower()}, {report.risk_level} risk)."
    if contradictions:
        parts.append("Mismatches found: " + " ".join(contradictions))
    elif report.status == "no_issues_found":
        parts.append("No mismatch was found between the documents.")
    if missing:
        parts.append("Missing evidence: " + "; ".join(missing) + ".")
    review = report.human_review
    if review.status in ("approved", "rejected", "modified"):
        decision = f"Human decision: {review.status}"
        if review.note:
            decision += f" ({review.note})"
        parts.append(decision + ".")
    parts.append(f"Question asked: {report.question.strip()[:200]}")
    return " ".join(parts)[:MAX_TEXT_CHARS]


async def remember_investigation(
    store: MemoryStore | None, settings: Settings, user_id: str, report: InvestigationReport
) -> bool:
    """Store the note for a finished investigation. Returns True when a note was written.

    Called by the service only AFTER the result is safely in the database, so
    memory never tells of a decision the database does not have. Memory is a
    convenience: if writing the note fails, the investigation is still done.
    """
    if store is None or not settings.ENABLE_MEMORY:
        return False
    note = investigation_note(report)
    if note is None:
        return False  # small talk and plain look-ups are not worth remembering
    try:
        await store.add_note(
            user_id, redact_secrets(note, settings), source="investigation", investigation_id=report.investigation_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("remember_investigation failed: %s: %s", type(exc).__name__, exc)
        tracing.record("error", "memory", error=type(exc).__name__)
        return False
    tracing.record("memory", "save", memory_type="investigation")
    return True
