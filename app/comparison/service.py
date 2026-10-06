"""The stand-alone comparison: put documents side by side without asking a question.

    POST /comparisons {"document_ids": [...]}

It returns a table with one row per fact and one column per document, the
line items of each document, and the findings of the checks in ``engine.py``.
Differences between numbers are worked out by the calculator tool, exactly as
in an investigation.

This is not a second investigation. There is no search, no risk level and no
recommendation: only what the documents state and where they disagree.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.comparison.engine import ProfileFinding, compare
from app.comparison.names import compare_names, same_identifier
from app.comparison.profile import FIELD_LABELS, PROFILED_TYPES, DocumentProfile, ProfileField
from app.comparison.store import ProfileSource
from app.database.models import Document
from app.errors import NotFound
from app.formatting import format_number, format_value
from app.schemas.comparison import (
    CellOut,
    ComparedDocument,
    ComparisonFindingOut,
    ComparisonOut,
    DocumentLines,
    LineItemOut,
    RowOut,
)
from app.schemas.response import CitationOut
from app.tools.registry import execute_tool

NOT_PROFILED = "Only contracts, purchase orders and invoices have a profile. Set the document type when uploading if it was guessed wrongly."


def show(name: str, value: ProfileField) -> str:
    """A profile value as a person would write it."""
    if value.number is not None:
        if name == "term_months":
            return f"{format_number(value.number)} months"
        return format_value(value.number, value.unit)
    return value.text or ""


def _differs(name: str, values: list[ProfileField]) -> bool:
    """Do the documents that state this fact disagree? Each document's own number is expected to differ."""
    if name in ("number", "date", "contract_reference", "purchase_order_reference", "subtotal", "tax", "term_start", "term_end", "term_months"):
        return False
    if len(values) < 2:
        return False
    first = values[0]
    for other in values[1:]:
        if first.number is not None and other.number is not None:
            if abs(first.number - other.number) > 0.005 or (first.unit and other.unit and first.unit != other.unit):
                return True
        elif name in ("supplier", "customer"):
            if compare_names(first.text, other.text) in ("different", "similar", "short_form"):
                return True
        elif not same_identifier(first.text, other.text) and (first.text or "").strip().lower() != (other.text or "").strip().lower():
            return True
    return False


def build_rows(profiles: list[DocumentProfile | None]) -> list[RowOut]:
    rows = []
    for name, label in FIELD_LABELS.items():
        values: list[ProfileField | None] = []
        for profile in profiles:
            if profile is None:
                values.append(None)
            elif name in ("supplier", "customer"):
                values.append(profile.party(name))  # type: ignore[arg-type]
            else:
                values.append(getattr(profile, name))
        if not any(values):
            continue
        rows.append(RowOut(
            field=name, label=label,
            cells=[CellOut(value=show(name, value), page=value.page, quote=value.quote) if value else None for value in values],
            differs=_differs(name, [value for value in values if value is not None]),
        ))
    # Contract parties whose role the document does not state.
    others = [[party for party in profile.parties if party.role is None] if profile else [] for profile in profiles]
    if any(others):
        rows.append(RowOut(
            field="parties", label="Parties",
            cells=[
                CellOut(value=" and ".join(party.text or "" for party in parties), page=parties[0].page, quote=parties[0].quote)
                if parties else None
                for parties in others
            ],
        ))
    return rows


def finding_out(finding: ProfileFinding) -> ComparisonFindingOut:
    difference = percentage = None
    numeric = finding.value_a is not None and finding.value_b is not None and finding.type != "unit_mismatch"
    if numeric:
        # value_b - value_a, by the tool: the same rule as in an investigation (no arithmetic outside the calculator).
        result = execute_tool("calculate_difference", {"value_a": finding.value_b, "value_b": finding.value_a})
        difference = result.result["value"] if result.result else None
        if finding.source_a != finding.source_b and finding.value_a:
            result = execute_tool("calculate_percentage_change", {"old": finding.value_a, "new": finding.value_b})
            percentage = result.result["value"] if result.result else None
    citations: list[CitationOut] = []
    for cited in finding.cited:
        citation = CitationOut(document=cited.filename, page=cited.page, document_id=cited.document_id, quote=cited.quote)
        if citation not in citations:
            citations.append(citation)
    return ComparisonFindingOut(
        type=finding.type, severity=finding.severity, description=finding.description, key=finding.key,
        source_a=finding.source_a, source_b=finding.source_b, value_a=finding.value_a, value_b=finding.value_b,
        unit_a=finding.unit_a, unit_b=finding.unit_b, text_a=finding.text_a, text_b=finding.text_b,
        difference=difference, percentage_change=percentage, citations=citations,
    )


async def load_documents(session: AsyncSession, user_id: uuid.UUID, document_ids: list[uuid.UUID]) -> list[Document]:
    """The user's own documents, in the order asked for. 404 if any is missing or belongs to someone else."""
    wanted = list(dict.fromkeys(document_ids))
    rows = (await session.execute(
        select(Document).where(Document.id.in_(wanted), Document.user_id == user_id, Document.status == "ready")
    )).scalars().all()
    by_id = {document.id: document for document in rows}
    if len(by_id) != len(wanted):
        raise NotFound("Document not found.")  # the same answer for "does not exist" and "not yours"
    return [by_id[document_id] for document_id in wanted]


async def compare_documents(
    session: AsyncSession, source: ProfileSource, user_id: uuid.UUID, document_ids: list[uuid.UUID]
) -> ComparisonOut:
    documents = await load_documents(session, user_id, document_ids)
    identities = [(str(document.id), document.filename, document.document_type) for document in documents]
    await session.rollback()  # nothing to save here; release the connection while the model works

    batch = await source.profiles(str(user_id), [document_id for document_id, _, _ in identities])
    by_id = {profile.document_id: profile for profile in batch.profiles}
    profiles = [by_id.get(document_id) for document_id, _, _ in identities]

    notes = []
    if any(document_type not in PROFILED_TYPES for _, _, document_type in identities):
        notes.append(NOT_PROFILED)
    if batch.used_rules:
        notes.append("The language model was unavailable, so the documents were read with simple rules. Only common layouts are understood that way.")
    if sum(profile is not None and not profile.is_empty for profile in profiles) < 2:
        notes.append("Fewer than two of these documents could be read in enough detail to compare them with each other.")

    known = [profile for profile in profiles if profile is not None]
    # Someone who puts ONE invoice next to a contract wants the two compared, whatever the supplier
    # names say. With several invoices, each is checked against the contract it belongs to, and the
    # notes say which were left alone.
    one_invoice = sum(profile.document_type == "invoice" for profile in known) == 1
    comparison = compare(known, basic=True, assume_related=one_invoice)
    findings = comparison.findings
    notes += [note.replace(" To have the two compared anyway, name both in the question or select just these documents.",
                           " To have the two compared anyway, select just these two documents.")
              for note in comparison.notes]
    return ComparisonOut(
        documents=[
            ComparedDocument(
                document_id=document_id, filename=filename, document_type=document_type,
                profiled=profile is not None, built_by=profile.built_by if profile else None,
            )
            for (document_id, filename, document_type), profile in zip(identities, profiles)
        ],
        rows=build_rows(profiles),
        line_items=[
            DocumentLines(
                document_id=profile.document_id, filename=profile.filename,
                items=[LineItemOut(**item.model_dump()) for item in profile.line_items],
            )
            for profile in profiles
            if profile is not None and profile.line_items
        ],
        findings=[finding_out(finding) for finding in findings],
        notes=notes,
    )
