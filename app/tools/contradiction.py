"""Deterministic contradiction detection (Node 4).

How it works
    An evidence item may carry a comparison ``key`` and a numeric value.
    Items with the same key describe the same quantity. If two documents give
    different values for one key, that is a contradiction. No LLM is involved
    in the comparison itself, so the result is repeatable and explainable.

What is compared
    ``total_amount``        the overall value of a document (contract value, invoice total)
    ``payment_terms_days``  the number of days allowed for payment

    These two come straight from the evidence that the search found. Parties,
    reference numbers, dates and line items are compared as well, but they
    need to be matched to each other first; that is the job of the document
    profiles in ``app/comparison`` (Version 3).

What is not a contradiction
    * Two invoices with different totals. They are two bills, not two
      statements of one quantity.
    * An invoice for LESS than the contract or order it is checked against.
      That is what a monthly invoice, an instalment and a part delivery look
      like. More than the reference is reported; several invoices that only
      exceed it together are added up by the document comparison.

A contradiction is a reason to review, not proof that something is wrong:
an invoice can legitimately differ from a contract when a signed variation
exists. Whether such support exists is checked later (evidence sufficiency).
"""

from __future__ import annotations

from itertools import combinations

from app.formatting import format_value, key_label
from app.numbers import normalize_unit
from app.schemas.investigation import (
    COMPARABLE_KEYS,
    KEY_PAYMENT_TERMS,
    KEY_TOTAL_AMOUNT,
    Contradiction,
    EvidenceItem,
)

# Lower number = more authoritative reference. The reference value becomes
# "value_a" and the document being checked against it becomes "value_b".
_REFERENCE_PRIORITY = {
    "contract": 0,
    "purchase_order": 1,
    "policy": 2,
    "quotation": 3,
    "other": 4,
    "report": 5,
    "financial_statement": 5,
    "receipt": 6,
    "invoice": 7,
}

# Documents that ask for money. Their totals are compared with what was agreed, never with each other.
_BILLING_TYPES = {"invoice", "receipt"}
# Documents that say what was agreed: an invoice below their total is not a contradiction.
_AGREEMENT_TYPES = {"contract", "purchase_order", "quotation"}

# A policy sets rules and thresholds; it has no "total amount" of its own, so
# an amount found in a policy is never compared with a contract or invoice total.
_NO_TOTAL_DOCUMENT_TYPES = {"policy"}


def _side_rank(items: list[EvidenceItem]) -> tuple[int, str]:
    best = min(_REFERENCE_PRIORITY.get(item.document_type, 4) for item in items)
    return best, min(item.filename for item in items)


def _sources(items: list[EvidenceItem]) -> str:
    names = list(dict.fromkeys(item.filename for item in items))  # unique, order kept
    return ", ".join(names)


def _unit(key: str, items: list[EvidenceItem]) -> str | None:
    if key == KEY_PAYMENT_TERMS:
        return "days"  # by definition of the key, whatever spelling the document used
    for item in items:
        unit = normalize_unit(item.unit)
        if unit:
            return unit
    return None


def classify_difference(key: str, value_a: float, value_b: float) -> tuple[str, str]:
    """Return (type, severity) for two different values of the same key."""
    difference = abs(value_b - value_a)
    relative = difference / max(abs(value_a), abs(value_b), 1e-9)

    if key == KEY_TOTAL_AMOUNT:
        if relative >= 0.10:
            return "amount_mismatch", "high"
        if relative >= 0.01:
            return "amount_mismatch", "medium"
        return "amount_mismatch", "low"
    return "payment_terms_mismatch", "medium" if difference >= 15 else "low"


def find_contradictions(evidence: list[EvidenceItem]) -> list[Contradiction]:
    """Compare evidence items that share a comparison key."""
    # key -> document_id -> the items that document contributed
    per_document: dict[str, dict[str, list[EvidenceItem]]] = {}
    for item in evidence:
        if item.key not in COMPARABLE_KEYS or item.value_number is None:
            continue
        if item.key == KEY_TOTAL_AMOUNT and item.document_type in _NO_TOTAL_DOCUMENT_TYPES:
            continue
        per_document.setdefault(item.key, {}).setdefault(item.document_id, []).append(item)

    across: list[dict] = []
    within: list[dict] = []
    for key in COMPARABLE_KEYS:  # money first, then payment terms
        # ---- Step 1: one value per document.
        by_value: dict[float, list[EvidenceItem]] = {}
        for items in per_document.get(key, {}).values():
            distinct = sorted({round(float(item.value_number), 2) for item in items})  # type: ignore[arg-type]
            if key == KEY_TOTAL_AMOUNT:
                chosen = distinct[-1]  # a total is never smaller than its parts
            else:
                chosen = round(float(items[0].value_number), 2)  # type: ignore[arg-type]  # first one stated
            by_value.setdefault(chosen, []).extend(
                item for item in items if round(float(item.value_number), 2) == chosen  # type: ignore[arg-type]
            )
            if len(distinct) > 1:
                # One document, several values for the same quantity. Often an
                # extraction mix-up (subtotal vs total), so it is only a low-
                # severity prompt to check, never the basis for a high risk level.
                unit = _unit(key, items)
                listed = " and ".join(format_value(value, unit) for value in distinct)
                within.append(dict(
                    type="internal_inconsistency", key=key, severity="low",
                    description=(
                        f"{items[0].filename} appears to give more than one value for the {key_label(key)}: "
                        f"{listed}. {format_value(chosen, unit)} was used for comparison; check which one is meant."
                    ),
                    value_a=distinct[0], value_b=distinct[-1], unit_a=unit, unit_b=unit,
                    source_a=items[0].filename, source_b=items[0].filename,
                    evidence_ids=[item.id for item in items],
                ))

        # ---- Step 2: compare documents with each other.
        for first, second in combinations(sorted(by_value), 2):
            # Put the more authoritative source on side A.
            sides = sorted([(first, by_value[first]), (second, by_value[second])], key=lambda s: _side_rank(s[1]))
            (value_a, items_a), (value_b, items_b) = sides
            bills_a = all(item.document_type in _BILLING_TYPES for item in items_a)
            bills_b = all(item.document_type in _BILLING_TYPES for item in items_b)
            if bills_a and bills_b:
                continue  # two different bills
            agreed = any(item.document_type in _AGREEMENT_TYPES for item in items_a)
            if key == KEY_TOTAL_AMOUNT and agreed and bills_b and value_b < value_a:
                continue  # billing less than was agreed: an instalment or a part delivery
            unit_a, unit_b = _unit(key, items_a), _unit(key, items_b)
            source_a, source_b = _sources(items_a), _sources(items_b)

            if unit_a and unit_b and unit_a != unit_b:
                kind, severity = "unit_mismatch", "medium"
                description = (
                    f"{source_a} states the {key_label(key)} as {format_value(value_a, unit_a)} but "
                    f"{source_b} states {format_value(value_b, unit_b)}. The currencies differ, so the "
                    "values cannot be compared directly."
                )
            else:
                kind, severity = classify_difference(key, value_a, value_b)
                description = (
                    f"{source_a} states the {key_label(key)} as {format_value(value_a, unit_a)} but "
                    f"{source_b} states {format_value(value_b, unit_b or unit_a)}."
                )
            across.append(dict(
                type=kind, key=key, severity=severity, description=description,
                value_a=value_a, value_b=value_b, unit_a=unit_a, unit_b=unit_b or unit_a,
                source_a=source_a, source_b=source_b,
                evidence_ids=[item.id for item in items_a + items_b],
            ))

    # Differences between documents come first; notes about a single document last.
    return [Contradiction(id=f"X{index}", **fields) for index, fields in enumerate(across + within, start=1)]
