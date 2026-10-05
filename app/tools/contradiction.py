"""Deterministic contradiction detection (Node 4).

How it works
    Every evidence item may carry a comparison ``key`` (for example
    ``total_amount``) and a numeric value. Items with the same key describe
    the same quantity. If two sources give different values for one key, that
    is a contradiction. No LLM is involved in the comparison itself, so the
    result is repeatable and explainable.

A contradiction is a reason to review, not proof that something is wrong:
an invoice can legitimately differ from a contract when a signed variation
exists. Whether such support exists is checked later (evidence sufficiency).
"""

from __future__ import annotations

from itertools import combinations

from app.formatting import format_value, key_label
from app.schemas.investigation import (
    KEY_OTHER,
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


def _side_rank(items: list[EvidenceItem]) -> tuple[int, str]:
    best = min(_REFERENCE_PRIORITY.get(item.document_type, 4) for item in items)
    return best, min(item.filename for item in items)


def _sources(items: list[EvidenceItem]) -> str:
    names = list(dict.fromkeys(item.filename for item in items))  # unique, order kept
    return ", ".join(names)


def _unit(items: list[EvidenceItem]) -> str | None:
    for item in items:
        if item.unit:
            return item.unit.strip()
    return None


def _classify(key: str, value_a: float, value_b: float) -> tuple[str, str]:
    """Return (type, severity) for two different values of the same key."""
    difference = abs(value_b - value_a)
    relative = difference / max(abs(value_a), abs(value_b), 1e-9)

    if key == KEY_TOTAL_AMOUNT:
        if relative >= 0.10:
            return "amount_mismatch", "high"
        if relative >= 0.01:
            return "amount_mismatch", "medium"
        return "amount_mismatch", "low"
    if key == KEY_PAYMENT_TERMS:
        return "payment_terms_mismatch", "medium" if difference >= 15 else "low"
    return "value_mismatch", "medium" if relative >= 0.10 else "low"


def find_contradictions(evidence: list[EvidenceItem]) -> list[Contradiction]:
    """Compare evidence items that share a comparison key."""
    by_key: dict[str, dict[float, list[EvidenceItem]]] = {}
    for item in evidence:
        if item.key == KEY_OTHER or not item.key or item.value_number is None:
            continue
        value = round(float(item.value_number), 2)
        by_key.setdefault(item.key, {}).setdefault(value, []).append(item)

    contradictions: list[Contradiction] = []
    for key in sorted(by_key):
        values = by_key[key]
        if len(values) < 2:
            continue  # everyone agrees, or only one source mentions it

        for first, second in combinations(sorted(values), 2):
            # Put the more authoritative source on side A.
            sides = sorted([(first, values[first]), (second, values[second])], key=lambda s: _side_rank(s[1]))
            (value_a, items_a), (value_b, items_b) = sides
            unit_a, unit_b = _unit(items_a), _unit(items_b)
            source_a, source_b = _sources(items_a), _sources(items_b)

            documents = {item.document_id for item in items_a + items_b}
            if unit_a and unit_b and unit_a.lower() != unit_b.lower():
                kind, severity = "unit_mismatch", "medium"
                description = (
                    f"{source_a} states the {key_label(key)} as {format_value(value_a, unit_a)} but "
                    f"{source_b} states {format_value(value_b, unit_b)}. The units differ, so the values "
                    "cannot be compared directly."
                )
            elif len(documents) == 1:
                kind, severity = "internal_inconsistency", "medium"
                description = (
                    f"{source_a} gives two different values for the {key_label(key)}: "
                    f"{format_value(value_a, unit_a)} and {format_value(value_b, unit_b or unit_a)}."
                )
            else:
                kind, severity = _classify(key, value_a, value_b)
                description = (
                    f"{source_a} states the {key_label(key)} as {format_value(value_a, unit_a)} but "
                    f"{source_b} states {format_value(value_b, unit_b or unit_a)}."
                )

            contradictions.append(
                Contradiction(
                    id=f"X{len(contradictions) + 1}",
                    type=kind,
                    key=key,
                    severity=severity,  # type: ignore[arg-type]
                    description=description,
                    value_a=value_a,
                    value_b=value_b,
                    unit_a=unit_a,
                    unit_b=unit_b or unit_a,
                    source_a=source_a,
                    source_b=source_b,
                    evidence_ids=[item.id for item in items_a + items_b],
                )
            )
    return contradictions
