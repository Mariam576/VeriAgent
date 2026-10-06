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
    * Two documents of the same kind: two invoices, two contracts, two orders.
      They are two pieces of business, not two statements of one quantity.
    * A document that states LESS than the one it is measured against: an
      invoice below its contract (a monthly invoice, an instalment, a part
      delivery), a call-off order below its framework agreement, a receipt for
      a part payment. More than the reference is reported; several invoices
      that only exceed it together are added up by the document comparison.
    * Two agreements that do not belong together. A contract and an order are
      only compared when one cites the other, or when they are the only two.
    * Documents that are known not to belong together (``unrelated``): an
      invoice from another supplier, an earlier version of a revised order.

    Currencies are looked at first. "GBP 40,000" and "KWD 39,000" cannot be
    compared at all, and saying so is more useful than saying nothing.

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
    "invoice": 6,
    "receipt": 7,  # a receipt is measured against the invoice it pays
}

# Documents that say what was agreed.
_AGREEMENT_TYPES = {"contract", "purchase_order", "quotation"}

# A policy sets rules and thresholds; it has no "total amount" of its own, so
# an amount found in a policy is never compared with a contract or invoice total.
_NO_TOTAL_DOCUMENT_TYPES = {"policy"}

Pair = tuple[str, str]


# Documents whose kind says nothing about their role: a memo, a budget report, a statement of account.
# What they state is compared in both directions - a contract above the approved budget matters as
# much as a budget above the contract.
_STATEMENT_TYPES = {"other", "report", "financial_statement"}


def _in_order(first: tuple[float, list[EvidenceItem]], second: tuple[float, list[EvidenceItem]]):
    """The two sides with the reference first: the more authoritative document, with one exception."""
    kinds = {first[1][0].document_type, second[1][0].document_type}
    if kinds == {"purchase_order", "quotation"}:
        # An order is placed on the strength of a quotation: the quotation is what the order is measured against.
        return sorted((first, second), key=lambda side: side[1][0].document_type != "quotation")
    return sorted((first, second), key=lambda side: _side_rank(side[1]))


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


def _relation(
    key: str, reference: list[EvidenceItem], checked: list[EvidenceItem], counts: dict[str, int],
    unrelated: set[frozenset[str]], linked: set[frozenset[str]],
) -> str | None:
    """May two documents' values for ``key`` be held against each other? (``reference`` ranks higher.)

    "reference"  the second is measured against the first (an invoice against its contract)
    None         not compared
    """
    pair = frozenset((reference[0].document_id, checked[0].document_id))
    if pair in unrelated:
        return None
    kind_a, kind_b = reference[0].document_type, checked[0].document_type
    if kind_a == kind_b:
        if kind_a == "policy":
            return "reference"  # two policies are meant to say the same thing
        if kind_a in ("invoice", "receipt"):
            return None  # two bills are two bills
        # Two contracts, two quotations: two pieces of business - unless something says that they
        # belong together. (A shared file name does not: every scanner calls its output "scan.pdf".)
        return "reference" if pair in linked else None
    if kind_a in _AGREEMENT_TYPES and kind_b in _AGREEMENT_TYPES:
        # A framework agreement and the cleaning contract next to it have nothing to say about each
        # other. An order is compared with a contract when it cites it - or when there is exactly one
        # of each, so that no guess is needed which belongs to which.
        return "reference" if pair in linked or (counts[kind_a] == 1 and counts[kind_b] == 1) else None
    if kind_b == "receipt" and counts.get("invoice"):
        # A receipt is measured against the invoice it pays, not against the contract behind the invoice
        # (which states its value before tax). With several invoices there is no telling which one it pays.
        return "reference" if kind_a == "invoice" and counts["invoice"] == 1 else None
    return "reference"


def find_contradictions(
    evidence: list[EvidenceItem], *, unrelated: set[Pair] | None = None, linked: set[Pair] | None = None,
    net_totals: dict[str, float] | None = None,
) -> list[Contradiction]:
    """Compare evidence items that share a comparison key.

    ``unrelated``: pairs of document ids that are known not to belong together
    and are never compared. ``linked``: pairs that are known to belong together
    (an invoice and the contract it cites, an order that cites its contract).
    ``net_totals``: an invoice's amount before tax, where it states one - an
    invoice for the agreed value plus VAT bills what was agreed. All three come
    from the document profiles (app/comparison); without them only what the
    document types say is used.
    """
    nets = net_totals or {}
    apart = {frozenset(pair) for pair in unrelated or ()}
    together = {frozenset(pair) for pair in linked or ()}
    # key -> document_id -> the items that document contributed
    per_document: dict[str, dict[str, list[EvidenceItem]]] = {}
    for item in evidence:
        if item.key not in COMPARABLE_KEYS or item.value_number is None:
            continue
        if item.key == KEY_TOTAL_AMOUNT and item.document_type in _NO_TOTAL_DOCUMENT_TYPES:
            continue
        per_document.setdefault(item.key, {}).setdefault(item.document_id, []).append(item)

    across: dict[tuple, dict] = {}
    within: list[dict] = []
    for key in COMPARABLE_KEYS:  # money first, then payment terms
        # ---- Step 1: one value per document.
        stated: list[tuple[float, list[EvidenceItem]]] = []
        for items in per_document.get(key, {}).values():
            distinct = sorted({round(float(item.value_number), 2) for item in items})  # type: ignore[arg-type]
            if key == KEY_TOTAL_AMOUNT:
                chosen = distinct[-1]  # a total is never smaller than its parts
            else:
                chosen = round(float(items[0].value_number), 2)  # type: ignore[arg-type]  # first one stated
            stated.append((chosen, [item for item in items if round(float(item.value_number), 2) == chosen]))  # type: ignore[arg-type]
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

        # ---- Step 2: compare the documents with each other, two at a time.
        counts: dict[str, int] = {}
        for _, items in stated:
            counts[items[0].document_type] = counts.get(items[0].document_type, 0) + 1
        # What each kind of agreement states: an invoice that agrees exactly with one contract is that
        # contract's invoice, and is not held against the other contracts.
        agreed: dict[str, set[float]] = {}
        for value, items in stated:
            agreed.setdefault(items[0].document_type, set()).add(value)
        for first, second in combinations(stated, 2):
            # The reference is side A.
            (value_a, items_a), (value_b, items_b) = _in_order(first, second)
            if not _relation(key, items_a, items_b, counts, apart, together):
                continue
            kind_a, kind_b = items_a[0].document_type, items_b[0].document_type
            known_pair = frozenset((items_a[0].document_id, items_b[0].document_id)) in together
            if (
                kind_a in _AGREEMENT_TYPES and kind_b in ("invoice", "receipt") and value_b != value_a
                and value_b in agreed[kind_a] and not known_pair
            ):
                # Only a guess, for when nothing better is known. An invoice that the profiles tie to THIS
                # contract is measured against it, whatever another contract in the folder happens to total
                # (third review: an over-billed invoice passed because a second contract had that total).
                continue
            unit_a, unit_b = _unit(key, items_a), _unit(key, items_b)
            if unit_a and unit_b and unit_a != unit_b:
                kind = "unit_mismatch"
            elif value_a == value_b:
                continue
            elif key == KEY_TOTAL_AMOUNT and kind_a != kind_b and kind_b not in _STATEMENT_TYPES and (
                min(value_b, nets.get(items_b[0].document_id, value_b)) <= value_a + 0.005
            ):
                # Not more than the reference: an instalment, a part delivery, a call-off, a part payment -
                # or exactly the agreed value with the tax on top.
                continue
            else:
                kind = "difference"
            # Documents that say the same thing are reported together: "Contract and order state X but the invoice states Y".
            slot = across.setdefault((key, kind, value_a, value_b, unit_a, unit_b), dict(items_a=[], items_b=[]))
            slot["items_a"] += [item for item in items_a if item not in slot["items_a"]]
            slot["items_b"] += [item for item in items_b if item not in slot["items_b"]]

    found: list[dict] = []
    for (key, kind, value_a, value_b, unit_a, unit_b), slot in across.items():
        source_a, source_b = _sources(slot["items_a"]), _sources(slot["items_b"])
        if kind == "unit_mismatch":
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
        found.append(dict(
            type=kind, key=key, severity=severity, description=description,
            value_a=value_a, value_b=value_b, unit_a=unit_a, unit_b=unit_b or unit_a,
            source_a=source_a, source_b=source_b,
            evidence_ids=[item.id for item in slot["items_a"] + slot["items_b"]],
        ))

    # Differences between documents come first; notes about a single document last.
    return [Contradiction(id=f"X{index}", **fields) for index, fields in enumerate(found + within, start=1)]
