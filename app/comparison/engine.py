"""The checks that compare document profiles. Plain Python, no LLM.

What is checked

    one document on its own
        line_arithmetic_mismatch    quantity x unit price is not the line amount
        line_items_total_mismatch   the lines do not add up to the stated total

    an invoice against the contract or purchase order it belongs to
        reference_mismatch          the invoice cites a different contract / order number
        party_mismatch              the supplier or customer is not the one in the reference
        date_outside_term           dated before the contract started, long after it ended,
                                    or before the purchase order existed
        quantity_mismatch           a matched line has a different quantity
        unit_price_mismatch         a matched line has a different unit price
        unlisted_line_item          the invoice bills something the purchase order does not list

    invoices against each other
        duplicate_invoice           the same invoice number twice, or the same supplier,
                                    date and total under two numbers

The guiding rule is the one from Version 1: a finding is a reason for a person
to look, never proof of wrongdoing, and a missing field means "not checked",
never "mismatch". A check only fires when both sides state the value.

``basic=True`` adds the total-amount and payment-terms comparison. Inside an
investigation those two come from the retrieved evidence, so the graph leaves
them out here; the stand-alone comparison endpoint switches them on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import combinations
from typing import Literal

from app.comparison.dates import add_months, format_date
from app.comparison.names import compare_names, same_identifier
from app.comparison.profile import REFERENCE_TYPES, DocumentProfile, LineItem, ProfileField
from app.formatting import format_number, format_value
from app.schemas.investigation import KEY_PAYMENT_TERMS, KEY_TOTAL_AMOUNT
from app.tools.contradiction import classify_difference

GRACE_AFTER_TERM_DAYS = 60  # invoices in arrears legitimately arrive after a term has ended
LINE_MATCH_THRESHOLD = 0.5
MAX_CITED_LINES = 8
MAX_UNLISTED_LINES = 3
MAX_FINDINGS = 12

_KIND = {"contract": "contract", "purchase_order": "purchase order", "invoice": "invoice"}
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_STOPWORDS = {"the", "and", "for", "per", "with", "from", "each", "item", "items", "unit", "units", "pcs", "nos"}


@dataclass(frozen=True)
class Cited:
    """One profile value that a finding rests on. Becomes an evidence item with a real citation."""

    document_id: str
    filename: str
    document_type: str
    key: str
    claim: str
    page: int
    quote: str
    value_number: float | None = None
    unit: str | None = None


@dataclass
class ProfileFinding:
    type: str
    key: str
    severity: Literal["low", "medium", "high"]
    description: str
    source_a: str  # the reference (contract, purchase order), or the document itself
    source_b: str  # the document being checked
    value_a: float | None = None
    value_b: float | None = None
    unit_a: str | None = None
    unit_b: str | None = None
    text_a: str | None = None
    text_b: str | None = None
    cited: list[Cited] = field(default_factory=list)
    # Numbers that must be combined to get value_a (the sum of the lines, quantity x price).
    # The calculate step runs them through the calculator tool, so the result is on record.
    operands: list[float] = field(default_factory=list)
    operation: Literal["sum", "multiply"] | None = None


def _close(first: float, second: float) -> bool:
    return abs(first - second) <= 0.005


def _kind(profile: DocumentProfile) -> str:
    return _KIND.get(profile.document_type, "document")


# ------------------------------------------------------------------ citations
def _cite(profile: DocumentProfile, key: str, value: ProfileField, claim: str) -> Cited:
    return Cited(
        document_id=profile.document_id, filename=profile.filename, document_type=profile.document_type,
        key=key, claim=claim, page=value.page, quote=value.quote, value_number=value.number, unit=value.unit,
    )


def cite_number(profile: DocumentProfile) -> Cited:
    assert profile.number is not None
    return _cite(profile, f"{profile.document_type}_number", profile.number,
                 f"{profile.filename} gives its {_kind(profile)} number as {profile.number.text}.")


def cite_party(profile: DocumentProfile, party: ProfileField) -> Cited:
    role = f"the {party.role}" if party.role else "a party"
    return _cite(profile, party.role or "party", party, f"{profile.filename} names {party.text} as {role}.")


def cite_date(profile: DocumentProfile) -> Cited:
    assert profile.date is not None
    return _cite(profile, "document_date", profile.date, f"{profile.filename} is dated {profile.date.text}.")


def cite_total(profile: DocumentProfile, which: str = "total") -> Cited:
    value: ProfileField = getattr(profile, which)
    return _cite(profile, f"document_{which}", value,
                 f"{profile.filename} states a {which} of {format_value(value.number, value.unit)}.")


def cite_terms(profile: DocumentProfile) -> Cited:
    value = profile.payment_terms_days
    assert value is not None and value.number is not None
    return _cite(profile, "document_payment_terms", value,
                 f"{profile.filename} states payment terms of {format_number(value.number)} days.")


def describe_line(item: LineItem) -> str:
    """'44 x sterilisation pouches cartons at $205 each' / 'Lift maintenance, annual: £40,000'"""
    text = item.description
    if item.quantity is not None:
        text = f"{format_number(item.quantity)} x {text}"
    if item.unit_price is not None:
        text += f" at {format_value(item.unit_price, item.unit)} each"
    if item.amount is not None:
        text += f": {format_value(item.amount, item.unit)}"
    return text


def cite_line(profile: DocumentProfile, item: LineItem) -> Cited:
    return Cited(
        document_id=profile.document_id, filename=profile.filename, document_type=profile.document_type,
        key="line_item", claim=f"{profile.filename} lists {describe_line(item)}.", page=item.page, quote=item.quote,
        # Only an amount the document states itself is a fact; quantity x price is a calculation.
        value_number=item.amount, unit=item.unit if item.amount is not None else None,
    )


# -------------------------------------------------------- one document alone
def check_document(profile: DocumentProfile) -> list[ProfileFinding]:
    findings: list[ProfileFinding] = []
    name = profile.filename

    for item in profile.line_items:
        if item.quantity is None or item.unit_price is None or item.amount is None:
            continue
        expected = round(item.quantity * item.unit_price, 2)
        if not _close(expected, item.amount):
            findings.append(ProfileFinding(
                type="line_arithmetic_mismatch", key="line_amount", severity="medium",
                description=(
                    f"{name} bills {format_value(item.amount, item.unit)} for {item.description}, but "
                    f"{format_number(item.quantity)} x {format_value(item.unit_price, item.unit)} is "
                    f"{format_value(expected, item.unit)}."
                ),
                source_a=name, source_b=name, value_a=expected, value_b=item.amount, unit_a=item.unit, unit_b=item.unit,
                cited=[cite_line(profile, item)], operands=[item.quantity, item.unit_price], operation="multiply",
            ))

    amounts = [item.effective_amount for item in profile.line_items]
    total = profile.total
    # One line is only worth adding up when its amount had to be worked out (quantity x price).
    enough = len(amounts) >= 2 or (len(amounts) == 1 and profile.line_items[0].amount is None)
    if total is not None and total.number is not None and enough and all(amount is not None for amount in amounts):
        units = {item.unit for item in profile.line_items if item.unit}
        if len(units) <= 1 and (not units or not total.unit or units == {total.unit}):
            line_sum = round(sum(amount for amount in amounts if amount is not None), 2)
            subtotal, tax = profile.subtotal, profile.tax
            accepted = [total.number]
            if subtotal is not None and subtotal.number is not None:
                accepted.append(subtotal.number)
            if tax is not None and tax.number is not None:
                accepted.append(total.number - tax.number)
            if not any(_close(line_sum, value) for value in accepted):
                against, label = (subtotal, "subtotal") if subtotal is not None and subtotal.number is not None else (total, "total")
                assert against.number is not None
                unexplained_extra = line_sum < against.number and tax is None and label == "total"
                description = (
                    f"The line items on {name} add up to {format_value(line_sum, total.unit)}, but its stated "
                    f"{label} is {format_value(against.number, against.unit or total.unit)}."
                )
                if unexplained_extra:
                    description += " Tax or other charges that are not listed as lines could explain the difference."
                findings.append(ProfileFinding(
                    type="line_items_total_mismatch", key="line_items_total",
                    severity="low" if unexplained_extra else "medium",
                    description=description, source_a=name, source_b=name,
                    value_a=line_sum, value_b=against.number, unit_a=total.unit, unit_b=against.unit or total.unit,
                    cited=[cite_line(profile, item) for item in profile.line_items[:MAX_CITED_LINES]]
                    + [cite_total(profile, label)],
                    operands=[amount for amount in amounts if amount is not None], operation="sum",
                ))
    return findings


# ------------------------------------------------- invoice against reference
def _line_words(description: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", description.lower())
    return {word.rstrip("s") if len(word) > 3 else word for word in words if len(word) >= 3 and word not in _STOPWORDS}


def match_lines(reference: list[LineItem], invoice: list[LineItem]) -> tuple[list[tuple[LineItem, LineItem]], list[LineItem]]:
    """Pair up lines that describe the same thing. Returns (pairs, invoice lines without a partner).

    Similarity is the share of words two descriptions have in common. The best
    pairs are taken first, and every line is used at most once.
    """
    scored = []
    for i, first in enumerate(reference):
        for j, second in enumerate(invoice):
            a, b = _line_words(first.description), _line_words(second.description)
            if a and b:
                similarity = len(a & b) / len(a | b)
                if similarity >= LINE_MATCH_THRESHOLD:
                    scored.append((similarity, i, j))
    used_reference: set[int] = set()
    used_invoice: set[int] = set()
    pairs = []
    for _, i, j in sorted(scored, key=lambda row: (-row[0], row[1], row[2])):
        if i not in used_reference and j not in used_invoice:
            used_reference.add(i)
            used_invoice.add(j)
            pairs.append((reference[i], invoice[j]))
    return pairs, [item for j, item in enumerate(invoice) if j not in used_invoice]


def _term_end(reference: DocumentProfile):
    """(last day of the term, the fields that state it), or (None, [])."""
    if reference.term_end is not None and reference.term_end.dates:
        return max(reference.term_end.dates), [("term_end", reference.term_end, f"{reference.filename} states that the term ends on {reference.term_end.text}.")]
    start, months = reference.term_start, reference.term_months
    if start is not None and start.dates and months is not None and months.number:
        last_day = add_months(max(start.dates), int(months.number)) - timedelta(days=1)
        return last_day, [
            ("term_start", start, f"{reference.filename} states that the term starts on {start.text}."),
            ("term_length", months, f"{reference.filename} states the length of the term: {months.quote}"),
        ]
    return None, []


def check_pair(reference: DocumentProfile, invoice: DocumentProfile, *, basic: bool = False) -> list[ProfileFinding]:
    """Check one invoice against one contract or purchase order."""
    findings: list[ProfileFinding] = []
    ref_name, inv_name, kind = reference.filename, invoice.filename, _kind(reference)

    # ---- the two classic quantities (stand-alone comparison only, see the module text)
    if basic:
        a, b = reference.total, invoice.total
        if a is not None and b is not None and a.number is not None and b.number is not None:
            if a.unit and b.unit and a.unit != b.unit:
                findings.append(ProfileFinding(
                    type="unit_mismatch", key=KEY_TOTAL_AMOUNT, severity="medium",
                    description=(
                        f"{ref_name} states the total amount as {format_value(a.number, a.unit)} but {inv_name} states "
                        f"{format_value(b.number, b.unit)}. The currencies differ, so the values cannot be compared directly."
                    ),
                    source_a=ref_name, source_b=inv_name, value_a=a.number, value_b=b.number, unit_a=a.unit, unit_b=b.unit,
                    cited=[cite_total(reference), cite_total(invoice)],
                ))
            elif not _close(a.number, b.number):
                kind_of, severity = classify_difference(KEY_TOTAL_AMOUNT, a.number, b.number)
                unit = a.unit or b.unit
                findings.append(ProfileFinding(
                    type=kind_of, key=KEY_TOTAL_AMOUNT, severity=severity,  # type: ignore[arg-type]
                    description=(
                        f"{ref_name} states the total amount as {format_value(a.number, unit)} but {inv_name} states "
                        f"{format_value(b.number, unit)}."
                    ),
                    source_a=ref_name, source_b=inv_name, value_a=a.number, value_b=b.number, unit_a=unit, unit_b=unit,
                    cited=[cite_total(reference), cite_total(invoice)],
                ))
        a, b = reference.payment_terms_days, invoice.payment_terms_days
        if a is not None and b is not None and a.number is not None and b.number is not None and not _close(a.number, b.number):
            kind_of, severity = classify_difference(KEY_PAYMENT_TERMS, a.number, b.number)
            findings.append(ProfileFinding(
                type=kind_of, key=KEY_PAYMENT_TERMS, severity=severity,  # type: ignore[arg-type]
                description=(
                    f"{ref_name} states the payment terms as {format_value(a.number, 'days')} but {inv_name} states "
                    f"{format_value(b.number, 'days')}."
                ),
                source_a=ref_name, source_b=inv_name, value_a=a.number, value_b=b.number, unit_a="days", unit_b="days",
                cited=[cite_terms(reference), cite_terms(invoice)],
            ))

    # ---- parties
    for role in ("supplier", "customer"):
        mine = invoice.party(role)  # type: ignore[arg-type]
        if mine is None:
            continue
        named = reference.party(role)  # type: ignore[arg-type]
        candidates = [named] if named is not None else [party for party in reference.parties if party.role is None]
        results = [compare_names(mine.text, candidate.text) for candidate in candidates]
        if not candidates or "same" in results or all(result == "unknown" for result in results):
            continue
        look_alike = "similar" in results
        theirs = " and ".join(candidate.text or "" for candidate in candidates)
        if role == "supplier":
            description = f"{inv_name} is issued by {mine.text}, but the {kind} {ref_name} "
        else:
            description = f"{inv_name} is addressed to {mine.text}, but the {kind} {ref_name} "
        description += f"names {theirs} as the {role}." if named is not None else f"is between {theirs}."
        if look_alike:
            description += " The names are similar but not identical."
        findings.append(ProfileFinding(
            type="party_mismatch", key=role,
            # A different company asking for the money is the more serious case; a near-match may be a typing error.
            severity="high" if role == "supplier" and not look_alike else "medium",
            description=description, source_a=ref_name, source_b=inv_name, text_a=theirs, text_b=mine.text,
            cited=[cite_party(reference, candidate) for candidate in candidates] + [cite_party(invoice, mine)],
        ))

    # ---- dates
    if invoice.date is not None and invoice.date.dates:
        invoice_dates = invoice.date.dates
        if reference.document_type == "contract":
            start = reference.term_start or reference.date
            if start is not None and start.dates and max(invoice_dates) < min(start.dates):
                if start is reference.term_start:
                    what, claim = "the term of", f"{ref_name} states that the term starts on {start.text}."
                    cited_start = _cite(reference, "term_start", start, claim)
                else:
                    what, cited_start = "", cite_date(reference)
                verb = "starts" if start is reference.term_start else "was made"
                findings.append(ProfileFinding(
                    type="date_outside_term", key="document_date", severity="medium",
                    description=f"{inv_name} is dated {invoice.date.text}, before {what} {ref_name} {verb} ({start.text}).".replace("  ", " "),
                    source_a=ref_name, source_b=inv_name, text_a=start.text, text_b=invoice.date.text,
                    cited=[cited_start, cite_date(invoice)],
                ))
            last_day, stated = _term_end(reference)
            if last_day is not None and min(invoice_dates) > last_day + timedelta(days=GRACE_AFTER_TERM_DAYS):
                findings.append(ProfileFinding(
                    type="date_outside_term", key="document_date", severity="low",
                    description=(
                        f"{inv_name} is dated {invoice.date.text}, more than {GRACE_AFTER_TERM_DAYS} days after the term of "
                        f"{ref_name} ended ({format_date(last_day)})."
                    ),
                    source_a=ref_name, source_b=inv_name, text_a=format_date(last_day), text_b=invoice.date.text,
                    cited=[_cite(reference, key, value, claim) for key, value, claim in stated] + [cite_date(invoice)],
                ))
        elif reference.date is not None and reference.date.dates and max(invoice_dates) < min(reference.date.dates):
            findings.append(ProfileFinding(
                type="date_outside_term", key="document_date", severity="medium",
                description=f"{inv_name} is dated {invoice.date.text}, before the {kind} {ref_name} was issued ({reference.date.text}).",
                source_a=ref_name, source_b=inv_name, text_a=reference.date.text, text_b=invoice.date.text,
                cited=[cite_date(reference), cite_date(invoice)],
            ))

    # ---- line items
    if reference.line_items and invoice.line_items:
        pairs, unmatched = match_lines(reference.line_items, invoice.line_items)
        for ordered, billed in pairs:
            cited = [cite_line(reference, ordered), cite_line(invoice, billed)]
            if ordered.quantity is not None and billed.quantity is not None and not _close(ordered.quantity, billed.quantity):
                findings.append(ProfileFinding(
                    type="quantity_mismatch", key="quantity", severity="medium",
                    description=(
                        f"{ref_name} lists {format_number(ordered.quantity)} x {ordered.description}, but {inv_name} "
                        f"bills {format_number(billed.quantity)}."
                    ),
                    source_a=ref_name, source_b=inv_name, value_a=ordered.quantity, value_b=billed.quantity, cited=cited,
                ))
            same_currency = not ordered.unit or not billed.unit or ordered.unit == billed.unit
            if (
                ordered.unit_price is not None and billed.unit_price is not None and same_currency
                and not _close(ordered.unit_price, billed.unit_price)
            ):
                unit = ordered.unit or billed.unit
                findings.append(ProfileFinding(
                    type="unit_price_mismatch", key="unit_price", severity="medium",
                    description=(
                        f"{ref_name} gives a unit price of {format_value(ordered.unit_price, unit)} for "
                        f"{ordered.description}, but {inv_name} charges {format_value(billed.unit_price, unit)}."
                    ),
                    source_a=ref_name, source_b=inv_name, value_a=ordered.unit_price, value_b=billed.unit_price,
                    unit_a=unit, unit_b=unit, cited=cited,
                ))
        # Only when at least one line matched: that shows the two documents describe items in comparable words.
        if pairs and reference.document_type == "purchase_order":
            for item in unmatched[:MAX_UNLISTED_LINES]:
                findings.append(ProfileFinding(
                    type="unlisted_line_item", key="line_item", severity="medium",
                    description=f"{inv_name} bills an item that the {kind} {ref_name} does not list: {describe_line(item)}.",
                    source_a=ref_name, source_b=inv_name, text_b=item.description,
                    value_b=item.effective_amount, unit_b=item.unit, cited=[cite_line(invoice, item)],
                ))
    return findings


# ------------------------------------------------------ invoices among each other
def _identifier_key(value: ProfileField | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value.text or "").upper()) if value is not None else ""


def _total_of(profile: DocumentProfile) -> tuple[float, str | None] | None:
    if profile.total is None or profile.total.number is None:
        return None
    return profile.total.number, profile.total.unit


def check_duplicates(invoices: list[DocumentProfile]) -> list[ProfileFinding]:
    findings = []
    for first, second in combinations(invoices, 2):
        if first.document_id == second.document_id:
            continue
        suppliers = compare_names(
            first.supplier.text if first.supplier else None, second.supplier.text if second.supplier else None
        )
        if suppliers in ("different", "similar"):
            continue  # two suppliers may well use the same numbering
        number_a, number_b = _identifier_key(first.number), _identifier_key(second.number)
        if not number_a or not number_b:
            continue
        total_a, total_b = _total_of(first), _total_of(second)
        same_total = total_a is not None and total_b is not None and _close(total_a[0], total_b[0]) and total_a[1] == total_b[1]
        shown_a, shown_b = first.number.text, second.number.text  # type: ignore[union-attr]

        if number_a == number_b:
            description = f"{first.filename} and {second.filename} carry the same invoice number ({shown_a})"
            cited = [cite_number(first), cite_number(second)]
            if total_a is not None and total_b is not None:
                cited += [cite_total(first), cite_total(second)]
                if same_total:
                    description += f" and the same total ({format_value(*total_a)}). The invoice may have been submitted twice."
                else:
                    description += (
                        f" but different totals ({format_value(*total_a)} and {format_value(*total_b)}). "
                        "Check whether one of them is a correction or a credit note."
                    )
            else:
                description += "."
            findings.append(ProfileFinding(
                type="duplicate_invoice", key="invoice_number", severity="high" if same_total else "medium",
                description=description, source_a=first.filename, source_b=second.filename,
                text_a=shown_a, text_b=shown_b, cited=cited,
            ))
            continue

        # The same bill under a new number: same supplier, same date, same total.
        same_date = (
            first.date is not None and second.date is not None and bool(set(first.date.dates) & set(second.date.dates))
        )
        if suppliers == "same" and same_total and same_date and total_a is not None:
            findings.append(ProfileFinding(
                type="duplicate_invoice", key="invoice_number", severity="medium",
                description=(
                    f"{first.filename} ({shown_a}) and {second.filename} ({shown_b}) come from the same supplier, carry "
                    f"the same date and the same total ({format_value(*total_a)}), but have different invoice numbers. "
                    "Check that the same work is not billed twice."
                ),
                source_a=first.filename, source_b=second.filename, text_a=shown_a, text_b=shown_b,
                cited=[cite_number(first), cite_number(second), cite_total(first), cite_total(second),
                       cite_date(first), cite_date(second)],
            ))
    return findings


# --------------------------------------------------------------- the whole set
def _richness(profile: DocumentProfile) -> int:
    return sum(
        value is not None
        for value in (profile.number, profile.total, profile.date, profile.term_start, profile.payment_terms_days)
    ) + len(profile.parties) + min(len(profile.line_items), 3)


def find_reference(invoice: DocumentProfile, kind: str, profiles: list[DocumentProfile]) -> tuple[DocumentProfile | None, ProfileFinding | None]:
    """The contract (or purchase order) an invoice belongs to, and a finding if its reference number points elsewhere."""
    candidates = sorted(
        (profile for profile in profiles if profile.document_type == kind and not profile.is_empty),
        key=_richness, reverse=True,
    )
    if not candidates:
        return None, None
    cited_reference = invoice.contract_reference if kind == "contract" else invoice.purchase_order_reference

    if cited_reference is not None and cited_reference.text:
        for candidate in candidates:
            if candidate.number is not None and same_identifier(candidate.number.text, cited_reference.text):
                return candidate, None
        numbered = [candidate for candidate in candidates if candidate.number is not None]
        if numbered:
            label = _KIND[kind]
            key = "contract_reference" if kind == "contract" else "purchase_order_reference"
            theirs = ", ".join(f"{candidate.number.text} ({candidate.filename})" for candidate in numbered[:3])  # type: ignore[union-attr]
            provided = f"the {label} provided is" if len(numbered) == 1 else f"the {label}s provided are"
            finding = ProfileFinding(
                type="reference_mismatch", key=key, severity="medium",
                description=f"{invoice.filename} refers to {label} {cited_reference.text}, but {provided} {theirs}.",
                source_a=", ".join(candidate.filename for candidate in numbered[:3]), source_b=invoice.filename,
                text_a=", ".join(candidate.number.text or "" for candidate in numbered[:3]),  # type: ignore[union-attr]
                text_b=cited_reference.text,
                cited=[cite_number(candidate) for candidate in numbered[:3]]
                + [_cite(invoice, key, cited_reference, f"{invoice.filename} refers to {label} {cited_reference.text}.")],
            )
            # With a single candidate the other checks still say something useful about the pair.
            return (candidates[0] if len(candidates) == 1 else None), finding

    if len(candidates) == 1:
        return candidates[0], None
    # Several candidates and no usable reference number: take the one with the same supplier, if exactly one has it.
    if invoice.supplier is not None:
        same = [
            candidate for candidate in candidates
            if any(compare_names(invoice.supplier.text, party.text) == "same" for party in candidate.parties)
        ]
        if len(same) == 1:
            return same[0], None
    return None, None


def compare_profiles(profiles: list[DocumentProfile], *, basic: bool = False) -> list[ProfileFinding]:
    """Run every check on a set of documents. The most serious findings come first."""
    usable = [profile for profile in profiles if not profile.is_empty]
    invoices = [profile for profile in usable if profile.document_type == "invoice"]
    findings: list[ProfileFinding] = []

    for invoice in invoices:
        for kind in REFERENCE_TYPES:
            reference, mismatch = find_reference(invoice, kind, usable)
            if mismatch is not None:
                findings.append(mismatch)
            if reference is not None:
                findings += check_pair(reference, invoice, basic=basic)
    findings += check_duplicates(invoices)
    for profile in usable:
        findings += check_document(profile)

    findings.sort(key=lambda finding: _SEVERITY_ORDER[finding.severity])  # stable: equal severities keep their order
    return findings[:MAX_FINDINGS]
