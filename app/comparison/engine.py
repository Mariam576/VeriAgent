"""The checks that compare document profiles. Plain Python, no LLM.

What is checked

    one document on its own
        line_arithmetic_mismatch    quantity x unit price is not the line amount
        line_items_total_mismatch   the lines do not add up to the stated total, and no tax,
                                    discount or charge on the page explains the difference

    an invoice against the contract or purchase order it belongs to
        reference_mismatch          the invoice cites a different contract / order number
        party_mismatch              the supplier or customer is not the one in the reference
        date_outside_term           dated before the contract started, long after it ended,
                                    or before the purchase order existed
        quantity_mismatch           more is billed than was ordered - on one invoice, or on
                                    several invoices for the same order taken together
        unit_price_mismatch         a matched line has a different unit price
        line_amount_mismatch        a matched line is billed at more than was ordered
        unit_mismatch               a matched line is priced in another currency
        unlisted_line_item          the invoice bills something the purchase order does not list

    invoices against each other
        duplicate_invoice           the same invoice number twice, or the same supplier,
                                    date and total under two numbers

The guiding rule is the one from Version 1: a finding is a reason for a person
to look, never proof of wrongdoing, and a missing field means "not checked",
never "mismatch". A check only fires when both sides state the value.

A second rule came out of the review of Version 3: do not accuse what you do
not model. Ordinary business looks like a mismatch to a naive check - a
partial delivery bills less than was ordered, twelve monthly invoices each
bill a twelfth of the contract, a discount stands between the lines and the
total, a second invoice from another supplier lies in the same folder. So:

    * an invoice is only checked against a contract or order it belongs to
      (it cites its number, or comes from its supplier),
    * billing less than the reference is not a finding; billing more is, also
      when it takes several invoices together to get there,
    * arithmetic is only a finding when no tax, discount, charge or deposit
      on the page explains the difference, and rounding to the cent is not one.

A third rule came out of the second review: leniency needs bounds, and silence
needs a reason. Each of the rules above can be turned into a way of hiding
something - a charge that is counted as a line and again as "an adjustment
that explains the total", an invoice from a look-alike supplier that simply
leaves out the contract number, the words "credit note" somewhere near the
top. So:

    * an amount explains the total once, and only if it is not one of the lines,
    * a name that keeps the contract supplier's words with small changes is
      compared, not skipped (names.may_belong_together),
    * when the person names both documents, or selects a single invoice with
      its contract, they are compared whatever the names say,
    * and whenever a document is left unchecked, the result says so
      (``Comparison.notes``): "not checked" must never look like "no problem".

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

from app.comparison.adjustments import ADDED
from app.comparison.dates import add_months, format_date
from app.comparison.names import RELATED, compare_names, may_belong_together, same_identifier
from app.comparison.profile import PER_MANY, REFERENCE_TYPES, DocumentProfile, LineItem, ProfileField
from app.formatting import format_number, format_value
from app.schemas.investigation import KEY_PAYMENT_TERMS, KEY_TOTAL_AMOUNT
from app.tools.contradiction import classify_difference

GRACE_AFTER_TERM_DAYS = 60  # invoices in arrears legitimately arrive after a term has ended
LINE_MATCH_THRESHOLD = 0.5
MAX_CITED_LINES = 8
MAX_UNLISTED_LINES = 3
MAX_FINDINGS = 12

MAX_NOTES = 4

CENT = 0.0101  # one cent (or penny, or paisa), with room for the way computers store decimals
MAX_ADJUSTMENTS_TRIED = 7  # each can be added, subtracted or left out: 3^7 combinations at most
# Delivery and postage ride along with an order without being on it - up to a point. A "handling
# charge" of four times the goods is not an incidental.
MAX_INCIDENTAL_SHARE = 0.25
# The highest sales taxes are a little under 30%. A larger gap between lines and total is not "probably the tax".
MAX_UNLISTED_TAX_SHARE = 0.3
# The same invoice number this many days apart is more likely a numbering that starts again
# (every year, every month) than the same bill sent twice.
NUMBERING_RESTART_DAYS = 60

_KIND = {"contract": "contract", "purchase_order": "purchase order", "invoice": "invoice"}
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_STOPWORDS = {
    "the", "and", "for", "per", "with", "from", "each", "item", "items", "unit", "units", "pcs", "nos", "of", "to", "in",
    "on", "at", "a", "an", "x", "no", "qty",
}
# A line that carries its own discount or tax ("Disc 10%", "less 5%", "incl. VAT 20%"): quantity x price
# is then not meant to equal the line amount.
_LINE_ADJUSTED = re.compile(r"%|\b(?:disc(?:ount)?|rebate|less|net\s+of|incl(?:uding|\.)?|excl(?:uding|\.)?)\b", re.I)
# Charges that ride along with an order without being ordered.
_INCIDENTAL = re.compile(r"\b(?:delivery|shipping|freight|carriage|postage|courier|handling|transport|packing)\b", re.I)


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


def _close(first: float, second: float, tolerance: float = CENT) -> bool:
    """Equal to the cent. Amounts are rounded to the cent, and 2.5 x 19.99 is 49.975: 49.97 and 49.98 are both right."""
    return abs(first - second) <= tolerance


def _same_count(first: float, second: float) -> bool:
    return abs(first - second) <= 0.0005


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
def _explained_by_adjustments(line_sum: float, target: float, adjustments: list[tuple[float, str]], tolerance: float) -> bool:
    """Can ``target`` be reached from ``line_sum`` with some of the adjustments on the page?

    1,000 in lines, a discount of 100 and VAT of 180 explain a total of 1,080
    (1,000 - 100 + 180). A tax or charge may be added, a discount or deposit
    taken off, a rounding line either; each may also be left out. With at most
    seven adjustments that is a couple of thousand sums.
    """
    reachable = {0.0}
    for value, direction in adjustments[:MAX_ADJUSTMENTS_TRIED]:
        moves = {"add": (value,), "subtract": (-value,)}.get(direction, (value, -value))
        reachable |= {round(known + move, 2) for known in reachable for move in moves}
    need = target - line_sum
    return any(abs(change - need) <= tolerance for change in reachable)


def check_document(profile: DocumentProfile) -> list[ProfileFinding]:
    findings: list[ProfileFinding] = []
    name = profile.filename

    for item in profile.line_items:
        if item.quantity is None or item.unit_price is None or item.amount is None:
            continue
        if _LINE_ADJUSTED.search(item.quote) or PER_MANY.search(item.quote):
            continue  # a discount or tax of its own, or a price "per 1,000": quantity x price is not the amount
        expected = round(item.quantity * item.unit_price, 2)
        if profile.minimum_charge and item.amount > expected:
            continue  # "minimum call-out charge applies": billed above quantity x price on purpose
        # Compared before rounding as well: 2.5 x 19.99 = 49.975 is 49.97 to one program and 49.98 to another.
        if not _close(item.quantity * item.unit_price, item.amount):
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
            targets = [total.number]
            if subtotal is not None and subtotal.number is not None:
                targets.append(subtotal.number)
            # Everything on the page that may stand between the lines and the total. The tax comes first:
            # only the first few are tried, and the tax is the one that is nearly always needed.
            listed = [(value.number, value.text or "either") for value in profile.adjustments if value.number]
            adjustments = [entry for entry in listed if entry[1] == "add"] + [entry for entry in listed if entry[1] != "add"]
            tax_is_a_line = tax is not None and tax.number and any(
                amount is not None and _close(amount, tax.number) and ADDED.search(item.description)
                for item, amount in zip(profile.line_items, amounts)
            )
            if tax is not None and tax.number and not tax_is_a_line:
                # The tax field is the same statement as one of the amounts found in the text: counted once.
                same = next((entry for entry in adjustments if entry[1] == "add" and _close(entry[0], tax.number)), None)
                if same is not None:
                    adjustments.remove(same)
                adjustments.insert(0, same or (tax.number, "add"))
            tolerance = min(CENT * max(len(amounts), 1), 0.1)  # every line may be rounded by up to a cent
            explained = any(
                _close(line_sum, target, tolerance) or _explained_by_adjustments(line_sum, target, adjustments, tolerance)
                for target in targets
            )
            if not explained:
                against, label = (subtotal, "subtotal") if subtotal is not None and subtotal.number is not None else (total, "total")
                assert against.number is not None
                description = (
                    f"The line items on {name} add up to {format_value(line_sum, total.unit)}, but its stated "
                    f"{label} is {format_value(against.number, against.unit or total.unit)}."
                )
                # Something that could explain it is mentioned, or may simply not be listed: say so, and say it softly.
                # ... but only a difference of the size a tax can have. No tax turns 6,200 into 11,200.
                unlisted_extra = (
                    not adjustments and label == "total" and 0 < against.number - line_sum <= MAX_UNLISTED_TAX_SHARE * abs(line_sum)
                )
                if profile.unquantified_adjustment:
                    description += " The document mentions a discount, tax or charge without an amount, which could explain the difference."
                elif unlisted_extra:
                    description += " Tax or other charges that are not listed as lines could explain the difference."
                findings.append(ProfileFinding(
                    type="line_items_total_mismatch", key="line_items_total",
                    severity="low" if unlisted_extra or profile.unquantified_adjustment else "medium",
                    description=description, source_a=name, source_b=name,
                    value_a=line_sum, value_b=against.number, unit_a=total.unit, unit_b=against.unit or total.unit,
                    cited=[cite_line(profile, item) for item in profile.line_items[:MAX_CITED_LINES]]
                    + [cite_total(profile, label)],
                    operands=[amount for amount in amounts if amount is not None], operation="sum",
                ))
    return findings


# ------------------------------------------------- invoice against reference
def _singular(word: str) -> str:
    """'glasses' -> 'glass', 'cartons' -> 'carton', 'batteries' -> 'battery'. 'glass' stays 'glass'."""
    if len(word) <= 3 or not word.isalpha():
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _line_words(description: str) -> set[str]:
    """The words that say what a line is. Short codes count: "size M" and "size L" are different goods."""
    words = re.findall(r"[a-z0-9]+", description.lower())
    return {_singular(word) for word in words if word not in _STOPWORDS}


def _agreement(first: LineItem, second: LineItem) -> int:
    """How many of quantity, unit price and amount two lines have in common."""
    return sum(
        a is not None and b is not None and _close(a, b)
        for a, b in ((first.quantity, second.quantity), (first.unit_price, second.unit_price), (first.amount, second.amount))
    )


def match_lines(reference: list[LineItem], invoice: list[LineItem]) -> tuple[list[tuple[LineItem, LineItem]], list[LineItem]]:
    """Pair up lines that describe the same thing. Returns (pairs, invoice lines without a partner).

    Similarity is the share of words two descriptions have in common. The best
    pairs are taken first, and every line is used at most once. Between two
    equally similar candidates the one whose numbers agree is taken: when an
    order lists "gloves size M" and "gloves size L" and the invoice words them
    a little differently, the line with the same quantity and price is the
    partner, not the one that happens to come first.
    """
    scored = []
    for i, first in enumerate(reference):
        for j, second in enumerate(invoice):
            a, b = _line_words(first.description), _line_words(second.description)
            if a and b:
                similarity = len(a & b) / len(a | b)
                if similarity >= LINE_MATCH_THRESHOLD:
                    scored.append((similarity, _agreement(first, second), i, j))
    used_reference: set[int] = set()
    used_invoice: set[int] = set()
    pairs = []
    for _, _, i, j in sorted(scored, key=lambda row: (-row[0], -row[1], row[2], row[3])):
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


def check_pair(
    reference: DocumentProfile, invoice: DocumentProfile, *, basic: bool = False, parties_only: bool = False
) -> list[ProfileFinding]:
    """Check one invoice against one contract or purchase order.

    ``parties_only``: for a credit note. It gives money back, so its amounts,
    dates and lines are not an invoice's - but who it comes from still matters.
    """
    findings: list[ProfileFinding] = []
    ref_name, inv_name, kind = reference.filename, invoice.filename, _kind(reference)

    # ---- the two classic quantities (stand-alone comparison only, see the module text)
    if basic and not parties_only:
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
            elif b.number > a.number + CENT:
                # Only "more than the reference" is a finding. An invoice for less is an instalment, a
                # partial delivery or a monthly fee; several of them are added up in check_series.
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
        # Every party of that role (a joint venture has two suppliers), and every party whose role the
        # reference does not state: an invoice from any of them comes from a party to the agreement.
        named = reference.parties_with(role)  # type: ignore[arg-type]
        unstated = [party for party in reference.parties if party.role is None]
        candidates = named + unstated
        results = [compare_names(mine.text, candidate.text) for candidate in candidates]
        if not candidates or "same" in results or all(result == "unknown" for result in results):
            continue
        shown = named or unstated
        short_form, look_alike = "short_form" in results, "similar" in results
        theirs = " and ".join(candidate.text or "" for candidate in shown)
        if role == "supplier":
            description = f"{inv_name} is issued by {mine.text}, but the {kind} {ref_name} "
        else:
            description = f"{inv_name} is addressed to {mine.text}, but the {kind} {ref_name} "
        description += f"names {theirs} as the {role}." if named else f"is between {theirs}."
        if short_form:
            description += " One name is a shorter form of the other; check that they are the same company."
            severity = "low"  # usually a letterhead that leaves out a word; sometimes a look-alike that adds one
        elif look_alike:
            description += " The names are similar but not identical."
            severity = "medium"
        else:
            # A different company asking for the money is the most serious case.
            severity = "high" if role == "supplier" else "medium"
        findings.append(ProfileFinding(
            type="party_mismatch", key=role, severity=severity,  # type: ignore[arg-type]
            description=description, source_a=ref_name, source_b=inv_name, text_a=theirs, text_b=mine.text,
            cited=[cite_party(reference, candidate) for candidate in shown] + [cite_party(invoice, mine)],
        ))
    if parties_only:
        return findings

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
            # A contract that renews by itself has no last day to be late for.
            if last_day is not None and not reference.renews and min(invoice_dates) > last_day + timedelta(days=GRACE_AFTER_TERM_DAYS):
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
            same_currency = not (ordered.unit and billed.unit and ordered.unit != billed.unit)
            if (
                same_currency and ordered.effective_amount is not None and billed.effective_amount is not None
                and _close(ordered.effective_amount, billed.effective_amount)
            ):
                # The money agrees. If quantity and price both differ, the two documents most likely count in
                # different units (10 boxes of 100 at £50, or 1,000 pieces at £0.50): a note, not an accusation.
                if (
                    ordered.quantity is not None and billed.quantity is not None and not _same_count(ordered.quantity, billed.quantity)
                    and ordered.unit_price is not None and billed.unit_price is not None
                ):
                    unit = ordered.unit or billed.unit
                    findings.append(ProfileFinding(
                        type="quantity_mismatch", key="quantity", severity="low",
                        description=(
                            f"{ref_name} lists {format_number(ordered.quantity)} x {ordered.description} at "
                            f"{format_value(ordered.unit_price, unit)}, and {inv_name} bills {format_number(billed.quantity)} at "
                            f"{format_value(billed.unit_price, unit)}. The line amounts agree "
                            f"({format_value(billed.effective_amount, unit)}), so the two probably count in different units."
                        ),
                        source_a=ref_name, source_b=inv_name, value_a=ordered.quantity, value_b=billed.quantity, cited=cited,
                    ))
                continue
            # Billing fewer than ordered is a partial delivery, not a finding.
            if ordered.quantity is not None and billed.quantity is not None and billed.quantity > ordered.quantity + 0.0005:
                findings.append(ProfileFinding(
                    type="quantity_mismatch", key="quantity", severity="medium",
                    description=(
                        f"{ref_name} lists {format_number(ordered.quantity)} x {ordered.description}, but {inv_name} "
                        f"bills {format_number(billed.quantity)}."
                    ),
                    source_a=ref_name, source_b=inv_name, value_a=ordered.quantity, value_b=billed.quantity, cited=cited,
                ))
            if ordered.unit and billed.unit and ordered.unit != billed.unit:
                priced = ordered.unit_price is not None and billed.unit_price is not None
                first = ordered.unit_price if priced else ordered.effective_amount
                second = billed.unit_price if priced else billed.effective_amount
                if first is not None and second is not None:
                    findings.append(ProfileFinding(
                        type="unit_mismatch", key="unit_price" if priced else "line_amount", severity="medium",
                        description=(
                            f"{ref_name} prices {ordered.description} at {format_value(first, ordered.unit)}, but {inv_name} "
                            f"charges {format_value(second, billed.unit)}. The currencies differ, so the values cannot be compared directly."
                        ),
                        source_a=ref_name, source_b=inv_name, value_a=first, value_b=second,
                        unit_a=ordered.unit, unit_b=billed.unit, cited=cited,
                    ))
                continue
            unit = ordered.unit or billed.unit
            if ordered.unit_price is not None and billed.unit_price is not None:
                if not _close(ordered.unit_price, billed.unit_price, 0.00005):
                    findings.append(ProfileFinding(
                        type="unit_price_mismatch", key="unit_price", severity="medium",
                        description=(
                            f"{ref_name} gives a unit price of {format_value(ordered.unit_price, unit)} for "
                            f"{ordered.description}, but {inv_name} charges {format_value(billed.unit_price, unit)}."
                        ),
                        source_a=ref_name, source_b=inv_name, value_a=ordered.unit_price, value_b=billed.unit_price,
                        unit_a=unit, unit_b=unit, cited=cited,
                    ))
            elif (
                ordered.effective_amount is not None and billed.effective_amount is not None
                and billed.effective_amount > ordered.effective_amount + CENT
            ):
                # Lines without quantities ("Machine service: £1,350"): the amounts themselves are compared.
                findings.append(ProfileFinding(
                    type="line_amount_mismatch", key="line_amount", severity="medium",
                    description=(
                        f"{ref_name} lists {ordered.description} at {format_value(ordered.effective_amount, unit)}, but "
                        f"{inv_name} bills {format_value(billed.effective_amount, unit)} for it."
                    ),
                    source_a=ref_name, source_b=inv_name, value_a=ordered.effective_amount, value_b=billed.effective_amount,
                    unit_a=unit, unit_b=unit, cited=cited,
                ))
        # Only when at least one line matched: that shows the two documents describe items in comparable words.
        if pairs and reference.document_type == "purchase_order":
            # Delivery, postage and the like ride along with an order without being on it - as long as
            # they stay small next to what was ordered.
            matched_value = sum(abs(billed.effective_amount or 0) for _, billed in pairs)
            incidental = [item for item in unmatched if _INCIDENTAL.search(item.description)]
            charged = sum(abs(item.effective_amount or 0) for item in incidental)
            # Without amounts on the matched lines there is nothing to measure against: left alone.
            small_enough = matched_value == 0 or charged <= matched_value * MAX_INCIDENTAL_SHARE
            unlisted = [item for item in unmatched if not (small_enough and item in incidental)]
            for item in unlisted[:MAX_UNLISTED_LINES]:
                findings.append(ProfileFinding(
                    type="unlisted_line_item", key="line_item", severity="medium",
                    description=f"{inv_name} bills an item that the {kind} {ref_name} does not list: {describe_line(item)}.",
                    source_a=ref_name, source_b=inv_name, text_b=item.description,
                    value_b=item.effective_amount, unit_b=item.unit, cited=[cite_line(invoice, item)],
                ))
    return findings


def _names(profiles: list[DocumentProfile]) -> str:
    names = [profile.filename for profile in profiles]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def check_series(reference: DocumentProfile, invoices: list[DocumentProfile], *, basic: bool = False) -> list[ProfileFinding]:
    """Several invoices for one contract or order, taken together.

    Each of them alone may bill less than was ordered - that is what partial
    deliveries and monthly invoices look like. Together they must not bill
    more. Only cases that no single invoice already shows are reported here.
    """
    findings: list[ProfileFinding] = []
    if len(invoices) < 2:
        return findings
    ref_name, kind = reference.filename, _kind(reference)

    # ---- quantities per ordered line
    billed: dict[int, list[tuple[DocumentProfile, LineItem]]] = {}
    for invoice in invoices:
        pairs, _ = match_lines(reference.line_items, invoice.line_items)
        for ordered, line in pairs:
            if ordered.quantity is None or line.quantity is None:
                continue
            # Quantities are only added up when they count the same thing: a line priced per piece
            # is not added to an order counted in boxes.
            if ordered.unit_price is not None and line.unit_price is not None and not _close(ordered.unit_price, line.unit_price, 0.00005):
                continue
            billed.setdefault(reference.line_items.index(ordered), []).append((invoice, line))
    for index, lines in billed.items():
        ordered = reference.line_items[index]
        assert ordered.quantity is not None
        quantities = [line.quantity for _, line in lines if line.quantity is not None]
        if len(lines) < 2 or any(quantity > ordered.quantity + 0.0005 for quantity in quantities):
            continue  # one invoice alone already bills too much: reported by check_pair
        together = round(sum(quantities), 4)
        if together > ordered.quantity + 0.0005:
            sources = [invoice for invoice, _ in lines]
            findings.append(ProfileFinding(
                type="quantity_mismatch", key="quantity", severity="medium",
                description=(
                    f"{ref_name} lists {format_number(ordered.quantity)} x {ordered.description}, but {_names(sources)} "
                    f"together bill {format_number(together)}."
                ),
                source_a=ref_name, source_b=_names(sources), value_a=ordered.quantity, value_b=together,
                cited=[cite_line(reference, ordered)] + [cite_line(invoice, line) for invoice, line in lines[:MAX_CITED_LINES]],
                operands=quantities, operation="sum",
            ))

    # ---- totals (stand-alone comparison only, like the single totals in check_pair)
    total = reference.total
    if basic and total is not None and total.number is not None:
        stated = [
            invoice for invoice in invoices
            if invoice.total is not None and invoice.total.number is not None
            and (not total.unit or not invoice.total.unit or invoice.total.unit == total.unit)
        ]
        amounts = [invoice.total.number for invoice in stated]  # type: ignore[union-attr]
        together = round(sum(amounts), 2)
        if len(stated) >= 2 and all(amount <= total.number + CENT for amount in amounts) and together > total.number + CENT:
            findings.append(ProfileFinding(
                type="amount_mismatch", key=KEY_TOTAL_AMOUNT, severity="medium",
                description=(
                    f"{ref_name} states the total amount as {format_value(total.number, total.unit)}, but {_names(stated)} "
                    f"together bill {format_value(together, total.unit)}."
                ),
                source_a=ref_name, source_b=_names(stated), value_a=total.number, value_b=together,
                unit_a=total.unit, unit_b=total.unit,
                cited=[cite_total(reference)] + [cite_total(invoice) for invoice in stated[:MAX_CITED_LINES]],
            ))
    return findings


# ------------------------------------------------------ invoices among each other
def _identifier_key(value: ProfileField | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value.text or "").upper()) if value is not None else ""


def _total_of(profile: DocumentProfile) -> tuple[float, str | None] | None:
    if profile.total is None or profile.total.number is None:
        return None
    return profile.total.number, profile.total.unit


def _same_day(first: list, second: list) -> bool:
    """Are two dates the same day, given that "03/02/2026" may be 3 February or 2 March?

    A date written in digits keeps both readings, the day-first one first.
    "03/02/2026" and "02/03/2026" have the same two readings - and are two
    different days whichever way this supplier writes dates. So two such dates
    are compared reading by reading; a date written in words is the same day
    as a numeric one if it is one of its readings.
    """
    if not first or not second:
        return False
    if len(first) == len(second):
        return list(first) == list(second)
    single, several = (first, second) if len(first) == 1 else (second, first)
    return len(single) == 1 and single[0] in several


def _days_apart(first: DocumentProfile, second: DocumentProfile) -> int | None:
    """The smallest number of days between the two documents' dates, over every reading. None if a date is unknown."""
    if first.date is None or second.date is None or not first.date.dates or not second.date.dates:
        return None
    return min(abs((a - b).days) for a in first.date.dates for b in second.date.dates)


def _for_different_orders(first: DocumentProfile, second: DocumentProfile) -> bool:
    """Do the two invoices say themselves that they belong to different orders (or contracts)?"""
    for name in ("purchase_order_reference", "contract_reference"):
        a, b = getattr(first, name), getattr(second, name)
        if a is not None and b is not None and a.text and b.text and not same_identifier(a.text, b.text):
            return True
    return False


def _bill_the_same_things(first: DocumentProfile, second: DocumentProfile) -> bool:
    """Do both invoices list the same lines? Unknown (no lines read) counts as "may well be"."""
    if not first.line_items or not second.line_items:
        return True
    a = sorted(sorted(_line_words(item.description)) for item in first.line_items)
    b = sorted(sorted(_line_words(item.description)) for item in second.line_items)
    return a == b


def check_duplicates(invoices: list[DocumentProfile]) -> list[ProfileFinding]:
    findings = []
    for first, second in combinations(invoices, 2):
        if first.document_id == second.document_id:
            continue
        suppliers = compare_names(
            first.supplier.text if first.supplier else None, second.supplier.text if second.supplier else None
        )
        if suppliers == "different":
            continue  # two suppliers may well use the same numbering
        # "Tolland Electrical" and "Tolland Electricals": a resubmitted bill is exactly where a name gets retyped.
        related = suppliers in RELATED
        number_a, number_b = _identifier_key(first.number), _identifier_key(second.number)
        if not number_a or not number_b:
            continue
        total_a, total_b = _total_of(first), _total_of(second)
        same_total = total_a is not None and total_b is not None and _close(total_a[0], total_b[0]) and total_a[1] == total_b[1]
        shown_a, shown_b = first.number.text, second.number.text  # type: ignore[union-attr]
        days_apart = _days_apart(first, second)
        far_apart = days_apart is not None and days_apart > NUMBERING_RESTART_DAYS

        if number_a == number_b:
            if first.pro_forma != second.pro_forma:
                continue  # a pro-forma invoice and the invoice that follows it carry the same number
            if not same_total and (not related or far_apart):
                # Different amounts under one number: worth a look from one supplier within a few weeks
                # (a correction?). From suppliers whose names were not read, or a year apart, it is two
                # invoices that happen to share a number.
                continue
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
            if far_apart:
                cited += [cite_date(first), cite_date(second)]
                description += (
                    f" They are dated {first.date.text} and {second.date.text}; a supplier that starts its numbering "  # type: ignore[union-attr]
                    "again every year would explain it."
                )
            if suppliers in ("short_form", "similar"):
                description += f" The supplier is written differently on the two ({first.supplier.text} and {second.supplier.text})."  # type: ignore[union-attr]
            findings.append(ProfileFinding(
                type="duplicate_invoice", key="invoice_number",
                severity="low" if far_apart else "high" if same_total else "medium",
                description=description, source_a=first.filename, source_b=second.filename,
                text_a=shown_a, text_b=shown_b, cited=cited,
            ))
            continue

        # The same bill under a new number: same supplier, same date, same total - and nothing on the
        # two that says they are for different things (another order, other lines).
        same_date = first.date is not None and second.date is not None and _same_day(first.date.dates, second.date.dates)
        if related and same_total and same_date and total_a is not None:
            if first.pro_forma != second.pro_forma or _for_different_orders(first, second) or not _bill_the_same_things(first, second):
                continue
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


def _supplier_link(invoice: DocumentProfile, candidate: DocumentProfile) -> str:
    """What the supplier names say about an invoice and a contract or order: related | unrelated | unknown."""
    if invoice.supplier is None or not candidate.parties:
        return "unknown"
    # The reference's suppliers, and every party whose role it does not state.
    others = [party for party in candidate.parties if party.role != "customer"]
    if not others:
        return "unknown"
    if any(may_belong_together(invoice.supplier.text, party.text) for party in others):
        return "related"
    if all(compare_names(invoice.supplier.text, party.text) == "different" for party in others):
        return "unrelated"
    return "unknown"


def _latest_revisions(candidates: list[DocumentProfile]) -> list[DocumentProfile]:
    """Of several orders (or contracts) that carry one number, keep the one with the latest date.

    An order that was revised lies in the library twice. The invoice is for
    the latest version; which one was uploaded first must not decide.
    """
    by_number: dict[str, list[DocumentProfile]] = {}
    for candidate in candidates:
        by_number.setdefault(_identifier_key(candidate.number) or f"document {candidate.document_id}", []).append(candidate)
    kept = []
    for group in by_number.values():
        dated = [candidate for candidate in group if candidate.date is not None and candidate.date.dates]
        kept.append(max(dated, key=lambda candidate: max(candidate.date.dates)) if dated else group[0])  # type: ignore[union-attr]
    return [candidate for candidate in candidates if candidate in kept]


def _candidates(kind: str, profiles: list[DocumentProfile]) -> list[DocumentProfile]:
    usable = sorted(
        (profile for profile in profiles if profile.document_type == kind and not profile.is_empty),
        key=_richness, reverse=True,
    )
    return _latest_revisions(usable)


def _cites(invoice: DocumentProfile, reference: DocumentProfile) -> bool:
    cited = invoice.contract_reference if reference.document_type == "contract" else invoice.purchase_order_reference
    return bool(
        cited is not None and cited.text and reference.number is not None
        and same_identifier(reference.number.text, cited.text)
    )


def find_reference(
    invoice: DocumentProfile, kind: str, profiles: list[DocumentProfile], *,
    assume_related: bool = False, forced: frozenset[tuple[str, str]] | set[tuple[str, str]] = frozenset(),
) -> tuple[DocumentProfile | None, ProfileFinding | None]:
    """The contract (or purchase order) an invoice belongs to, and a finding if its reference number points elsewhere.

    An invoice belongs to a reference when it cites its number, or when it
    comes from its supplier - or from a name that looks like its supplier's.
    An invoice from quite another company that cites nothing is simply another
    piece of business that lies in the same folder: it is left alone. (Before
    the first review it was checked against whatever contract was there, and
    accused of coming from the wrong supplier.)

    Two things overrule the names, because the person asking knows better than
    a name comparison (second review):

    ``forced``          (invoice id, reference id) pairs the person named.
    ``assume_related``  the person put one invoice next to these documents on
                        purpose: the only contract (or order) there is the one
                        it is to be checked against.
    """
    candidates = _candidates(kind, profiles)
    if not candidates:
        return None, None
    cited_reference = invoice.contract_reference if kind == "contract" else invoice.purchase_order_reference
    links = {candidate.document_id: _supplier_link(invoice, candidate) for candidate in candidates}
    named = [candidate for candidate in candidates if (invoice.document_id, candidate.document_id) in forced]
    related = [candidate for candidate in candidates if links[candidate.document_id] == "related"]
    possible = [candidate for candidate in candidates if links[candidate.document_id] != "unrelated" or candidate in named]
    if assume_related and len(candidates) == 1:
        possible = candidates

    if cited_reference is not None and cited_reference.text:
        for candidate in candidates:
            if candidate.number is not None and same_identifier(candidate.number.text, cited_reference.text):
                return candidate, None  # it says so itself; a different supplier is then a finding of check_pair
        # It cites a number that none of the documents carries. Worth saying only about references of
        # its own supplier (or when no supplier is known): otherwise it is simply about something else.
        numbered = [candidate for candidate in possible if candidate.number is not None]
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
            return (possible[0] if len(possible) == 1 else None), finding
        return None, None

    # No reference number cited: the person's own words decide first, then the supplier.
    if len(named) == 1:
        return named[0], None
    if len(related) == 1:
        return related[0], None
    if not related and len(possible) == 1 and len(candidates) == 1:
        return possible[0], None  # nothing is known that speaks against the only candidate
    return None, None


_IDENTIFIER_IN_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9/_.-]*\d[A-Za-z0-9/_-]*")


def named_in(question: str, profiles: list[DocumentProfile]) -> set[str]:
    """The ids of the documents a question names, by their number ("INV-355") or by their file name."""
    lowered = question.lower()
    tokens = _IDENTIFIER_IN_TEXT.findall(question)
    named = set()
    for profile in profiles:
        stem = profile.filename.rsplit(".", 1)[0].lower()
        if profile.filename.lower() in lowered or (len(stem) >= 4 and stem in lowered):
            named.add(profile.document_id)
        elif profile.number is not None and any(same_identifier(profile.number.text, token) for token in tokens):
            named.add(profile.document_id)
    return named


def named_pairs(question: str, profiles: list[DocumentProfile]) -> set[tuple[str, str]]:
    """(invoice id, reference id) for every invoice and contract or order that the question names together."""
    named = named_in(question, profiles)
    return {
        (invoice.document_id, reference.document_id)
        for invoice in profiles if invoice.document_type == "invoice" and invoice.document_id in named
        for reference in profiles if reference.document_type in REFERENCE_TYPES and reference.document_id in named
    }


@dataclass
class Comparison:
    """What comparing a set of documents gave."""

    findings: list[ProfileFinding] = field(default_factory=list)  # the most serious first
    # What was NOT checked, and why. Shown to the reader: a pair that was left alone must not read as "no problem".
    notes: list[str] = field(default_factory=list)
    # (invoice id, contract or order id): this invoice is not held against that document - not here,
    # and not in the comparison of totals that comes from the search (app/tools/contradiction.py).
    unrelated: set[tuple[str, str]] = field(default_factory=set)
    # Documents that belong together: (invoice, its contract or order) and (order, the contract it cites).
    linked: set[tuple[str, str]] = field(default_factory=set)


def _left_alone(invoice: DocumentProfile, candidate: DocumentProfile) -> str:
    kind = _kind(candidate)
    theirs = " and ".join(party.text or "" for party in candidate.parties if party.role != "customer")
    return (
        f"{invoice.filename} was not checked against the {kind} {candidate.filename}: it is issued by "
        f"{invoice.supplier.text if invoice.supplier else 'another company'}, the {kind} names {theirs}, and the invoice "
        f"does not quote the {kind}'s number. To have the two compared anyway, name both in the question or select just these documents."
    )


def compare(
    profiles: list[DocumentProfile], *, basic: bool = False, assume_related: bool = False,
    forced: frozenset[tuple[str, str]] | set[tuple[str, str]] = frozenset(),
) -> Comparison:
    """Run every check on a set of documents."""
    usable = [profile for profile in profiles if not profile.is_empty]
    bills = [profile for profile in usable if profile.document_type == "invoice"]
    # A credit note quotes the invoice it corrects and gives money back: it is not checked like an invoice.
    invoices = [profile for profile in bills if not profile.credit_note]
    result = Comparison()
    findings = result.findings
    series: dict[str, tuple[DocumentProfile, list[DocumentProfile]]] = {}

    for bill in bills:
        if bill.credit_note:
            result.notes.append(
                f"{bill.filename} calls itself a credit note. Who it comes from was checked; its amounts, dates and lines "
                "were not checked like an invoice's."
            )
        for kind in REFERENCE_TYPES:
            candidates = [profile for profile in usable if profile.document_type == kind]
            reference, mismatch = find_reference(bill, kind, usable, assume_related=assume_related, forced=forced)
            if mismatch is not None and not bill.credit_note:
                findings.append(mismatch)
            if reference is not None:
                findings += check_pair(reference, bill, basic=basic, parties_only=bill.credit_note)
                result.linked.add((bill.document_id, reference.document_id))
                if not bill.credit_note:
                    series.setdefault(reference.document_id, (reference, []))[1].append(bill)
                # It belongs to this one - so not to the others of the same kind (another contract, an
                # earlier version of the order).
                result.unrelated |= {
                    (bill.document_id, other.document_id) for other in candidates if other.document_id != reference.document_id
                }
                continue
            for candidate in candidates:
                if not _cites(bill, candidate) and _supplier_link(bill, candidate) == "unrelated":
                    result.unrelated.add((bill.document_id, candidate.document_id))
                    if not bill.credit_note:
                        result.notes.append(_left_alone(bill, candidate))
    for reference, group in series.values():
        findings += check_series(reference, group, basic=basic)
    findings += check_duplicates(invoices)
    for profile in usable:
        if not profile.credit_note:
            findings += check_document(profile)

    # An order that cites a contract belongs to it: the two totals may be compared (the order must not exceed the contract).
    for order in usable:
        cited = order.contract_reference if order.document_type == "purchase_order" else None
        for contract in usable:
            if cited is not None and contract.document_type == "contract" and _cites_number(cited, contract):
                result.linked.add((order.document_id, contract.document_id))

    findings.sort(key=lambda finding: _SEVERITY_ORDER[finding.severity])  # stable: equal severities keep their order
    del findings[MAX_FINDINGS:]
    result.notes = list(dict.fromkeys(result.notes))[:MAX_NOTES]
    return result


def _cites_number(cited: ProfileField, reference: DocumentProfile) -> bool:
    return bool(cited.text and reference.number is not None and same_identifier(reference.number.text, cited.text))


def unrelated_pairs(profiles: list[DocumentProfile]) -> set[tuple[str, str]]:
    """(invoice id, contract or order id) for invoices that do not belong to that reference."""
    return compare(profiles).unrelated


def compare_profiles(profiles: list[DocumentProfile], *, basic: bool = False) -> list[ProfileFinding]:
    """Run every check on a set of documents. The most serious findings come first."""
    return compare(profiles, basic=basic).findings
