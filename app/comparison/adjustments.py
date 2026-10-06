"""What stands between the lines of an invoice and its total.

Concept - do not accuse what you do not model:
    "The line items add up to £1,000 but the total is £1,080" is only a
    finding if nothing on the page explains the £80. Real invoices explain it
    all the time: VAT, two taxes instead of one (CGST and SGST), a discount, a
    delivery charge, a deposit that was already paid, a rounding line.

    Modelling every one of these as its own field would never be complete. So
    this module does something simpler and more honest: it collects every
    amount that stands in a sentence or line about such an adjustment, and
    the comparison (``engine.py``) then asks one question - can the difference
    be made from these amounts, each added, subtracted or left out? If yes,
    the document explains itself and nothing is reported.

    The second review showed where that generosity has to stop. An amount is
    only an adjustment if it is not already one of the invoice's lines (a
    courier's "Delivery to site A" lines are what is billed, and counting them
    again would explain any total), if it is stated and not a condition ("free
    delivery on orders over £750"), and once - a VAT summary box repeats the
    VAT, it does not add a second one.

Everything here is plain text search, done by code for every profile, whether
the profile itself was built by the model or by the rules. The same goes for
the three things a document says about itself that change what is checked:
"this is a credit note", "this is a pro-forma invoice", "this contract renews".
"""

from __future__ import annotations

import re

from app.comparison.profile import MAX_QUOTE_CHARS, LineItem, ProfileField
from app.patterns import MONEY, NOT_A_STATEMENT, find_amounts, iter_segments, money_amount, money_currency
from app.rag.citations import normalize
from app.rag.loaders import PageText
from app.security.guardrails import trusted_part

MAX_ADJUSTMENTS = 12

# Which way an adjustment moves the sum on its way from the lines to the total.
_ADDED = (
    r"vat|gst|cgst|sgst|igst|hst|pst|sales\s+tax|tax(?:es)?|duty|levy|shipping|delivery|freight|carriage|postage|courier|handling|surcharge|"
    # an unpaid balance from an earlier statement, added to what this invoice bills
    r"(?:brought|carried)\s+forward|b/f|(?:previous|opening|outstanding)\s+balance|arrears"
)
_TAKEN_OFF = (
    r"discount|rebate|less(?!\s+than)|deduction|retention|allowance|deposit|advance|prepaid|pre-paid|paid|tds|cis|withholding|"
    r"payment\s+received|credit(?!\s+(?:note|memo|terms|period|limit))"
)
_EITHER_WAY = r"round(?:ing|ed)?(?:[\s-]off)?|adjustment"
# Taxes only (not delivery or handling): what turns an amount "before tax" into the total.
TAX = re.compile(r"\b(?:vat|gst|cgst|sgst|igst|hst|pst|sales\s+tax|further\s+tax|tax(?:es)?|duty|levy)\b", re.I)
ADDED = re.compile(rf"\b(?:{_ADDED})\b", re.I)
TAKEN_OFF = re.compile(rf"\b(?:{_TAKEN_OFF})\b", re.I)
ADJUSTMENT_WORDS = re.compile(rf"\b(?:{_ADDED}|{_TAKEN_OFF}|{_EITHER_WAY})\b", re.I)
_PERCENT = re.compile(r"\d\s?%")
# Words that turn an adjustment round: a tax that is withheld is taken off, a retention that is released is added.
_TURNED_ROUND = re.compile(r"\b(?:deduct\w*|withh\w+|tds|released?|refund\w*|revers\w+|credited|reclaim\w*)\b", re.I)
# The condition word stands directly in front of the amount: "orders over £750", "up to £500".
_CONDITION = re.compile(NOT_A_STATEMENT.pattern + r"(?:\s+of)?\W{0,3}$", re.I)

TITLE_CHARS = 300  # how far into the first page a document's title is looked for
TITLE_LINES = 6
# A heading: the words stand at the start of a line. "Credit note requests: accounts@..." and
# "No credit note will be issued after 30 days" are sentences about credit notes, not titles.
_CREDIT_NOTE_HEADING = re.compile(
    r"^\W*(?:(?:tax|vat|sales)\s+)?credit\s+(?:note|memo(?:randum)?)\b"
    r"(?!\s+(?:requests?|enquir\w+|quer\w+|polic\w+|terms|will|may|can|cannot|is|are|must|shall)\b)",
    re.I,
)
_PRO_FORMA_HEADING = re.compile(r"^\W*pro[\s-]?forma\b", re.I)
_ASKS_FOR_PAYMENT = re.compile(r"\b(?:due|payable|to\s+pay|please\s+pay|outstanding)\b", re.I)
_GIVES_BACK = re.compile(r"\b(?:credit\w*|refund\w*)\b", re.I)
_RENEWS = re.compile(
    r"\b(?:renews?\s+automatically|automatically\s+renew\w*|auto[\s-]?renew\w*|(?:shall|will)\s+(?:be\s+)?renew\w*|"
    r"renewed\s+automatically|evergreen|successive\s+(?:periods?|terms?)|rolling\s+(?:basis|contract|term)|until\s+terminated)\b",
    re.I,
)


def find_adjustments(pages: list[PageText]) -> tuple[list[ProfileField], bool]:
    """(the adjustment amounts, True if an adjustment is mentioned without an amount).

    Each amount is positive; its ``text`` says which way it moves the sum: "add" (a tax, a
    delivery charge), "subtract" (a discount, a deposit already paid) or "either" (rounding).
    """
    found: list[ProfileField] = []
    seen: set[tuple[frozenset[str], float]] = set()
    unquantified = False
    for page in pages:
        text = trusted_part(page.text)
        for _, _, segment in iter_segments(text):
            if not segment or not ADJUSTMENT_WORDS.search(segment):
                continue
            hits = find_amounts(segment)
            if not hits:
                # "Less 10% discount", "VAT at 20% is included": something changes the sum, by an unknown amount.
                unquantified = unquantified or bool(_PERCENT.search(segment))
                continue
            if any(hit.priority or hit.subtotal for hit in hits):
                continue  # "Total due including VAT: £1,080" states a total, not an adjustment
            last = list(MONEY.finditer(segment))[-1]
            amount = money_amount(last)
            if amount is None or amount == 0 or len(found) >= MAX_ADJUSTMENTS:
                continue
            if _CONDITION.search(segment[:last.start()]):
                continue  # "Free delivery on orders over £750.00": a condition, not a charge
            # The same adjustment said twice ("VAT 20%: £200" and, in a summary box, "VAT analysis: ... = £200")
            # is one adjustment. Two different ones of the same size (CGST and SGST) stay two.
            words = frozenset(word.lower() for word in ADJUSTMENT_WORDS.findall(segment))
            if (words, abs(amount)) in seen:
                continue
            seen.add((words, abs(amount)))
            added, taken_off = bool(ADDED.search(segment)), bool(TAKEN_OFF.search(segment))
            direction = "add" if added and not taken_off else "subtract" if taken_off and not added else "either"
            if _TURNED_ROUND.search(segment):
                direction = "either"  # "CIS tax deducted", "retention released": the word alone does not say which way
            found.append(ProfileField(
                text=direction, number=abs(amount), unit=money_currency(last) or None, page=page.page_number,
                quote=segment[:MAX_QUOTE_CHARS],
            ))
    return found, unquantified


def not_a_line(adjustment: ProfileField, lines: list[LineItem]) -> bool:
    """False when this "adjustment" is one of the document's own line items, read a second time.

    The same words, and a figure of that line: its amount, or its unit price
    ("2 delivery runs at £400.00 each" is a line of £800, not a £400 charge).
    A model may shorten the row it quotes with "...", so the words of the
    line's description are enough - the quote need not be the whole sentence.
    """
    words = normalize(adjustment.quote)
    for line in lines:
        if line.page != adjustment.page or adjustment.number is None:
            continue
        figures = [figure for figure in (line.amount, line.effective_amount, line.unit_price) if figure is not None]
        if not any(abs(abs(figure) - adjustment.number) <= 0.005 for figure in figures):
            continue
        quote = normalize(line.quote)
        described = [word for word in re.findall(r"[a-z0-9]+", line.description.lower()) if len(word) >= 3]
        same_words = bool(described) and all(word in words for word in described)
        if (quote and (quote in words or words in quote)) or same_words:
            return False
    return True


def _title_lines(pages: list[PageText]) -> list[str]:
    first = next((page.text for page in pages if page.text.strip()), "")
    top = trusted_part(first)[:TITLE_CHARS]
    return [segment for _, _, segment in iter_segments(top) if segment][:TITLE_LINES]


def is_credit_note(pages: list[PageText]) -> bool:
    """Does the document call itself a credit note in its heading?"""
    return any(_CREDIT_NOTE_HEADING.match(line) for line in _title_lines(pages))


def asks_for_payment(total: ProfileField | None) -> bool:
    """A positive total that is "due" or "payable": whatever the heading says, this document wants money."""
    if total is None or total.number is None or total.number <= 0:
        return False
    return bool(_ASKS_FOR_PAYMENT.search(total.quote)) and not _GIVES_BACK.search(total.quote)


# A minimum CHARGE (for a call-out, for attendance). "Minimum order value £50" is a condition of sale and
# lifts no line above quantity x price.
_MINIMUM_CHARGE = re.compile(r"\bminimum\s+(?:call[\s-]?out\s+|attendance\s+|service\s+|labou?r\s+)?(?:charge|fee|billing)\b", re.I)


def has_minimum_charge(pages: list[PageText]) -> bool:
    """ "Minimum call-out charge applies": a line may then be billed at more than quantity x price."""
    return any(_MINIMUM_CHARGE.search(trusted_part(page.text)) for page in pages)


def is_pro_forma(pages: list[PageText]) -> bool:
    return any(_PRO_FORMA_HEADING.match(line) for line in _title_lines(pages))


_NEGATED = re.compile(r"\b(?:not|no|never|nor|without|cannot|won't|shall\s+not)\b[^.;\n]{0,25}$", re.I)


def renews_automatically(pages: list[PageText]) -> bool:
    """Does the contract say that it renews by itself? ("does not renew automatically" says the opposite.)"""
    for page in pages:
        text = trusted_part(page.text)
        for match in _RENEWS.finditer(text):
            if not _NEGATED.search(text[max(match.start() - 40, 0):match.start()]):
                return True
    return False
