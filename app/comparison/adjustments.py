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

Everything here is plain text search, done by code for every profile, whether
the profile itself was built by the model or by the rules.
"""

from __future__ import annotations

import re

from app.comparison.profile import MAX_QUOTE_CHARS, ProfileField
from app.patterns import MONEY, find_amounts, iter_segments, money_amount, money_currency
from app.rag.loaders import PageText
from app.security.guardrails import trusted_part

MAX_ADJUSTMENTS = 12

# Which way an adjustment moves the sum on its way from the lines to the total.
_ADDED = r"vat|gst|cgst|sgst|igst|hst|pst|sales\s+tax|tax(?:es)?|duty|levy|shipping|delivery|freight|carriage|postage|courier|handling|surcharge"
_TAKEN_OFF = (
    r"discount|rebate|less(?!\s+than)|deduction|retention|allowance|deposit|advance|prepaid|pre-paid|paid|"
    r"payment\s+received|credit(?!\s+(?:note|memo|terms|period|limit))"
)
_EITHER_WAY = r"round(?:ing|ed)?(?:[\s-]off)?|adjustment"
ADDED = re.compile(rf"\b(?:{_ADDED})\b", re.I)
TAKEN_OFF = re.compile(rf"\b(?:{_TAKEN_OFF})\b", re.I)
ADJUSTMENT_WORDS = re.compile(rf"\b(?:{_ADDED}|{_TAKEN_OFF}|{_EITHER_WAY})\b", re.I)
_PERCENT = re.compile(r"\d\s?%")
_CREDIT_NOTE_TITLE = re.compile(r"\bcredit\s+(?:note|memo(?:randum)?|invoice)\b", re.I)
TITLE_CHARS = 300  # how far into the first page a document's title is looked for


def find_adjustments(pages: list[PageText]) -> tuple[list[ProfileField], bool]:
    """(the adjustment amounts, True if an adjustment is mentioned without an amount).

    Each amount is positive; its ``text`` says which way it moves the sum: "add" (a tax, a
    delivery charge), "subtract" (a discount, a deposit already paid) or "either" (rounding).
    """
    found: list[ProfileField] = []
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
            added, taken_off = bool(ADDED.search(segment)), bool(TAKEN_OFF.search(segment))
            direction = "add" if added and not taken_off else "subtract" if taken_off and not added else "either"
            found.append(ProfileField(
                text=direction, number=abs(amount), unit=money_currency(last) or None, page=page.page_number,
                quote=segment[:MAX_QUOTE_CHARS],
            ))
    return found, unquantified


def is_credit_note(pages: list[PageText]) -> bool:
    """Does the document call itself a credit note at its top?"""
    first = next((page.text for page in pages if page.text.strip()), "")
    return bool(_CREDIT_NOTE_TITLE.search(trusted_part(first)[:TITLE_CHARS]))
