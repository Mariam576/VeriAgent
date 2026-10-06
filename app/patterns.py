"""Patterns for reading amounts, totals and payment terms out of text.

Shared by the two rule-based readers: the fact extractor that stands in for
the LLM (``app/agents/heuristics.py``) and the document profile builder
(``app/comparison/rules.py``). Keeping the patterns in one place means both
agree on what "a total" is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.numbers import CURRENCY_CODES, NUMBER_CORE, SCALE_WORDS, normalize_unit, parse_number

# A currency marker directly before or after a number token (see app.numbers for the number formats).
# "USD 1.2 million" is left unread here: reading it as $1.20 would be a false alarm waiting to happen,
# and an unread amount only means that one check is skipped.
NUMBER = NUMBER_CORE + rf"(?![\w]|[.,]\d)(?!\s?(?:{SCALE_WORDS})\b)"
MONEY = re.compile(
    rf"(?P<cur>£|\$|€|₹|\b(?:{CURRENCY_CODES}|Rs\.?))\s?(?P<neg>[-−])?(?P<num>{NUMBER})"
    rf"|(?<!\w)(?<!\d[.,])(?P<num2>{NUMBER})\s?(?P<cur2>{CURRENCY_CODES})\b",
    re.IGNORECASE,
)

# (pattern, priority). A higher priority wins when a document has several candidates.
TOTAL_LABELS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\b(?:grand total|total amount due|amount due|total due|balance due|total payable|invoice total|total invoice (?:amount|value))\b", re.I), 3),
    (re.compile(r"\b(?:total (?:contract |order |purchase )?(?:value|price|sum|amount|fee|charges?|cost)|contract (?:value|price|sum|amount)|order (?:total|value))\b", re.I), 2),
    (re.compile(r"(?<![a-z-])total\b", re.I), 1),
]
CEILING = re.compile(r"\b(?:not\s+(?:to\s+)?exceed|maximum|up\s+to|capped|limited\s+to|no\s+more\s+than)\b", re.I)
PERIODIC = re.compile(r"\b(?:monthly|weekly|daily|hourly|quarterly|yearly|per\s+(?:month|week|day|hour|quarter|year|annum|visit|unit))\b", re.I)
PER_PERIOD = re.compile(r"\s*(?:per|a|each|/)\s*(?:calendar\s+)?(?:month|week|day|hour|quarter|year|annum|visit|unit)\b|\s*p\.?\s?[am]\.?(?!\w)", re.I)
# The amount before tax, under the names it goes by: "Subtotal", "Net amount", "Value excluding sales tax".
SUBTOTAL = re.compile(
    r"sub[\s-]?total|\bnet\s+(?:amount|total|value)\b|(?:^|\n)\W*net\s*:?\s*$|\bgoods\s+total\b|\btaxable\s+(?:value|amount)\b"
    r"|\b(?:value|amount|total)\s+(?:excluding|excl\.?|before|ex\.?)\s+(?:sales\s+)?(?:tax|vat|gst)\b",
    re.I,
)
# Words that make an amount a condition instead of a statement: "orders over £750", "up to £500".
NOT_A_STATEMENT = re.compile(r"\b(?:above|exceed\w*|over|under|below|up to|more than|less than|at least|minimum|maximum|limit|threshold)\b", re.I)

PAYMENT_TERMS = [
    re.compile(r"payment terms?\b[^.]{0,80}?\b(?P<days>\d{1,3})\s*(?:calendar |working |business )?days", re.I),
    re.compile(r"\b(?:pay|paid|payable|payment)\b[^.]{0,80}?\bwithin\s+(?P<days>\d{1,3})\s*(?:calendar |working |business )?days", re.I),
    re.compile(r"\b(?P<days>\d{1,3})\s*days\s+(?:from|of|after)\s+(?:the\s+)?(?:date of\s+)?(?:receipt of\s+)?(?:the\s+)?invoice", re.I),
    re.compile(r"\bnet\s?(?P<days>\d{1,3})\b", re.I),
]


def money_currency(match: re.Match[str]) -> str:
    return normalize_unit(match.group("cur") or match.group("cur2")) or ""


_SIGN_BEFORE = re.compile(r"(?:^|[\s:(=|])[-−–]$")


def is_negative(match: re.Match[str]) -> bool:
    """Is the amount written as negative: "-£1,200", "£-1,200" or, as accountants do, "(£1,200)"?

    A dash only counts as a minus sign when it stands directly in front of the
    amount and is not part of a word or a range ("A-£5", "£100 - £200").
    """
    text = match.string
    before = text[max(match.start() - 2, 0) : match.start()]
    if _SIGN_BEFORE.search(before) or match.groupdict().get("neg"):
        return True
    return before.endswith("(") and text[match.end() : match.end() + 1] == ")"


def money_amount(match: re.Match[str]) -> float | None:
    """The amount a money match states, negative when it is written as negative."""
    readings = parse_number(match.group("num") or match.group("num2"))
    if not readings:
        return None
    # For money, "40.000" means forty thousand (the first reading), not forty.
    return -readings[0] if is_negative(match) else readings[0]


def money_readings(match: re.Match[str]) -> list[float]:
    """Every reading of an ambiguous amount ("3.459": three thousand, or three and a bit)."""
    sign = -1.0 if is_negative(match) else 1.0
    return [sign * reading for reading in parse_number(match.group("num") or match.group("num2"))]


def iter_segments(text: str):
    """The sentences and lines of ``text``, one at a time: (start, end, the piece without its edges)."""
    position = 0
    while position < len(text):
        left, right = sentence_bounds(text, position, position)
        position = right + (2 if text.startswith(". ", right) else 1)
        yield left, right, text[left:right].strip(" .\n")


# "Rs. 1,150.00": the full stop belongs to the currency, the sentence goes on.
_NOT_AN_END = re.compile(r"(?:^|[^A-Za-z])Rs$", re.I)


def _stop_before(text: str, position: int) -> int:
    """Index of the last ". " that ends a sentence before ``position``, or -1."""
    while True:
        position = text.rfind(". ", 0, position)
        if position == -1 or not _NOT_AN_END.search(text[max(position - 3, 0):position]):
            return position


def _stop_after(text: str, position: int) -> int:
    """Index of the first ". " that ends a sentence at or after ``position``, or -1."""
    while True:
        position = text.find(". ", position)
        if position == -1 or not _NOT_AN_END.search(text[max(position - 3, 0):position]):
            return position
        position += 2


def sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Start and end index of the sentence (or line) that contains text[start:end]."""
    left = max(_stop_before(text, start), text.rfind("\n", 0, start)) + 1
    right_candidates = [pos for pos in (_stop_after(text, end), text.find("\n", end)) if pos != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return left, right


def sentence_around(text: str, start: int, end: int) -> str:
    left, right = sentence_bounds(text, start, end)
    return text[left:right].strip(" .\n")


@dataclass(frozen=True)
class AmountHit:
    """One money amount found in a text."""

    value: float
    unit: str
    start: int
    end: int
    sentence: str  # the sentence or line it stands in
    before: str  # the words of that sentence in front of the amount (at most 90 characters)
    priority: int  # 0 = no "total" label in front of it; otherwise the rank of the label (see TOTAL_LABELS)
    subtotal: bool  # the label in front of it is "subtotal"


def find_amounts(text: str) -> list[AmountHit]:
    """Every money amount in ``text``, with the label that introduces it (if any)."""
    hits: list[AmountHit] = []
    for match in MONEY.finditer(text):
        value = money_amount(match)
        if value is None:
            continue
        # Look only at the words directly before the amount, within the same sentence.
        sentence_start, _ = sentence_bounds(text, match.start(), match.end())
        before = text[max(sentence_start, match.start() - 90) : match.start()]
        is_subtotal = bool(SUBTOTAL.search(before))
        priority = 0
        if not is_subtotal:
            for pattern, level in TOTAL_LABELS:
                label = None
                for label in pattern.finditer(before):
                    pass  # keep the last (closest) label
                if label is None:
                    continue
                gap = before[label.end():]
                # The amount must follow the label closely: "total ... is £40,000", "Total due: £58,000".
                # A rate is not a total: "the total monthly charge is £3,000", "total fee of £40,000 per annum".
                rate = PERIODIC.search(gap) or PER_PERIOD.match(text, match.end())
                # "The total contract value shall not exceed £40,000": a ceiling is a total for our purpose,
                # because the only thing ever reported is "more than the total".
                condition = NOT_A_STATEMENT.search(gap) and not CEILING.search(gap)
                if len(gap) <= 34 and not re.search(r"\d", gap) and not condition and not rate:
                    priority = level
                    break
        hits.append(AmountHit(
            value=value, unit=money_currency(match), start=match.start(), end=match.end(),
            sentence=sentence_around(text, match.start(), match.end()), before=before,
            priority=priority, subtotal=is_subtotal,
        ))
    return hits


def find_payment_terms(text: str) -> tuple[float, str] | None:
    """(days, the words that state them) for the first payment-terms statement, or None."""
    for pattern in PAYMENT_TERMS:
        match = pattern.search(text)
        if match:
            return float(match.group("days")), match.group(0)
    return None
