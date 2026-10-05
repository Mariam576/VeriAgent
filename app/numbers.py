"""Reading numbers and units out of text, in the formats business documents use.

One parser for the whole application, so grounding checks, the rule-based
extractor and the reviewer all agree on what a piece of text says.

Formats understood:

    40,000        40,000.50      Western grouping
    1,50,000      1,50,000.75    Indian / Pakistani grouping (lakh, crore)
    40.000,00     1.234.567      European grouping, decimal comma
    40 000        40 000,50      space grouping (also non-breaking spaces)
    40000         40000.5        plain

One case is truly ambiguous: "40.000" is forty in the UK but forty thousand
in Germany. ``parse_number`` therefore returns every reasonable reading, and
callers decide (a grounding check accepts either; money prefers thousands).
"""

from __future__ import annotations

import re

_SPACES = "    "

# A number: digits, optionally grouped by , . ' or spaces, optionally with decimals.
NUMBER_CORE = r"(?:\d{1,3}(?:[,.'\u00a0\u202f\u2009 ]\d{2,3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
# ... standing on its own: not glued to a word ("v2") and not cut out of a longer number.
NUMBER_TOKEN = re.compile(rf"(?<!\w)(?<!\d[.,]){NUMBER_CORE}(?![\w]|[.,]\d)")

_WESTERN = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_INDIAN = re.compile(r"^\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?$")
_EUROPEAN = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$")
_SPACED = re.compile(rf"^\d{{1,3}}(?:[{_SPACES}']\d{{3}})+(?:[.,]\d+)?$")
_PLAIN = re.compile(r"^\d+(?:[.,]\d+)?$")


def parse_number(token: str) -> list[float]:
    """Every reasonable numeric reading of one token. Empty list = not a number."""
    token = token.strip()
    if not token or not token[0].isdigit():
        return []

    if _INDIAN.match(token) or _WESTERN.match(token):
        return [float(token.replace(",", ""))]

    if _SPACED.match(token):
        compact = re.sub(rf"[{_SPACES}']", "", token).replace(",", ".")
        return [float(compact)]

    if _EUROPEAN.match(token):
        as_thousands = float(token.replace(".", "").replace(",", "."))
        if "," in token or token.count(".") > 1:
            return [as_thousands]  # "40.000,00" or "1.234.567" can only be European
        return [as_thousands, float(token)]  # "40.000": thousands, or forty with three decimals

    if _PLAIN.match(token):
        if "," in token:
            return [float(token.replace(",", "."))]  # decimal comma: "40,5"
        return [float(token)]
    return []


def find_number_tokens(text: str) -> list[tuple[str, int, int]]:
    """(token, start, end) for every number written in ``text``."""
    tokens = []
    for match in NUMBER_TOKEN.finditer(text):
        token = match.group(0)
        # A space-grouped match such as "30 202" is only a number if every group has 3 digits.
        if re.search(rf"[{_SPACES}]", token) and not _SPACED.match(token):
            for part in re.finditer(r"\d+(?:[.,]\d+)*", token):
                tokens.append((part.group(0), match.start() + part.start(), match.start() + part.end()))
            continue
        tokens.append((token, match.start(), match.end()))
    return tokens


def numbers_in_text(text: str) -> list[float]:
    """All numbers written in the text, with every reading of ambiguous ones.

    '£40,000 within 30 days' -> [40000.0, 30.0]
    """
    values: list[float] = []
    for token, _, _ in find_number_tokens(text):
        for value in parse_number(token):
            values.append(value)
    return values


def number_appears(value: float, text: str) -> bool:
    """Is ``value`` written somewhere in ``text``, in any supported format?"""
    for token, _, _ in find_number_tokens(text):
        if any(abs(candidate - value) < 0.005 for candidate in parse_number(token)):
            return True
        # Last resort for whole numbers: compare the bare digits ("1,50,000" -> "150000").
        if float(value).is_integer():
            digits = re.sub(r"[.,]0+$", "", token)
            if re.sub(r"\D", "", digits) == str(int(abs(value))) and re.fullmatch(rf"[\d,.'{_SPACES}]+", digits):
                return True
    return False


# ------------------------------------------------------------------- units
_CURRENCY_WORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"£|\bgbp\b|pounds?(?:\s+sterling)?|\bsterling\b|british\s+pounds?", re.I), "GBP"),
    (re.compile(r"€|\beur\b|\beuros?\b", re.I), "EUR"),
    (re.compile(r"\bpkr\b|pakistani\s+rupees?|\brs\.?(?!\w)|\brupees?\b", re.I), "PKR"),
    (re.compile(r"₹|\binr\b|indian\s+rupees?", re.I), "INR"),
    (re.compile(r"us\s?\$|\$|\busd\b|(?:us\s+)?dollars?", re.I), "USD"),
    (re.compile(r"¥|\bjpy\b|\byen\b", re.I), "JPY"),
    (re.compile(r"\baed\b|dirhams?", re.I), "AED"),
    (re.compile(r"\bsar\b|riyals?", re.I), "SAR"),
]


def normalize_unit(unit: str | None) -> str | None:
    """One spelling per unit: '£' / 'pounds sterling' -> 'GBP', 'calendar days' -> 'days'."""
    if unit is None or not unit.strip():
        return None
    cleaned = unit.strip()
    lowered = cleaned.lower()
    if "day" in lowered:
        return "days"
    if lowered in ("%", "percent", "per cent", "percentage"):
        return "%"
    # "Indian rupees" must win over the generic "rupees" rule, so check INR first.
    for pattern, code in sorted(_CURRENCY_WORDS, key=lambda item: item[1] != "INR"):
        if pattern.fullmatch(lowered) or pattern.search(lowered):
            return code
    if len(cleaned) == 3 and cleaned.isalpha():
        return cleaned.upper()  # any other ISO currency code
    return lowered
