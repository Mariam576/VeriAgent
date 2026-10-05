"""Small helpers for showing numbers and labels to people."""

from __future__ import annotations

import re

_CURRENCY_SYMBOLS = {"GBP": "£", "USD": "$", "EUR": "€", "INR": "₹", "JPY": "¥"}
# Shown as "PKR 1,200" because they have no single well-known symbol.
_CURRENCY_CODES = {"PKR", "AED", "SAR", "CAD", "AUD", "CHF", "CNY", "SGD", "ZAR", "NZD", "SEK", "NOK", "DKK"}

_KEY_LABELS = {
    "total_amount": "total amount",
    "payment_terms_days": "payment terms",
    "document_total": "total amount",
    "document_subtotal": "subtotal",
    "document_payment_terms": "payment terms",
    "document_date": "date",
}


def format_number(value: float) -> str:
    """58000.0 -> '58,000'; 1234.5 -> '1,234.50'."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def format_value(value: float | None, unit: str | None) -> str:
    """Human-readable value with its unit: '£58,000', '30 days', 'PKR 1,200', '45'."""
    if value is None:
        return "unknown"
    number = format_number(value)
    if not unit:
        return number
    code = unit.strip().upper()
    if code in _CURRENCY_SYMBOLS:
        sign = "-" if value < 0 else ""
        return f"{sign}{_CURRENCY_SYMBOLS[code]}{format_number(abs(value))}"
    if code in _CURRENCY_CODES:
        return f"{code} {number}"
    if unit.strip() == "%":
        return f"{number}%"
    return f"{number} {unit.strip()}"


def key_label(key: str) -> str:
    return _KEY_LABELS.get(key, key.replace("_", " "))


# Every character Markdown (or a Markdown viewer's extensions: $ for formulas) gives a meaning to.
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+!|~$])")
_URL_SCHEME = re.compile(r"\b(https?|ftp|mailto|file|javascript|data)(:)(/{0,2})", re.IGNORECASE)
_WWW = re.compile(r"\bwww\.", re.IGNORECASE)
_ZERO_WIDTH = "\u200b"


def escape_markdown(text: object) -> str:
    """Make text show literally when a report is rendered as Markdown.

    Text taken from documents is untrusted. Without this, an invoice that
    contains ``![x](http://attacker/pixel.png)`` would make the reader's browser
    call that address, ``[Pay here](...)`` would put a phishing link into the
    report, and "$40,000 ... $58,000" would be displayed as a formula.
    Used wherever a report leaves VeriAgent as Markdown, for example in the
    text of a follow-up issue (app/actions/proposals.py). The web interface
    needs no such step: React writes text into the page as text.
    """
    value = " ".join(str(text if text is not None else "").split())
    # A bare address would be turned into a link automatically; an invisible break prevents that.
    value = _URL_SCHEME.sub(lambda match: f"{match.group(1)}{match.group(2)}{_ZERO_WIDTH}{match.group(3)}", value)
    value = _WWW.sub(f"www{_ZERO_WIDTH}.", value)
    return _MARKDOWN_SPECIAL.sub(r"\\\1", value)
