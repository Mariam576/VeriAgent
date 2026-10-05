"""Small helpers for showing numbers and labels to people."""

from __future__ import annotations

_CURRENCY_SYMBOLS = {"GBP": "£", "USD": "$", "EUR": "€", "INR": "₹", "JPY": "¥"}
# Shown as "PKR 1,200" because they have no single well-known symbol.
_CURRENCY_CODES = {"PKR", "AED", "SAR", "CAD", "AUD", "CHF", "CNY", "SGD", "ZAR", "NZD", "SEK", "NOK", "DKK"}

_KEY_LABELS = {
    "total_amount": "total amount",
    "payment_terms_days": "payment terms",
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
