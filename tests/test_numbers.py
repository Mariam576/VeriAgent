"""Number and unit parsing in the formats real business documents use."""

from __future__ import annotations

import pytest

from app.numbers import normalize_unit, number_appears, numbers_in_text, parse_number


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("40,000", [40000.0]),
        ("1,234,567.89", [1234567.89]),
        ("1,50,000", [150000.0]),  # lakh grouping (Pakistan, India)
        ("12,50,000.75", [1250000.75]),
        ("40.000,00", [40000.0]),  # European
        ("1.234.567", [1234567.0]),
        ("40 000", [40000.0]),  # space grouping
        ("40 000,50", [40000.5]),  # non-breaking space, decimal comma
        ("40000", [40000.0]),
        ("40,5", [40.5]),
        ("7.3", [7.3]),
        ("40.000", [40000.0, 40.0]),  # ambiguous: both readings are returned
        ("abc", []),
        ("", []),
    ],
)
def test_parse_number(token, expected):
    assert parse_number(token) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The total contract value is £40,000 payable within 30 days.", [40000.0, 30.0]),
        ("Total: 1,234,567.89 for INV-102", [1234567.89, 102.0]),
        ("Rs.1,50,000 and Rs 1,58,000.50", [150000.0, 158000.5]),
        ("EUR 40.000,00", [40000.0]),
        ("Clause 7.3 on 3 March 2026", [7.3, 3.0, 2026.0]),
        ("30 2026", [30.0, 2026.0]),  # two numbers, not "30 202" + "6"
        ("version v2 of file1", []),  # digits glued to words are not numbers
        ("£58,000.", [58000.0]),  # the full stop ends the sentence
        ("no digits", []),
    ],
)
def test_numbers_in_text(text, expected):
    assert numbers_in_text(text) == expected


def test_number_appears_in_any_supported_format():
    assert number_appears(150000, "Total: Rs. 1,50,000 only")
    assert number_appears(40000, "EUR 40.000,00")
    assert number_appears(40000, "40 000 EUR")
    assert number_appears(40000, "£40,000.00")
    assert number_appears(1234.5, "USD 1,234.50")
    assert number_appears(30, "within 30 days")
    assert not number_appears(58000, "£46,000 and £12,000")
    assert not number_appears(30, "within 300 days")
    assert not number_appears(40000, "no numbers here")


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("£", "GBP"), ("GBP", "GBP"), ("gbp", "GBP"), ("pounds sterling", "GBP"), ("British pounds", "GBP"),
        ("$", "USD"), ("US$", "USD"), ("dollars", "USD"), ("€", "EUR"), ("euros", "EUR"),
        ("Rs.", "PKR"), ("rupees", "PKR"), ("PKR", "PKR"), ("Indian rupees", "INR"),
        ("days", "days"), ("day", "days"), ("calendar days", "days"), ("Working Days", "days"),
        ("chf", "CHF"), ("%", "%"), ("km", "km"), (None, None), ("  ", None),
    ],
)
def test_normalize_unit(unit, expected):
    assert normalize_unit(unit) == expected
