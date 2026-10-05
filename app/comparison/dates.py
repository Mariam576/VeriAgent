"""Reading dates out of document text.

Formats understood:

    3 March 2026      3rd March 2026     3 Mar 2026      day, month name, year
    March 3, 2026     Mar 3 2026                          month name, day, year
    2026-03-03                                            ISO
    03/03/2026        3.3.2026           03-03-2026       numeric

A numeric date is ambiguous when both numbers could be a month: 04/05/2026 is
4 May in Europe and 5 April in the United States. ``FoundDate`` then carries
both readings, and a check only reports a problem when every reading shows it.
Guessing one reading would produce false alarms on half of all documents.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
    "nov": 11, "dec": 12,
}
_MONTH = "(?P<month>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_DAY = r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(?P<year>(?:19|20)\d{2})"

_DAY_MONTH_YEAR = re.compile(rf"(?<![\d/.-]){_DAY}\s+(?:of\s+)?{_MONTH}\b\.?,?\s+{_YEAR}(?!\d)", re.I)
_MONTH_DAY_YEAR = re.compile(rf"(?<![a-z]){_MONTH}\b\.?\s+{_DAY},?\s+{_YEAR}(?!\d)", re.I)
_ISO = re.compile(r"(?<![\d/.-])(?P<year>(?:19|20)\d{2})-(?P<month>\d{2})-(?P<day>\d{2})(?![\d/-])")
_NUMERIC = re.compile(r"(?<![\d/.-])(?P<first>\d{1,2})(?P<sep>[/.-])(?P<second>\d{1,2})(?P=sep)" + _YEAR + r"(?![\d/-])")


@dataclass(frozen=True)
class FoundDate:
    readings: tuple[date, ...]  # one date, or two when a numeric date can be read both ways
    text: str  # the date as it is written in the document
    start: int
    end: int

    @property
    def ambiguous(self) -> bool:
        return len(self.readings) > 1


def _make(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None  # 31 February, month 13 ...


def find_dates(text: str) -> list[FoundDate]:
    """Every date written in ``text``, in reading order."""
    found: list[FoundDate] = []

    def add(match: re.Match[str], readings: list[date | None]) -> None:
        valid = tuple(dict.fromkeys(reading for reading in readings if reading is not None))
        if valid:
            found.append(FoundDate(valid, match.group(0), match.start(), match.end()))

    for pattern in (_DAY_MONTH_YEAR, _MONTH_DAY_YEAR):
        for match in pattern.finditer(text):
            month = _MONTHS[match.group("month").lower()]
            add(match, [_make(int(match.group("year")), month, int(match.group("day")))])
    for match in _ISO.finditer(text):
        add(match, [_make(int(match.group("year")), int(match.group("month")), int(match.group("day")))])
    for match in _NUMERIC.finditer(text):
        first, second, year = int(match.group("first")), int(match.group("second")), int(match.group("year"))
        add(match, [_make(year, second, first), _make(year, first, second)])  # day first, then month first

    # "3 March 2026" must not also be read as "March 2026 ..." by another pattern: keep the earliest, longest match.
    found.sort(key=lambda item: (item.start, -(item.end - item.start)))
    kept: list[FoundDate] = []
    for item in found:
        if not kept or item.start >= kept[-1].end:
            kept.append(item)
    return kept


def parse_date(text: str | None) -> FoundDate | None:
    """The first date written in ``text``, or None."""
    dates = find_dates(text or "")
    return dates[0] if dates else None


def format_date(value: date) -> str:
    """date(2026, 3, 3) -> '3 March 2026' (the same on every operating system)."""
    return f"{value.day} {calendar.month_name[value.month]} {value.year}"


def add_months(value: date, months: int) -> date:
    """The same day ``months`` later; the last day of the month if that day does not exist."""
    index = value.month - 1 + months
    year, month = value.year + index // 12, index % 12 + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


# ------------------------------------------------------------------ durations
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "eighteen": 18, "twenty-four": 24, "twenty four": 24, "thirty-six": 36,
    "thirty six": 36, "forty-eight": 48, "forty eight": 48, "sixty": 60,
}
_DURATION = re.compile(
    r"(?<![\w.,])(?P<count>\d{1,3}|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"
    r"(?:\s*\(\d{1,3}\))?\s+(?:calendar\s+)?(?P<unit>months?|years?)\b",
    re.I,
)


def months_in_text(text: str) -> list[tuple[int, int, int]]:
    """Durations written in months or years: ``(months, start, end)`` for each.

    'continues for twelve months' -> 12, 'a term of 2 years' -> 24, 'thirty-six (36) months' -> 36
    """
    found = []
    for match in _DURATION.finditer(text):
        raw = match.group("count").lower()
        count = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
        if match.group("unit").lower().startswith("year"):
            count *= 12
        found.append((count, match.start(), match.end()))
    return found
