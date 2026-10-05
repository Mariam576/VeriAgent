"""Indicators, and how they are combined into one score.

Concept - combining weak signals:
    Each indicator has a weight between 0 and 1: how strongly it suggests on
    its own that the file was altered. Two rules keep the result honest:

    1. Within one category only the strongest indicator counts. "Saved again
       after it was created", "modified date later than created date" and
       "produced by an editing tool" are three views of the same fact, so they
       must not add up.
    2. Across categories the weights combine as  1 - (1 - a)(1 - b)...
       Two independent signs are stronger than either one, but the score
       never reaches 1: there is no certainty here.

    The result is a score, not a probability. A probability would have to be
    measured on thousands of documents that are known to be genuine or forged.
    These weights are set by reasoning, and the report says "risk indicator".
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Raised whenever the checks change, so that stored reports made by an older version are made again
# (app/forensics/store.py). 2: visibility is decided by comparing pictures of the page.
FORENSICS_VERSION = 2

Category = Literal["content", "typography", "consistency", "history"]
Level = Literal["none", "low", "medium", "high"]

MEDIUM_SCORE = 0.30
HIGH_SCORE = 0.60
MAX_SCORE = 0.95
MAX_INDICATORS = 12
MAX_REGION_CHARS = 160

# The weight of each indicator type and the category it belongs to.
WEIGHTS: dict[str, tuple[Category, float]] = {
    # --- content: the file holds text that a reader does not see, or text printed over text
    "hidden_instruction": ("content", 0.85),  # hidden text that addresses an AI system
    "replaced_text": ("content", 0.70),  # text covered by a shape, with other text printed on top
    "hidden_text": ("content", 0.55),  # invisible, background-coloured or microscopic text
    "covered_text": ("content", 0.50),  # text covered by a shape or an image
    "overlapping_text": ("content", 0.45),  # different characters printed on the same spot
    "text_over_scan": ("content", 0.40),  # a few typed words on top of a scanned page
    "off_page_text": ("content", 0.30),  # text placed outside the visible page
    # --- typography: a value set in a font that the rest of the document does not use
    "font_mismatch": ("typography", 0.35),
    # --- consistency: the document contradicts itself
    "conflicting_field": ("consistency", 0.35),
    # --- history: the file was changed after it was first written. Common and usually innocent,
    #     so these stay below the "medium" level on their own and only strengthen other indicators.
    "editing_tool": ("history", 0.25),
    "tracked_changes": ("history", 0.25),
    "metadata_inconsistent": ("history", 0.25),
    "incremental_saves": ("history", 0.20),
    "modified_after_creation": ("history", 0.15),
    "different_last_editor": ("history", 0.10),
}


# Indicators whose ``region`` is text that a reader of the page cannot see. That text is shown
# to the person who reviews the report, and to nobody else: it is never given to the language
# model and its numbers never count as known facts. Whoever hid a sentence in a file wrote it
# for a machine to find.
HIDDEN_CONTENT = frozenset({"hidden_instruction", "hidden_text", "covered_text", "replaced_text", "off_page_text"})


class Indicator(BaseModel):
    type: str
    category: Category
    weight: float
    reason: str  # one sentence a person can act on
    page: int | None = None
    region: str | None = None  # the text concerned, cut to a short length. Untrusted: taken from the document.


class ForensicReport(BaseModel):
    version: int = FORENSICS_VERSION
    analysed: bool = True  # False: this kind of file cannot be inspected (plain text) or the analysis failed
    score: float = 0.0
    level: Level = "none"
    indicators: list[Indicator] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)  # author, producer, dates ... as found in the file
    note: str | None = None
    # Things a reader of an investigation should know about how this file was read. They are not
    # signs of editing and do not change the score: "these pages are scans", "this page could not be examined".
    caveats: list[str] = Field(default_factory=list)


def indicator(kind: str, reason: str, *, page: int | None = None, region: str | None = None) -> Indicator:
    category, weight = WEIGHTS[kind]
    cleaned = " ".join(region.split())[:MAX_REGION_CHARS] if region else None
    return Indicator(type=kind, category=category, weight=weight, reason=reason, page=page, region=cleaned or None)


def score_of(indicators: list[Indicator]) -> float:
    strongest: dict[str, float] = {}
    for item in indicators:
        strongest[item.category] = max(strongest.get(item.category, 0.0), item.weight)
    remaining = 1.0
    for weight in strongest.values():
        remaining *= 1.0 - weight
    return round(min(1.0 - remaining, MAX_SCORE), 2)


def level_of(score: float) -> Level:
    if score >= HIGH_SCORE:
        return "high"
    if score >= MEDIUM_SCORE:
        return "medium"
    return "low" if score > 0 else "none"


def build_report(
    indicators: list[Indicator], metadata: dict[str, Any] | None = None, note: str | None = None, caveats: list[str] | None = None
) -> ForensicReport:
    ordered = sorted(indicators, key=lambda item: -item.weight)[:MAX_INDICATORS]
    score = score_of(ordered)
    return ForensicReport(
        score=score, level=level_of(score), indicators=ordered, metadata=metadata or {}, note=note, caveats=caveats or [],
    )


def page_list(numbers: list[int]) -> str:
    """[1, 2, 3, 7] -> "pages 1-3 and 7"."""
    runs: list[list[int]] = []
    for number in sorted(set(numbers)):
        if runs and number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    parts = [str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs]
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"{'page' if len(set(numbers)) == 1 else 'pages'} {joined}"
