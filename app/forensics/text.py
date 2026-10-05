"""A check that needs only the text: is a key field stated twice with different values?

An invoice has one "Total amount due". If the same label appears with two
different amounts, either a part of the document was changed and another was
forgotten, or the document is unusual. Both are worth a look.

Only documents that normally carry exactly one such total are checked. A
statement of account or a report may list many totals for good reasons.
"""

from __future__ import annotations

from app.formatting import format_value
from app.forensics.model import Indicator, indicator
from app.patterns import TOTAL_LABELS, find_amounts
from app.rag.loaders import PageText

SINGLE_TOTAL_TYPES = ("invoice", "purchase_order", "receipt", "quotation")
_FINAL_TOTAL = TOTAL_LABELS[0][0]  # "total amount due", "grand total", "balance due" ...


def conflicting_fields(pages: list[PageText], document_type: str) -> list[Indicator]:
    if document_type not in SINGLE_TOTAL_TYPES:
        return []
    stated: dict[str, dict[tuple[float, str], int]] = {}
    for page in pages:
        for hit in find_amounts(page.text):
            if hit.priority != 3:
                continue
            labels = list(_FINAL_TOTAL.finditer(hit.before))
            if labels:
                label = " ".join(labels[-1].group(0).lower().split())
                stated.setdefault(label, {}).setdefault((hit.value, hit.unit), page.page_number)
    found = []
    for label, values in stated.items():
        if len(values) > 1:
            listed = " and ".join(format_value(value, unit) for value, unit in values)
            found.append(indicator(
                "conflicting_field",
                f"'{label.capitalize()}' is stated with different values in the same document: {listed}.",
                page=next(iter(values.values())), region=f"{label}: {listed}",
            ))
    return found
