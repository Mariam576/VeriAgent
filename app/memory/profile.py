"""Structured long-term memory: the user's standing preferences (spec section 25).

Concept - structured memory:
    A preference such as "keep answers brief" has a name and a value. It is
    stored in an ordinary table and read back by its key. No embedding and no
    vector search are involved: when you know exactly what you are looking
    for, a direct lookup is faster, cheaper and never returns the wrong row.

Every preference here changes something real. Adding one means adding an
entry to ``PREFERENCES`` and using it where the report is rendered.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.errors import ValidationFailed


@dataclass(frozen=True)
class Preference:
    description: str
    allowed_values: tuple[str, ...]
    default: str


PREFERENCES: dict[str, Preference] = {
    "detail_level": Preference(
        description=(
            "How much of the report is written out in the answer text: brief = verdict, contradictions and "
            "recommendation; standard = everything; detailed = also the calculations and run statistics."
        ),
        allowed_values=("brief", "standard", "detailed"),
        default="standard",
    ),
    "response_format": Preference(
        description="Layout of the answer text: text = plain text; markdown = headings and bold labels.",
        allowed_values=("text", "markdown"),
        default="text",
    ),
}


def normalize_preference(key: str | None, value: str | None) -> tuple[str, str] | None:
    """A (key, value) pair that is allowed, or None. Used for values proposed by the model."""
    key = (key or "").strip().lower().replace(" ", "_").replace("-", "_")
    value = (value or "").strip().lower()
    preference = PREFERENCES.get(key)
    if preference is None or value not in preference.allowed_values:
        return None
    return key, value


def validate_preference(key: str, value: str) -> tuple[str, str]:
    """Same check for values sent by a person through the API: explains what is wrong."""
    cleaned_key = key.strip().lower()
    preference = PREFERENCES.get(cleaned_key)
    if preference is None:
        raise ValidationFailed(f"Unknown preference '{key}'. Available: {', '.join(PREFERENCES)}.")
    cleaned_value = value.strip().lower()
    if cleaned_value not in preference.allowed_values:
        raise ValidationFailed(
            f"'{value}' is not a valid value for {cleaned_key}. Allowed: {', '.join(preference.allowed_values)}."
        )
    return cleaned_key, cleaned_value


def with_defaults(stored: dict[str, str] | None) -> dict[str, str]:
    """Every preference with its stored value, or its default. Unknown or invalid rows are ignored."""
    stored = stored or {}
    result = {}
    for key, preference in PREFERENCES.items():
        value = stored.get(key)
        result[key] = value if value in preference.allowed_values else preference.default
    return result


def describe(key: str, value: str) -> str:
    """A short confirmation sentence for the user."""
    return f"Saved your preference: {key.replace('_', ' ')} = {value}."
