"""Schemas for long-term memory: the LLM's extraction output and the API payloads."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ----------------------------------------------------------- LLM output schema
class MemoryCandidate(BaseModel):
    """Is there something in the user's message that is worth remembering? (spec section 27)

    Every field is required (nullable where it may be absent), because some
    providers reject structured-output schemas with optional fields.
    """

    should_save: bool = Field(
        description=(
            "True only if the message states something durable that will be useful in later sessions: "
            "a standing preference, or a lasting fact about the user's vendors, contracts or past findings. "
            "False for questions, requests, greetings and anything that only matters right now."
        )
    )
    memory_type: Literal["profile", "semantic", "none"] = Field(
        description=(
            "profile = a preference about how answers should look (stored as key/value); "
            "semantic = a lasting fact in free text (stored with an embedding); none = nothing to save."
        )
    )
    key: str | None = Field(description="For profile: the preference name, exactly one of the allowed keys. Otherwise null.")
    value: str | None = Field(description="For profile: the preference value, exactly one of the allowed values. Otherwise null.")
    text: str | None = Field(
        description="For semantic: the fact as one self-contained sentence in the third person. Otherwise null."
    )
    only_memory: bool = Field(
        description="True when the message contains nothing else to answer or investigate besides the thing to remember."
    )


# ------------------------------------------------------------------ API models
class PreferenceInfo(BaseModel):
    key: str
    description: str
    allowed_values: list[str]
    default: str
    value: str  # the stored value, or the default
    is_set: bool


class PreferenceUpdate(BaseModel):
    value: str = Field(min_length=1, max_length=50)


class MemoryNoteCreate(BaseModel):
    text: str = Field(min_length=10, max_length=500, description="A lasting fact to remember, as one sentence.")


class MemoryNote(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    text: str
    source: str  # investigation | user
    investigation_id: uuid.UUID | None = None
    created_at: datetime


class MemoryOverview(BaseModel):
    enabled: bool
    preferences: list[PreferenceInfo]
    notes: list[MemoryNote]


class RecalledNote(BaseModel):
    id: str
    text: str
    source: str
    similarity: float
    created_at: str | None = None
    investigation_id: str | None = None


class MemorySearchResult(BaseModel):
    query: str
    results: list[RecalledNote]
