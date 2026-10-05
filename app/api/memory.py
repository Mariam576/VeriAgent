"""Long-term memory endpoints: see, change and delete what VeriAgent remembers about you.

Anything a system remembers about a person should be visible to that person
and removable by them. These endpoints are that window: preferences
(structured memory) and notes (semantic memory).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, Response

from app.config import get_settings
from app.errors import NotFound, ValidationFailed
from app.memory.extraction import clean_note
from app.memory.profile import PREFERENCES, validate_preference
from app.memory.store import get_memory_store
from app.schemas.memory import (
    MemoryNote,
    MemoryNoteCreate,
    MemoryOverview,
    MemorySearchResult,
    PreferenceInfo,
    PreferenceUpdate,
    RecalledNote,
)
from app.security.permissions import CurrentUser

router = APIRouter(prefix="/memory", tags=["memory"])


def _preference_list(stored: dict[str, str]) -> list[PreferenceInfo]:
    result = []
    for key, preference in PREFERENCES.items():
        is_set = stored.get(key) in preference.allowed_values
        result.append(
            PreferenceInfo(
                key=key, description=preference.description, allowed_values=list(preference.allowed_values),
                default=preference.default, value=stored[key] if is_set else preference.default, is_set=is_set,
            )
        )
    return result


@router.get("", response_model=MemoryOverview)
async def get_memory(user: CurrentUser) -> MemoryOverview:
    """Everything remembered about the current user."""
    store = get_memory_store()
    user_id = str(user.id)
    return MemoryOverview(
        enabled=get_settings().ENABLE_MEMORY,
        preferences=_preference_list(await store.get_preferences(user_id)),
        notes=[MemoryNote.model_validate(note) for note in await store.list_notes(user_id)],
    )


@router.put("/preferences/{key}", response_model=PreferenceInfo)
async def set_preference(key: str, request: PreferenceUpdate, user: CurrentUser) -> PreferenceInfo:
    """Set a preference, for example ``detail_level`` = ``brief``."""
    key, value = validate_preference(key, request.value)
    store = get_memory_store()
    await store.set_preference(str(user.id), key, value)
    return next(item for item in _preference_list({key: value}) if item.key == key)


@router.delete("/preferences/{key}", status_code=204)
async def reset_preference(key: str, user: CurrentUser) -> Response:
    """Forget a preference; its default applies again."""
    if key not in PREFERENCES:
        raise ValidationFailed(f"Unknown preference '{key}'. Available: {', '.join(PREFERENCES)}.")
    await get_memory_store().delete_preference(str(user.id), key)
    return Response(status_code=204)


@router.post("/notes", response_model=MemoryNote, status_code=201)
async def add_note(request: MemoryNoteCreate, user: CurrentUser) -> MemoryNote:
    """Remember a lasting fact, for example "Vendor ABC was flagged for duplicate invoices in 2025"."""
    text = clean_note(request.text, get_settings())
    if text is None:
        raise ValidationFailed("A note must be a statement of at least 10 characters, not a question.")
    store = get_memory_store()
    note_id = await store.add_note(str(user.id), text, source="user")
    note = await store.get_note(str(user.id), note_id)
    assert note is not None
    return MemoryNote.model_validate(note)


@router.get("/search", response_model=MemorySearchResult)
async def search_memory(
    user: CurrentUser,
    q: str = Query(min_length=2, max_length=500, description="What to look for, in your own words."),
    min_similarity: float | None = Query(default=None, ge=0.0, le=1.0),
) -> MemorySearchResult:
    """Semantic search over your notes: the same recall the agent uses, so you can see what it would find."""
    settings = get_settings()
    results = await get_memory_store().search_notes(
        str(user.id), q, top_k=settings.MEMORY_TOP_K,
        min_similarity=settings.MEMORY_MIN_SIMILARITY if min_similarity is None else min_similarity,
    )
    return MemorySearchResult(query=q, results=[RecalledNote(**item) for item in results])


@router.delete("/notes/{note_id}", status_code=204)
async def delete_note(note_id: uuid.UUID, user: CurrentUser) -> Response:
    if not await get_memory_store().delete_note(str(user.id), str(note_id)):
        raise NotFound("Note not found.")  # same answer whether it does not exist or belongs to someone else
    return Response(status_code=204)


@router.delete("", response_model=dict[str, int])
async def forget_everything(user: CurrentUser) -> dict[str, int]:
    """Delete all preferences and notes of the current user. Documents and investigations are not touched."""
    return await get_memory_store().forget_everything(str(user.id))
