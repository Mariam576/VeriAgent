"""Request / response shapes for documents."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.forensics.model import ForensicReport


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    document_type: str
    uploaded_at: datetime
    page_count: int
    status: str
    chunk_count: int = 0
    error: str | None = None
    # True when the same file had already been uploaded and was not processed again.
    duplicate: bool = False
    # Signs that the file was altered: none | low | medium | high. None = not inspected.
    integrity_level: str | None = None
    integrity_score: float | None = None


class ForensicsOut(BaseModel):
    document_id: str
    filename: str
    report: ForensicReport | None = None
    note: str | None = None


class DocumentList(BaseModel):
    documents: list[DocumentOut]
