"""Request / response shapes for the stand-alone document comparison."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.response import CitationOut

MAX_COMPARED_DOCUMENTS = 6


class ComparisonRequest(BaseModel):
    document_ids: list[uuid.UUID] = Field(min_length=1, max_length=MAX_COMPARED_DOCUMENTS)


class ComparedDocument(BaseModel):
    document_id: str
    filename: str
    document_type: str
    profiled: bool  # False: this type of document has no profile (only contracts, purchase orders, invoices do)
    built_by: str | None = None  # "llm" or "rules"


class CellOut(BaseModel):
    value: str
    page: int
    quote: str


class RowOut(BaseModel):
    """One fact, across all compared documents (None where a document does not state it)."""

    field: str
    label: str
    cells: list[CellOut | None]
    differs: bool = False  # the documents that state this fact do not all agree


class LineItemOut(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None
    unit: str | None = None
    page: int
    quote: str


class DocumentLines(BaseModel):
    document_id: str
    filename: str
    items: list[LineItemOut]


class ComparisonFindingOut(BaseModel):
    type: str
    severity: str
    description: str
    key: str
    source_a: str
    source_b: str
    value_a: float | None = None
    value_b: float | None = None
    unit_a: str | None = None
    unit_b: str | None = None
    text_a: str | None = None
    text_b: str | None = None
    difference: float | None = None  # calculated by the calculator tool
    percentage_change: float | None = None
    citations: list[CitationOut] = Field(default_factory=list)


class ComparisonOut(BaseModel):
    documents: list[ComparedDocument]
    rows: list[RowOut]
    line_items: list[DocumentLines]
    findings: list[ComparisonFindingOut]
    notes: list[str] = Field(default_factory=list)


class ProfileOut(BaseModel):
    document_id: str
    filename: str
    document_type: str
    profile: dict[str, Any] | None = None
    note: str | None = None
