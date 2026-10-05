"""API request and response shapes for investigations and chat."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.investigation import RiskLevel

FindingType = Literal["fact", "contradiction", "missing_evidence", "risk", "calculation", "recommendation"]

STATUS_LABELS = {
    "review_required": "Review Required",
    "no_issues_found": "No Issues Found",
    "insufficient_evidence": "Insufficient Evidence",
    "incomplete": "Incomplete",
    "not_applicable": "Not Applicable",
    "running": "Running",
    "failed": "Failed",
}


class CitationOut(BaseModel):
    document: str
    page: int
    document_id: str | None = None


class FindingOut(BaseModel):
    type: FindingType
    statement: str
    severity: str | None = None
    confidence: float | None = None
    citations: list[CitationOut] = Field(default_factory=list)


class ContradictionOut(BaseModel):
    id: str
    type: str
    severity: str
    description: str
    value_a: float
    value_b: float
    source_a: str
    source_b: str
    difference: float | None = None
    percentage_change: float | None = None
    citations: list[CitationOut] = Field(default_factory=list)


class CalculationOut(BaseModel):
    id: str
    description: str
    tool_name: str
    arguments: dict[str, Any]
    result: float | None
    requested_by: str


ApprovalStatus = Literal["not_required", "not_requested", "pending", "approved", "rejected", "modified"]


class HumanReview(BaseModel):
    """Human-in-the-loop state of a report.

    not_required   nothing in the findings calls for a person
    not_requested  a person should look at it, but pausing for approval is switched off
    pending        the investigation is paused and waits for a decision
    approved / rejected / modified   the decision a person made
    """

    required: bool = False
    status: ApprovalStatus = "not_required"
    reviewer: str | None = None
    note: str | None = None
    decided_at: str | None = None
    # The agent's own recommendation, kept when a person rejected or replaced it.
    original_action: str | None = None


class ReportStats(BaseModel):
    iterations: int = 0
    llm_calls: int = 0
    retrieval_rounds: int = 0
    retrieved_chunks: int = 0
    evidence_items: int = 0
    discarded_facts: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0
    llm_provider: str = ""
    # Token usage as reported by the provider (0 when the provider does not report it).
    input_tokens: int = 0
    output_tokens: int = 0


class InvestigationReport(BaseModel):
    investigation_id: str
    question: str
    status: str
    status_label: str
    risk_level: RiskLevel
    confidence: float
    evidence_quality: Literal["strong", "moderate", "weak", "none"]
    requires_human_review: bool
    human_review: HumanReview = Field(default_factory=HumanReview)
    summary: str
    key_findings: list[FindingOut] = Field(default_factory=list)
    contradictions: list[ContradictionOut] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    calculations: list[CalculationOut] = Field(default_factory=list)
    risk_reasons: list[FindingOut] = Field(default_factory=list)
    recommended_action: str = ""
    sources: list[CitationOut] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stats: ReportStats = Field(default_factory=ReportStats)
    # The same content as one readable text block (what a chat window shows).
    answer_text: str = ""


# ------------------------------------------------------------------ requests
class InvestigationRequest(BaseModel):
    question: str = Field(min_length=1, examples=["Review this invoice against the contract and tell me whether it should be approved."])
    document_ids: list[uuid.UUID] = Field(
        default_factory=list, description="Optional: only search these documents. Empty = all your documents."
    )
    document_type: str | None = Field(
        default=None, description="Optional metadata filter: only search documents of this type, e.g. 'invoice'."
    )
    year: int | None = Field(
        default=None, ge=1980, le=2100, description="Optional metadata filter: only search documents from this year."
    )


class ChatRequest(BaseModel):
    conversation_id: uuid.UUID | None = Field(default=None, description="Omit to start a new conversation.")
    message: str = Field(min_length=1, examples=["Compare the invoice amount with the contract."])


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    investigation_id: uuid.UUID
    answer: str
    risk_level: RiskLevel
    confidence: float
    status: str
    requires_human_review: bool
    # "pending" means: call POST /investigations/{investigation_id}/approve to finish.
    approval_status: ApprovalStatus = "not_required"
    citations: list[CitationOut]


class ApprovalRequest(BaseModel):
    """A person's decision on a paused investigation."""

    decision: Literal["approve", "reject", "modify"] = Field(
        description="approve = accept the agent's recommendation; reject = do not accept it; modify = replace it."
    )
    note: str | None = Field(default=None, max_length=1000, description="Why. Shown in the final report.")
    modified_action: str | None = Field(
        default=None, max_length=1000, description="Required for 'modify': the recommendation to use instead."
    )


# ----------------------------------------------------------------- listings
class InvestigationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    question: str
    status: str
    approval_status: str = "not_required"
    risk_level: str | None = None
    confidence_score: float | None = None
    created_at: datetime
    completed_at: datetime | None = None


class InvestigationList(BaseModel):
    investigations: list[InvestigationSummary]


class InvestigationDetail(InvestigationSummary):
    report: InvestigationReport | None = None


class StoredFinding(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    finding_type: str
    statement: str
    severity: str | None = None
    confidence: float | None = None
    source_document_id: uuid.UUID | None = None
    page_number: int | None = None


class FindingList(BaseModel):
    investigation_id: uuid.UUID
    findings: list[StoredFinding]


class TraceEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run: int
    seq: int
    kind: str
    name: str
    latency_ms: float | None = None
    offset_ms: float = 0.0
    data: dict[str, Any] = Field(default_factory=dict)


class TraceOut(BaseModel):
    investigation_id: uuid.UUID
    totals: dict[str, Any]
    events: list[TraceEventOut]
