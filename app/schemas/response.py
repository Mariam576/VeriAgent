"""API request and response shapes for investigations and chat."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.investigation import RiskLevel

FindingType = Literal["fact", "contradiction", "missing_evidence", "risk", "calculation", "recommendation"]

STATUS_LABELS = {
    "review_required": "Review Required",
    "no_issues_found": "No Issues Found",
    "insufficient_evidence": "Insufficient Evidence",
    "incomplete": "Incomplete",
    "not_applicable": "Not Applicable",
    "running": "Running",
    "interrupted": "Interrupted",
    "failed": "Failed",
}


class CitationOut(BaseModel):
    document: str
    page: int
    document_id: str | None = None
    # The exact words on that page that support the statement (set for findings, not for the source list).
    quote: str | None = None


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
    key: str = ""
    # Numbers for amounts and quantities, text for names, reference numbers and dates.
    value_a: float | None = None
    value_b: float | None = None
    unit_a: str | None = None
    unit_b: str | None = None
    text_a: str | None = None
    text_b: str | None = None
    source_a: str
    source_b: str
    difference: float | None = None
    percentage_change: float | None = None
    citations: list[CitationOut] = Field(default_factory=list)


class IntegrityIndicatorOut(BaseModel):
    type: str
    reason: str
    page: int | None = None
    region: str | None = None  # the text concerned, as found in the file (untrusted)


class DocumentIntegrityOut(BaseModel):
    """Signs that one file may have been altered. A risk indicator, never proof."""

    id: str  # "F1", "F2", ...
    document: str
    document_id: str | None = None
    level: str  # medium | high
    score: float
    indicators: list[IntegrityIndicatorOut] = Field(default_factory=list)


class ActionOut(BaseModel):
    """Something to be done outside VeriAgent, and what became of it (see app/actions)."""

    id: str  # "A1"
    kind: str  # create_issue
    server: str  # tickets | github
    target: str  # where it goes, in words
    title: str
    body: str | None = None  # the exact text that is (or was) sent
    # proposed = waiting for a person; not_approved = the person did not tick it;
    # done / failed = carried out, or tried; unknown = started, result never recorded
    status: Literal["proposed", "not_approved", "done", "failed", "unknown"] = "proposed"
    reference: str | None = None  # "T-1A2B3C4D", "#17"
    url: str | None = None
    detail: str | None = None


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


class RecalledMemory(BaseModel):
    """A note from an earlier session that is related to this question. Context, not evidence."""

    text: str
    source: str = "investigation"  # investigation | user
    similarity: float = 0.0
    created_at: str | None = None
    investigation_id: str | None = None


class MemoryInfo(BaseModel):
    # What this request added to long-term memory, as sentences for the user.
    saved: list[str] = Field(default_factory=list)
    recalled: list[RecalledMemory] = Field(default_factory=list)


class Presentation(BaseModel):
    """The user's preferences that shaped ``answer_text`` (from structured long-term memory)."""

    detail_level: Literal["brief", "standard", "detailed"] = "standard"
    response_format: Literal["text", "markdown"] = "text"


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
    document_integrity: list[DocumentIntegrityOut] = Field(default_factory=list)
    actions: list[ActionOut] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    calculations: list[CalculationOut] = Field(default_factory=list)
    risk_reasons: list[FindingOut] = Field(default_factory=list)
    recommended_action: str = ""
    sources: list[CitationOut] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stats: ReportStats = Field(default_factory=ReportStats)
    memory: MemoryInfo = Field(default_factory=MemoryInfo)
    presentation: Presentation = Field(default_factory=Presentation)
    # The same content as one readable text block (what a chat window shows).
    answer_text: str = ""

    @model_validator(mode="before")
    @classmethod
    def _reports_from_version_1(cls, data: Any) -> Any:
        """Version 1 stored only a yes/no flag. Keep its meaning when such a report is read back."""
        if isinstance(data, dict) and "human_review" not in data and data.get("requires_human_review"):
            return {**data, "human_review": {"required": True, "status": "not_requested"}}
        return data


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
    approved_actions: list[str] = Field(
        default_factory=list, max_length=5,
        description=(
            "Ids of the proposed follow-up actions to carry out (see 'actions' in the pending report), for example "
            "[\"A1\"]. Nothing outside VeriAgent is done unless it is listed here. Ignored for 'reject'."
        ),
    )


# ----------------------------------------------------------------- listings
class InvestigationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    question: str
    status: str
    approval_status: str = "not_required"
    error: str | None = None  # why it failed or was interrupted, when it did
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
