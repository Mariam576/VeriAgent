"""Data shapes used inside an investigation.

Two groups live here:

1. Domain objects the pipeline produces itself (retrieved chunks, evidence
   items, contradictions, calculations). Source fields such as ``filename``
   and ``page`` on these objects are always copied from the database by code.
   The LLM never supplies them, which is why it cannot invent a citation.

2. LLM output schemas (``RequestAnalysis``, ``EvidenceExtraction``,
   ``SufficiencyAssessment``, ``RiskAssessment``). The model must fill in
   these typed fields; nothing important is parsed out of free prose.
   The ``Field(description=...)`` texts are sent to the model as part of the
   schema, so they double as instructions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high", "critical"]
Severity = Literal["low", "medium", "high"]

RISK_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# Comparison keys with a fixed meaning. The model may also invent its own
# snake_case keys for other quantities that should agree across documents.
KEY_TOTAL_AMOUNT = "total_amount"
KEY_PAYMENT_TERMS = "payment_terms_days"
KEY_OTHER = "other"


# ============================================================ domain objects
class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    document_type: str = "other"
    page: int
    chunk_index: int = 0
    text: str
    score: float = 0.0  # vector similarity, 1.0 = identical direction
    rerank_score: float = 0.0
    ref: str = ""  # short alias used in prompts, e.g. "C3"
    injection_suspected: bool = False


class EvidenceItem(BaseModel):
    id: str  # "E1", "E2", ...
    claim: str
    key: str = KEY_OTHER
    value_number: float | None = None
    unit: str | None = None
    quote: str = ""
    extracted_by: Literal["llm", "rules"] = "llm"
    # --- source, copied from the retrieved chunk by code
    chunk_id: str
    chunk_ref: str = ""
    document_id: str
    filename: str
    document_type: str = "other"
    page: int


class Calculation(BaseModel):
    id: str  # "K1", "K2", ...
    tool_name: str
    arguments: dict[str, Any]
    result: float | None = None
    label: str = ""
    status: Literal["ok", "error", "rejected"] = "ok"
    error: str | None = None
    requested_by: Literal["system", "llm"] = "system"


class Contradiction(BaseModel):
    id: str  # "X1", "X2", ...
    type: str  # amount_mismatch | payment_terms_mismatch | value_mismatch | unit_mismatch | internal_inconsistency
    key: str
    severity: Severity
    description: str
    value_a: float
    value_b: float
    unit_a: str | None = None
    unit_b: str | None = None
    source_a: str
    source_b: str
    difference: float | None = None  # value_b - value_a, filled in by the calculator tool
    percentage_change: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """What was asked of a tool and what came back. Stored in the tool_calls table."""

    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    status: Literal["ok", "error", "rejected"]
    latency_ms: float = 0.0
    error: str | None = None


# ========================================================= LLM output schemas
class RequestAnalysis(BaseModel):
    """Node 1 output: what does the user want?"""

    task_type: Literal["document_investigation", "document_question", "calculation", "general"] = Field(
        description=(
            "document_investigation = review/compare/verify documents; document_question = look up a fact "
            "in the documents; calculation = mainly arithmetic on document values; general = greeting or a "
            "question that is not about the uploaded documents."
        )
    )
    needs_rag: bool = Field(description="True when answering requires searching the uploaded documents.")
    needs_calculation: bool = Field(description="True when amounts, differences, totals or percentages matter.")
    needs_external_verification: bool = Field(
        default=False, description="True only if the answer would need information from outside the documents."
    )
    decision_requested: bool = Field(
        default=False,
        description="True when the user asks whether something should be approved, paid, accepted or rejected.",
    )
    standalone_question: str = Field(
        description="The user's request rewritten so it is understandable without the earlier conversation."
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description=(
            "1 to 4 short search queries for finding the relevant passages. Use one query per topic or "
            "per document, e.g. 'contract total value', 'invoice total amount', 'payment terms'."
        ),
    )


class ExtractedFact(BaseModel):
    chunk_ref: str = Field(description="The reference of the chunk the fact comes from, exactly as given, e.g. 'C2'.")
    claim: str = Field(description="One factual statement, in plain words, that the chunk supports.")
    key: str = Field(
        default=KEY_OTHER,
        description=(
            "'total_amount' for the overall total value/price/amount due of the document; "
            "'payment_terms_days' for the number of days allowed for payment; another short snake_case "
            "name for any other quantity that the same quantity in a different document should match; "
            "otherwise 'other'."
        ),
    )
    value_number: float | None = Field(
        default=None, description="The numeric value of the fact if it has one (40000 for '£40,000'), else null."
    )
    unit: str | None = Field(
        default=None, description="ISO currency code (GBP, USD, EUR...) for money, 'days' for durations, else null."
    )
    quote: str = Field(description="A short passage copied word for word from the chunk that supports the claim.")


class EvidenceExtraction(BaseModel):
    """Node 3 output: factual claims found in the retrieved chunks."""

    facts: list[ExtractedFact] = Field(default_factory=list)


class SufficiencyAssessment(BaseModel):
    """Node 6 output: is there enough evidence to answer?"""

    sufficient: bool = Field(description="True only if the evidence is enough to answer the question reliably.")
    missing: list[str] = Field(
        default_factory=list,
        description="Each piece of evidence that would be needed but was not found, e.g. 'approved purchase order'.",
    )
    follow_up_queries: list[str] = Field(
        default_factory=list,
        description="Up to 3 new search queries that might find the missing evidence. Empty if none would help.",
    )


class RiskReason(BaseModel):
    text: str = Field(description="One reason for the risk level, in one sentence.")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Ids of the items that support this reason (E.. evidence, X.. contradiction, K.. calculation, M.. missing evidence).",
    )


class RiskAssessment(BaseModel):
    """Node 7 output: how risky is the situation, and why?"""

    level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0, description="How sure you are, between 0 and 1.")
    reasons: list[RiskReason] = Field(default_factory=list)
    decision: Literal["proceed", "review", "reject"] = Field(
        description="proceed = no issue found; review = a person must look at it first; reject = should not go ahead."
    )
    recommended_action: str = Field(description="One or two sentences telling the user what to do next.")
    summary: str = Field(description="A direct answer to the user's question in at most three sentences.")
