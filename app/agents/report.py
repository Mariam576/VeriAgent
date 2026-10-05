"""Node 10 helpers: assemble the final structured report.

The report is built by code from the structured findings in the state. The
only free text that came from the model (summary, recommended action, risk
reasons) has already passed the reviewer. Citations are attached here from
the evidence records, never written by the model.
"""

from __future__ import annotations

from typing import Any

from app.agents.assessment import evidence_quality
from app.formatting import format_number, format_value, key_label
from app.memory.profile import with_defaults
from app.rag.citations import citations_from_evidence, format_citation
from app.schemas.investigation import (
    KEY_OTHER,
    KEY_PAYMENT_TERMS,
    KEY_TOTAL_AMOUNT,
    RISK_ORDER,
    Calculation,
    Contradiction,
    EvidenceItem,
)
from app.schemas.response import (
    STATUS_LABELS,
    CalculationOut,
    CitationOut,
    ContradictionOut,
    FindingOut,
    HumanReview,
    InvestigationReport,
    MemoryInfo,
    Presentation,
    RecalledMemory,
    ReportStats,
)

MAX_OTHER_FACTS = 6
MAX_QUOTE_CHARS = 300
_KEY_ORDER = {KEY_TOTAL_AMOUNT: 0, KEY_PAYMENT_TERMS: 1}
_DOC_ORDER = {"contract": 0, "purchase_order": 1, "quotation": 2, "invoice": 3, "receipt": 4, "policy": 5}

CAPABILITIES = (
    "I am VeriAgent, an evidence-verification agent. Upload business documents (contracts, invoices, "
    "policies, purchase orders ...) and ask me to review them, compare them, find contradictions or check "
    "whether there is enough evidence to approve something. Every finding comes with its source page."
)


def _cite(item: EvidenceItem) -> CitationOut:
    # The quote was checked against the chunk when the fact was extracted, so it is safe to show.
    return CitationOut(
        document=item.filename, page=item.page, document_id=item.document_id,
        quote=item.quote[:MAX_QUOTE_CHARS] or None,
    )


def _cite_suspicious(item: dict[str, Any]) -> CitationOut:
    return CitationOut(document=item["document"], page=item["page"], document_id=item.get("document_id"))


def _unique(citations: list[CitationOut]) -> list[CitationOut]:
    # Two different files may share a name, so the document id is part of the identity.
    seen: set[tuple[str, int]] = set()
    result = []
    for citation in citations:
        identity = (citation.document_id or citation.document, citation.page)
        if identity not in seen:
            seen.add(identity)
            result.append(citation)
    return result


def determine_status(
    *,
    task_type: str,
    evidence: list[EvidenceItem],
    contradictions: list[Contradiction],
    missing_evidence: list[str],
    suspicious: list[dict[str, Any]],
    risk_level: str,
    iteration_limit_reached: bool,
) -> str:
    if task_type == "general":
        return "not_applicable"
    if not evidence:
        return "insufficient_evidence"
    if iteration_limit_reached and missing_evidence:
        return "incomplete"
    if contradictions or missing_evidence or suspicious or RISK_ORDER[risk_level] >= RISK_ORDER["medium"]:
        return "review_required"
    return "no_issues_found"


def needs_human_review(
    *, enabled: bool, status: str, risk_level: str, confidence: float, missing_evidence: list[str],
    decision_requested: bool,
) -> bool:
    """Spec section 20, Node 9: when must a person look at this before anything is done?"""
    if not enabled or status == "not_applicable":
        return False
    return (
        RISK_ORDER[risk_level] >= RISK_ORDER["high"]  # risk is high
        or confidence < 0.6  # confidence is low
        or bool(missing_evidence)  # evidence is incomplete
        or decision_requested  # the user asks for an approval-type decision
        or status in ("insufficient_evidence", "incomplete")
    )


def _calculation_sentence(contradiction: Contradiction) -> str | None:
    if contradiction.difference is None:
        return None
    unit = contradiction.unit_b or contradiction.unit_a
    amount = format_value(abs(contradiction.difference), unit)
    direction = "above" if contradiction.difference > 0 else "below"
    sentence = (
        f"Difference in {key_label(contradiction.key)}: {amount}. {contradiction.source_b} is {amount} "
        f"{direction} {contradiction.source_a}"
    )
    if contradiction.percentage_change is not None:
        change = contradiction.percentage_change
        sentence += f" ({format_value(abs(change), '%')} {'more' if change > 0 else 'less'})"
    return sentence + "."


def human_review_block(required: bool, approval: dict[str, Any] | None, agent_action: str) -> tuple[HumanReview, str]:
    """Combine "is a person needed?" with the decision a person made, if any.

    Returns the review block and the recommended action to show. A person can
    replace the agent's recommendation (modify) or refuse it (reject); the
    agent's own wording is then kept in ``original_action`` for the record.
    """
    approval = approval or {}
    status = approval.get("status") or ("not_requested" if required else "not_required")
    review = HumanReview(
        required=required, status=status, reviewer=approval.get("reviewer"),
        note=approval.get("note"), decided_at=approval.get("decided_at"),
    )
    action = agent_action
    if status == "modified" and approval.get("modified_action"):
        review.original_action = agent_action
        action = approval["modified_action"]
    elif status == "rejected":
        review.original_action = agent_action
        action = "The reviewer rejected the agent's recommendation." + (f" Reason: {review.note}" if review.note else "")
    return review, action


def build_report(state: dict[str, Any], *, human_approval_enabled: bool, llm_provider: str, duration_ms: float) -> InvestigationReport:
    evidence = [EvidenceItem(**item) for item in state.get("evidence", [])]
    contradictions = [Contradiction(**item) for item in state.get("contradictions", [])]
    calculations = [Calculation(**item) for item in state.get("calculations", [])]
    missing = list(state.get("missing_evidence", []))
    suspicious = list(state.get("risk_findings", []))
    analysis = state.get("analysis", {})
    draft = state.get("risk_draft", {})
    by_id = {item.id: item for item in evidence}

    risk_level = draft.get("level", "low")
    confidence = float(draft.get("confidence", 0.0))
    status = determine_status(
        task_type=analysis.get("task_type", "document_investigation"),
        evidence=evidence, contradictions=contradictions, missing_evidence=missing, suspicious=suspicious,
        risk_level=risk_level, iteration_limit_reached=bool(state.get("iteration_limit_reached")),
    )
    human_review = needs_human_review(
        enabled=human_approval_enabled, status=status, risk_level=risk_level, confidence=confidence,
        missing_evidence=missing, decision_requested=bool(analysis.get("decision_requested")),
    )

    # ----- key findings: typed facts, then calculations, then other facts, then gaps
    typed = sorted(
        (item for item in evidence if item.key != KEY_OTHER),
        key=lambda item: (_KEY_ORDER.get(item.key, 9), item.key, _DOC_ORDER.get(item.document_type, 9), item.filename),
    )
    others = [item for item in evidence if item.key == KEY_OTHER][:MAX_OTHER_FACTS]
    findings: list[FindingOut] = [
        FindingOut(type="fact", statement=item.claim, confidence=confidence, citations=[_cite(item)])
        for item in typed
    ]
    for contradiction in contradictions:
        sentence = _calculation_sentence(contradiction)
        if sentence:
            findings.append(
                FindingOut(
                    type="calculation", statement=sentence, severity=contradiction.severity,
                    citations=_unique([_cite(by_id[i]) for i in contradiction.evidence_ids if i in by_id]),
                )
            )
    findings += [
        FindingOut(type="fact", statement=item.claim, confidence=confidence, citations=[_cite(item)])
        for item in others
    ]
    findings += [
        FindingOut(type="missing_evidence", statement=f"Supporting evidence was not found: {text}", severity="medium")
        for text in missing
    ]

    contradictions_out = [
        ContradictionOut(
            id=item.id, type=item.type, severity=item.severity, description=item.description,
            value_a=item.value_a, value_b=item.value_b, source_a=item.source_a, source_b=item.source_b,
            difference=item.difference, percentage_change=item.percentage_change,
            citations=_unique([_cite(by_id[i]) for i in item.evidence_ids if i in by_id]),
        )
        for item in contradictions
    ]
    calculations_out = [
        CalculationOut(
            id=item.id, description=item.label, tool_name=item.tool_name, arguments=item.arguments,
            result=item.result, requested_by=item.requested_by,
        )
        for item in calculations
        if item.status == "ok"
    ]

    # ----- risk reasons (inferences) with the citations of the evidence they point to
    suspicious_by_id = {item["id"]: item for item in suspicious}
    contradiction_by_id = {item.id: item for item in contradictions}
    reasons_out = []
    for reason in draft.get("reasons", []):
        cited: list[CitationOut] = []
        for identifier in reason.get("evidence_ids", []):
            if identifier in by_id:
                cited.append(_cite(by_id[identifier]))
            elif identifier in contradiction_by_id:
                cited += [_cite(by_id[i]) for i in contradiction_by_id[identifier].evidence_ids if i in by_id]
            elif identifier in suspicious_by_id:
                cited.append(_cite_suspicious(suspicious_by_id[identifier]))
        reasons_out.append(FindingOut(type="risk", statement=reason["text"], severity=risk_level, citations=_unique(cited)))

    sources = [
        CitationOut(document=c.document, page=c.page, document_id=c.document_id)
        for c in citations_from_evidence(evidence)
    ]
    sources = _unique(sources + [_cite_suspicious(item) for item in suspicious])

    memory = MemoryInfo(
        saved=list(state.get("memory_notes", [])),
        recalled=[RecalledMemory(**item) for item in state.get("memories", [])],
    )
    if status == "not_applicable" and memory.saved:
        # The message only asked to remember something: confirm it instead of introducing the agent.
        summary, action = " ".join(memory.saved), "Ask a question about your documents whenever you are ready."
    elif status == "not_applicable":
        summary, action = CAPABILITIES, "Upload documents and ask a question about them."
    else:
        summary, action = draft.get("summary", ""), draft.get("recommended_action", "")
    if status == "insufficient_evidence" and memory.recalled:
        # Memory is not evidence, but it is worth pointing to when the documents are silent.
        summary = (
            f"{summary} Memory holds {len(memory.recalled)} related note(s) from earlier sessions "
            "(listed under 'From Earlier Investigations'); they are not evidence from the current documents."
        ).strip()
    review, action = human_review_block(human_review, state.get("approval"), action)

    report = InvestigationReport(
        investigation_id=state.get("investigation_id", ""),
        question=state.get("question", ""),
        status=status,
        status_label=STATUS_LABELS.get(status, status),
        risk_level=risk_level,
        confidence=round(confidence, 2),
        evidence_quality=evidence_quality(evidence, int(state.get("discarded_facts", 0)), missing),  # type: ignore[arg-type]
        requires_human_review=human_review,
        human_review=review,
        summary=summary,
        key_findings=findings,
        contradictions=contradictions_out,
        missing_evidence=missing,
        calculations=calculations_out,
        risk_reasons=reasons_out,
        recommended_action=action,
        sources=sources,
        limitations=list(dict.fromkeys(state.get("limitations", []))),
        stats=ReportStats(
            iterations=int(state.get("iteration", 0)),
            llm_calls=int(state.get("llm_calls", 0)),
            retrieval_rounds=int(state.get("retrieval_rounds", 0)),
            retrieved_chunks=len(state.get("retrieved_documents", [])),
            evidence_items=len(evidence),
            discarded_facts=int(state.get("discarded_facts", 0)),
            tool_calls=len(state.get("tool_calls", [])),
            duration_ms=round(duration_ms, 1),
            llm_provider=llm_provider,
        ),
        memory=memory,
        presentation=Presentation(**with_defaults(state.get("preferences"))),
    )
    report.answer_text = render_report(report)
    return report


def _human_review_line(review: HumanReview) -> str:
    if review.status == "pending":
        return "Required - waiting for a decision (approve / reject / modify)"
    if review.status in ("approved", "rejected", "modified"):
        line = f"{review.status.capitalize()} by {review.reviewer or 'reviewer'}"
        return line + (f" - {review.note}" if review.note else "")
    return "Required" if review.required else "Not required"


def render_report(report: InvestigationReport) -> str:
    """The report as readable text, in the layout of spec section 3.

    Two stored preferences shape it (see app/memory/profile.py):
    ``detail_level`` decides which sections are written out and
    ``response_format`` whether section titles are plain or Markdown.
    The structured report itself is always complete; only this text changes.
    """
    if report.status == "not_applicable":
        return report.summary
    detail = report.presentation.detail_level
    markdown = report.presentation.response_format == "markdown"

    def cites(citations: list[CitationOut]) -> str:
        return "".join(" " + format_citation(c.document, c.page) for c in citations)

    def title(text: str) -> str:
        return f"## {text}" if markdown else f"{text}:"

    def field(label: str, value: str) -> str:
        return f"**{label}:** {value}" if markdown else f"{label}: {value}"

    lines = [
        field("Investigation Status", report.status_label),
        field("Overall Risk Level", report.risk_level.capitalize()),
        field("Confidence Score", f"{round(report.confidence * 100)}%"),
        field("Human Review", _human_review_line(report.human_review)),
        "",
        title("Summary"),
        report.summary,
    ]
    labels = {"fact": "Fact", "calculation": "Calculation", "missing_evidence": "Missing evidence"}
    if report.key_findings and detail != "brief":
        lines += ["", title("Key Findings")]
        for number, finding in enumerate(report.key_findings, start=1):
            lines.append(f"{number}. [{labels.get(finding.type, finding.type)}] {finding.statement}{cites(finding.citations)}")
    if report.contradictions:
        lines += ["", title("Contradictions Detected")]
        lines += [f"- ({item.severity}) {item.description}{cites(item.citations)}" for item in report.contradictions]
    if report.missing_evidence:
        lines += ["", title("Missing Evidence")]
        lines += [f"- {text}" for text in report.missing_evidence]
    if report.calculations and detail == "detailed":
        lines += ["", title("Calculations (deterministic tools)")]
        for item in report.calculations:
            arguments = ", ".join(f"{name}={_number_text(value)}" for name, value in item.arguments.items())
            lines.append(f"- {item.description}: {item.tool_name}({arguments}) = {_number_text(item.result)}")
    if report.risk_reasons and detail != "brief":
        lines += ["", title("Risk Reasoning (inference)")]
        lines += [f"- {reason.statement}{cites(reason.citations)}" for reason in report.risk_reasons]
    if detail != "brief":
        lines += ["", field("Evidence Quality", report.evidence_quality.capitalize())]
    lines += ["", title("Recommended Action"), report.recommended_action]
    if report.human_review.original_action:
        lines.append(f"(The agent had recommended: {report.human_review.original_action})")
    if report.sources:
        lines += ["", title("Sources")]
        lines += [f"- {source.document} — Page {source.page}" for source in report.sources]
    if report.memory.recalled:
        lines += ["", title("From Earlier Investigations (memory, not evidence)")]
        lines += [f"- {note.text}" for note in report.memory.recalled]
    if report.memory.saved:
        lines += ["", title("Memory")]
        lines += [f"- {text}" for text in report.memory.saved]
    if report.limitations:
        lines += ["", title("Limitations")]
        lines += [f"- {text}" for text in report.limitations]
    if detail == "detailed":
        stats = report.stats
        lines += [
            "",
            title("Run Statistics"),
            f"- LLM calls: {stats.llm_calls}, tool calls: {stats.tool_calls}, searches: {stats.retrieval_rounds}, "
            f"chunks read: {stats.retrieved_chunks}, facts kept/discarded: {stats.evidence_items}/{stats.discarded_facts}",
        ]
    return "\n".join(lines)


def _number_text(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format_number(value)
    return str(value)


def findings_for_storage(report: InvestigationReport) -> list[dict[str, Any]]:
    """Flatten the report into rows for the ``findings`` table."""
    import uuid

    def source(citations: list[CitationOut]) -> dict[str, Any]:
        if not citations:
            return {"source_document_id": None, "page_number": None}
        first = citations[0]
        try:
            document_id = uuid.UUID(first.document_id) if first.document_id else None
        except ValueError:
            document_id = None
        return {"source_document_id": document_id, "page_number": first.page}

    rows: list[dict[str, Any]] = []
    for finding in report.key_findings:
        rows.append({
            "finding_type": finding.type, "statement": finding.statement, "severity": finding.severity,
            "confidence": finding.confidence, **source(finding.citations),
        })
    for item in report.contradictions:
        rows.append({
            "finding_type": "contradiction", "statement": item.description, "severity": item.severity,
            "confidence": report.confidence, **source(item.citations),
        })
    for reason in report.risk_reasons:
        rows.append({
            "finding_type": "risk", "statement": reason.statement, "severity": reason.severity,
            "confidence": report.confidence, **source(reason.citations),
        })
    if report.recommended_action and report.status != "not_applicable":
        rows.append({
            "finding_type": "recommendation", "statement": report.recommended_action, "severity": None,
            "confidence": report.confidence, "source_document_id": None, "page_number": None,
        })
    return rows
