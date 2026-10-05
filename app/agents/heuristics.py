"""Rule-based stand-ins for the LLM steps.

They have two real jobs:

1. Fallback. If the LLM is unavailable after its bounded retries, the
   investigation continues with these rules and says so in its limitations,
   instead of failing or inventing an answer (spec section 38).
2. The offline mock provider (``LLM_PROVIDER=mock``) is built from them, so
   the whole pipeline can run in tests and demos with no API key.

The rules only understand common wordings such as "Total amount due: £58,000"
or "payment within 30 days". A real LLM reads far more varied documents.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from app.agents import prompts
from app.formatting import format_value
from app.llm.base import ToolCallRequest
from app.patterns import find_amounts, find_payment_terms, sentence_around
from app.llm.mock_provider import MockLLM
from app.memory.extraction import mock_memory
from app.rag.citations import normalize
from app.schemas.investigation import (
    KEY_OTHER,
    KEY_PAYMENT_TERMS,
    KEY_TOTAL_AMOUNT,
    EvidenceExtraction,
    ExtractedFact,
    RequestAnalysis,
    RetrievedChunk,
    RiskAssessment,
    RiskReason,
    SufficiencyAssessment,
)
from app.schemas.memory import MemoryCandidate
from app.security.guardrails import UNTRUSTED_CLOSE, dedupe_keep_order

# =========================================================== fact extraction
_APPROVAL = re.compile(
    r"\b(?:approved|authorised|authorized|signed off)\s+by\b|\bapproval\s+(?:granted|given|confirmed)\b", re.I
)
# A rule that says approval is needed (typical for policies and contracts).
_APPROVAL_REQUIREMENT = re.compile(
    r"\bmust not be paid (?:until|unless)\b|\b(?:not payable|not be paid) unless\b|"
    r"\brequires?\s+(?:the\s+)?(?:prior\s+|written\s+)*approval\b|"
    r"\b(?:prior|written)\s+approval\b[^.]{0,80}\b(?:is|are)\s+(?:required|needed|attached)\b",
    re.I,
)


def extract_facts_with_rules(chunk: RetrievedChunk) -> list[tuple[ExtractedFact, int]]:
    """Find amounts, payment terms and approval statements in one chunk.

    Returns ``(fact, priority)`` pairs. The priority is only meaningful for
    ``total_amount`` facts (see ``rule_based_extraction``).
    """
    text = chunk.text
    facts: list[tuple[ExtractedFact, int]] = []

    # --- amounts
    labelled: list[tuple[ExtractedFact, int]] = []
    unlabelled: list[tuple[ExtractedFact, int]] = []
    for hit in find_amounts(text):
        if hit.priority:
            labelled.append((
                ExtractedFact(
                    chunk_ref=chunk.ref,
                    claim=f"{chunk.filename} states a total amount of {format_value(hit.value, hit.unit)}.",
                    key=KEY_TOTAL_AMOUNT, value_number=hit.value, unit=hit.unit, quote=hit.sentence,
                ),
                hit.priority,
            ))
        elif len(unlabelled) < 4:
            unlabelled.append((
                ExtractedFact(
                    chunk_ref=chunk.ref,
                    claim=f"{chunk.filename} lists {format_value(hit.value, hit.unit)}: {hit.sentence[:160]}",
                    key=KEY_OTHER, value_number=hit.value, unit=hit.unit, quote=hit.sentence,
                ),
                0,
            ))
    facts += labelled
    if labelled:
        facts += unlabelled  # amounts next to a total are its line items; elsewhere they are noise

    # --- payment terms
    flattened = re.sub(r"\s+", " ", text)
    terms = find_payment_terms(flattened)
    if terms:
        days, wording = terms
        facts.append((
            ExtractedFact(
                chunk_ref=chunk.ref,
                claim=f"{chunk.filename} states payment terms of {int(days)} days.",
                key=KEY_PAYMENT_TERMS, value_number=days, unit="days", quote=wording,
            ),
            0,
        ))

    # --- approval requirements ("must not be paid until written approval ...")
    match = _APPROVAL_REQUIREMENT.search(flattened)
    if match:
        sentence = sentence_around(flattened, match.start(), match.end())
        facts.append((
            ExtractedFact(
                chunk_ref=chunk.ref,
                claim=f"{chunk.filename} sets an approval requirement: {sentence[:240]}",
                key=KEY_OTHER, value_number=None, unit=None, quote=sentence,
            ),
            0,
        ))

    # --- approval statements
    match = _APPROVAL.search(flattened)
    if match:
        sentence = sentence_around(flattened, match.start(), match.end())
        facts.append((
            ExtractedFact(
                chunk_ref=chunk.ref,
                claim=f"{chunk.filename} contains an approval statement: {sentence[:200]}",
                key=KEY_OTHER, value_number=None, unit=None, quote=sentence,
            ),
            0,
        ))
    return facts


def rule_based_extraction(chunks: list[RetrievedChunk]) -> EvidenceExtraction:
    """Run the rules over all chunks; keep only the best "total" candidate(s) per document."""
    collected: list[tuple[ExtractedFact, int, str]] = []
    for chunk in chunks:
        for fact, priority in extract_facts_with_rules(chunk):
            collected.append((fact, priority, chunk.document_id))

    best_total: dict[str, int] = {}
    for fact, priority, document_id in collected:
        if fact.key == KEY_TOTAL_AMOUNT:
            best_total[document_id] = max(best_total.get(document_id, 0), priority)

    facts = []
    for fact, priority, document_id in collected:
        if fact.key == KEY_TOTAL_AMOUNT and priority < best_total[document_id]:
            fact = fact.model_copy(update={"key": KEY_OTHER})  # e.g. a "total" for one phase only
        facts.append(fact)
    return EvidenceExtraction(facts=facts)


# ========================================================== request analysis
_GENERAL = re.compile(r"^\s*(?:hi|hello|hey|thanks|thank you|good (?:morning|afternoon|evening)|help|who are you|what can you do)\b[\s!.?]*$", re.I)
_DECISION = re.compile(r"\b(?:approve|approved|approval|pay|paid|accept|accepted|reject|rejected|authori[sz]ed?|sign off)\b", re.I)
_INVESTIGATE = re.compile(r"\b(?:review|compare|comparison|inconsisten\w*|contradict\w*|agree\w*|match\w*|verify|verification|check|approve\w*|risk\w*|audit\w*|discrepanc\w*|mismatch\w*|differ\w*)\b", re.I)
_CALCULATE = re.compile(r"\b(?:calculate|calculation|difference|how much|percentage|percent|ratio|sum|add up|exceed\w*|variance|overcharg\w*)\b", re.I)
_AMOUNT_TOPIC = re.compile(r"\b(?:amount|value|total|price|cost|invoice\w*|charge\w*|fee|sum|difference|exceed\w*|overcharg\w*)\b", re.I)
_TERMS_TOPIC = re.compile(r"\b(?:payment|terms?|days|due|net)\b", re.I)
_FOLLOW_UP = re.compile(r"^\s*(?:and|also|what about|how about|and what about|what of|same for)\b", re.I)


def fallback_analysis(question: str, history: list[dict[str, str]] | None = None) -> RequestAnalysis:
    history = history or []
    if _GENERAL.match(question):
        return RequestAnalysis(
            task_type="general", needs_rag=False, needs_calculation=False, needs_external_verification=False,
            decision_requested=False, standalone_question=question, search_queries=[],
        )

    standalone = question
    previous_user = next((turn["content"] for turn in reversed(history) if turn.get("role") == "user"), None)
    if previous_user and (_FOLLOW_UP.match(question) or len(question.split()) <= 6):
        standalone = f"{question} (follow-up to: {previous_user})"

    investigating = bool(_INVESTIGATE.search(standalone))
    calculating = bool(_CALCULATE.search(standalone))
    decision = bool(_DECISION.search(question))

    queries = [question]
    if investigating or calculating or _AMOUNT_TOPIC.search(standalone):
        queries.append("total amount value price")
    if investigating or _TERMS_TOPIC.search(standalone):
        queries.append("payment terms days")
    if decision:
        queries.append("approval authorisation signed variation")

    if investigating:
        task_type = "document_investigation"
    elif calculating:
        task_type = "calculation"
    else:
        task_type = "document_question"

    return RequestAnalysis(
        task_type=task_type,
        needs_rag=True,
        needs_calculation=investigating or calculating or bool(_AMOUNT_TOPIC.search(standalone)),
        needs_external_verification=False,
        decision_requested=decision,
        standalone_question=standalone,
        search_queries=dedupe_keep_order(queries)[:4],
    )


# ============================================================== sufficiency
MISSING_APPROVAL = "Written approval or a signed variation covering the difference(s) between the documents"
MISSING_SUPPLIER_CONFIRMATION = (
    "Confirmation that the company on the invoice is the contracted party (for example a signed change-of-name "
    "or assignment notice)"
)
MISSING_REFERENCED_DOCUMENT = "The contract or purchase order that the invoice refers to"
# Findings that an approval cannot settle: they question which document or which company this is,
# or the document's own arithmetic. What is missing for them, if anything, is named separately.
_NOT_A_DEVIATION = {
    "party_mismatch", "reference_mismatch", "date_outside_term", "duplicate_invoice",
    "line_items_total_mismatch", "line_arithmetic_mismatch",
}
MISSING_APPROVAL_RECORD = "An approval record for this item (for example a signed approval or sign-off)"
_ASKS_IF_APPROVED = re.compile(r"\b(?:was|were|is|has|have|been|already)\b[^?.]{0,60}\b(?:approved|authori[sz]ed|signed off)\b", re.I)
MISSING_ANY_EVIDENCE = "No relevant evidence was found in the uploaded documents for this request."
# Documents written by the party that wants to be paid. What they say about approval is a claim.
_ISSUED_BY_THE_PAYEE = {"invoice", "receipt", "quotation"}


def fallback_sufficiency(
    *,
    question: str = "",
    decision_requested: bool,
    investigating: bool,
    evidence: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    executed_queries: list[str],
) -> SufficiencyAssessment:
    if not evidence:
        return SufficiencyAssessment(sufficient=False, missing=[MISSING_ANY_EVIDENCE], follow_up_queries=[])

    missing: list[str] = []
    follow_up: list[str] = []
    # Who says so matters. "Approved by the Finance Director" on the supplier's own invoice is the
    # supplier's claim; an approval is a record of the organisation that pays.
    has_approval = any(
        _APPROVAL.search(str(item.get("claim", ""))) and item.get("document_type") not in _ISSUED_BY_THE_PAYEE
        for item in evidence
    )
    kinds = {str(item.get("type", "")) for item in contradictions}
    deviations = kinds - _NOT_A_DEVIATION
    if contradictions and (decision_requested or investigating):
        if deviations and not has_approval:
            missing.append(MISSING_APPROVAL)
            follow_up = ["written approval of invoice deviation", "signed variation change order"]
        if "party_mismatch" in kinds:
            missing.append(MISSING_SUPPLIER_CONFIRMATION)
            follow_up.append("change of name assignment novation notice")
        if "reference_mismatch" in kinds:
            missing.append(MISSING_REFERENCED_DOCUMENT)
    if not missing and _ASKS_IF_APPROVED.search(question) and not has_approval:
        # "Was this invoice approved?" with no approval document: say so, never assume.
        missing.append(MISSING_APPROVAL_RECORD)
        follow_up = ["approved by signature sign-off", "approval record authorisation"]

    already_run = {normalize(query) for query in executed_queries}
    follow_up = [query for query in follow_up if normalize(query) not in already_run]
    return SufficiencyAssessment(sufficient=not missing, missing=missing, follow_up_queries=follow_up)


# ===================================================================== risk
def fallback_risk(
    *,
    baseline_level: str,
    contradictions: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    suspicious: list[dict[str, Any]],
    has_evidence: bool,
    integrity: list[dict[str, Any]] | None = None,
) -> RiskAssessment:
    """A cautious, fully templated assessment built only from structured findings."""
    integrity = integrity or []
    reasons = [
        RiskReason(text=item["description"], evidence_ids=[item["id"], *item.get("evidence_ids", [])])
        for item in contradictions
    ]
    reasons += [
        RiskReason(text=f"Supporting evidence was not found: {item['missing']}", evidence_ids=[item["id"]])
        for item in missing
    ]
    reasons += [
        RiskReason(
            text=f"{item['document']} (page {item['page']}) contains text that reads like instructions to an AI system.",
            evidence_ids=[item["id"]],
        )
        for item in suspicious
    ]
    reasons += [
        RiskReason(text=integrity_sentence(item), evidence_ids=[item["id"]])
        for item in integrity
    ]

    if not has_evidence:
        return RiskAssessment(
            level=baseline_level, confidence=0.3, reasons=reasons, decision="review",  # type: ignore[arg-type]
            recommended_action="Upload the relevant documents, or rephrase the question, and run the investigation again.",
            summary="No relevant evidence was found in the uploaded documents, so no conclusion can be drawn.",
        )
    if contradictions or missing or suspicious or integrity:
        sentences = [item["description"] for item in contradictions[:2]]
        if len(contradictions) > 2:
            sentences.append(f"{len(contradictions) - 2} further contradiction(s) are listed below.")
        if missing:
            sentences.append("Evidence supporting these points was not found: " + "; ".join(item["missing"] for item in missing[:2]) + ".")
        if suspicious:
            sentences.append("A document contains text that reads like instructions to an AI system; it was not followed.")
        if integrity:
            names = ", ".join(item["document"] for item in integrity[:3])
            sentences.append(f"{names} shows signs that the file may have been altered; this is a risk indicator, not proof.")
        action = "Do not approve automatically. Obtain clarification"
        action += " and formal written approval before processing." if missing else " before processing."
        if integrity:
            action += " Ask the issuer to confirm the original of each flagged file."
        return RiskAssessment(
            level=baseline_level, confidence=0.9, reasons=reasons, decision="review",  # type: ignore[arg-type]
            recommended_action=action,
            summary="Review required. " + " ".join(sentences),
        )
    return RiskAssessment(
        level=baseline_level, confidence=0.85, reasons=[], decision="proceed",  # type: ignore[arg-type]
        recommended_action="No action is required based on the retrieved evidence.",
        summary="No contradictions or missing evidence were identified in the retrieved evidence.",
    )


def integrity_sentence(item: dict[str, Any]) -> str:
    """One sentence for a document with forensic indicators (wording: an indicator, never an accusation)."""
    indicators = item.get("indicators") or []
    first = f" {indicators[0]['reason']}" if indicators else ""
    more = f" ({len(indicators) - 1} further indicator(s) are listed under Document Integrity.)" if len(indicators) > 1 else ""
    return f"{item['document']} shows signs that it may have been altered ({item['level']} risk indicator).{first}{more}"


# ================================================== offline mock LLM handlers
def _parse_chunks(user_prompt: str) -> list[RetrievedChunk]:
    """Read the UNTRUSTED DOCUMENT CONTENT blocks back out of an extraction prompt."""
    pattern = re.compile(
        r'<<<UNTRUSTED DOCUMENT CONTENT ref=(?P<ref>\S+) file="(?P<file>[^"]*)" page=(?P<page>\d+)>>>\n'
        rf"(?P<text>.*?)\n{re.escape(UNTRUSTED_CLOSE)}",
        re.DOTALL,
    )
    return [
        RetrievedChunk(
            chunk_id=match.group("ref"), document_id=match.group("file"), filename=match.group("file"),
            page=int(match.group("page")), text=match.group("text"), ref=match.group("ref"),
        )
        for match in pattern.finditer(user_prompt)
    ]


def _history_from_prompt(user_prompt: str) -> list[dict[str, str]]:
    body = prompts.read_section(user_prompt, "CONVERSATION SO FAR") or ""
    history = []
    for line in body.splitlines():
        role, _, content = line.partition(": ")
        if content:
            history.append({"role": role.strip(), "content": content.strip()})
    return history


def _mock_analysis(system: str, user: str) -> RequestAnalysis:
    question = prompts.read_section(user, "USER REQUEST") or ""
    return fallback_analysis(question, _history_from_prompt(user))


def _mock_extraction(system: str, user: str) -> EvidenceExtraction:
    return rule_based_extraction(_parse_chunks(user))


def _mock_sufficiency(system: str, user: str) -> SufficiencyAssessment:
    context = prompts.read_json_section(user, "CONTEXT") or {}
    question = prompts.read_section(user, "USER REQUEST") or ""
    return fallback_sufficiency(
        question=question,
        decision_requested=bool(context.get("decision_requested")),
        investigating=bool(_INVESTIGATE.search(question)),
        evidence=prompts.read_json_section(user, "UNTRUSTED EVIDENCE") or [],
        contradictions=prompts.read_json_section(user, "CONTRADICTIONS") or [],
        executed_queries=context.get("search_queries_already_run", []),
    )


def _mock_risk(system: str, user: str) -> RiskAssessment:
    context = prompts.read_json_section(user, "CONTEXT") or {}
    evidence = prompts.read_json_section(user, "UNTRUSTED EVIDENCE") or []
    return fallback_risk(
        baseline_level=context.get("minimum_risk_level_from_rules", "low"),
        contradictions=prompts.read_json_section(user, "CONTRADICTIONS") or [],
        missing=prompts.read_json_section(user, "MISSING EVIDENCE") or [],
        suspicious=prompts.read_json_section(user, "SUSPICIOUS DOCUMENT CONTENT") or [],
        has_evidence=bool(evidence),
        integrity=prompts.read_json_section(user, "UNTRUSTED DOCUMENT INTEGRITY INDICATORS") or [],
    )


def _mock_tool_choice(system: str, user: str, tools: list[Any]) -> list[ToolCallRequest]:
    """Ask for a ratio when the user mentions one; otherwise nothing beyond the standard calculations."""
    question = (prompts.read_section(user, "USER REQUEST") or "").lower()
    numeric = prompts.read_json_section(user, "NUMERIC EVIDENCE") or []
    totals = [row for row in numeric if row.get("key") == KEY_TOTAL_AMOUNT]
    available = {getattr(tool, "name", "") for tool in tools}
    if "ratio" in question and len(totals) >= 2 and "calculate_ratio" in available:
        reference = next((row for row in totals if row.get("document_type") == "contract"), totals[0])
        other = next(row for row in totals if row is not reference)
        return [ToolCallRequest(
            name="calculate_ratio",
            args={"numerator": other["value"], "denominator": reference["value"]},
            id="mock-call-1",
        )]
    return []


def _mock_profile(system: str, user: str) -> BaseModel:
    """The document profile as the rules read it, in the shape the model would answer in."""
    from app.comparison.llm import to_extraction
    from app.comparison.rules import build_profile_with_rules
    from app.rag.loaders import PageText

    context = prompts.read_json_section(user, "CONTEXT") or {}
    pages = [PageText(chunk.page, chunk.text) for chunk in _parse_chunks(user)]
    profile = build_profile_with_rules(
        document_id=str(context.get("document", "")), filename=str(context.get("document", "")),
        document_type=str(context.get("document_type", "other")), pages=pages,
    )
    return to_extraction(profile)


def create_mock_llm() -> MockLLM:
    from app.agents.reviewer import ClaimReview, mock_review
    from app.comparison.llm import ProfileExtraction  # imported here: app.comparison uses this module's prompts

    return MockLLM(
        handlers={
            RequestAnalysis: _mock_analysis,
            EvidenceExtraction: _mock_extraction,
            SufficiencyAssessment: _mock_sufficiency,
            RiskAssessment: _mock_risk,
            MemoryCandidate: mock_memory,
            ProfileExtraction: _mock_profile,
            ClaimReview: mock_review,
        },
        tool_chooser=_mock_tool_choice,
    )
