"""Deterministic scoring and review rules.

"Risk must be based on evidence, not model intuition alone" (spec, Node 7).
The LLM proposes a risk assessment, but these rules decide the limits it has
to stay within, and the reviewer (Node 8) uses them to check the draft.

Everything here is ordinary Python with fixed thresholds, so the same
evidence always gives the same result and every number can be explained.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from app.agents.heuristics import fallback_risk
from app.numbers import NUMBER_CORE, numbers_in_text, parse_number
from app.rag.citations import citations_from_evidence, unsupported_citations
from app.schemas.investigation import RISK_ORDER, Calculation, Contradiction, EvidenceItem, RetrievedChunk
from app.security.guardrails import detect_prompt_injection

RISK_LEVELS = ["low", "medium", "high", "critical"]

_SEVERITY_POINTS = {"low": 1, "medium": 2, "high": 3}


def clean_ids(identifiers: list[Any] | None) -> list[str]:
    """Normalise reference ids from the model: "x1", "[E2]" and " m1 " become X1, E2, M1."""
    cleaned = [
        re.sub(r"[^A-Za-z0-9]", "", str(identifier)).upper()
        for identifier in identifiers or []
        if identifier is not None
    ]
    return [identifier for identifier in cleaned if identifier]


# -------------------------------------------------------------- risk baseline
def risk_points(contradictions: list[Contradiction], missing_evidence: list[str], suspicious: list[dict]) -> int:
    points = sum(_SEVERITY_POINTS.get(item.severity, 1) for item in contradictions)
    points += min(len(missing_evidence), 2)  # each missing item adds 1, at most 2 in total
    points += 2 if suspicious else 0  # a document that tries to steer the reviewer is itself a warning sign
    return points


def baseline_risk(contradictions: list[Contradiction], missing_evidence: list[str], suspicious: list[dict]) -> str:
    """The minimum risk level that the structured findings justify.

        0 points      -> low
        1 - 2 points  -> medium
        3 - 6 points  -> high
        7+ points     -> critical
    """
    points = risk_points(contradictions, missing_evidence, suspicious)
    if points == 0:
        return "low"
    if points <= 2:
        return "medium"
    if points <= 6:
        return "high"
    return "critical"


def raise_level(level: str, steps: int = 1) -> str:
    return RISK_LEVELS[min(RISK_ORDER[level] + steps, len(RISK_LEVELS) - 1)]


# ----------------------------------------------------------------- confidence
def confidence_ceiling(
    *,
    evidence: list[EvidenceItem],
    discarded_facts: int,
    contradictions: list[Contradiction],
    missing_evidence: list[str],
    iteration_limit_reached: bool,
    used_fallback: bool,
) -> float:
    """The highest confidence the evidence can justify. The model may report less, never more.

    Starts at 0.55 and moves with things that can be counted:
      + sources:      +0.15 when at least two documents contribute, +0.05 for one
      + grounding:    +0.15 x share of extracted statements that survived the grounding check
      + calculations: +0.10 when every numeric contradiction has a tool-computed difference
      - gaps:         -0.05 per missing piece of evidence (at most -0.15)
      - limits:       -0.10 when the iteration limit stopped the investigation
      - fallback:     -0.05 when rule-based fallbacks replaced an unavailable LLM
    """
    if not evidence:
        return 0.30
    score = 0.55
    documents = {item.document_id for item in evidence}
    score += 0.15 if len(documents) >= 2 else 0.05
    score += 0.15 * (len(evidence) / (len(evidence) + discarded_facts))
    numeric = [item for item in contradictions if item.type not in ("unit_mismatch", "internal_inconsistency")]
    if all(item.difference is not None for item in numeric):
        score += 0.10
    score -= min(0.05 * len(missing_evidence), 0.15)
    if iteration_limit_reached:
        score -= 0.10
    if used_fallback:
        score -= 0.05
    return round(max(0.20, min(score, 0.95)), 2)


def evidence_quality(evidence: list[EvidenceItem], discarded_facts: int, missing_evidence: list[str]) -> str:
    if not evidence:
        return "none"
    documents = {item.document_id for item in evidence}
    if len(evidence) < 2 or discarded_facts > len(evidence):
        return "weak"
    if missing_evidence or discarded_facts or len(documents) < 2:
        return "moderate"
    return "strong"


# --------------------------------------------------------- suspicious content
def suspicious_content(retrieved: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Chunks that contain text which looks like a prompt-injection attempt."""
    found = []
    for chunk in retrieved:
        phrases = detect_prompt_injection(chunk.text)
        if phrases:
            found.append(
                {
                    "id": f"S{len(found) + 1}",
                    "document": chunk.filename,
                    "page": chunk.page,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "phrases": phrases[:3],
                }
            )
    return found


# -------------------------------------------------------------- number checks
def known_numbers(
    question: str,
    retrieved: list[RetrievedChunk],
    evidence: list[EvidenceItem],
    contradictions: list[Contradiction],
    calculations: list[Calculation],
) -> set[float]:
    """Every number that legitimately exists in this investigation."""
    numbers: set[float] = set(numbers_in_text(question))
    for chunk in retrieved:
        numbers.update(numbers_in_text(chunk.text))
    for item in evidence:
        if item.value_number is not None:
            numbers.add(float(item.value_number))
    for item in contradictions:
        numbers.update(v for v in (item.value_a, item.value_b, item.difference, item.percentage_change) if v is not None)
    for item in calculations:
        if item.result is not None:
            numbers.add(float(item.result))
    return {abs(value) for value in numbers}


# "£18k", "1.2m", "3bn": a number followed directly by a magnitude suffix.
_SUFFIXED_NUMBER = re.compile(rf"(?<!\w)(?<!\d[.,])(?P<number>{NUMBER_CORE})\s?(?P<suffix>k|m|mn|bn|million|billion|thousand)?(?![\w]|[.,]\d)", re.I)
_MAGNITUDE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}


def _is_known(value: float, known: set[float]) -> bool:
    """Is ``value`` one of the known numbers, allowing for ordinary rounding in prose?

    A model that writes "about 31%" for a tool result of 31.25, or "£18.3k"
    for 18,250, is rounding, not inventing. The value is accepted when it
    equals a known number rounded to a whole number or to one or two decimals.
    """
    for candidate in known:
        if abs(value - candidate) <= 0.005 * max(1.0, abs(candidate)):
            return True
        if any(abs(value - round(candidate, digits)) < 1e-9 for digits in (0, 1, 2)):
            return True
    return False


def unknown_numbers(text: str, known: set[float]) -> list[float]:
    """Numbers in ``text`` that appear nowhere in the documents or tool results.

    Small whole numbers (up to 10) are allowed: they are counts such as
    "2 contradictions" or list numbering, not facts from a document.
    """
    unknown = []
    for match in _SUFFIXED_NUMBER.finditer(text):
        readings = parse_number(match.group("number"))
        if not readings:
            continue
        scale = _MAGNITUDE.get((match.group("suffix") or "").lower(), 1.0)
        candidates = [reading * scale for reading in readings]
        if scale == 1.0 and any(value <= 10 and float(value).is_integer() for value in candidates):
            continue
        if not any(_is_known(value, known) for value in candidates):
            unknown.append(candidates[0])
    return unknown


def recompute(calculation: Calculation) -> float | None:
    """Recalculate a tool result with exact fractions, independently of the tool's own code."""
    args = calculation.arguments
    try:
        if calculation.tool_name == "calculate_difference":
            value = Fraction(str(args["value_a"])) - Fraction(str(args["value_b"]))
        elif calculation.tool_name == "calculate_percentage_change":
            old = Fraction(str(args["old"]))
            value = (Fraction(str(args["new"])) - old) / abs(old) * 100
        elif calculation.tool_name == "calculate_ratio":
            value = Fraction(str(args["numerator"])) / Fraction(str(args["denominator"]))
        elif calculation.tool_name == "sum_values":
            value = sum((Fraction(str(v)) for v in args["values"]), Fraction(0))
        else:
            return None
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    return round(float(value), 6)


# --------------------------------------------------------------------- review
def review_draft(
    draft: dict[str, Any],
    *,
    question: str,
    retrieved: list[RetrievedChunk],
    evidence: list[EvidenceItem],
    contradictions: list[Contradiction],
    calculations: list[Calculation],
    missing_evidence: list[str],
    suspicious: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any], dict[str, list[str]]]:
    """Node 8: check the draft assessment against the evidence.

    Returns ``(issues, corrected_draft, drops)``. ``issues`` is empty when the
    draft is approved. ``corrected_draft`` is the draft with every issue fixed
    by rule, and ``drops`` lists evidence / calculation ids that must be removed.
    """
    issues: list[str] = []
    corrected = dict(draft)
    drops: dict[str, list[str]] = {"evidence": [], "calculations": []}

    baseline = draft["baseline_level"]
    ceiling = draft["confidence_ceiling"]
    has_findings = bool(contradictions or missing_evidence or suspicious)
    safe = fallback_risk(
        baseline_level=baseline,
        contradictions=[item.model_dump() for item in contradictions],
        missing=[{"id": f"M{i}", "missing": text} for i, text in enumerate(missing_evidence, start=1)],
        suspicious=suspicious,
        has_evidence=bool(evidence),
    )

    # 1. Missing citations: every evidence item must point at a retrieved chunk.
    bad_sources = unsupported_citations(citations_from_evidence(evidence), retrieved)
    if bad_sources:
        bad_keys = {(c.document_id, c.page) for c in bad_sources}
        for item in evidence:
            if (item.document_id, item.page) in bad_keys:
                drops["evidence"].append(item.id)
                issues.append(f"Evidence {item.id} cites a source that was not retrieved.")
    surviving_evidence = {item.id for item in evidence} - set(drops["evidence"])

    # 2. Wrong calculations: recompute every tool result independently.
    for calculation in calculations:
        if calculation.status != "ok":
            continue
        expected = recompute(calculation)
        if expected is None or calculation.result is None or abs(expected - calculation.result) > 2e-6:
            drops["calculations"].append(calculation.id)
            issues.append(f"Calculation {calculation.id} could not be verified independently.")

    # 3. Risk level must stay within what the evidence supports.
    level = draft["level"]
    if RISK_ORDER[level] < RISK_ORDER[baseline]:
        issues.append(f"Risk level '{level}' is lower than the evidence supports (minimum '{baseline}').")
        corrected["level"] = baseline
    elif RISK_ORDER[level] > RISK_ORDER[baseline] + 1:
        allowed = raise_level(baseline, 1)
        issues.append(f"Risk level '{level}' is more than one step above what the evidence supports ('{baseline}').")
        corrected["level"] = allowed

    # 4. Overconfidence.
    if draft["confidence"] > ceiling + 1e-9:
        issues.append(f"Confidence {draft['confidence']:.2f} exceeds what the evidence supports ({ceiling:.2f}).")
        corrected["confidence"] = ceiling

    # 5. Unsupported claims: every reason needs valid supporting ids and only known numbers.
    valid_ids = (
        surviving_evidence
        | {item.id for item in contradictions}
        | {item.id for item in calculations if item.id not in drops["calculations"]}
        | {f"M{i}" for i in range(1, len(missing_evidence) + 1)}
        | {item["id"] for item in suspicious}
    )
    numbers = known_numbers(question, retrieved, evidence, contradictions, calculations)
    kept_reasons = []
    for index, reason in enumerate(draft.get("reasons", []), start=1):
        ids = clean_ids(reason.get("evidence_ids"))
        if not ids or any(identifier not in valid_ids for identifier in ids):
            issues.append(f"Reason {index} has no supporting citation.")
            continue
        strangers = unknown_numbers(reason.get("text", ""), numbers)
        if strangers:
            issues.append(f"Reason {index} mentions a figure that is not in the evidence ({strangers[0]:g}).")
            continue
        kept_reasons.append({**reason, "evidence_ids": ids})
    if len(kept_reasons) != len(draft.get("reasons", [])):
        corrected["reasons"] = kept_reasons or [reason.model_dump() for reason in safe.reasons]

    # 6. Contradictory conclusion: "proceed" is impossible while issues are open.
    if draft["decision"] == "proceed" and (has_findings or RISK_ORDER[corrected["level"]] >= RISK_ORDER["high"]):
        issues.append("The decision 'proceed' contradicts the contradictions or missing evidence that were found.")
        corrected["decision"] = "review"
        corrected["recommended_action"] = safe.recommended_action
        corrected["summary"] = safe.summary

    # 7. Free text must not contain figures that exist nowhere in the evidence.
    for field in ("summary", "recommended_action"):
        strangers = unknown_numbers(str(corrected.get(field, "")), numbers)
        if strangers:
            issues.append(f"The {field.replace('_', ' ')} mentions a figure that is not in the evidence ({strangers[0]:g}).")
            corrected[field] = getattr(safe, field)

    return issues, corrected, drops
