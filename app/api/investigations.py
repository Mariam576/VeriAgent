"""Investigation endpoints: start one, decide a paused one, list them, read report, findings and trace."""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from app.agents.service import decide_investigation, resume_investigation, run_investigation
from app.database import repositories
from app.errors import NotFound
from app.schemas.response import (
    ApprovalRequest,
    FindingList,
    InvestigationDetail,
    InvestigationList,
    InvestigationReport,
    InvestigationRequest,
    InvestigationSummary,
    StoredFinding,
    TraceEventOut,
    TraceOut,
)
from app.security.permissions import CurrentUser, DbSession

router = APIRouter(prefix="/investigations", tags=["investigations"])


@router.post("", response_model=InvestigationReport, status_code=201)
async def create_investigation(request: InvestigationRequest, user: CurrentUser, session: DbSession) -> InvestigationReport:
    """Run a full investigation over your uploaded documents and return the structured report.

    When the findings need a human decision, the investigation pauses and the
    report comes back as a draft with ``human_review.status = "pending"``.
    Send the decision to ``POST /investigations/{id}/approve`` to finish it.
    """
    _, report = await run_investigation(
        session,
        user=user,
        question=request.question,
        document_ids=request.document_ids,
        document_type=request.document_type,
        year=request.year,
    )
    return report


@router.post("/{investigation_id}/approve", response_model=InvestigationReport)
async def approve_investigation(
    investigation_id: uuid.UUID, request: ApprovalRequest, user: CurrentUser, session: DbSession
) -> InvestigationReport:
    """Approve, reject or modify a paused investigation. Returns the final report.

    Answers 409 when the investigation is not waiting for a decision (for
    example because someone else already decided).
    """
    return await decide_investigation(
        session,
        user=user,
        investigation_id=investigation_id,
        decision=request.decision,
        note=request.note,
        modified_action=request.modified_action,
    )


@router.post("/{investigation_id}/resume", response_model=InvestigationReport)
async def resume_interrupted_investigation(
    investigation_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> InvestigationReport:
    """Continue an investigation that was cut off (for example by a server restart).

    The work done before the interruption is not repeated: the graph continues
    from the last node whose result was saved by the checkpointer.
    """
    return await resume_investigation(session, user=user, investigation_id=investigation_id)


@router.get("", response_model=InvestigationList)
async def list_investigations(user: CurrentUser, session: DbSession) -> InvestigationList:
    rows = await repositories.list_investigations(session, user.id)
    return InvestigationList(investigations=[InvestigationSummary.model_validate(row) for row in rows])


@router.get("/{investigation_id}", response_model=InvestigationDetail)
async def get_investigation(investigation_id: uuid.UUID, user: CurrentUser, session: DbSession) -> InvestigationDetail:
    investigation = await repositories.get_investigation(session, user.id, investigation_id)
    if investigation is None:
        raise NotFound("Investigation not found.")  # same answer whether it does not exist or belongs to someone else
    return InvestigationDetail.model_validate(investigation)


@router.get("/{investigation_id}/findings", response_model=FindingList)
async def get_findings(investigation_id: uuid.UUID, user: CurrentUser, session: DbSession) -> FindingList:
    investigation = await repositories.get_investigation(session, user.id, investigation_id)
    if investigation is None:
        raise NotFound("Investigation not found.")
    findings = await repositories.list_findings(session, investigation.id)
    return FindingList(
        investigation_id=investigation.id,
        findings=[StoredFinding.model_validate(finding) for finding in findings],
    )


@router.get("/{investigation_id}/trace", response_model=TraceOut)
async def get_trace(investigation_id: uuid.UUID, user: CurrentUser, session: DbSession) -> TraceOut:
    """What the agent actually did: nodes, LLM calls with token usage, tool calls, searches, errors."""
    investigation = await repositories.get_investigation(session, user.id, investigation_id)
    if investigation is None:
        raise NotFound("Investigation not found.")
    events = await repositories.list_trace(session, investigation.id)
    llm = [event for event in events if event.kind == "llm"]
    totals = {
        "events": len(events),
        "runs": max((event.run for event in events), default=0),
        "llm_calls": sum(1 for event in llm if event.data.get("status") == "ok"),
        "llm_failures": sum(1 for event in llm if event.data.get("status") != "ok"),
        "llm_latency_ms": round(sum(event.latency_ms or 0 for event in llm), 1),
        "input_tokens": sum(int(event.data.get("input_tokens") or 0) for event in llm),
        "output_tokens": sum(int(event.data.get("output_tokens") or 0) for event in llm),
        "tool_calls": sum(1 for event in events if event.kind == "tool"),
        "errors": sum(1 for event in events if event.kind == "error"),
    }
    return TraceOut(
        investigation_id=investigation.id,
        totals=totals,
        events=[TraceEventOut.model_validate(event) for event in events],
    )
