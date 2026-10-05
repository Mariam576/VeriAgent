"""Investigation endpoints: start one, list them, read the report and findings."""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from app.agents.service import run_investigation
from app.database import repositories
from app.errors import NotFound
from app.schemas.response import (
    FindingList,
    InvestigationDetail,
    InvestigationList,
    InvestigationReport,
    InvestigationRequest,
    InvestigationSummary,
    StoredFinding,
)
from app.security.permissions import CurrentUser, DbSession

router = APIRouter(prefix="/investigations", tags=["investigations"])


@router.post("", response_model=InvestigationReport, status_code=201)
async def create_investigation(request: InvestigationRequest, user: CurrentUser, session: DbSession) -> InvestigationReport:
    """Run a full investigation over your uploaded documents and return the structured report."""
    _, report = await run_investigation(
        session, user=user, question=request.question, document_ids=request.document_ids
    )
    return report


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
