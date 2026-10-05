"""Follow-up actions: what is configured, and what was done.

    GET /actions/config   where an approved follow-up issue would be created
    GET /actions          the actions carried out for your investigations

Actions are never started from here. They are proposed with a report, approved
with POST /investigations/{id}/approve and carried out by the agent.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.actions.runner import configured_target
from app.config import get_settings
from app.database.models import ExternalAction
from app.security.permissions import CurrentUser, DbSession

router = APIRouter(prefix="/actions", tags=["actions"])


class ActionConfig(BaseModel):
    enabled: bool
    server: str | None = None  # tickets | github
    target: str | None = None  # in words: "the local ticket list", "GitHub repository owner/repo"
    problem: str | None = None  # set when the configuration is incomplete


class ActionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    investigation_id: uuid.UUID
    kind: str
    server: str
    target: str
    title: str
    status: str
    reference: str | None = None
    url: str | None = None
    detail: str | None = None
    created_at: datetime
    finished_at: datetime | None = None


class ActionList(BaseModel):
    actions: list[ActionRecord]


@router.get("/config", response_model=ActionConfig)
async def action_config(user: CurrentUser) -> ActionConfig:
    try:
        target = configured_target(get_settings())
    except ValueError as exc:
        return ActionConfig(enabled=False, problem=str(exc))
    if target is None:
        return ActionConfig(enabled=False)
    return ActionConfig(enabled=True, server=target.server, target=target.label)


@router.get("", response_model=ActionList)
async def list_actions(user: CurrentUser, session: DbSession) -> ActionList:
    rows = (await session.execute(
        select(ExternalAction).where(ExternalAction.user_id == user.id).order_by(ExternalAction.created_at.desc()).limit(100)
    )).scalars().all()
    return ActionList(actions=[ActionRecord.model_validate(row) for row in rows])
