"""Carry out an approved action exactly once, and keep a record of it.

Concept - at most once:
    Creating an issue cannot be undone by the agent, so doing it twice is a
    real fault. But a graph node can run twice: if the server stops right
    after the issue was created and before that was saved, the node runs again
    when the investigation is resumed.

    The record makes this safe. Before the call the action is written down as
    "started". After the call it becomes "done" or "failed". When the node
    finds an action that is already "done", it reuses the stored result. When
    it finds one that is still "started", nobody knows whether the call went
    through; it is reported as "unknown" for a person to check, and it is NOT
    tried again. A missing issue is an inconvenience; a duplicate is noise in
    someone's tracker with the agent's name on it.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.actions.client import McpEndpoint, ToolOutcome, call_tool, github_endpoint, ticket_endpoint
from app.actions.proposals import propose_actions
from app.config import Settings
from app.database.models import ExternalAction
from app.observability import tracing
from app.schemas.response import InvestigationReport

logger = logging.getLogger(__name__)

_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100}$")
UNKNOWN_DETAIL = (
    "The action was started but its result was never recorded (the server stopped in between). "
    "It was not tried again. Check the target system to see whether it exists."
)


SENT_DETAIL = "The request had already been sent, so it may have been carried out. It was not tried again. Check the target system before creating it by hand."
NO_REFERENCE_DETAIL = "The server answered without naming what it created, so it could not be confirmed. It was not tried again. Check the target system."


class ActionLog(Protocol):
    async def start(self, investigation_id: str, user_id: str, proposal: dict[str, Any]) -> dict[str, Any] | None:
        """Write the action down as started. Returns None if that worked, or the existing record."""
        ...

    async def finish(self, investigation_id: str, action_id: str, outcome: dict[str, Any]) -> None: ...


class MemoryActionLog:
    """A record in a dict, for tests and for runs without a database."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], dict[str, Any]] = {}

    async def start(self, investigation_id: str, user_id: str, proposal: dict[str, Any]) -> dict[str, Any] | None:
        key = (investigation_id, proposal["id"])
        if key in self.records:
            return dict(self.records[key])
        self.records[key] = {"status": "started"}
        return None

    async def finish(self, investigation_id: str, action_id: str, outcome: dict[str, Any]) -> None:
        self.records[(investigation_id, action_id)] = dict(outcome)


class PgActionLog:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def start(self, investigation_id: str, user_id: str, proposal: dict[str, Any]) -> dict[str, Any] | None:
        values = dict(
            investigation_id=uuid.UUID(investigation_id), user_id=uuid.UUID(user_id), action_key=proposal["id"],
            kind=proposal["kind"], server=proposal["server"], target=proposal["target"][:200],
            title=proposal["title"][:300], status="started",
        )
        async with self._session_factory() as session:
            # One statement decides who goes first: the unique (investigation, action) pair lets only one insert through.
            created = await session.execute(
                insert(ExternalAction).values(**values)
                .on_conflict_do_nothing(index_elements=["investigation_id", "action_key"])
                .returning(ExternalAction.id)
            )
            if created.scalar_one_or_none() is not None:
                await session.commit()
                return None
            row = (await session.execute(
                select(ExternalAction).where(
                    ExternalAction.investigation_id == values["investigation_id"], ExternalAction.action_key == proposal["id"]
                )
            )).scalar_one()
            return {"status": row.status, "reference": row.reference, "url": row.url, "detail": row.detail}

    async def finish(self, investigation_id: str, action_id: str, outcome: dict[str, Any]) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(ExternalAction)
                .where(ExternalAction.investigation_id == uuid.UUID(investigation_id), ExternalAction.action_key == action_id)
                .values(
                    status=outcome["status"], reference=outcome.get("reference"), url=outcome.get("url"),
                    detail=(outcome.get("detail") or "")[:500] or None, finished_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()


@dataclass
class ActionTarget:
    """Where issues go: which server, how it is shown to people, and how an issue maps onto its tool."""

    server: str  # "tickets" | "github"
    label: str  # "the local ticket list" / "GitHub repository owner/repo"
    endpoint: McpEndpoint
    repository: str = ""

    def calls(self, proposal: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """The tool calls that would create this issue, in order of preference."""
        title, body = proposal["title"], proposal["body"]
        if self.server == "github":
            owner, repo = self.repository.split("/", 1)
            return [
                ("issue_write", {"method": "create", "owner": owner, "repo": repo, "title": title, "body": body}),
                ("create_issue", {"owner": owner, "repo": repo, "title": title, "body": body}),  # older server versions
            ]
        return [("create_ticket", {"title": title, "body": body, "severity": proposal.get("severity", "medium")})]

    def read_result(self, outcome: ToolOutcome) -> tuple[str | None, str | None]:
        """(reference, link) of what was created. A link is only accepted when it points where we sent the issue."""
        data = outcome.data or {}
        if self.server == "github":
            expected = re.compile(rf"https://github\.com/{re.escape(self.repository)}/issues/(\d+)", re.I)
            match = expected.search(f"{data.get('html_url') or ''} {data.get('url') or ''} {outcome.text}")
            return (f"#{match.group(1)}", match.group(0)) if match else (None, None)
        reference = str(data.get("id") or "")
        return (reference if re.fullmatch(r"T-[0-9A-F]{8}", reference) else None), None


def configured_target(settings: Settings) -> ActionTarget | None:
    """The target named by ACTION_SERVER, or None when actions are switched off.

    Raises ``ValueError`` with a readable message when the setting is incomplete.
    """
    if settings.ACTION_SERVER == "tickets":
        return ActionTarget("tickets", "the local ticket list", ticket_endpoint(settings.TICKETS_FILE))
    if settings.ACTION_SERVER == "github":
        if settings.GITHUB_TOKEN is None or not _REPOSITORY.match(settings.GITHUB_REPOSITORY):
            raise ValueError("ACTION_SERVER=github needs GITHUB_TOKEN and GITHUB_REPOSITORY (in the form owner/repository).")
        return ActionTarget(
            "github", f"GitHub repository {settings.GITHUB_REPOSITORY}",
            github_endpoint(settings.GITHUB_MCP_URL, settings.GITHUB_TOKEN.get_secret_value()),
            repository=settings.GITHUB_REPOSITORY,
        )
    return None


class ActionRunner:
    def __init__(self, target: ActionTarget, log: ActionLog, *, timeout: float = 30.0) -> None:
        self.target = target
        self._log = log
        self._timeout = timeout

    def propose(self, report: InvestigationReport) -> list[dict[str, Any]]:
        return propose_actions(report, server=self.target.server, target=self.target.label)

    async def execute(self, investigation_id: str, user_id: str, proposal: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Carry out one approved proposal. Returns (outcome, tool-call record or None). Never raises."""
        outcome = {key: proposal[key] for key in ("id", "kind", "server", "target", "title")}
        try:
            existing = await self._log.start(investigation_id, user_id, proposal)
        except Exception as exc:  # noqa: BLE001 - without a record there is no "exactly once", so nothing is sent
            logger.error("action_log_failed investigation=%s error=%s", investigation_id, type(exc).__name__)
            return {**outcome, "status": "failed", "detail": "The action could not be recorded, so it was not carried out."}, None
        if existing is not None:
            if existing.get("status") == "started":
                await self._log.finish(investigation_id, proposal["id"], {**outcome, "status": "unknown", "detail": UNKNOWN_DETAIL})
                return {**outcome, "status": "unknown", "detail": UNKNOWN_DETAIL}, None
            return {**outcome, **{key: existing.get(key) for key in ("status", "reference", "url", "detail")}}, None

        started = time.perf_counter()
        calls = self.target.calls(proposal)
        result = await call_tool(self.target.endpoint, calls, timeout=self._timeout)
        latency = round((time.perf_counter() - started) * 1000, 1)
        reference, url = self.target.read_result(result) if result.ok else (None, None)
        if result.ok and reference:
            outcome.update(status="done", reference=reference, url=url, detail=None)
        elif result.ok:
            # The server did not report an error, but named nothing that it created. "Done" is
            # only ever said about something that can be pointed at.
            outcome.update(status="unknown", reference=None, url=None, detail=NO_REFERENCE_DETAIL)
        elif result.uncertain:
            # The request left and no answer came back: the issue may exist. Saying "failed"
            # here would invite someone to create it a second time by hand.
            outcome.update(status="unknown", reference=None, url=None, detail=f"{result.error} {SENT_DETAIL}"[:500])
        else:
            outcome.update(status="failed", reference=None, url=None, detail=result.error[:300])
        await self._log.finish(investigation_id, proposal["id"], outcome)

        tool = result.tool or calls[0][0]
        tracing.record(
            "action", proposal["kind"], latency, server=self.target.server, tool=tool,
            status=outcome["status"], reference=outcome.get("reference"),
        )
        logger.info(
            "external_action investigation=%s server=%s tool=%s status=%s reference=%s latency_ms=%.0f",
            investigation_id, self.target.server, tool, outcome["status"], outcome.get("reference"), latency,
        )
        record = {
            "tool_name": f"mcp:{self.target.server}/{tool}",
            "arguments": {"title": proposal["title"], "body_chars": len(proposal["body"]), "target": self.target.label},
            "result": {"reference": outcome.get("reference"), "url": outcome.get("url")} if outcome["status"] == "done" else None,
            "status": "ok" if outcome["status"] == "done" else "error",
            "latency_ms": latency,
            "error": None if outcome["status"] == "done" else outcome["detail"],
        }
        return outcome, record


_runner: ActionRunner | None = None
_runner_built = False


def get_action_runner() -> ActionRunner | None:
    """The runner for the configured target; None when ACTION_SERVER is empty."""
    global _runner, _runner_built
    if not _runner_built:
        from app.config import get_settings
        from app.database.session import get_session_factory

        settings = get_settings()
        target = configured_target(settings)
        _runner = ActionRunner(target, PgActionLog(get_session_factory()), timeout=settings.ACTION_TIMEOUT_SECONDS) if target else None
        _runner_built = True
    return _runner


def reset_action_runner() -> None:
    global _runner, _runner_built
    _runner, _runner_built = None, False
