"""Run investigations and store their results.

This is the bridge between the API layer and the LangGraph agent:

    API route -> service -> graph.astream(...) -> report -> database

An investigation can now take more than one run:

    start    run the graph until it finishes or pauses for a human decision
    decide   a person approved / rejected / modified -> continue from the pause
    resume   a run was cut off (server restart, lost connection) -> continue
             from the last node that completed

All three go through ``_run``, which streams progress events while the graph
works and ends with the report. The non-streaming API simply waits for the
last event; the streaming API forwards every event to the browser.

Concept - thread_id:
    The checkpointer stores the graph state per ``thread_id``. VeriAgent uses
    the investigation id, so "continue investigation X" needs nothing but X.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import datetime, timedelta, timezone
from typing import Any

from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.checkpoint import get_checkpointer
from app.agents.graph import build_investigation_graph, recursion_limit
from app.agents.nodes import STAGE_LABELS, AgentDeps
from app.agents.report import findings_for_storage, render_report
from app.agents.state import initial_state
from app.config import get_settings
from app.database import repositories
from app.database.models import Investigation, User
from app.database.session import get_session_factory
from app.embeddings.factory import get_embedding_provider
from app.errors import Conflict, NotFound, ServiceUnavailable, TooManyRequests, ValidationFailed, safe_error
from app.llm.factory import get_llm
from app.memory.semantic import remember_investigation
from app.memory.store import get_memory_store
from app.observability.tracing import json_safe, start_trace
from app.rag.retriever import PgVectorRetriever
from app.schemas.response import InvestigationReport
from app.security.guardrails import redact_secrets
from app.security.validation import validate_document_type, validate_question

logger = logging.getLogger(__name__)

_graph: CompiledStateGraph | None = None
_graph_lock = asyncio.Lock()
# Investigations this process is working on right now. Stops a second request
# from running the same investigation at the same time.
_active: set[str] = set()


async def get_graph() -> CompiledStateGraph:
    """Build the graph once and reuse it. It holds no per-request data; that lives in the state."""
    global _graph
    if _graph is not None:
        return _graph
    async with _graph_lock:
        if _graph is None:
            settings = get_settings()
            deps = AgentDeps(
                llm=get_llm(),
                retriever=PgVectorRetriever(
                    get_session_factory(),
                    get_embedding_provider(),
                    hybrid=settings.HYBRID_SEARCH,
                    keyword_top_k=settings.KEYWORD_TOP_K,
                ),
                settings=settings,
                pause_for_approval=True,  # still subject to ENABLE_HUMAN_APPROVAL
                memory=get_memory_store(),  # still subject to ENABLE_MEMORY
            )
            _graph = build_investigation_graph(deps, checkpointer=await get_checkpointer())
    return _graph


def reset_graph() -> None:
    """Forget the built graph (used at shutdown and by tests)."""
    global _graph
    _graph = None


def _config(investigation_id: uuid.UUID) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": str(investigation_id)},
        "recursion_limit": recursion_limit(get_settings().MAX_AGENT_ITERATIONS),
    }


def _waiting_requests(snapshot: Any) -> list[dict[str, Any]]:
    """The approval requests a paused graph is waiting on (empty when it is not paused)."""
    return [
        interrupt.value
        for task in snapshot.tasks
        for interrupt in task.interrupts
        if isinstance(interrupt.value, dict) and "draft_report" in interrupt.value
    ]


async def _delete_checkpoints(graph: CompiledStateGraph | None, investigation_id: uuid.UUID) -> None:
    """Remove the saved graph state of an investigation that will not continue."""
    if graph is None or graph.checkpointer is None or get_settings().KEEP_CHECKPOINTS:
        return
    try:
        await graph.checkpointer.adelete_thread(str(investigation_id))
    except Exception as exc:  # noqa: BLE001 - cleanup must never fail a finished investigation
        logger.warning("Could not delete checkpoints of %s: %s", investigation_id, type(exc).__name__)


async def _run(
    session: AsyncSession, investigation: Investigation, graph_input: Any, *, mode: str
) -> AsyncIterator[dict[str, Any]]:
    """Run (or continue) the graph for one investigation.

    ``graph_input`` is the initial state for a new investigation (mode
    "start"), ``Command(resume=...)`` to deliver a human decision ("decide"),
    or ``None`` to continue from the last checkpoint ("resume").

    Yields ``{"type": "progress", "node", "message"}`` and
    ``{"type": "tool", "tool_name", "status"}`` events while the graph works,
    and finally ``{"type": "report", "report": InvestigationReport}``.
    Progress messages are fixed, safe sentences; the model's reasoning is
    never streamed.
    """
    key = str(investigation.id)
    if key in _active:
        raise Conflict("This investigation is being worked on right now. Try again in a moment.")
    _active.add(key)
    try:
        # aclosing(): when our caller stops reading, close the inner generator NOW, so it
        # can record the interruption. Python would otherwise close it "some time later".
        async with aclosing(_run_exclusive(session, investigation, graph_input, mode)) as events:
            async for item in events:
                yield item
    finally:
        _active.discard(key)


async def _run_exclusive(
    session: AsyncSession, investigation: Investigation, graph_input: Any, mode: str
) -> AsyncIterator[dict[str, Any]]:
    settings = get_settings()
    investigation_id = investigation.id
    user_id = str(investigation.user_id)
    was_waiting = investigation.approval_status in ("pending", "deciding")
    earlier_stats = (investigation.report or {}).get("stats", {})
    started = time.perf_counter()
    trace = start_trace()  # collects every node, LLM call, tool call and search of this run
    config = _config(investigation_id)
    graph: CompiledStateGraph | None = None

    try:
        # Hand the request's database connection back to the pool while the graph
        # works: the graph's own searches and checkpoints need connections too.
        await session.commit()
        graph = await get_graph()  # inside the try: a missing API key surfaces here
        skip = False
        if graph_input is None:
            before = await graph.aget_state(config)
            # Nothing left to run: the graph finished, only storing the result was cut off.
            # (A node that was cut off mid-way shows up in ``tasks`` even when ``next`` is empty.)
            skip = not before.next and not before.tasks
        if not skip:
            # Two streams at once: "custom" carries "node X started" (sent by the node
            # itself while it works), "updates" carries what each node returned.
            announced: dict[str, str] = {}
            async for stream_mode, chunk in graph.astream(graph_input, config=config, stream_mode=["updates", "custom"]):
                if stream_mode == "custom":
                    if isinstance(chunk, dict) and chunk.get("type") == "node_started":
                        node = str(chunk.get("node", ""))
                        label = STAGE_LABELS.get(node)
                        if label:
                            announced[node] = label
                            yield {"type": "progress", "node": node, "message": label}
                    continue
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue  # e.g. the "__interrupt__" marker
                    for item in update.get("events", []):
                        message = item.get("message", "")
                        if message and message != announced.get(node):  # do not repeat the "started" line
                            yield {"type": "progress", "node": item.get("node", node), "message": message}
                    for call in update.get("tool_calls", []):
                        yield {"type": "tool", "tool_name": str(call.get("tool_name", ""))[:100], "status": call.get("status", "")}
                    announced.pop(node, None)
        snapshot = await graph.aget_state(config)
        waiting = _waiting_requests(snapshot)
        state = snapshot.values
        raw_report = waiting[0]["draft_report"] if waiting else state["report"]
        if waiting:
            yield {"type": "progress", "node": "human_approval", "message": "Waiting for a human decision..."}

        # Output validation: the whole report passes through the redaction filter,
        # so nothing that looks like a key or password leaves the system.
        report = InvestigationReport.model_validate_json(
            redact_secrets(InvestigationReport(**raw_report).model_dump_json(), settings)
        )
        totals = trace.totals()
        # An investigation can span several runs (pause -> decision); add them up.
        report.stats.duration_ms = round(
            float(earlier_stats.get("duration_ms") or 0) + (time.perf_counter() - started) * 1000, 1
        )
        report.stats.input_tokens = int(earlier_stats.get("input_tokens") or 0) + totals["input_tokens"]
        report.stats.output_tokens = int(earlier_stats.get("output_tokens") or 0) + totals["output_tokens"]
        if report.stats.llm_calls == 0 and state.get("used_fallback"):
            # Be explicit when no model answered at all, so a rule-based result is never mistaken for an LLM one.
            report.stats.llm_provider = f"{report.stats.llm_provider} (unavailable - rules used)"
        report.answer_text = render_report(report)

        paused = report.human_review.status == "pending"
        tool_calls = [
            {
                "tool_name": str(call["tool_name"])[:100],  # the name may come from the model
                **{key: json_safe(call[key]) for key in ("arguments", "result", "status", "latency_ms")},
            }
            for call in state.get("tool_calls", [])
        ]
        await repositories.save_investigation_result(
            session,
            investigation,
            status=report.status,
            risk_level=report.risk_level,
            confidence_score=report.confidence,
            report=report.model_dump(mode="json"),
            findings=findings_for_storage(report),
            tool_calls=tool_calls,
            approval_status=report.human_review.status,
            approval_note=report.human_review.note,
            completed=not paused,
        )
    except (asyncio.CancelledError, GeneratorExit):
        # The caller went away (the browser closed the stream, or the server is stopping).
        # shield(): finish recording what happened even though this task is being cancelled.
        await asyncio.shield(_record_stop(session, graph, investigation, trace, mode, was_waiting, None))
        raise
    except Exception as exc:
        logger.error(
            "investigation_failed id=%s error=%s: %s", investigation_id, type(exc).__name__,
            redact_secrets(str(exc), settings),
        )
        await _record_stop(session, graph, investigation, trace, mode, was_waiting, exc)
        if isinstance(exc, GraphRecursionError):
            raise ServiceUnavailable(
                "The investigation was stopped because it needed too many steps. Try a narrower question."
            ) from exc
        raise

    # ---- the result is stored; what follows can no longer fail the investigation
    if not paused:
        # Only now, so memory never tells of an outcome the database does not have.
        await remember_investigation(get_memory_store(), settings, user_id, report)
        await _delete_checkpoints(graph, investigation_id)  # the result is in the normal tables now
    await _save_trace(session, investigation_id, trace)
    logger.info(
        "investigation_%s id=%s status=%s risk=%s confidence=%.2f approval=%s llm_calls=%d tool_calls=%d duration_ms=%.0f",
        "paused" if paused else "completed", investigation_id, report.status, report.risk_level, report.confidence,
        report.human_review.status, report.stats.llm_calls, report.stats.tool_calls, report.stats.duration_ms,
    )
    yield {"type": "report", "report": report}


async def _save_trace(session: AsyncSession, investigation_id: uuid.UUID, trace: Any) -> None:
    try:
        await repositories.save_trace(session, investigation_id, trace.events)
    except Exception as exc:  # noqa: BLE001 - a trace is diagnostics; losing it must not lose the result
        await session.rollback()
        logger.warning("Could not save the trace of %s: %s", investigation_id, type(exc).__name__)


async def _record_stop(
    session: AsyncSession,
    graph: CompiledStateGraph | None,
    investigation: Investigation,
    trace: Any,
    mode: str,
    was_waiting: bool,
    error: BaseException | None,
) -> None:
    """A run ended without a stored result. Leave the investigation in a state that says truthfully what can happen next.

    ``error`` is None when the run was cancelled (the caller went away).

        still waiting for the decision  -> "pending": the decision was not applied and can be made again
        cancelled, or the decision was
        recorded and a later step failed -> "interrupted": POST /investigations/{id}/resume continues at once
        anything else                    -> "failed": a start that failed, or a resume that failed again
                                            (so a broken investigation is not retried forever)
    """
    investigation_id = investigation.id
    try:
        await session.rollback()
        still_waiting, has_progress = False, False
        if graph is not None:
            try:
                snapshot = await graph.aget_state(_config(investigation_id))
                still_waiting, has_progress = bool(_waiting_requests(snapshot)), bool(snapshot.values)
            except Exception:  # noqa: BLE001
                pass
        if error is None:
            reason = "The run was cut off before it finished (the connection was closed)."
        else:
            reason = safe_error(error)[1]["message"]
            if isinstance(error, GraphRecursionError):
                reason = "The investigation needed too many steps."

        if was_waiting and still_waiting:
            await repositories.set_approval_status(session, investigation_id, "pending")
        elif has_progress and (error is None or mode == "decide"):
            await repositories.mark_investigation_interrupted(session, investigation_id, reason)
        else:
            await repositories.mark_investigation_failed(session, investigation_id, reason)
            await _delete_checkpoints(graph, investigation_id)
        await _save_trace(session, investigation_id, trace)  # the trace matters most when something went wrong
    except Exception as exc:  # noqa: BLE001 - never hide the original error behind a bookkeeping error
        logger.error("Could not record why investigation %s stopped: %s", investigation_id, type(exc).__name__)


async def _last_report(events: AsyncIterator[dict[str, Any]]) -> InvestigationReport:
    report: InvestigationReport | None = None
    async for item in events:
        if item["type"] == "report":
            report = item["report"]
    if report is None:  # cannot happen: _run either yields a report or raises
        raise ServiceUnavailable("The investigation ended without a report.")
    return report


# ---------------------------------------------------------------------- start
async def stream_investigation(
    session: AsyncSession,
    *,
    user: User,
    question: str,
    document_ids: list[uuid.UUID] | None = None,
    document_type: str | None = None,
    year: int | None = None,
    conversation_id: uuid.UUID | None = None,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Start an investigation and yield its events as they happen.

    The first event is ``{"type": "started", "investigation": Investigation}``,
    then progress events, then ``{"type": "report", ...}``.
    """
    settings = get_settings()
    question = validate_question(question, settings)  # input validation
    filters = {"document_type": validate_document_type(document_type), "year": year}

    # Ownership check: a user can only investigate their own documents.
    document_ids = list(dict.fromkeys(document_ids or []))
    if document_ids:
        owned = await repositories.owned_document_ids(session, user.id, document_ids)
        if owned != set(document_ids):
            raise NotFound("One or more of the selected documents were not found.")

    if settings.MAX_INVESTIGATIONS_PER_DAY:
        # Every investigation spends model calls. On a public deployment this cap keeps
        # one account from using up the whole quota.
        since = datetime.now(timezone.utc) - timedelta(days=1)
        used = await repositories.count_recent_investigations(session, user.id, since)
        if used >= settings.MAX_INVESTIGATIONS_PER_DAY:
            raise TooManyRequests(
                f"You have reached the limit of {settings.MAX_INVESTIGATIONS_PER_DAY} investigations in 24 hours. "
                "Try again later."
            )

    investigation = await repositories.create_investigation(session, user.id, question, conversation_id)
    logger.info("investigation_started id=%s user=%s", investigation.id, user.id)
    yield {"type": "started", "investigation": investigation}

    state = initial_state(
        user_id=str(user.id),
        investigation_id=str(investigation.id),  # also the checkpoint thread_id
        question=question,
        history=history,
        document_ids=[str(document_id) for document_id in document_ids],
        filters={name: value for name, value in filters.items() if value is not None},
    )
    async with aclosing(_run(session, investigation, state, mode="start")) as events:
        async for item in events:
            yield item


async def run_investigation(session: AsyncSession, **kwargs: Any) -> tuple[Investigation, InvestigationReport]:
    """Start an investigation and wait for its report (final, or a draft waiting for approval)."""
    investigation: Investigation | None = None
    report: InvestigationReport | None = None
    async for item in stream_investigation(session, **kwargs):
        if item["type"] == "started":
            investigation = item["investigation"]
        elif item["type"] == "report":
            report = item["report"]
    assert investigation is not None and report is not None
    return investigation, report


# --------------------------------------------------------------------- decide
async def decide_investigation(
    session: AsyncSession,
    *,
    user: User,
    investigation_id: uuid.UUID,
    decision: str,
    note: str | None = None,
    modified_action: str | None = None,
) -> InvestigationReport:
    """Deliver a person's decision to a paused investigation and return the final report."""
    if decision == "modify" and not (modified_action or "").strip():
        raise ValidationFailed("A 'modify' decision needs 'modified_action': the recommendation to use instead.")
    investigation = await repositories.get_investigation(session, user.id, investigation_id)
    if investigation is None:
        raise NotFound("Investigation not found.")
    if str(investigation_id) in _active:
        raise Conflict("This investigation is being worked on right now. Try again in a moment.")
    if investigation.status == "interrupted":
        raise Conflict("A decision was already recorded and the run was cut off. Use POST /investigations/{id}/resume to finish it.")
    if not await repositories.claim_pending_approval(session, investigation_id):
        raise Conflict("This investigation is not waiting for a decision.")
    investigation.approval_status = "deciding"  # what the claim just wrote

    # ``Command(resume=...)`` continues the paused graph; the value becomes the
    # return value of ``interrupt()`` inside the human_approval node.
    command = Command(
        resume={"decision": decision, "note": note, "modified_action": modified_action, "reviewer": user.name}
    )
    report = await _last_report(_run(session, investigation, command, mode="decide"))
    if report.human_review.status == "pending":
        # The graph refused the decision and is waiting again (see the human_approval node).
        raise ValidationFailed("The decision could not be applied. Check 'decision' and 'modified_action'.")
    return report


# --------------------------------------------------------------------- resume
async def resume_investigation(
    session: AsyncSession,
    *,
    user: User,
    investigation_id: uuid.UUID,
    stale_after_seconds: float | None = None,
) -> InvestigationReport:
    """Continue an investigation whose run was cut off, from its last checkpoint.

    Two situations:

    * status "interrupted": the server itself recorded that the run stopped
      (the stream was closed, or a step failed after the decision was saved).
      It can be resumed at once.
    * status "running" (or a decision stuck in "deciding") with no progress for
      STALE_RUN_SECONDS: the process died without being able to record
      anything. The waiting time is what stops two workers from running the
      same investigation at once.

    A resume that fails marks the investigation failed, so nothing is retried forever.
    """
    settings = get_settings()
    investigation = await repositories.get_investigation(session, user.id, investigation_id)
    if investigation is None:
        raise NotFound("Investigation not found.")
    interrupted = investigation.status == "interrupted"
    cut_off = investigation.status == "running" or investigation.approval_status == "deciding"
    if investigation.status == "failed" or investigation.completed_at is not None or not (interrupted or cut_off):
        raise Conflict("This investigation was not interrupted, so there is nothing to resume.")
    if str(investigation_id) in _active:
        raise Conflict("This investigation is being worked on right now. Try again in a moment.")

    graph = await get_graph()
    snapshot = await graph.aget_state(_config(investigation_id))
    if interrupted:
        if not await repositories.claim_interrupted(session, investigation_id):  # atomic: only one resume wins
            raise Conflict("This investigation is already being resumed.")
        investigation.status = "running"
    else:
        last_progress = investigation.created_at
        if snapshot.created_at:
            last_progress = datetime.fromisoformat(snapshot.created_at)
        idle_seconds = (datetime.now(timezone.utc) - last_progress).total_seconds()
        limit = settings.STALE_RUN_SECONDS if stale_after_seconds is None else stale_after_seconds
        if idle_seconds < limit:
            raise Conflict(
                f"This investigation made progress {int(idle_seconds)} seconds ago and may still be running. "
                f"It can be resumed after {int(limit)} seconds without progress."
            )
    if not snapshot.values:
        # It stopped before the first node finished: there is nothing saved to continue from.
        await repositories.mark_investigation_failed(
            session, investigation_id, "It stopped before anything was saved, so it could not be resumed."
        )
        raise Conflict("Nothing was saved for this investigation before it stopped. Please start it again.")

    logger.info("investigation_resumed id=%s next=%s", investigation_id, ",".join(snapshot.next) or "-")
    return await _last_report(_run(session, investigation, None, mode="resume"))
