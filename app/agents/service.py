"""Run one investigation from start to finish and store the result.

This is the bridge between the API layer and the LangGraph agent:

    API route -> run_investigation() -> graph.ainvoke(state) -> report -> database
"""

from __future__ import annotations

import logging
import time
import uuid
from functools import lru_cache

from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import build_investigation_graph, recursion_limit
from app.agents.nodes import AgentDeps
from app.agents.report import findings_for_storage
from app.agents.state import initial_state
from app.config import get_settings
from app.database import repositories
from app.database.models import Investigation, User
from app.database.session import get_session_factory
from app.embeddings.factory import get_embedding_provider
from app.errors import NotFound, ServiceUnavailable
from app.llm.factory import get_llm
from app.observability.tracing import json_safe, start_trace
from app.rag.retriever import PgVectorRetriever
from app.schemas.response import InvestigationReport
from app.security.guardrails import redact_secrets
from app.security.validation import validate_document_type, validate_question

logger = logging.getLogger(__name__)


@lru_cache
def get_graph() -> CompiledStateGraph:
    """Build the graph once and reuse it. It holds no per-request data; that lives in the state."""
    deps = AgentDeps(
        llm=get_llm(),
        retriever=PgVectorRetriever(
            get_session_factory(),
            get_embedding_provider(),
            hybrid=get_settings().HYBRID_SEARCH,
            keyword_top_k=get_settings().KEYWORD_TOP_K,
        ),
        settings=get_settings(),
    )
    return build_investigation_graph(deps)


async def run_investigation(
    session: AsyncSession,
    *,
    user: User,
    question: str,
    document_ids: list[uuid.UUID] | None = None,
    document_type: str | None = None,
    year: int | None = None,
    conversation_id: uuid.UUID | None = None,
    history: list[dict[str, str]] | None = None,
) -> tuple[Investigation, InvestigationReport]:
    settings = get_settings()
    question = validate_question(question, settings)  # input validation
    filters = {"document_type": validate_document_type(document_type), "year": year}

    # Ownership check: a user can only investigate their own documents.
    document_ids = list(dict.fromkeys(document_ids or []))
    if document_ids:
        owned = await repositories.owned_document_ids(session, user.id, document_ids)
        if owned != set(document_ids):
            raise NotFound("One or more of the selected documents were not found.")

    investigation = await repositories.create_investigation(session, user.id, question, conversation_id)
    investigation_id = investigation.id
    started = time.perf_counter()
    trace = start_trace()  # collects every node, LLM call, tool call and search of this run
    logger.info("investigation_started id=%s user=%s", investigation_id, user.id)

    try:
        graph = get_graph()
        state = await graph.ainvoke(
            initial_state(
                user_id=str(user.id),
                investigation_id=str(investigation_id),  # also the future checkpoint thread_id
                question=question,
                history=history,
                document_ids=[str(document_id) for document_id in document_ids],
                filters={name: value for name, value in filters.items() if value is not None},
            ),
            config={"recursion_limit": recursion_limit(settings.MAX_AGENT_ITERATIONS)},
        )
    except Exception as exc:
        # Record the failure, then let the API layer turn it into a clean error response.
        logger.error("investigation_failed id=%s error=%s: %s", investigation_id, type(exc).__name__, redact_secrets(str(exc), settings))
        await repositories.save_investigation_result(
            session, investigation, status="failed", risk_level=None, confidence_score=None,
            report=None, findings=[], tool_calls=[],
        )
        await repositories.save_trace(session, investigation_id, trace.events)  # the trace matters most when it failed
        if isinstance(exc, GraphRecursionError):
            raise ServiceUnavailable(
                "The investigation was stopped because it needed too many steps. Try a narrower question."
            ) from exc
        raise

    # Output validation: the whole report passes through the redaction filter,
    # so nothing that looks like a key or password leaves the system.
    report = InvestigationReport.model_validate_json(
        redact_secrets(InvestigationReport(**state["report"]).model_dump_json(), settings)
    )
    report.stats.duration_ms = round((time.perf_counter() - started) * 1000, 1)
    totals = trace.totals()
    report.stats.input_tokens = totals["input_tokens"]
    report.stats.output_tokens = totals["output_tokens"]
    if report.stats.llm_calls == 0 and state.get("used_fallback"):
        # Be explicit when no model answered at all, so a rule-based result is never mistaken for an LLM one.
        report.stats.llm_provider = f"{report.stats.llm_provider} (unavailable - rules used)"

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
    )
    await repositories.save_trace(session, investigation_id, trace.events)
    logger.info(
        "investigation_completed id=%s status=%s risk=%s confidence=%.2f llm_calls=%d tool_calls=%d duration_ms=%.0f",
        investigation_id, report.status, report.risk_level, report.confidence,
        report.stats.llm_calls, report.stats.tool_calls, report.stats.duration_ms,
    )
    return investigation, report
