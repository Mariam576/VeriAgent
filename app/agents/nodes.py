"""The ten nodes of the investigation graph (spec section 20).

Concept - node:
    A node is one step of work. It is an async function that receives the
    current state and returns a dict containing only the keys it changed.

Where the LLM is used, and where it is not:

    analyze_request        LLM   (structured output)   fallback: rules
    retrieve               no LLM: embeddings + pgvector + reranker
    extract_evidence       LLM   (structured output)   fallback: rules
                           + code verifies every fact against its source chunk
    detect_contradictions  no LLM: typed comparison
    calculate              no LLM for the arithmetic; the LLM may request extra
                           calculator calls (tool calling), Python validates and runs them
    check_sufficiency      LLM   (structured output)   fallback: rules
    assess_risk            LLM   (structured output)   fallback: rules
                           + rule-based minimum risk level and confidence ceiling
    review_findings        no LLM: rule-based reviewer
    correct_findings       no LLM: applies the reviewer's fixes
    final_report           no LLM: report assembled from structured findings

Two kinds of LLM failure are handled differently (see app/llm/base.py):

* Not usable at all - missing key, or the provider rejects the request (wrong
  key, unknown model: HTTP 400/401/403/404). The investigation stops with a
  clear error. Carrying on with rules would look like success and hide the cause.
* Temporarily failing - timeout, rate limit, server error, unusable output -
  after the bounded retries. The step falls back to rules, and the report says
  so under "Limitations".
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from app.agents import heuristics, prompts
from app.agents.assessment import (
    baseline_risk,
    clean_ids,
    confidence_ceiling,
    known_numbers,
    review_draft,
    suspicious_content,
)
from app.agents.report import build_report
from app.agents.state import InvestigationState
from app.config import Settings
from app.formatting import key_label
from app.llm.base import BaseLLM, LLMConfigurationError, LLMError
from app.memory.extraction import MEMORY_TASK, accepted_memory, build_memory_prompt, fallback_memory, might_contain_memory
from app.memory.profile import describe as describe_preference
from app.memory.profile import with_defaults
from app.memory.semantic import investigation_note
from app.memory.store import MemoryStore
from app.numbers import normalize_unit
from app.observability import tracing
from app.rag.citations import is_grounded, normalize
from app.rag.reranker import select_diverse
from app.rag.retriever import Retriever, SearchFilters
from app.schemas.investigation import (
    COMPARABLE_KEYS,
    KEY_OTHER,
    KEY_PAYMENT_TERMS,
    Calculation,
    Contradiction,
    EvidenceExtraction,
    EvidenceItem,
    ExtractedFact,
    RequestAnalysis,
    RetrievedChunk,
    RiskAssessment,
    SufficiencyAssessment,
)
from app.schemas.memory import MemoryCandidate
from app.schemas.response import InvestigationReport
from app.security.guardrails import dedupe_keep_order, redact_secrets
from app.tools.calculator import CALCULATOR_TOOLS
from app.tools.contradiction import find_contradictions
from app.tools.registry import execute_tool

logger = logging.getLogger(__name__)

MAX_SEARCH_QUERIES = 4
MAX_FOLLOW_UP_QUERIES = 3
MAX_EVIDENCE_ITEMS = 40
MAX_LLM_TOOL_CALLS = 5
MAX_REVIEW_ROUNDS = 2

_CHUNK_REF = re.compile(r"\bC\d+\b", re.IGNORECASE)
# Answers such as "none" in a list of missing evidence mean "nothing is missing".
_NOTHING = {"none", "n/a", "na", "nothing", "nil", "no", "-", "not applicable", "nothing is missing", "none identified"}

NodeFunction = Callable[[Any, InvestigationState], Awaitable[dict[str, Any]]]


@dataclass
class AgentDeps:
    """Everything the nodes need from outside. Passed in, so tests can swap any part."""

    llm: BaseLLM
    retriever: Retriever
    settings: Settings
    # True only when the graph has a checkpointer, because pausing needs somewhere to save the state.
    pause_for_approval: bool = False
    # Long-term memory. None = the agent runs without memory (unit tests, evaluation).
    memory: MemoryStore | None = None


def traced(name: str) -> Callable[[NodeFunction], NodeFunction]:
    """Log when a node starts and ends, with its duration (observability)."""

    def decorator(function: NodeFunction) -> NodeFunction:
        @functools.wraps(function)
        async def wrapper(self: Any, state: InvestigationState) -> dict[str, Any]:
            investigation = state.get("investigation_id", "-")
            logger.info("node_entered node=%s investigation=%s iteration=%s", name, investigation, state.get("iteration", 0))
            started = time.perf_counter()
            try:
                update = await function(self, state)
            except GraphBubbleUp:
                # Not a failure: the node paused the graph on purpose (human approval).
                logger.info("node_paused node=%s investigation=%s", name, investigation)
                tracing.record("node", name, (time.perf_counter() - started) * 1000, paused=True)
                raise
            except Exception as exc:
                logger.exception("node_failed node=%s investigation=%s", name, investigation)
                tracing.record(
                    "error", name, (time.perf_counter() - started) * 1000,
                    error=type(exc).__name__, iteration=state.get("iteration", 0),
                )
                raise
            latency = (time.perf_counter() - started) * 1000
            logger.info("node_exited node=%s investigation=%s latency_ms=%.0f", name, investigation, latency)
            tracing.record("node", name, latency, iteration=state.get("iteration", 0))
            return update

        return wrapper

    return decorator


def event(node: str, message: str) -> dict[str, Any]:
    """A safe progress message for the user interface. Never contains model reasoning."""
    return {"node": node, "message": message}


def _clean_key(key: str | None) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (key or "").strip().lower()).strip("_")
    return cleaned if cleaned in COMPARABLE_KEYS else KEY_OTHER


def _find_chunk(reference: str, by_ref: dict[str, RetrievedChunk]) -> RetrievedChunk | None:
    """Resolve the model's chunk reference. Accepts "C2", "c2", "[C2]", "Chunk C2", "ref=C2"."""
    match = _CHUNK_REF.search(reference or "")
    return by_ref.get(match.group(0).upper()) if match else None


def _question(state: InvestigationState) -> str:
    """The request as later steps should read it: the standalone rewrite when there is one."""
    return state.get("standalone_question") or state["question"]


class InvestigationNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.llm = deps.llm
        self.retriever = deps.retriever
        self.settings = deps.settings
        self.pause_for_approval = deps.pause_for_approval and deps.settings.ENABLE_HUMAN_APPROVAL
        self.memory = deps.memory if deps.settings.ENABLE_MEMORY else None

    # ================================================================ Node 0
    @traced("load_memory")
    async def load_memory(self, state: InvestigationState) -> dict[str, Any]:
        """Read long-term memory before the investigation starts.

        Three things happen, all for the current user only:
        1. if the message states something to remember, store it (see app/memory/extraction.py)
        2. load the user's preferences (structured memory: a direct key/value lookup)
        3. recall notes from earlier sessions that are close in meaning (semantic memory)

        Memory is a convenience. If it fails, the investigation continues without it.
        """
        if self.memory is None:
            return {}
        user_id, question = state["user_id"], state["question"]
        update: dict[str, Any] = {"events": [event("load_memory", "Checking memory...")]}
        notes: list[str] = []
        saved_ids: list[str] = []
        memory_only = False
        try:
            if might_contain_memory(question):
                try:
                    candidate = await self.llm.structured(
                        MemoryCandidate,
                        system=prompts.system_prompt(MEMORY_TASK),
                        user=build_memory_prompt(question),
                        purpose="extract_memory",
                    )
                    update["llm_calls"] = 1
                except LLMConfigurationError:
                    raise
                except LLMError as exc:
                    logger.warning("extract_memory fell back to rules: %s", exc)
                    candidate = fallback_memory(question)
                accepted = accepted_memory(candidate, self.settings)  # the model proposes, code decides
                if accepted is not None:
                    kind, first, second = accepted
                    if kind == "profile":
                        await self.memory.set_preference(user_id, first, second)
                        notes.append(describe_preference(first, second))
                    else:
                        saved_ids.append(await self.memory.add_note(user_id, first, source="user"))
                        notes.append(f"Noted for later: {first}")
                    tracing.record("memory", "save", memory_type=kind)
                    memory_only = candidate.only_memory

            update["preferences"] = with_defaults(await self.memory.get_preferences(user_id))
            if not memory_only:
                recalled = await self.memory.search_notes(
                    user_id, question, top_k=self.settings.MEMORY_TOP_K,
                    min_similarity=self.settings.MEMORY_MIN_SIMILARITY, exclude_ids=saved_ids,
                )
                update["memories"] = recalled
                tracing.record("memory", "recall", notes=[note["id"] for note in recalled])
        except LLMConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - memory must never stop an investigation
            logger.warning("load_memory failed: %s: %s", type(exc).__name__, exc)
            tracing.record("error", "memory", error=type(exc).__name__)
            update["limitations"] = ["Long-term memory was unavailable for this request."]

        update["memory_notes"] = notes
        if memory_only:
            # Nothing to investigate: skip straight to the report, which confirms what was saved.
            update["memory_only"] = True
            update["analysis"] = RequestAnalysis(
                task_type="general", needs_rag=False, needs_calculation=False, needs_external_verification=False,
                decision_requested=False, standalone_question=question, search_queries=[],
            ).model_dump()
        return update

    # ============================================================== Node 11
    @traced("save_memory")
    async def save_memory(self, state: InvestigationState) -> dict[str, Any]:
        """After the final report: leave one note about this investigation in semantic memory."""
        if self.memory is None or not state.get("report"):
            return {}
        note = investigation_note(InvestigationReport(**state["report"]))
        if note is None:
            return {}  # small talk and plain look-ups are not worth remembering
        try:
            await self.memory.add_note(
                state["user_id"], redact_secrets(note, self.settings),
                source="investigation", investigation_id=state.get("investigation_id"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("save_memory failed: %s: %s", type(exc).__name__, exc)
            tracing.record("error", "memory", error=type(exc).__name__)
            return {}
        tracing.record("memory", "save", memory_type="investigation")
        return {"events": [event("save_memory", "Saved a note about this investigation to memory.")]}

    # ================================================================ Node 1
    @traced("analyze_request")
    async def analyze_request(self, state: InvestigationState) -> dict[str, Any]:
        question = state["question"]
        history = state.get("history", [])
        update: dict[str, Any] = {}
        try:
            analysis = await self.llm.structured(
                RequestAnalysis,
                system=prompts.system_prompt(prompts.ANALYZE_TASK),
                user=prompts.build_analysis_prompt(question, history),
                purpose="analyze_request",
            )
            update["llm_calls"] = 1
        except LLMConfigurationError:
            raise
        except LLMError as exc:
            logger.warning("analyze_request fell back to rules: %s", exc)
            analysis = heuristics.fallback_analysis(question, history)
            update["used_fallback"] = True
            update["limitations"] = ["The language model was unavailable while analysing the request; simple rules were used instead."]

        # Guardrail: the documents are the only source of evidence, so anything
        # that is not small talk must search them, whatever the model said.
        if analysis.task_type != "general" and not analysis.needs_rag:
            analysis = analysis.model_copy(update={"needs_rag": True})

        standalone = analysis.standalone_question.strip() or question
        queries = dedupe_keep_order([standalone, *analysis.search_queries])[:MAX_SEARCH_QUERIES]
        update.update(
            analysis=analysis.model_dump(),
            # Later steps read this, so a follow-up like "what about the payment
            # terms?" keeps the context of the earlier turns.
            standalone_question=standalone,
            search_queries=queries if analysis.needs_rag else [],
            events=[event("analyze_request", "Analyzing request...")],
        )
        return update

    # ================================================================ Node 2
    @traced("retrieve")
    async def retrieve(self, state: InvestigationState) -> dict[str, Any]:
        executed = list(state.get("executed_queries", []))
        already_run = {normalize(query) for query in executed}
        pending = [query for query in state.get("search_queries", []) if normalize(query) not in already_run]
        existing = [RetrievedChunk(**chunk) for chunk in state.get("retrieved_documents", [])]
        update: dict[str, Any] = {
            "retrieval_rounds": state.get("retrieval_rounds", 0) + 1,
            "executed_queries": executed + pending,
            "retrieve_again": False,
            "new_chunk_ids": [],
        }
        if not pending:
            update["events"] = [event("retrieve", "No new search queries to run.")]
            return update

        chosen = state.get("filters", {})
        filters = SearchFilters(
            document_ids=list(state.get("document_ids", [])),
            document_type=chosen.get("document_type"),
            year=chosen.get("year"),
        )
        # Independent searches run at the same time (async I/O, spec section 32).
        results = await asyncio.gather(
            *(
                self.retriever.search(
                    state["user_id"], query,
                    top_k=self.settings.RAG_TOP_K, rerank_top_k=self.settings.RERANK_TOP_K, filters=filters,
                )
                for query in pending
            ),
            return_exceptions=True,
        )

        known_ids = {chunk.chunk_id for chunk in existing}
        candidates: dict[str, RetrievedChunk] = {}
        errors: list[str] = []
        for query, result in zip(pending, results):
            if isinstance(result, BaseException):
                logger.error("retrieval_failed query=%r error=%s: %s", query[:80], type(result).__name__, result)
                errors.append(f"Search failed for query '{query[:80]}': {type(result).__name__}")
                tracing.record("error", "retrieval", query=query[:200], error=type(result).__name__)
                continue
            tracing.record(
                "retrieval", "search", query=query[:200],
                chunks=[
                    {"chunk_id": chunk.chunk_id, "document": chunk.filename, "page": chunk.page,
                     "score": chunk.score, "matched_by": chunk.matched_by}
                    for chunk in result
                ],
            )
            for chunk in result:
                best = candidates.get(chunk.chunk_id)
                if chunk.chunk_id not in known_ids and (best is None or chunk.rerank_score > best.rerank_score):
                    candidates[chunk.chunk_id] = chunk

        # Guardrail: never hold more than MAX_RETRIEVED_CHUNKS chunks in total.
        # The first search may not use the whole budget, so that a later
        # "retrieve again" round still has room for what it finds.
        budget = self.settings.MAX_RETRIEVED_CHUNKS
        if not existing:
            reserve = min(self.settings.RERANK_TOP_K, budget // 3)
            budget = max(budget - reserve, 1)
        room = max(budget - len(existing), 0)
        selected = select_diverse(sorted(candidates.values(), key=lambda c: c.rerank_score, reverse=True), room)
        new_chunks = [
            chunk.model_copy(update={"ref": f"C{len(existing) + index}"})
            for index, chunk in enumerate(selected, start=1)
        ]

        update["retrieved_documents"] = [chunk.model_dump() for chunk in existing + new_chunks]
        update["new_chunk_ids"] = [chunk.chunk_id for chunk in new_chunks]
        update["events"] = [
            event("retrieve", "Searching documents..."),
            event("retrieve", f"{len(new_chunks)} relevant evidence item(s) found..."),
        ]
        if errors:
            update["errors"] = errors
            update["limitations"] = ["Part of the document search failed, so some evidence may be missing."]
        return update

    # ================================================================ Node 3
    @traced("extract_evidence")
    async def extract_evidence(self, state: InvestigationState) -> dict[str, Any]:
        retrieved = [RetrievedChunk(**chunk) for chunk in state.get("retrieved_documents", [])]
        new_ids = set(state.get("new_chunk_ids", []))
        new_chunks = [chunk for chunk in retrieved if chunk.chunk_id in new_ids]
        if not new_chunks:
            return {"events": [event("extract_evidence", "No new document text to read.")]}

        update: dict[str, Any] = {}
        source = "llm"
        try:
            extraction = await self.llm.structured(
                EvidenceExtraction,
                system=prompts.system_prompt(prompts.EXTRACT_TASK),
                user=prompts.build_extraction_prompt(_question(state), new_chunks),
                purpose="extract_evidence",
            )
            update["llm_calls"] = 1
        except LLMConfigurationError:
            raise
        except LLMError as exc:
            logger.warning("extract_evidence fell back to rules: %s", exc)
            extraction = heuristics.rule_based_extraction(new_chunks)
            source = "rules"
            update["used_fallback"] = True
            update["limitations"] = [
                "The language model was unavailable while reading the documents; only amounts and payment "
                "terms in common wordings were extracted."
            ]

        evidence = [EvidenceItem(**item) for item in state.get("evidence", [])]
        accepted, discarded = self._ground_facts(extraction.facts, new_chunks, evidence, source)
        evidence += accepted

        update["evidence"] = [item.model_dump() for item in evidence]
        update["discarded_facts"] = state.get("discarded_facts", 0) + discarded
        update["events"] = [event("extract_evidence", f"{len(accepted)} supported fact(s) extracted...")]
        if discarded:
            update.setdefault("limitations", []).append(
                f"{discarded} extracted statement(s) were discarded because the cited text did not support them."
            )
        return update

    def _ground_facts(
        self, facts: list[ExtractedFact], chunks: list[RetrievedChunk], existing: list[EvidenceItem], source: str
    ) -> tuple[list[EvidenceItem], int]:
        """Turn model output into evidence items, keeping only what the source text supports."""
        by_ref = {chunk.ref.upper(): chunk for chunk in chunks}
        seen = {(item.document_id, item.key, item.value_number, normalize(item.claim)) for item in existing}
        seen_typed = {(item.document_id, item.key, item.value_number) for item in existing if item.key != KEY_OTHER}
        accepted: list[EvidenceItem] = []
        discarded = 0

        for fact in facts:
            if len(existing) + len(accepted) >= MAX_EVIDENCE_ITEMS:
                break
            chunk = _find_chunk(fact.chunk_ref, by_ref)
            claim = fact.claim.strip()
            # Unknown chunk reference = invented source. Not grounded = unsupported claim.
            if chunk is None or not claim or not is_grounded(fact.quote, fact.value_number, chunk.text):
                discarded += 1
                logger.info("fact_discarded ref=%s reason=%s", fact.chunk_ref, "unknown_ref" if chunk is None else "not_grounded")
                continue

            key = _clean_key(fact.key)
            if fact.value_number is None:
                key = KEY_OTHER  # a comparison key needs a number to compare
            unit = "days" if key == KEY_PAYMENT_TERMS else normalize_unit(fact.unit)
            identity = (chunk.document_id, key, fact.value_number, normalize(claim))
            typed_identity = (chunk.document_id, key, fact.value_number)
            if identity in seen or (key != KEY_OTHER and typed_identity in seen_typed):
                continue  # the same fact from an overlapping chunk
            seen.add(identity)
            if key != KEY_OTHER:
                seen_typed.add(typed_identity)

            accepted.append(
                EvidenceItem(
                    id=f"E{len(existing) + len(accepted) + 1}",
                    claim=claim[:400],
                    key=key,
                    value_number=fact.value_number,
                    unit=unit,
                    quote=fact.quote.strip()[:400],
                    extracted_by=source,  # type: ignore[arg-type]
                    # Source fields come from the database record, not from the model.
                    chunk_id=chunk.chunk_id,
                    chunk_ref=chunk.ref,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    document_type=chunk.document_type,
                    page=chunk.page,
                )
            )
        return accepted, discarded

    # ================================================================ Node 4
    @traced("detect_contradictions")
    async def detect_contradictions(self, state: InvestigationState) -> dict[str, Any]:
        evidence = [EvidenceItem(**item) for item in state.get("evidence", [])]
        contradictions = find_contradictions(evidence)
        return {
            "contradictions": [item.model_dump() for item in contradictions],
            "events": [
                event("detect_contradictions", "Checking contradictions..."),
                event("detect_contradictions", f"{len(contradictions)} contradiction(s) detected."),
            ],
        }

    # ================================================================ Node 5
    @traced("calculate")
    async def calculate(self, state: InvestigationState) -> dict[str, Any]:
        evidence = [EvidenceItem(**item) for item in state.get("evidence", [])]
        contradictions = [Contradiction(**item) for item in state.get("contradictions", [])]
        calculations = [Calculation(**item) for item in state.get("calculations", [])]
        analysis = state.get("analysis", {})
        new_records: list[dict[str, Any]] = []
        update: dict[str, Any] = {"events": [event("calculate", "Calculating differences...")]}

        def run(tool_name: str, arguments: dict[str, Any], label: str, requested_by: str) -> Calculation:
            for done in calculations:
                if done.tool_name == tool_name and done.arguments == arguments:
                    return done  # never pay for the same calculation twice
            record = execute_tool(tool_name, arguments)
            new_records.append(record.model_dump())
            calculation = Calculation(
                id=f"K{len(calculations) + 1}",
                tool_name=tool_name,
                arguments=arguments,
                result=record.result["value"] if record.result else None,
                label=label,
                status=record.status,
                error=record.error,
                requested_by=requested_by,  # type: ignore[arg-type]
            )
            calculations.append(calculation)
            return calculation

        # --- Part 1: calculations every numeric contradiction needs. Chosen by code, run by the tool.
        for item in contradictions:
            if item.type in ("unit_mismatch", "internal_inconsistency"):
                continue  # different currencies cannot be subtracted; one-document notes need no maths
            label = key_label(item.key)
            difference = run(
                "calculate_difference", {"value_a": item.value_b, "value_b": item.value_a},
                f"{label}: {item.source_b} minus {item.source_a}", "system",
            )
            item.difference = difference.result
            change = run(
                "calculate_percentage_change", {"old": item.value_a, "new": item.value_b},
                f"{label}: percentage change from {item.source_a} to {item.source_b}", "system",
            )
            item.percentage_change = change.result

        # --- Part 2: tool calling. The model may request further calculations.
        numeric_evidence = [item for item in evidence if item.value_number is not None]
        if analysis.get("needs_calculation") and not state.get("tool_selection_done") and len(numeric_evidence) >= 2:
            update["tool_selection_done"] = True
            try:
                requests = await self.llm.choose_tools(
                    system=prompts.system_prompt(prompts.TOOLS_TASK),
                    user=prompts.build_tools_prompt(_question(state), evidence, calculations),
                    tools=CALCULATOR_TOOLS,
                    purpose="choose_calculations",
                )
                update["llm_calls"] = 1
            except LLMConfigurationError:
                raise
            except LLMError as exc:
                logger.warning("choose_calculations failed, keeping the standard calculations: %s", exc)
                requests = []
                update["limitations"] = ["The language model was unavailable for selecting extra calculations; the standard ones were still performed."]

            if requests:
                requests = requests[:MAX_LLM_TOOL_CALLS]
                # Guardrail on tool arguments: the model may only pass numbers
                # that exist in the evidence or in earlier tool results.
                allowed = known_numbers(_question(state), [], evidence, contradictions, calculations)
                messages: list[Any] = [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": r.name, "args": r.args, "id": r.id or f"call-{i}"} for i, r in enumerate(requests, 1)],
                    )
                ]
                for index, request in enumerate(requests, start=1):
                    numbers = _flatten_numbers(request.args)
                    invented = [n for n in numbers if not any(abs(abs(n) - k) <= 0.005 * max(1.0, k) for k in allowed)]
                    if invented:
                        record = {
                            "tool_name": request.name, "arguments": request.args, "result": None, "status": "rejected",
                            "latency_ms": 0.0, "error": f"Argument {invented[0]:g} is not a value from the evidence.",
                        }
                        new_records.append(record)
                        content = f"rejected: {record['error']}"
                    else:
                        calculation = run(request.name, request.args, f"{request.name} requested by the model", "llm")
                        content = str(calculation.result) if calculation.status == "ok" else f"{calculation.status}: {calculation.error}"
                    # The tool result goes back into the message history as a ToolMessage.
                    messages.append(ToolMessage(content=content, tool_call_id=request.id or f"call-{index}"))
                update["messages"] = messages

        update["calculations"] = [item.model_dump() for item in calculations]
        update["contradictions"] = [item.model_dump() for item in contradictions]
        if new_records:
            update["tool_calls"] = new_records
        return update

    # ================================================================ Node 6
    @traced("check_sufficiency")
    async def check_sufficiency(self, state: InvestigationState) -> dict[str, Any]:
        evidence = [EvidenceItem(**item) for item in state.get("evidence", [])]
        contradictions = [Contradiction(**item) for item in state.get("contradictions", [])]
        calculations = [Calculation(**item) for item in state.get("calculations", [])]
        analysis = state.get("analysis", {})
        iteration = state.get("iteration", 0)
        update: dict[str, Any] = {"retrieve_again": False, "events": [event("check_sufficiency", "Checking whether the evidence is sufficient...")]}

        if analysis.get("task_type") == "general":
            update["sufficiency"] = {"sufficient": True, "missing": [], "follow_up_queries": []}
            update["missing_evidence"] = []
            return update

        previous = state.get("sufficiency")
        if not evidence:
            # Nothing was found at all: no need to spend an LLM call to learn that.
            assessment = SufficiencyAssessment(
                sufficient=False, missing=[heuristics.MISSING_ANY_EVIDENCE], follow_up_queries=[]
            )
        elif (
            previous
            and state.get("retrieval_rounds", 0) > 1
            and len(evidence) == state.get("evidence_count_assessed", -1)
        ):
            # The follow-up search added no new supported facts, so the earlier
            # assessment still stands. Asking the model again would cost a call
            # and could only start another fruitless search.
            assessment = SufficiencyAssessment(**previous).model_copy(update={"follow_up_queries": []})
            if state.get("new_chunk_ids"):
                update["limitations"] = [
                    "A second search for the missing evidence found more text in the uploaded documents, "
                    "but nothing in it supplied the missing evidence."
                ]
            else:
                update["limitations"] = ["A second search for the missing evidence found nothing new in the uploaded documents."]
        else:
            try:
                assessment = await self.llm.structured(
                    SufficiencyAssessment,
                    system=prompts.system_prompt(prompts.SUFFICIENCY_TASK),
                    user=prompts.build_sufficiency_prompt(
                        _question(state), bool(analysis.get("decision_requested")), evidence, contradictions,
                        calculations, state.get("executed_queries", []),
                        sorted({chunk["filename"] for chunk in state.get("retrieved_documents", [])}),
                    ),
                    purpose="check_sufficiency",
                )
                update["llm_calls"] = 1
            except LLMConfigurationError:
                raise
            except LLMError as exc:
                logger.warning("check_sufficiency fell back to rules: %s", exc)
                assessment = heuristics.fallback_sufficiency(
                    question=_question(state),
                    decision_requested=bool(analysis.get("decision_requested")),
                    investigating=analysis.get("task_type") == "document_investigation",
                    evidence=[item.model_dump() for item in evidence],
                    contradictions=[item.model_dump() for item in contradictions],
                    executed_queries=state.get("executed_queries", []),
                )
                update["used_fallback"] = True
                update["limitations"] = ["The language model was unavailable while judging evidence sufficiency; simple rules were used instead."]

        missing = [text for text in dedupe_keep_order(assessment.missing) if text.strip(" .").lower() not in _NOTHING]
        if not assessment.sufficient and not missing:
            # "Not sufficient" without saying why must still be visible in the report.
            missing = ["The evidence was judged insufficient, but what is missing was not specified."]
        sufficient = assessment.sufficient and not missing
        update["sufficiency"] = {"sufficient": sufficient, "missing": missing, "follow_up_queries": assessment.follow_up_queries}
        update["missing_evidence"] = missing
        update["evidence_count_assessed"] = len(evidence)

        # ---- Loop decision: search again, or move on? (agent loop with protection)
        already_run = {normalize(query) for query in state.get("executed_queries", [])}
        new_queries = [
            query for query in dedupe_keep_order(assessment.follow_up_queries)[:MAX_FOLLOW_UP_QUERIES]
            if normalize(query) not in already_run
        ]
        room_left = len(state.get("retrieved_documents", [])) < self.settings.MAX_RETRIEVED_CHUNKS
        if not sufficient and new_queries and not room_left:
            update.setdefault("limitations", []).append(
                f"The limit of {self.settings.MAX_RETRIEVED_CHUNKS} retrieved chunks was reached, so no further "
                "search for the missing evidence was run."
            )
        if not sufficient and new_queries and room_left:
            if iteration < self.settings.MAX_AGENT_ITERATIONS:
                update["retrieve_again"] = True
                update["iteration"] = iteration + 1
                update["search_queries"] = list(state.get("search_queries", [])) + new_queries
                update["events"].append(event("check_sufficiency", "Evidence incomplete - searching again..."))
            else:
                # Loop protection: stop, keep the partial findings, say so. Never fabricate.
                update["iteration_limit_reached"] = True
                update.setdefault("limitations", []).append(
                    f"The investigation stopped after the maximum of {self.settings.MAX_AGENT_ITERATIONS} "
                    "iteration(s). The findings are partial and some evidence was not searched for."
                )
        return update

    # ================================================================ Node 7
    @traced("assess_risk")
    async def assess_risk(self, state: InvestigationState) -> dict[str, Any]:
        retrieved = [RetrievedChunk(**chunk) for chunk in state.get("retrieved_documents", [])]
        evidence = [EvidenceItem(**item) for item in state.get("evidence", [])]
        contradictions = [Contradiction(**item) for item in state.get("contradictions", [])]
        calculations = [Calculation(**item) for item in state.get("calculations", [])]
        missing = list(state.get("missing_evidence", []))
        analysis = state.get("analysis", {})
        update: dict[str, Any] = {"events": [event("assess_risk", "Assessing risk...")]}

        suspicious = suspicious_content(retrieved)
        baseline = baseline_risk(contradictions, missing, suspicious)
        fallback_kwargs = dict(
            baseline_level=baseline,
            contradictions=[item.model_dump() for item in contradictions],
            missing=[{"id": f"M{i}", "missing": text} for i, text in enumerate(missing, start=1)],
            suspicious=suspicious,
            has_evidence=bool(evidence),
        )

        source = "rules"
        if analysis.get("task_type") == "general" or not evidence:
            assessment = heuristics.fallback_risk(**fallback_kwargs)  # nothing for a model to reason about
        else:
            try:
                assessment = await self.llm.structured(
                    RiskAssessment,
                    system=prompts.system_prompt(prompts.RISK_TASK),
                    user=prompts.build_risk_prompt(
                        _question(state), bool(analysis.get("decision_requested")), evidence, contradictions,
                        calculations, missing, suspicious, baseline,
                    ),
                    purpose="assess_risk",
                )
                update["llm_calls"] = 1
                source = "llm"
            except LLMConfigurationError:
                raise
            except LLMError as exc:
                logger.warning("assess_risk fell back to rules: %s", exc)
                assessment = heuristics.fallback_risk(**fallback_kwargs)
                update["used_fallback"] = True
                update["limitations"] = ["The language model was unavailable for the risk assessment; a rule-based assessment was used instead."]

        ceiling = confidence_ceiling(
            evidence=evidence,
            discarded_facts=state.get("discarded_facts", 0),
            contradictions=contradictions,
            missing_evidence=missing,
            iteration_limit_reached=bool(state.get("iteration_limit_reached")),
            used_fallback=bool(state.get("used_fallback") or update.get("used_fallback")),
        )
        draft = assessment.model_dump()
        for reason in draft["reasons"]:
            reason["evidence_ids"] = clean_ids(reason["evidence_ids"])  # "x1", "[E2]" -> X1, E2
        draft.update(source=source, baseline_level=baseline, confidence_ceiling=ceiling)
        if source == "rules":
            draft["confidence"] = ceiling  # rule-based drafts take the evidence-based value directly

        update["risk_draft"] = draft
        update["risk_findings"] = suspicious
        if suspicious:
            names = ", ".join(sorted({item["document"] for item in suspicious}))
            update.setdefault("limitations", []).append(
                f"Text that looks like instructions to an AI system was found in {names}. It was treated as "
                "document content and was not followed."
            )
        return update

    # ================================================================ Node 8
    @traced("review_findings")
    async def review_findings(self, state: InvestigationState) -> dict[str, Any]:
        issues, _, _ = self._run_review(state)
        rounds = state.get("review_rounds", 0) + 1
        update: dict[str, Any] = {
            "review": {"approved": not issues, "issues": issues},
            "review_rounds": rounds,
            "events": [event("review_findings", "Reviewing findings...")],
        }
        if issues:
            logger.info("review_rejected investigation=%s issues=%s", state.get("investigation_id"), issues)
        return update

    def _run_review(self, state: InvestigationState) -> tuple[list[str], dict[str, Any], dict[str, list[str]]]:
        return review_draft(
            state["risk_draft"],
            question=_question(state) + " " + state["question"],
            retrieved=[RetrievedChunk(**chunk) for chunk in state.get("retrieved_documents", [])],
            evidence=[EvidenceItem(**item) for item in state.get("evidence", [])],
            contradictions=[Contradiction(**item) for item in state.get("contradictions", [])],
            calculations=[Calculation(**item) for item in state.get("calculations", [])],
            missing_evidence=list(state.get("missing_evidence", [])),
            suspicious=list(state.get("risk_findings", [])),
        )

    # ------------------------------------------------- correction after review
    @traced("correct_findings")
    async def correct_findings(self, state: InvestigationState) -> dict[str, Any]:
        issues, corrected, drops = self._run_review(state)
        # This loop has its own bound (MAX_REVIEW_ROUNDS, enforced in routing.py).
        # "iteration" counts repeat searches only.
        update: dict[str, Any] = {
            "risk_draft": corrected,
            "events": [event("correct_findings", f"Correcting {len(issues)} issue(s) found in review...")],
            # Transparency: the reader should know that the model's draft was changed, and why.
            "limitations": [
                "The reviewer corrected the draft assessment: " + " ".join(issues[:4])
                + (f" (and {len(issues) - 4} more)" if len(issues) > 4 else "")
            ],
        }
        if drops["evidence"]:
            keep = [item for item in state.get("evidence", []) if item["id"] not in drops["evidence"]]
            update["evidence"] = keep
            update["contradictions"] = [item.model_dump() for item in find_contradictions([EvidenceItem(**i) for i in keep])]
        if drops["calculations"]:
            update["calculations"] = [item for item in state.get("calculations", []) if item["id"] not in drops["calculations"]]
        return update

    def _reviewed_state(self, state: InvestigationState) -> tuple[dict[str, Any], list[str]]:
        """The state as it should be reported: with the reviewer's last corrections applied."""
        working = dict(state)
        limitations: list[str] = []
        review = state.get("review", {})
        if not review.get("approved", True):
            # The reviewer still objects after the allowed correction rounds:
            # publish the rule-corrected version and say that this happened.
            _, corrected, _ = self._run_review(state)
            working["risk_draft"] = corrected
            limitations.append("The reviewer's remaining objections were resolved by rule: " + "; ".join(review.get("issues", [])[:3]))
            working["limitations"] = list(state.get("limitations", [])) + limitations
        return working, limitations

    def _build_report(self, working: dict[str, Any]):
        return build_report(
            working,
            human_approval_enabled=self.settings.ENABLE_HUMAN_APPROVAL,
            llm_provider=self.llm.provider,
            duration_ms=0.0,  # filled in by the service, which times the whole run
        )

    # ================================================================ Node 9
    @traced("human_approval")
    async def human_approval(self, state: InvestigationState) -> dict[str, Any]:
        """Pause for a person when the findings call for it.

        Concept - human-in-the-loop with ``interrupt``:
            ``interrupt(payload)`` stops the graph right here. LangGraph saves
            the whole state through the checkpointer and returns control to
            the caller, together with the payload (the draft report, so the
            person can see what they are deciding about). Nothing is running
            while the graph waits, so it can wait for seconds or for days.

            When the person decides, the API resumes the graph with
            ``Command(resume=decision)``. This node then runs again from its
            first line, and this time ``interrupt`` returns the decision
            instead of pausing. Code before ``interrupt`` therefore runs
            twice, which is why it only reads the state and has no side effects.
        """
        working, _ = self._reviewed_state(state)
        draft = self._build_report({**working, "approval": {"status": "pending"}})

        if not draft.requires_human_review:
            return {"approval": {"status": "not_required"}}
        if not self.pause_for_approval:
            # Pausing is switched off (or there is no checkpointer): only flag it.
            return {"approval": {"status": "not_requested"}}

        request = {
            "type": "approval_request",
            "investigation_id": state.get("investigation_id"),
            "options": ["approve", "reject", "modify"],
            "draft_report": draft.model_dump(mode="json"),
        }
        # Ask until the answer is usable. An answer that does not make sense
        # must not crash the node: LangGraph would remember it and replay it on
        # the next attempt. Asking again is the safe way to refuse it.
        while True:
            approval, problem = _parse_decision(interrupt(request))
            if approval is not None:
                break
            request = {**request, "error": problem}

        return {
            "approval": approval,
            "events": [event("human_approval", f"Human decision received: {approval['status']}.")],
        }

    # =============================================================== Node 10
    @traced("final_report")
    async def final_report(self, state: InvestigationState) -> dict[str, Any]:
        working, limitations = self._reviewed_state(state)
        report = self._build_report(working)
        update: dict[str, Any] = {
            "report": report.model_dump(),
            "final_answer": report.answer_text,
            "risk_level": report.risk_level,
            "confidence_score": report.confidence,
            "requires_human_review": report.requires_human_review,
            "messages": [AIMessage(content=report.answer_text)],
            "events": [event("final_report", "Generating report...")],
        }
        if "risk_draft" in working:  # absent when the message only asked to remember something
            update["risk_draft"] = working["risk_draft"]
        if limitations:
            update["limitations"] = limitations
        return update


def _parse_decision(decision: Any) -> tuple[dict[str, Any] | None, str]:
    """Validate what a person sent. Returns (approval, "") or (None, what is wrong)."""
    if not isinstance(decision, dict):
        return None, "The decision must be an object with a 'decision' field."
    choice = str(decision.get("decision", "")).strip().lower()
    status = {"approve": "approved", "reject": "rejected", "modify": "modified"}.get(choice)
    if status is None:
        return None, "The decision must be one of: approve, reject, modify."
    modified_action = _clean_text(decision.get("modified_action"))
    if status == "modified" and not modified_action:
        return None, "A 'modify' decision needs the replacement recommendation."
    return {
        "status": status,
        "reviewer": _clean_text(decision.get("reviewer"), 100) or "reviewer",
        "note": _clean_text(decision.get("note")),
        "modified_action": modified_action if status == "modified" else None,
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ""


def _clean_text(value: Any, limit: int = 1000) -> str | None:
    """Text typed by a reviewer: no control characters, bounded length."""
    if value is None:
        return None
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(value)).strip()
    return cleaned[:limit] or None


def _flatten_numbers(value: Any) -> list[float]:
    """All numbers inside tool arguments, however they are nested."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _flatten_numbers(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _flatten_numbers(item)]
    return []
