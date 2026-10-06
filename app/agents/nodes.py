"""The ten nodes of the investigation graph (spec section 20).

Concept - node:
    A node is one step of work. It is an async function that receives the
    current state and returns a dict containing only the keys it changed.

Where the LLM is used, and where it is not:

    analyze_request        LLM   (structured output)   fallback: rules
    retrieve               no LLM: embeddings + pgvector + reranker
    extract_evidence       LLM   (structured output)   fallback: rules
                           + code verifies every fact against its source chunk
    detect_contradictions  no LLM in the comparison itself. Document profiles (parties, dates,
                           line items) are read by the LLM once per document, verified by code
                           and stored                               fallback: rules
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
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from app.agents import heuristics, prompts
from app.actions.runner import ActionRunner
from app.agents.assessment import (
    baseline_risk,
    clean_ids,
    integrity_findings,
    confidence_ceiling,
    known_numbers,
    review_draft,
    suspicious_content,
)
from app.agents.report import action_list, build_report, render_report
from app.agents.reviewer import review_claims, reviewable
from app.agents.state import InvestigationState
from app.comparison.engine import Cited, ProfileFinding, compare, named_pairs
from app.comparison.profile import PROFILED_TYPES
from app.comparison.store import ProfileSource
from app.config import Settings
from app.forensics.store import ForensicsSource
from app.formatting import key_label
from app.llm.base import BaseLLM, LLMConfigurationError, LLMError
from app.memory.extraction import (
    MEMORY_TASK,
    accepted_memory,
    build_memory_prompt,
    fallback_memory,
    is_only_memory,
    might_contain_memory,
)
from app.memory.profile import describe as describe_preference
from app.memory.profile import with_defaults
from app.memory.store import MemoryStore
from app.numbers import normalize_unit
from app.observability import tracing
from app.rag.citations import is_grounded, normalize, numbers_not_in_source, quote_in_text
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
from app.security.guardrails import dedupe_keep_order, trusted_part
from app.tools.calculator import CALCULATOR_TOOLS
from app.tools.contradiction import find_contradictions
from app.tools.registry import execute_tool

logger = logging.getLogger(__name__)

MAX_SEARCH_QUERIES = 4
MAX_FOLLOW_UP_QUERIES = 3
MAX_EVIDENCE_ITEMS = 40
MAX_LLM_TOOL_CALLS = 5
MAX_REVIEW_ROUNDS = 2
MAX_PROFILED_DOCUMENTS = 6  # one model call per document the first time, so keep the set small
MAX_PROFILE_EVIDENCE = 24  # evidence items a comparison may add on top of the extracted facts
# How a passage is marked that the search did not find but the document comparison fetched to cite a value.
CITED_BY_COMPARISON = "comparison"
# A comparison finding may rest on a total or on payment terms that the evidence already holds under its own key.
_SAME_FACT = {"document_total": "total_amount", "document_payment_terms": "payment_terms_days"}

_CHUNK_REF = re.compile(r"\bC\d+\b", re.IGNORECASE)
# Answers such as "none" in a list of missing evidence mean "nothing is missing".
_NOTHING = {"none", "n/a", "na", "nothing", "nil", "no", "-", "not applicable", "nothing is missing", "none identified"}

NodeFunction = Callable[[Any, InvestigationState], Awaitable[dict[str, Any]]]


def _document_pairs(state: dict[str, Any], update: dict[str, Any], name: str) -> set[tuple[str, str]]:
    """Pairs of document ids from the latest comparison (``update``), or from an earlier round (``state``)."""
    return {(pair[0], pair[1]) for pair in update.get(name, state.get(name, [])) if len(pair) == 2}


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
    # Document profiles for the detailed comparison. None = only totals and payment terms are compared.
    profiles: ProfileSource | None = None
    # Forensic reports of the uploaded files. None = signs of editing are not considered.
    forensics: ForensicsSource | None = None
    # The model that double-checks extracted statements. None = they are not double-checked.
    reviewer: BaseLLM | None = None
    # Carries out approved follow-up actions through MCP. None = nothing is ever done outside VeriAgent.
    actions: ActionRunner | None = None


def traced(name: str) -> Callable[[NodeFunction], NodeFunction]:
    """Log when a node starts and ends, with its duration (observability)."""

    def decorator(function: NodeFunction) -> NodeFunction:
        @functools.wraps(function)
        async def wrapper(self: Any, state: InvestigationState) -> dict[str, Any]:
            investigation = state.get("investigation_id", "-")
            logger.info("node_entered node=%s investigation=%s iteration=%s", name, investigation, state.get("iteration", 0))
            _announce(name)
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


# What the user sees while a node is working. None = nothing worth announcing.
STAGE_LABELS: dict[str, str | None] = {
    "load_memory": "Checking memory...",
    "analyze_request": "Analyzing request...",
    "retrieve": "Searching documents...",
    "extract_evidence": "Reading the evidence...",
    "detect_contradictions": "Checking contradictions...",
    "calculate": "Calculating differences...",
    "check_sufficiency": "Checking whether the evidence is sufficient...",
    "assess_risk": "Assessing risk...",
    "review_findings": "Reviewing findings...",
    "correct_findings": "Correcting findings...",
    "human_approval": None,
    "follow_up": None,  # announced only when there is something to carry out
    "final_report": "Generating report...",
}


def _announce(node: str) -> None:
    """Tell a streaming client that a node has started.

    Concept - streaming from inside a node:
        ``get_stream_writer()`` returns a function that pushes a value into
        the graph's "custom" stream. Whoever iterates ``graph.astream(...,
        stream_mode=["updates", "custom"])`` receives it immediately, while
        the node is still working. Only the node's name is sent - never
        prompts, model output or document text.
    """
    if STAGE_LABELS.get(node) is None:
        return
    try:
        get_stream_writer()({"type": "node_started", "node": node})
    except Exception:  # noqa: BLE001 - not inside a streamed graph run (e.g. a unit test calling the node directly)
        pass


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
        self.profiles = deps.profiles if deps.settings.ENABLE_DOCUMENT_COMPARISON else None
        self.forensics = deps.forensics if deps.settings.ENABLE_FORENSICS else None
        self.reviewer = deps.reviewer if deps.settings.ENABLE_CLAIM_REVIEW else None
        self.actions = deps.actions

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
                    # Skipping the investigation is too important to leave to the model:
                    # it happens only when the model AND the code rule agree that nothing else was asked.
                    memory_only = candidate.only_memory and is_only_memory(question)

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

        # Passages that the document comparison fetched only to cite a value were never read for
        # facts. They do not count as "already found": a search may find them, and then they are read.
        cited_only = {chunk.chunk_id for chunk in existing if chunk.matched_by == [CITED_BY_COMPARISON]}
        known_ids = {chunk.chunk_id for chunk in existing} - cited_only
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
        # ... and they do not use up the budget either: it is a limit on what the search may bring in.
        room = max(budget - (len(existing) - len(cited_only)), 0)
        selected = select_diverse(sorted(candidates.values(), key=lambda c: c.rerank_score, reverse=True), room)
        found_again = {chunk.chunk_id: chunk for chunk in selected if chunk.chunk_id in cited_only}
        # A passage that was only cited keeps its place and its reference number; it now also counts as found.
        existing = [
            chunk if chunk.chunk_id not in found_again
            else found_again[chunk.chunk_id].model_copy(update={"ref": chunk.ref})
            for chunk in existing
        ]
        brand_new = [chunk for chunk in selected if chunk.chunk_id not in found_again]
        new_chunks = [
            chunk.model_copy(update={"ref": f"C{len(existing) + index}"})
            for index, chunk in enumerate(brand_new, start=1)
        ]

        update["retrieved_documents"] = [chunk.model_dump() for chunk in existing + new_chunks]
        update["new_chunk_ids"] = [chunk.chunk_id for chunk in selected]
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
        if discarded:
            update.setdefault("limitations", []).append(
                f"{discarded} extracted statement(s) were discarded because the cited text did not support them."
            )
        if source == "llm" and self.reviewer is not None:
            accepted, rejected = await self._second_opinion(accepted, len(evidence), update)
            discarded += rejected
        evidence += accepted

        update["evidence"] = [item.model_dump() for item in evidence]
        update["discarded_facts"] = state.get("discarded_facts", 0) + discarded
        update["events"] = [event("extract_evidence", f"{len(accepted)} supported fact(s) extracted...")]
        return update

    async def _second_opinion(
        self, accepted: list[EvidenceItem], already: int, update: dict[str, Any]
    ) -> tuple[list[EvidenceItem], int]:
        """Let the independent reviewer check the new free-text statements (see app/agents/reviewer.py).

        Returns the statements that remain, numbered without gaps, and how many were removed.
        """
        assert self.reviewer is not None
        if not reviewable(accepted):
            return accepted, 0  # nothing the reviewer looks at: no call, no cost
        try:
            rejected = await review_claims(self.reviewer, accepted)
            update["llm_calls"] = update.get("llm_calls", 0) + 1
        except LLMConfigurationError:
            raise
        except LLMError as exc:
            logger.warning("review_claims unavailable: %s", exc)
            update.setdefault("limitations", []).append(
                "The independent check of the extracted statements was unavailable; they were verified by rules only."
            )
            return accepted, 0
        if not rejected:
            return accepted, 0
        for identifier, reason in rejected.items():
            logger.info("fact_discarded id=%s reason=reviewer: %s", identifier, reason)
        tracing.record("review", "claims", checked=len(accepted), rejected=[{"id": k, "reason": v} for k, v in rejected.items()])
        kept = [item for item in accepted if item.id not in rejected]
        kept = [item.model_copy(update={"id": f"E{already + index}"}) for index, item in enumerate(kept, start=1)]
        update.setdefault("limitations", []).append(
            f"{len(rejected)} extracted statement(s) were discarded by the independent reviewer because the quoted "
            "text did not support them."
        )
        return kept, len(rejected)

    def _ground_facts(
        self, facts: list[ExtractedFact], chunks: list[RetrievedChunk], existing: list[EvidenceItem], source: str
    ) -> tuple[list[EvidenceItem], int]:
        """Turn model output into evidence items, keeping only what the source text supports."""
        by_ref = {chunk.ref.upper(): chunk for chunk in chunks}
        seen = {(item.document_id, item.key, item.value_number, normalize(item.claim)) for item in existing}
        seen_typed = {(item.document_id, item.key, item.value_number) for item in existing if item.key != KEY_OTHER}
        accepted: list[EvidenceItem] = []
        discarded = 0
        trusted: dict[str, str] = {}  # chunk id -> the part of its text that precedes any injected instruction

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
            # The quote is real, but does the claim add a number the document never states?
            if numbers_not_in_source(claim, chunk.text):
                discarded += 1
                logger.info("fact_discarded ref=%s reason=number_not_in_source", fact.chunk_ref)
                continue
            # Text that follows an instruction aimed at the agent is not evidence, whatever it claims.
            trusted_text = trusted.setdefault(chunk.chunk_id, trusted_part(chunk.text))
            if trusted_text != chunk.text and not is_grounded(fact.quote, fact.value_number, trusted_text):
                discarded += 1
                logger.info("fact_discarded ref=%s reason=inside_injected_text", fact.chunk_ref)
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
        """Find where the documents disagree. No LLM takes part in any comparison.

        Two sources feed this step:
        1. the evidence from the search: total amounts and payment terms (app/tools/contradiction.py)
        2. the document profiles: parties, reference numbers, dates and line items (app/comparison)
        """
        evidence = [EvidenceItem(**item) for item in state.get("evidence", [])]
        from_search = list(evidence)  # the comparison below adds its own evidence items to ``evidence``
        update: dict[str, Any] = {"events": [event("detect_contradictions", "Checking contradictions...")]}

        detailed: list[Contradiction] = []
        if self.profiles is not None and state.get("analysis", {}).get("task_type") == "document_investigation":
            detailed = await self._compare_documents(state, evidence, update)
        # The profiles also say which documents belong together. An invoice from another supplier is not
        # held against this contract's total, and two contracts are not held against each other.
        contradictions = find_contradictions(
            from_search,
            unrelated=_document_pairs(state, update, "unrelated_documents"),
            linked=_document_pairs(state, update, "linked_documents"),
            net_totals=update.get("net_totals", state.get("net_totals", {})),
        )
        contradictions += [
            item.model_copy(update={"id": f"X{len(contradictions) + index}"})
            for index, item in enumerate(detailed, start=1)
        ]

        update["contradictions"] = [item.model_dump() for item in contradictions]
        update["events"].append(event("detect_contradictions", f"{len(contradictions)} contradiction(s) detected."))
        return update

    async def _compare_documents(
        self, state: InvestigationState, evidence: list[EvidenceItem], update: dict[str, Any]
    ) -> list[Contradiction]:
        """Compare the profiles of the documents this investigation is about.

        Every finding must be citable: each value it rests on is turned into an
        evidence item that points at the chunk containing its quote. A finding
        with a value that cannot be found again in the stored text is dropped.
        """
        assert self.profiles is not None
        retrieved = [RetrievedChunk(**chunk) for chunk in state.get("retrieved_documents", [])]
        candidates = list(dict.fromkeys(
            chunk.document_id for chunk in retrieved if chunk.document_type in PROFILED_TYPES
        ))
        document_ids = candidates[:MAX_PROFILED_DOCUMENTS]
        if not document_ids:
            return []
        if len(candidates) > MAX_PROFILED_DOCUMENTS:
            # Said out loud (third review): a seventh invoice was left out of the comparison without a word.
            left_out = list(dict.fromkeys(
                chunk.filename for chunk in retrieved if chunk.document_id in candidates[MAX_PROFILED_DOCUMENTS:]
            ))
            note = (
                f"Parties, dates and line items were compared for {MAX_PROFILED_DOCUMENTS} documents only. Not compared in "
                f"detail: {', '.join(left_out[:6])}. Select fewer documents to have these checked."
            )
            if note not in state.get("limitations", []):
                update["limitations"] = [*update.get("limitations", []), note]
        earlier = [Contradiction(**item) for item in state.get("contradictions", []) if item.get("origin") == "profile"]
        if set(document_ids) == set(state.get("compared_documents", [])):
            return earlier  # same documents as in the previous round: the comparison cannot come out differently

        user_id = state["user_id"]
        try:
            batch = await self.profiles.profiles(user_id, document_ids)
        except LLMConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - the detailed comparison is an extra; the investigation goes on without it
            logger.warning("document comparison failed: %s: %s", type(exc).__name__, exc)
            tracing.record("error", "comparison", error=type(exc).__name__)
            update["limitations"] = [
                *update.get("limitations", []),
                "The detailed comparison of parties, dates, reference numbers and line items was unavailable for this request.",
            ]
            return earlier
        if batch.llm_calls:
            update["llm_calls"] = batch.llm_calls
        if batch.used_rules:
            update["used_fallback"] = True
            update["limitations"] = [
                *update.get("limitations", []),
                "The language model was unavailable while reading the document details (parties, dates, line items); "
                "only common layouts were understood.",
            ]

        # What the person asked for overrules what the supplier names suggest: documents that the question
        # names together, or a single invoice selected with its contract, are compared in any case.
        question = f"{state.get('question', '')} {state.get('standalone_question', '')}"
        selected = set(state.get("document_ids") or [])
        one_invoice_selected = bool(selected) and sum(
            profile.document_type == "invoice" and profile.document_id in selected for profile in batch.profiles
        ) == 1
        comparison = compare(batch.profiles, assume_related=one_invoice_selected, forced=named_pairs(question, batch.profiles))
        findings = comparison.findings
        update["unrelated_documents"] = [list(pair) for pair in sorted(comparison.unrelated)]
        update["linked_documents"] = [list(pair) for pair in sorted(comparison.linked)]
        update["net_totals"] = comparison.net_totals
        # "Not checked" must not read as "no problem": every pair that was left alone is said out loud.
        already_said = set(state.get("limitations", [])) | set(update.get("limitations", []))
        notes = [note for note in comparison.notes if note not in already_said]
        if notes:
            update["limitations"] = [*update.get("limitations", []), *notes]
        added_evidence = 0
        page_cache: dict[tuple[str, int], list[RetrievedChunk]] = {}

        async def locate(cited: Cited) -> RetrievedChunk | None:
            """The chunk that contains the quoted words: an already retrieved one if possible."""
            def holds(chunk: RetrievedChunk) -> bool:
                return chunk.document_id == cited.document_id and chunk.page == cited.page and quote_in_text(cited.quote, chunk.text, 2)

            known = next((chunk for chunk in retrieved if holds(chunk)), None)
            if known is not None:
                return known
            slot = (cited.document_id, cited.page)
            if slot not in page_cache:
                page_cache[slot] = await self.profiles.page_chunks(user_id, cited.document_id, cited.page)  # type: ignore[union-attr]
            return next((chunk for chunk in page_cache[slot] if holds(chunk)), None)

        def existing(cited: Cited) -> EvidenceItem | None:
            same_as = _SAME_FACT.get(cited.key)
            for item in evidence:
                if item.document_id != cited.document_id:
                    continue
                same_words = normalize(item.quote) == normalize(cited.quote)
                if item.key == cited.key and same_words:
                    return item
                # A line that the fact extraction already took from the same words: one evidence item, not two.
                if same_words and cited.value_number is not None and item.value_number is not None \
                        and abs(item.value_number - cited.value_number) < 0.005:
                    return item
                if same_as and item.key == same_as and item.value_number is not None and cited.value_number is not None \
                        and abs(item.value_number - cited.value_number) < 0.005:
                    return item
            return None

        results: list[Contradiction] = []
        for finding in findings:
            located = []
            for cited in finding.cited:
                item = existing(cited)
                chunk = None if item is not None else await locate(cited)
                if item is None and chunk is None:
                    break
                located.append((cited, item, chunk))
            new_items = sum(1 for _, item, _ in located if item is None)
            if len(located) != len(finding.cited) or added_evidence + new_items > MAX_PROFILE_EVIDENCE:
                logger.info("comparison_finding_dropped type=%s reason=not_citable", finding.type)
                continue

            evidence_ids = []
            for cited, item, chunk in located:
                if item is None:
                    assert chunk is not None
                    if all(known.chunk_id != chunk.chunk_id for known in retrieved):
                        chunk = chunk.model_copy(update={"ref": f"C{len(retrieved) + 1}", "matched_by": [CITED_BY_COMPARISON]})
                        retrieved.append(chunk)
                    else:
                        chunk = next(known for known in retrieved if known.chunk_id == chunk.chunk_id)
                    item = EvidenceItem(
                        id=f"E{len(evidence) + 1}", claim=cited.claim[:400], key=cited.key,
                        value_number=cited.value_number, unit=cited.unit, quote=cited.quote[:400], extracted_by="profile",
                        chunk_id=chunk.chunk_id, chunk_ref=chunk.ref, document_id=chunk.document_id,
                        filename=chunk.filename, document_type=chunk.document_type, page=chunk.page,
                    )
                    evidence.append(item)
                    added_evidence += 1
                if item.id not in evidence_ids:
                    evidence_ids.append(item.id)
            results.append(_as_contradiction(finding, evidence_ids))

        tracing.record(
            "comparison", "profiles", documents=[profile.filename for profile in batch.profiles],
            built_by_model=batch.llm_calls, findings=[finding.type for finding in findings], kept=len(results),
        )
        update["compared_documents"] = document_ids
        update["evidence"] = [item.model_dump() for item in evidence]
        update["retrieved_documents"] = [chunk.model_dump() for chunk in retrieved]
        update["events"].append(event("detect_contradictions", "Comparing parties, dates and line items..."))
        return results

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
            if not item.is_numeric:
                continue  # names and dates, different currencies, one-document notes: nothing to subtract
            label = key_label(item.key)
            within_one_document = item.source_a == item.source_b
            # Side A of a one-document check is a value that has to be worked out first.
            if item.operation == "sum" and item.operands:
                worked_out = run("sum_values", {"values": item.operands}, f"sum of the line items on {item.source_a}", "system")
                item.value_a = worked_out.result if worked_out.result is not None else item.value_a
            elif item.operation == "multiply" and len(item.operands) == 2:
                worked_out = run(
                    "multiply_values", {"value_a": item.operands[0], "value_b": item.operands[1]},
                    f"quantity x unit price on {item.source_a}", "system",
                )
                item.value_a = worked_out.result if worked_out.result is not None else item.value_a
            difference = run(
                "calculate_difference", {"value_a": item.value_b, "value_b": item.value_a},
                f"{label}: stated value minus calculated value ({item.source_a})" if within_one_document
                else f"{label}: {item.source_b} minus {item.source_a}",
                "system",
            )
            item.difference = difference.result
            if within_one_document:
                continue  # a percentage of a document's own arithmetic says nothing
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
        searched = [chunk for chunk in state.get("retrieved_documents", []) if chunk.get("matched_by") != [CITED_BY_COMPARISON]]
        room_left = len(searched) < self.settings.MAX_RETRIEVED_CHUNKS
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
        integrity = await self._integrity(state, retrieved, update)
        baseline = baseline_risk(contradictions, missing, suspicious, integrity)
        fallback_kwargs = dict(
            baseline_level=baseline,
            contradictions=[item.model_dump() for item in contradictions],
            missing=[{"id": f"M{i}", "missing": text} for i, text in enumerate(missing, start=1)],
            suspicious=suspicious,
            has_evidence=bool(evidence),
            integrity=integrity,
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
                        calculations, missing, suspicious, baseline, integrity,
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
                update.setdefault("limitations", []).append(
                    "The language model was unavailable for the risk assessment; a rule-based assessment was used instead."
                )

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
        update["integrity_findings"] = integrity
        if suspicious:
            names = ", ".join(sorted({item["document"] for item in suspicious}))
            update.setdefault("limitations", []).append(
                f"Text that looks like instructions to an AI system was found in {names}. It was treated as "
                "document content and was not followed."
            )
        return update

    async def _integrity(
        self, state: InvestigationState, retrieved: list[RetrievedChunk], update: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Forensic indicators of the documents this investigation read (app/forensics).

        The files were inspected when they were uploaded; this only looks the results up.
        """
        if self.forensics is None or state.get("analysis", {}).get("task_type") == "general":
            return []
        document_ids = list(dict.fromkeys(chunk.document_id for chunk in retrieved))
        if not document_ids:
            return []
        try:
            reports = await self.forensics.reports(state["user_id"], document_ids)
        except Exception as exc:  # noqa: BLE001 - an extra check must not stop the investigation
            logger.warning("forensic lookup failed: %s: %s", type(exc).__name__, exc)
            tracing.record("error", "forensics", error=type(exc).__name__)
            update.setdefault("limitations", []).append("The check of the files for signs of editing was unavailable for this request.")
            return []
        found = integrity_findings(reports)
        # Not signs of editing, but things the reader should know about how a file was read
        # (for example: its pages are scans, so their text could not be checked against the picture).
        for item in reports:
            for caveat in item.report.caveats[:3]:
                update.setdefault("limitations", []).append(f"{item.filename}: {caveat}")
        tracing.record(
            "forensics", "lookup", documents=len(reports),
            flagged=[{"document": item["document"], "level": item["level"], "score": item["score"]} for item in found],
        )
        return found

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
            integrity=list(state.get("integrity_findings", [])),
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
            kept_ids = {item["id"] for item in keep}
            contradictions = find_contradictions(
                [EvidenceItem(**i) for i in keep],
                unrelated=_document_pairs(state, {}, "unrelated_documents"),
                linked=_document_pairs(state, {}, "linked_documents"),
                net_totals=state.get("net_totals", {}),
            )
            # Findings from the document comparison stay as long as all their evidence survived.
            for item in state.get("contradictions", []):
                if item.get("origin") == "profile" and set(item.get("evidence_ids", [])) <= kept_ids:
                    contradictions.append(Contradiction(**{**item, "id": f"X{len(contradictions) + 1}"}))
            update["contradictions"] = [item.model_dump() for item in contradictions]
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

        # What the agent would like to do outside VeriAgent. Only proposed here: the person sees
        # the exact text with the draft and chooses (see app/actions and the follow_up node).
        proposals = self.actions.propose(draft) if self.actions is not None else []
        if proposals:
            draft.actions = action_list({"proposed_actions": proposals})
            draft.answer_text = render_report(draft)

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
            approval, problem = _parse_decision(interrupt(request), {item["id"] for item in proposals})
            if approval is not None:
                break
            request = {**request, "error": problem}

        return {
            "approval": approval,
            "proposed_actions": proposals,
            "events": [event("human_approval", f"Human decision received: {approval['status']}.")],
        }

    # ============================================================== Node 9b
    @traced("follow_up")
    async def follow_up(self, state: InvestigationState) -> dict[str, Any]:
        """Carry out the follow-up actions a person approved - and only those.

        This is the one place where VeriAgent changes something outside itself.
        It runs after the pause, reads which proposals were ticked, and hands
        each to the action runner, which talks to the MCP server and makes sure
        an action is never carried out twice (see app/actions/runner.py).
        """
        proposals = state.get("proposed_actions", [])
        if not proposals:
            return {}
        approval = state.get("approval", {})
        approved = set(approval.get("approved_actions", [])) if approval.get("status") in ("approved", "modified") else set()
        results: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        events = []
        for proposal in proposals:
            summary = {key: proposal[key] for key in ("id", "kind", "server", "target", "title")}
            if proposal["id"] not in approved or self.actions is None:
                results.append({**summary, "status": "not_approved"})
                continue
            events.append(event("follow_up", f"Creating an issue in {proposal['target']}..."))
            outcome, record = await self.actions.execute(str(state.get("investigation_id")), state["user_id"], proposal)
            results.append(outcome)
            if record is not None:
                records.append(record)
        update: dict[str, Any] = {"action_results": results}
        if events:
            update["events"] = events
        if records:
            update["tool_calls"] = records
        notes = [
            f"An approved follow-up action was not completed: {item.get('detail') or 'no details'}" if item["status"] == "failed"
            # "Unknown" is not "failed": the issue may exist, and creating it again by hand would duplicate it.
            else f"It is not known whether an approved follow-up action was carried out: {item.get('detail') or 'no details'}"
            for item in results if item["status"] in ("failed", "unknown")
        ]
        if notes:
            update["limitations"] = notes
        return update

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


def _as_contradiction(finding: ProfileFinding, evidence_ids: list[str]) -> Contradiction:
    return Contradiction(
        id="X0", type=finding.type, key=finding.key, severity=finding.severity, description=finding.description,
        value_a=finding.value_a, value_b=finding.value_b, unit_a=finding.unit_a, unit_b=finding.unit_b,
        text_a=finding.text_a, text_b=finding.text_b, source_a=finding.source_a, source_b=finding.source_b,
        evidence_ids=evidence_ids, origin="profile", operands=finding.operands, operation=finding.operation,
    )


def _parse_decision(decision: Any, offered: set[str] | None = None) -> tuple[dict[str, Any] | None, str]:
    """Validate what a person sent. Returns (approval, "") or (None, what is wrong).

    ``offered``: the ids of the follow-up actions that were proposed with the request.
    """
    if not isinstance(decision, dict):
        return None, "The decision must be an object with a 'decision' field."
    choice = str(decision.get("decision", "")).strip().lower()
    status = {"approve": "approved", "reject": "rejected", "modify": "modified"}.get(choice)
    if status is None:
        return None, "The decision must be one of: approve, reject, modify."
    modified_action = _clean_text(decision.get("modified_action"))
    if status == "modified" and not modified_action:
        return None, "A 'modify' decision needs the replacement recommendation."
    wanted = decision.get("approved_actions") or []
    if not isinstance(wanted, list) or any(not isinstance(item, str) for item in wanted):
        return None, "'approved_actions' must be a list of action ids."
    unknown = [item for item in wanted if item not in (offered or set())]
    if unknown:
        return None, f"Unknown action id: {unknown[0][:20]}. Only actions proposed with this report can be approved."
    return {
        "status": status,
        # Rejecting the recommendation rejects what was proposed with it.
        "approved_actions": list(dict.fromkeys(wanted)) if status != "rejected" else [],
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
