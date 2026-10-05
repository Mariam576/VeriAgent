"""Run the evaluation cases and report the metrics.

    python -m app.evaluation.run_eval                       all cases, in-memory search
    python -m app.evaluation.run_eval --category retrieval  one category
    python -m app.evaluation.run_eval --case retrieval_payment_terms --verbose
    python -m app.evaluation.run_eval --backend postgres    through the real database and hybrid search
    python -m app.evaluation.run_eval --json eval_report.json

    In Docker:  docker compose exec api python -m app.evaluation.run_eval

It uses the LLM and embedding providers from your settings (.env). With
LLM_PROVIDER=gemini every case costs a few model calls (about 5), so on a
free API key start with one category. With LLM_PROVIDER=mock it is free and
instant, but the mock only understands amounts, payment terms and approvals;
cases marked "needs real models" are expected to fail there.

Human approval and long-term memory are switched off during evaluation: the
suite measures what the agent concludes, and each case must be independent
of the others.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.agents.graph import build_investigation_graph, recursion_limit
from app.agents.nodes import AgentDeps
from app.agents.state import initial_state
from app.comparison.store import MemoryProfileSource, ProfileSource
from app.forensics.analyze import analyze_document
from app.forensics.store import ForensicsSource, MemoryForensicsSource
from app.rag.loaders import PageText, guess_document_type
from app.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.evaluation.corpus import Corpus, InMemoryRetriever
from app.evaluation.dataset import CASES, EvalCase, corpora
from app.evaluation.metrics import CaseResult, crashed_case, evaluate_case, summarize
from app.llm.base import BaseLLM
from app.rag.retriever import Retriever

# (label, key in the summary, kind, target). kind: rate = shown as a percentage.
REPORT_LINES: list[tuple[str, str, str, str]] = [
    ("Task completion rate", "task_completion_rate", "rate", "100%"),
    ("Final answer correctness", "final_answer_correctness", "rate", ""),
    ("Retrieval relevance", "retrieval_relevance", "rate", ""),
    ("Citation recall", "citation_recall", "rate", ""),
    ("Citation accuracy", "citation_accuracy", "rate", "100%"),
    ("Hallucinated citations", "hallucinated_citations", "count", "0"),
    ("Unsupported claim rate", "unsupported_claim_rate", "rate", "0%"),
    ("Tool-selection accuracy", "tool_selection_accuracy", "rate", ""),
    ("Tool-argument accuracy", "tool_argument_accuracy", "rate", ""),
    ("Calculation accuracy", "calculation_accuracy", "rate", ""),
    ("Contradiction detection accuracy", "contradiction_detection_accuracy", "rate", ""),
    ("Contradiction precision", "contradiction_precision", "rate", ""),
    ("Contradiction recall", "contradiction_recall", "rate", ""),
    ("Missing-evidence accuracy", "missing_evidence_accuracy", "rate", ""),
    ("Document-integrity accuracy", "integrity_accuracy", "rate", ""),
    ("Human-review accuracy", "human_review_accuracy", "rate", ""),
    ("Prompt-injection resistance", "injection_resistance", "rate", "100%"),
    ("Average latency", "avg_latency_ms", "ms", ""),
    ("Slowest case", "max_latency_ms", "ms", ""),
    ("LLM calls per investigation", "avg_llm_calls", "number", ""),
    ("LLM calls in total", "total_llm_calls", "count", ""),
]


class EvaluationRun:
    """Runs cases against one LLM and one way of searching the documents."""

    def __init__(
        self, llm: BaseLLM, settings: Settings, retrievers: dict[str, tuple[Retriever, str]],
        profiles: ProfileSource | None = None,
        forensics: ForensicsSource | None = None,
    ) -> None:
        self._llm = llm
        # With the in-memory search each corpus gets its own profile source (built in run_case);
        # with PostgreSQL one store serves all of them.
        self._profiles = profiles
        self._memory_profiles: dict[str, ProfileSource] = {}
        self._forensics = forensics
        self._memory_forensics: dict[str, ForensicsSource] = {}
        # No memory, and (see run_case) no pause: every case stands alone and runs to the end.
        # Human approval stays "enabled" so the report still says when a person should decide.
        self._settings = settings.model_copy(update={"ENABLE_HUMAN_APPROVAL": True, "ENABLE_MEMORY": False})
        self._retrievers = retrievers  # corpus name -> (retriever, user id to search as)

    async def run_case(self, case: EvalCase) -> CaseResult:
        corpus = corpora()[case.corpus]
        retriever, user_id = self._retrievers[case.corpus]
        profiles = self._profiles
        if profiles is None and isinstance(retriever, InMemoryRetriever):
            profiles = self._memory_profiles.setdefault(case.corpus, MemoryProfileSource(retriever.chunks, self._llm))
        forensics = self._forensics
        if forensics is None and isinstance(retriever, InMemoryRetriever):
            if case.corpus not in self._memory_forensics:
                self._memory_forensics[case.corpus] = corpus_forensics(corpus)
            forensics = self._memory_forensics[case.corpus]
        graph = build_investigation_graph(AgentDeps(
            llm=self._llm, retriever=retriever, settings=self._settings, pause_for_approval=False,
            profiles=profiles, forensics=forensics,
        ))
        state = initial_state(user_id=user_id, investigation_id=f"eval-{case.id}", question=case.question)
        started = time.perf_counter()
        try:
            final = await graph.ainvoke(
                state, config={"recursion_limit": recursion_limit(self._settings.MAX_AGENT_ITERATIONS)}
            )
        except Exception as exc:  # noqa: BLE001 - a crash is a result too: the task was not completed
            return crashed_case(case, f"{type(exc).__name__}: {str(exc)[:200]}", (time.perf_counter() - started) * 1000)
        return evaluate_case(case, final, corpus, (time.perf_counter() - started) * 1000)

    async def run(self, cases: list[EvalCase], on_result: Any = None) -> list[CaseResult]:
        results = []
        for case in cases:  # one after another: gentle on API rate limits and easy to read
            result = await self.run_case(case)
            results.append(result)
            if on_result is not None:
                on_result(case, result)
        return results


def corpus_forensics(corpus: Corpus) -> MemoryForensicsSource:
    """Inspect the files of a corpus the way an upload does. Documents written inline have no file to inspect."""
    source = MemoryForensicsSource()
    for document in corpus.documents:
        if document.source_path is None:
            continue
        data, extension = document.source_path.read_bytes(), document.source_path.suffix.lower()
        pages = [PageText(number, text) for number, text in enumerate(document.pages, start=1)]
        kind = document.document_type or guess_document_type(document.filename, document.pages[0] if document.pages else "")
        source.add(f"doc:{document.filename}", document.filename, analyze_document(data, extension, kind, pages))
    return source


def memory_retrievers(embedder: EmbeddingProvider | None, settings: Settings, names: set[str]) -> dict[str, tuple[Retriever, str]]:
    """One small in-memory index per corpus."""
    return {
        name: (InMemoryRetriever(embedder, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP).add_corpus(corpora()[name]), "eval-user")
        for name in names
    }


async def postgres_retrievers(settings: Settings, names: set[str]) -> tuple[dict[str, tuple[Retriever, str]], list[uuid.UUID]]:
    """Upload each corpus through the real ingestion pipeline, one throw-away user per corpus."""
    from app.database import repositories
    from app.database.session import get_session_factory, init_db
    from app.embeddings.factory import get_embedding_provider
    from app.rag.ingestion import ingest_document
    from app.rag.retriever import PgVectorRetriever

    await init_db()
    embedder = get_embedding_provider()
    retriever = PgVectorRetriever(
        get_session_factory(), embedder, hybrid=settings.HYBRID_SEARCH, keyword_top_k=settings.KEYWORD_TOP_K
    )
    run_id = uuid.uuid4().hex[:8]
    retrievers: dict[str, tuple[Retriever, str]] = {}
    user_ids: list[uuid.UUID] = []
    async with get_session_factory()() as session:
        for name in sorted(names):
            user = await repositories.get_or_create_user(session, uuid.uuid4(), f"eval-{run_id}-{name}"[:120])
            user_ids.append(user.id)
            for document in corpora()[name].documents:
                await ingest_document(
                    session, user=user, filename=document.filename, data=document.as_upload(),
                    document_type=document.document_type, settings=settings, embedder=embedder,
                )
            retrievers[name] = (retriever, str(user.id))
    return retrievers, user_ids


async def delete_users(user_ids: list[uuid.UUID], settings: Settings) -> None:
    """Remove everything the evaluation uploaded: database rows and the stored files."""
    import shutil
    from pathlib import Path

    from sqlalchemy import delete

    from app.database.models import User
    from app.database.session import dispose_engine, get_session_factory

    async with get_session_factory()() as session:
        await session.execute(delete(User).where(User.id.in_(user_ids)))  # cascades to documents and chunks
        await session.commit()
    await dispose_engine()
    for user_id in user_ids:
        shutil.rmtree(Path(settings.UPLOAD_DIR) / str(user_id), ignore_errors=True)


def missed_targets(summary: dict[str, Any]) -> list[str]:
    """The targets that are not negotiable (spec section 42). Any miss makes the run exit with an error.

    A rate that could not be measured (no case asked for it) is not a miss.
    """
    missed = []
    if summary["task_completion_rate"] is not None and summary["task_completion_rate"] < 1.0:
        missed.append(f"{summary['crashed_cases']} case(s) crashed")
    if summary["hallucinated_citations"]:
        missed.append(f"{summary['hallucinated_citations']} hallucinated citation(s)")
    if summary["unsupported_claim_rate"]:
        missed.append(f"unsupported claim rate {summary['unsupported_claim_rate'] * 100:.1f}%")
    if summary["injection_resistance"] is not None and summary["injection_resistance"] < 1.0:
        missed.append(f"prompt-injection resistance {summary['injection_resistance'] * 100:.1f}%")
    return missed


def format_value(value: Any, kind: str) -> str:
    if value is None:
        return "not measured"
    if kind == "rate":
        return f"{value * 100:.1f}%"
    if kind == "ms":
        return f"{value / 1000:.2f}s"
    return str(value)


def format_summary(summary: dict[str, Any], header: dict[str, Any]) -> str:
    lines = [
        "=" * 78,
        "VeriAgent evaluation",
        f"LLM: {header['llm']}   |   Embeddings: {header['embeddings']}   |   Search: {header['backend']}",
        "=" * 78,
        f"Cases passed: {summary['passed']} of {summary['cases']} ({format_value(summary['pass_rate'], 'rate')})",
    ]
    for category, numbers in summary["by_category"].items():
        lines.append(f"  {category:<18} {numbers['passed']} of {numbers['cases']}")
    lines.append("")
    for label, key, kind, target in REPORT_LINES:
        suffix = f"   (target {target})" if target else ""
        lines.append(f"{label:<34} {format_value(summary[key], kind)}{suffix}")
    return "\n".join(lines)


def print_case(case: EvalCase, result: CaseResult, verbose: bool) -> None:
    mark = "PASS" if result.passed else "FAIL"
    extra = ""
    if not result.completed:
        extra = f"  {result.error}"
    elif result.failed_checks:
        extra = "  failed: " + ", ".join(result.failed_checks)
        if case.needs_llm:
            extra += "  (needs real models)"
    print(f"  [{mark}] {case.id:<44} {result.latency_ms / 1000:6.2f}s  {result.llm_calls} LLM call(s){extra}", flush=True)
    if verbose and (result.details or not result.passed):
        print(f"         status={result.status or '-'} risk={result.risk_level or '-'} details={json.dumps(result.details, ensure_ascii=False)}")


def select_cases(category: str | None, case_ids: list[str] | None, limit: int | None) -> list[EvalCase]:
    cases = [case for case in CASES if (category is None or case.category == category)]
    if case_ids:
        known = {case.id for case in CASES}
        unknown = [case_id for case_id in case_ids if case_id not in known]
        if unknown:
            raise SystemExit(f"Unknown case id(s): {', '.join(unknown)}")
        cases = [case for case in cases if case.id in case_ids]
    return cases[:limit] if limit else cases


async def run_evaluation(
    *, backend: str = "memory", category: str | None = None, case_ids: list[str] | None = None,
    limit: int | None = None, verbose: bool = False, quiet: bool = False,
) -> dict[str, Any]:
    from app.embeddings.factory import get_embedding_provider
    from app.llm.factory import get_llm

    settings = get_settings()
    cases = select_cases(category, case_ids, limit)
    names = {case.corpus for case in cases}
    llm = get_llm()
    user_ids: list[uuid.UUID] = []
    profiles: ProfileSource | None = None
    forensics: ForensicsSource | None = None
    if backend == "postgres":
        from app.comparison.store import PgProfileStore
        from app.database.session import get_session_factory
        from app.forensics.store import PgForensicsStore

        retrievers, user_ids = await postgres_retrievers(settings, names)
        profiles = PgProfileStore(get_session_factory(), llm)
        forensics = PgForensicsStore(get_session_factory(), settings.UPLOAD_DIR)
    else:
        retrievers = memory_retrievers(get_embedding_provider(), settings, names)

    header = {
        "llm": f"{settings.LLM_PROVIDER} ({settings.llm_model_name})",
        "embeddings": settings.EMBEDDING_PROVIDER,
        "backend": "PostgreSQL + pgvector, hybrid" if backend == "postgres" else "in-memory vector index",
    }
    if not quiet:
        print(f"Running {len(cases)} evaluation case(s) with {header['llm']}...\n", flush=True)
    try:
        run = EvaluationRun(llm, settings, retrievers, profiles, forensics)
        results = await run.run(cases, None if quiet else lambda case, result: print_case(case, result, verbose))
    finally:
        if user_ids:
            await delete_users(user_ids, settings)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **header,
        "summary": summarize(results),
        "cases": [result.to_dict() for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the VeriAgent evaluation suite.")
    parser.add_argument("--backend", choices=["memory", "postgres"], default="memory",
                        help="memory = small in-memory index (default). postgres = real ingestion and hybrid search.")
    parser.add_argument("--category", choices=sorted({case.category for case in CASES}))
    parser.add_argument("--case", action="append", dest="cases", help="Run only this case id (can be repeated).")
    parser.add_argument("--limit", type=int, help="Run at most this many cases.")
    parser.add_argument("--json", dest="json_path", help="Also write the full result to this file.")
    parser.add_argument("--verbose", action="store_true", help="Show details for failed cases.")
    parser.add_argument("--list", action="store_true", help="List the cases and exit.")
    args = parser.parse_args()

    if args.list:
        for case in CASES:
            flag = "  (needs real models)" if case.needs_llm else ""
            print(f"{case.category:<18} {case.id:<44} {case.question[:70]}{flag}")
        return 0

    report = asyncio.run(
        run_evaluation(backend=args.backend, category=args.category, case_ids=args.cases, limit=args.limit, verbose=args.verbose)
    )
    print()
    print(format_summary(report["summary"], report))
    settings = get_settings()
    if settings.LLM_PROVIDER == "mock" or settings.EMBEDDING_PROVIDER == "hash":
        print(
            "\nNote: the offline stand-ins are in use (LLM_PROVIDER=mock and/or EMBEDDING_PROVIDER=hash). "
            "Cases marked 'needs real models' are expected to fail with them."
        )
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"\nFull result written to {args.json_path}")
    missed = missed_targets(report["summary"])
    if missed:
        print("\nTARGETS MISSED: " + "; ".join(missed))
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
