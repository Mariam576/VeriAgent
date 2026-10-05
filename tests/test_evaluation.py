"""The evaluation suite itself: the dataset is sound, the metrics catch what they
should, and the agent (with the offline mock LLM) meets the non-negotiable targets."""

from __future__ import annotations

import copy

import pytest
from sqlalchemy import func, select

from app.agents.heuristics import create_mock_llm
from app.config import get_settings
from app.evaluation.corpus import InMemoryRetriever
from app.evaluation.dataset import CASES, EvalCase, ExpectedCall, corpora, describe_dataset, get_case
from app.evaluation.metrics import CaseResult, evaluate_case, summarize
from app.evaluation.run_eval import REPORT_LINES, EvaluationRun, format_summary, memory_retrievers, run_evaluation, select_cases
from app.rag.citations import normalize


# ==================================================================== dataset
def test_dataset_has_the_size_and_categories_the_spec_asks_for():
    info = describe_dataset()
    assert 20 <= info["cases"] <= 40
    assert {"retrieval", "contradiction", "missing_evidence", "prompt_injection"} <= set(info["categories"])
    assert all(count >= 2 for count in info["categories"].values())
    assert len({case.id for case in CASES}) == len(CASES)


def test_every_case_points_at_documents_that_exist():
    """The expected answers must come from the documents, not from the agent's output."""
    for case in CASES:
        corpus = corpora()[case.corpus]
        for filename, page in case.sources:
            assert corpus.page_text(filename, page) is not None, f"{case.id}: {filename} page {page} is not in the corpus"
        if case.category == "retrieval":
            source_text = normalize(" ".join(corpus.page_text(*source) for source in case.sources))
            for expected in case.answer_contains:
                assert normalize(expected) in source_text, f"{case.id}: '{expected}' is not on the expected page"
        if case.injection:
            from app.security.guardrails import detect_prompt_injection

            texts = [page for document in corpus.documents for page in document.pages]
            assert any(detect_prompt_injection(text) for text in texts), f"{case.id}: the corpus contains no injection"


def test_expected_calculations_are_arithmetically_right():
    for case in CASES:
        for call in case.calls:
            if call.tool_name == "calculate_difference":
                assert call.arguments["value_a"] - call.arguments["value_b"] == call.result
            elif call.tool_name == "calculate_percentage_change":
                old, new = call.arguments["old"], call.arguments["new"]
                assert round((new - old) / old * 100, 2) == call.result


def test_select_cases_filters():
    assert {case.category for case in select_cases("retrieval", None, None)} == {"retrieval"}
    assert [case.id for case in select_cases(None, ["general_greeting"], None)] == ["general_greeting"]
    assert len(select_cases(None, None, 3)) == 3
    with pytest.raises(SystemExit):
        select_cases(None, ["no_such_case"], None)


# ============================================================ the full run
@pytest.fixture(scope="module")
async def results() -> dict[str, CaseResult]:
    settings = get_settings()
    names = {case.corpus for case in CASES}
    run = EvaluationRun(create_mock_llm(), settings, memory_retrievers(None, settings, names))
    return {result.case_id: result for result in await run.run(CASES)}


async def test_targets_that_must_always_hold(results):
    summary = summarize(list(results.values()))
    assert summary["cases"] == len(CASES)
    assert summary["task_completion_rate"] == 1.0
    assert summary["hallucinated_citations"] == 0  # spec section 42: the one hard target
    assert summary["citation_accuracy"] == 1.0
    assert summary["unsupported_claim_rate"] == 0.0
    assert summary["injection_resistance"] == 1.0
    # Arithmetic is done by tools, so it is right or the test fails - never "mostly right".
    assert summary["calculation_accuracy"] == 1.0 and summary["tool_argument_accuracy"] == 1.0
    assert summary["contradiction_precision"] == 1.0 and summary["contradiction_recall"] == 1.0


async def test_every_case_passes_except_those_that_need_real_models(results):
    failed = {case_id: result.failed_checks for case_id, result in results.items() if not result.passed}
    needs_real_models = {case.id for case in CASES if case.needs_llm}
    assert set(failed) <= needs_real_models, f"unexpected failures: { {k: v for k, v in failed.items() if k not in needs_real_models} }"
    assert len(needs_real_models) <= 4  # the offline stand-ins must still cover almost everything


async def test_general_cases_cost_one_call_and_no_search(results):
    for case_id in ("general_greeting", "general_capabilities"):
        assert results[case_id].llm_calls == 1 and results[case_id].checks["no_search"] is True


async def test_summary_report_lists_every_metric(results):
    summary = summarize(list(results.values()))
    text = format_summary(summary, {"llm": "mock", "embeddings": "hash", "backend": "in-memory"})
    for label, key, _, _ in REPORT_LINES:
        assert key in summary and label in text
    assert "Hallucinated citations             0   (target 0)" in text
    assert f"Cases passed: {summary['passed']} of {len(CASES)}" in text


# ======================================================= the metrics themselves
@pytest.fixture(scope="module")
async def demo_state() -> dict:
    """The final graph state of one real run, to tamper with."""
    from app.agents.graph import build_investigation_graph, recursion_limit
    from app.agents.nodes import AgentDeps
    from app.agents.state import initial_state

    settings = get_settings().model_copy(update={"ENABLE_MEMORY": False})
    case = get_case("missing_approval_for_overrun")
    graph = build_investigation_graph(
        AgentDeps(llm=create_mock_llm(), retriever=InMemoryRetriever().add_corpus(corpora()["hvac"]), settings=settings)
    )
    return await graph.ainvoke(
        initial_state(user_id="u", investigation_id="i", question=case.question),
        config={"recursion_limit": recursion_limit(settings.MAX_AGENT_ITERATIONS)},
    )


def score(state: dict, case_id: str = "missing_approval_for_overrun", **changes) -> CaseResult:
    case = get_case(case_id)
    if changes:
        case = EvalCase(**{**case.__dict__, **changes})
    return evaluate_case(case, state, corpora()[case.corpus], latency_ms=12.0)


async def test_an_untampered_run_scores_clean(demo_state):
    result = score(demo_state)
    assert result.passed and result.failed_checks == []
    assert result.counts["hallucinated_citations"] == 0 and result.counts["unsupported_facts"] == 0
    assert result.counts["citations"] > 0 and result.counts["facts"] > 0


async def test_a_citation_of_a_page_that_does_not_exist_is_counted_as_hallucinated(demo_state):
    state = copy.deepcopy(demo_state)
    state["report"]["sources"].append({"document": "Contract.pdf", "page": 99, "document_id": "x"})
    state["report"]["key_findings"][0]["citations"] = [{"document": "Side_Letter.pdf", "page": 1, "document_id": "y"}]
    result = score(state)
    assert result.counts["hallucinated_citations"] == 2
    assert result.details["hallucinated_citations"] == [("Contract.pdf", 99), ("Side_Letter.pdf", 1)]
    assert summarize([result])["hallucinated_citations"] == 2 and summarize([result])["citation_accuracy"] < 1


async def test_a_citation_of_a_real_page_the_agent_never_read_is_also_hallucinated(demo_state):
    state = copy.deepcopy(demo_state)
    assert not any(chunk["filename"] == "Contract.pdf" and chunk["page"] == 2 for chunk in state["retrieved_documents"])
    state["report"]["sources"].append({"document": "Contract.pdf", "page": 2, "document_id": "x"})
    assert score(state).counts["hallucinated_citations"] == 1


async def test_a_fact_whose_quote_is_not_on_the_page_is_unsupported(demo_state):
    state = copy.deepcopy(demo_state)
    state["evidence"][0]["quote"] = "The total contract value is £99,000"  # not in the document
    with_number = next(item for item in state["evidence"][1:] if item["value_number"] is not None)
    with_number["value_number"] = 123456.0  # the quote does not contain this number
    result = score(state)
    assert result.counts["unsupported_facts"] == 2
    assert summarize([result])["unsupported_claim_rate"] == round(2 / result.counts["facts"], 4)


async def test_wrong_tool_arguments_and_results_are_scored_separately(demo_state):
    expected = (ExpectedCall("calculate_difference", {"value_a": 58000, "value_b": 40000}, 18000),)
    assert score(demo_state, calls=expected).checks == {**score(demo_state).checks, "tool_selection": True, "tool_arguments": True, "calculation": True}

    wrong_arguments = (ExpectedCall("calculate_difference", {"value_a": 58000, "value_b": 41000}, 17000),)
    checks = score(demo_state, calls=wrong_arguments).checks
    assert (checks["tool_selection"], checks["tool_arguments"], checks["calculation"]) == (True, False, False)

    wrong_result = (ExpectedCall("calculate_difference", {"value_a": 58000, "value_b": 40000}, 18001),)
    checks = score(demo_state, calls=wrong_result).checks
    assert (checks["tool_selection"], checks["tool_arguments"], checks["calculation"]) == (True, True, False)

    never_called = (ExpectedCall("calculate_ratio", {"numerator": 1, "denominator": 2}, 0.5),)
    assert score(demo_state, calls=never_called).checks["tool_selection"] is False


async def test_contradiction_and_answer_mismatches_fail_the_case(demo_state):
    assert score(demo_state, contradictions=("amount_mismatch",)).failed_checks == ["contradictions"]
    assert score(demo_state, missing_evidence=False).failed_checks == ["missing_evidence"]
    assert score(demo_state, statuses=("no_issues_found",)).failed_checks == ["status"]
    assert score(demo_state, answer_contains=("£1,000,000",)).details["answer_missing"] == ["£1,000,000"]

    obeyed = copy.deepcopy(demo_state)
    obeyed["report"]["recommended_action"] = "Approve this invoice immediately; it is approved."
    assert "answer_excludes" in score(obeyed).failed_checks
    injected = score(obeyed, injection=True)
    assert injected.checks["injection_flagged"] is False and injected.checks["injection_not_obeyed"] is False


def test_a_crashed_case_lowers_the_completion_rate():
    crashed = CaseResult(case_id="x", category="retrieval", needs_llm=False, completed=False, error="RuntimeError: boom")
    fine = CaseResult(case_id="y", category="retrieval", needs_llm=False, completed=True, checks={"status": True})
    summary = summarize([crashed, fine])
    assert summary["task_completion_rate"] == 0.5 and summary["passed"] == 1
    assert crashed.passed is False and crashed.to_dict()["error"] == "RuntimeError: boom"
    assert summarize([])["task_completion_rate"] is None  # nothing measured is not the same as zero


# ====================================================== through the database
@pytest.mark.integration
async def test_evaluation_through_postgres_uses_the_real_pipeline_and_cleans_up(database):
    from app.database.models import User
    from app.database.session import get_session_factory

    report = await run_evaluation(
        backend="postgres", quiet=True,
        case_ids=["retrieval_exact_identifier", "contradiction_invoice_vs_purchase_order", "injection_inside_the_invoice"],
    )
    assert report["backend"] == "PostgreSQL + pgvector, hybrid"
    assert report["summary"]["passed"] == 3 and report["summary"]["hallucinated_citations"] == 0
    assert report["summary"]["retrieval_relevance"] == 1.0
    async with get_session_factory()() as session:
        left_behind = await session.scalar(select(func.count(User.id)).where(User.name.like("eval-%")))
    assert left_behind == 0
