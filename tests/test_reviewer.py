"""The independent reviewer (Version 3, milestone 18): a second agent checks the first one's reading."""

from __future__ import annotations

import pytest

from app.agents import heuristics
from app.agents.reviewer import ClaimReview, ClaimVerdict, build_review_prompt, mock_review, review_claims, reviewable
from app.llm.base import LLMConfigurationError
from app.schemas.investigation import EvidenceExtraction, EvidenceItem, ExtractedFact
from tests.helpers import DownLLM, ScriptedLLM, run_graph, sample_library

DEMO = "Review this invoice against the contract and tell me whether it should be approved."
MISREAD = "The additional charges on the invoice were approved by the Finance Director."


def item(identifier: str, claim: str, quote: str, key: str = "other", value: float | None = None, by: str = "llm") -> EvidenceItem:
    return EvidenceItem(
        id=identifier, claim=claim, key=key, value_number=value, quote=quote, extracted_by=by,  # type: ignore[arg-type]
        chunk_id="c1", document_id="d1", filename="Contract.pdf", document_type="contract", page=4,
    )


CLAIMS = [
    item("E1", "Contract.pdf states a total amount of £40,000.", "The total contract value is £40,000", "total_amount", 40000),
    item("E2", "The Client shall pay each valid invoice within 30 days.", "the Client shall pay each valid invoice within 30 days"),
    item("E3", MISREAD, "the Client shall pay each valid invoice within 30 days"),
    item("E4", "Contract.pdf lists £12,000: call-outs", "call-outs: £12,000", by="rules"),
]


# ====================================================================== unit
def test_only_free_text_statements_written_by_the_model_are_reviewed():
    assert [claim.id for claim in reviewable(CLAIMS)] == ["E2", "E3"]
    prompt = build_review_prompt(reviewable(CLAIMS))
    assert '"id": "E2"' in prompt and '"id": "E3"' in prompt
    assert "£40,000" not in prompt and "call-outs" not in prompt  # typed facts and rule-made statements are not sent


async def test_unsupported_statements_come_back_with_their_reason():
    answer = ClaimReview(verdicts=[
        ClaimVerdict(id="E2", verdict="supported", reason="Stated."),
        ClaimVerdict(id="[e3]", verdict="unsupported", reason="The passage is about payment,   not approval."),
        ClaimVerdict(id="E1", verdict="unsupported", reason="Not a total."),  # was never asked about: ignored
        ClaimVerdict(id="E9", verdict="unsupported", reason="Unknown id."),
    ])
    llm = ScriptedLLM({ClaimReview: answer})
    assert await review_claims(llm, CLAIMS) == {"E3": "The passage is about payment, not approval."}
    [(system, user)] = llm.prompts["ClaimReview"]
    assert "check another reader's work" in system and "UNTRUSTED STATEMENTS" in user

    silent = ScriptedLLM({ClaimReview: ClaimReview(verdicts=[ClaimVerdict(id="E3", verdict="unsupported", reason="  ")])})
    assert await review_claims(silent, CLAIMS) == {}  # a verdict without a reason is not acted on
    nothing_to_do = ScriptedLLM()
    assert await review_claims(nothing_to_do, [CLAIMS[0], CLAIMS[3]]) == {} and nothing_to_do.calls == []  # no call, no cost


def test_the_offline_reviewer_compares_words():
    review = mock_review("", build_review_prompt(reviewable(CLAIMS)))
    assert {verdict.id: verdict.verdict for verdict in review.verdicts} == {"E2": "supported", "E3": "unsupported"}


# ================================================================ in the graph
def misreading(system: str, user: str) -> EvidenceExtraction:
    """The honest extraction, plus one statement that a real passage does not make."""
    honest = heuristics.rule_based_extraction(heuristics._parse_chunks(user))
    payment = next((chunk for chunk in heuristics._parse_chunks(user) if "within 30 days of the invoice date" in chunk.text), None)
    if payment is not None:  # the contract's payment clause is among the chunks of this round
        honest.facts.append(ExtractedFact(
            chunk_ref=payment.ref, claim=MISREAD, key="other", value_number=None, unit=None,
            quote="the Client shall pay each valid invoice within 30 days of the invoice date",
        ))
    return honest


async def test_without_a_second_opinion_a_misread_passage_becomes_evidence(samples_dir):
    """The gap this milestone closes: the quote is real, the statement is not what it says."""
    final = await run_graph(DEMO, sample_library(samples_dir), ScriptedLLM({EvidenceExtraction: misreading}))
    assert any(evidence["claim"] == MISREAD for evidence in final["evidence"])
    assert final["report"]["missing_evidence"] == []  # the invented approval satisfied the check


async def test_the_reviewer_removes_the_misread_statement(samples_dir):
    llm = ScriptedLLM({EvidenceExtraction: misreading})
    final = await run_graph(DEMO, sample_library(samples_dir), llm, reviewer=llm)
    report = final["report"]

    assert all(evidence["claim"] != MISREAD for evidence in final["evidence"])
    assert [evidence["id"] for evidence in final["evidence"]] == [f"E{n}" for n in range(1, len(final["evidence"]) + 1)]  # no gaps
    assert report["missing_evidence"] and report["requires_human_review"] is True  # the approval really is missing
    assert "approved by the Finance Director" not in report["answer_text"]
    assert any("discarded by the independent reviewer" in text for text in report["limitations"])
    assert final["discarded_facts"] >= 1 and "ClaimReview" in llm.calls
    assert final["llm_calls"] == len(llm.calls)
    # The contradictions rest on typed facts, which the reviewer does not touch.
    assert sorted(c["type"] for c in report["contradictions"]) == ["amount_mismatch", "payment_terms_mismatch"]


async def test_the_reviewer_cannot_remove_totals_or_payment_terms(samples_dir):
    reject_everything = lambda system, user: ClaimReview(verdicts=[  # noqa: E731
        ClaimVerdict(id=f"E{n}", verdict="unsupported", reason="I disagree with all of it.") for n in range(1, 41)
    ])
    llm = ScriptedLLM({ClaimReview: reject_everything})
    final = await run_graph(DEMO, sample_library(samples_dir), llm, reviewer=llm)
    typed = [evidence for evidence in final["evidence"] if evidence["key"] in ("total_amount", "payment_terms_days")]
    assert len(typed) == 4
    assert sorted(c["type"] for c in final["report"]["contradictions"]) == ["amount_mismatch", "payment_terms_mismatch"]
    assert final["report"]["risk_level"] == "high"


async def test_a_reviewer_that_is_down_does_not_stop_the_investigation(samples_dir):
    final = await run_graph(DEMO, sample_library(samples_dir), ScriptedLLM({EvidenceExtraction: misreading}), reviewer=DownLLM())
    assert any(evidence["claim"] == MISREAD for evidence in final["evidence"])  # kept, but ...
    assert any("independent check of the extracted statements was unavailable" in text for text in final["report"]["limitations"])  # ... said so

    wrong_key = ScriptedLLM({ClaimReview: LLMConfigurationError("REVIEWER key is not valid")})
    with pytest.raises(LLMConfigurationError):  # a set-up mistake must be visible, not silently skipped
        await run_graph(DEMO, sample_library(samples_dir), ScriptedLLM({EvidenceExtraction: misreading}), reviewer=wrong_key)


async def test_the_second_opinion_can_be_switched_off(samples_dir, settings):
    llm = ScriptedLLM({EvidenceExtraction: misreading})
    off = settings.model_copy(update={"ENABLE_CLAIM_REVIEW": False})
    final = await run_graph(DEMO, sample_library(samples_dir), llm, off, reviewer=llm)
    assert "ClaimReview" not in llm.calls and any(evidence["claim"] == MISREAD for evidence in final["evidence"])


async def test_rule_based_extraction_is_not_sent_to_the_reviewer(samples_dir):
    """When the main model is down, the statements come from fixed templates; there is nothing to second-guess."""
    reviewer = ScriptedLLM()
    final = await run_graph(DEMO, sample_library(samples_dir), DownLLM(), reviewer=reviewer)
    assert reviewer.calls == [] and final["used_fallback"] is True


# ============================================================ which model reviews
def test_the_reviewer_is_the_main_model_unless_another_is_configured(monkeypatch):
    from app.config import get_settings
    from app.llm import factory

    settings = get_settings()
    factory.get_llm.cache_clear()
    factory.get_reviewer_llm.cache_clear()
    try:
        assert factory.get_reviewer_llm() is factory.get_llm()

        factory.get_reviewer_llm.cache_clear()
        monkeypatch.setattr(settings, "REVIEWER_LLM_PROVIDER", "gemini")
        with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
            factory.get_reviewer_llm()  # another provider needs its own key

        from pydantic import SecretStr

        monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr("test-key-not-real"))
        monkeypatch.setattr(settings, "GEMINI_MODEL", "main-model")
        monkeypatch.setattr(settings, "REVIEWER_MODEL", "reviewer-model")
        factory.get_reviewer_llm.cache_clear()
        reviewer = factory.get_reviewer_llm()
        assert (reviewer.provider, reviewer.model) == ("gemini", "reviewer-model")
        assert settings.GEMINI_MODEL == "main-model"  # the main model's setting is untouched
    finally:
        factory.get_llm.cache_clear()
        factory.get_reviewer_llm.cache_clear()
