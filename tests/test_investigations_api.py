"""The whole product through its HTTP API: upload -> investigate -> read results.

Uses the real PostgreSQL + pgvector, the offline mock LLM and hash embeddings.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.database.models import Investigation, Message, ToolCall
from app.database.session import get_session_factory

pytestmark = pytest.mark.integration

DEMO_QUESTION = "Review this invoice against the contract and tell me whether it should be approved."


def pages(citations: list[dict]) -> set[tuple[str, int]]:
    """Citations as (document, page) pairs."""
    return {(c["document"], c["page"]) for c in citations}

async def upload(client, headers, path):
    with path.open("rb") as handle:
        response = await client.post("/documents/upload", headers=headers, files={"file": (path.name, handle)})
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
async def library(client, user_headers, samples_dir):
    return {
        name: await upload(client, user_headers, samples_dir / name)
        for name in ("Contract.pdf", "Invoice_102.pdf", "Procurement_Policy.pdf")
    }


async def test_demo_investigation_end_to_end(client, user_headers, library):
    response = await client.post("/investigations", headers=user_headers, json={"question": DEMO_QUESTION})
    assert response.status_code == 201, response.text
    report = response.json()

    assert report["status"] == "review_required"
    assert report["risk_level"] == "high"
    assert report["requires_human_review"] is True
    amount = next(c for c in report["contradictions"] if c["type"] == "amount_mismatch")
    assert (amount["value_a"], amount["value_b"], amount["difference"]) == (40000, 58000, 18000)
    assert any(c["type"] == "payment_terms_mismatch" for c in report["contradictions"])
    assert report["missing_evidence"]
    assert ("Contract.pdf", 4) in pages(report["sources"])
    assert ("Invoice_102.pdf", 1) in pages(report["sources"])
    assert "£18,000" in report["answer_text"]
    assert report["stats"]["duration_ms"] > 0
    assert report["stats"]["llm_provider"] == "mock"

    # --- the stored investigation can be read back
    investigation_id = report["investigation_id"]
    detail = (await client.get(f"/investigations/{investigation_id}", headers=user_headers)).json()
    assert detail["status"] == "review_required" and detail["risk_level"] == "high"
    assert detail["question"] == DEMO_QUESTION
    assert detail["completed_at"] is not None
    assert detail["report"]["answer_text"] == report["answer_text"]

    # --- findings are stored one per row, typed, with their source document and page
    findings = (await client.get(f"/investigations/{investigation_id}/findings", headers=user_headers)).json()["findings"]
    types = {finding["finding_type"] for finding in findings}
    assert {"fact", "contradiction", "calculation", "missing_evidence", "risk", "recommendation"} <= types
    contract_fact = next(f for f in findings if f["finding_type"] == "fact" and "£40,000" in f["statement"])
    assert contract_fact["source_document_id"] == library["Contract.pdf"]["id"]
    assert contract_fact["page_number"] == 4

    # --- every tool call was logged (observability)
    async with get_session_factory()() as session:
        calls = (await session.execute(
            select(ToolCall).where(ToolCall.investigation_id == uuid.UUID(investigation_id))
        )).scalars().all()
    assert {"calculate_difference", "calculate_percentage_change"} <= {call.tool_name for call in calls}
    difference = next(c for c in calls if c.arguments == {"value_a": 58000.0, "value_b": 40000.0})
    assert difference.result == {"value": 18000.0} and difference.status == "ok"

    listing = (await client.get("/investigations", headers=user_headers)).json()["investigations"]
    assert [item["id"] for item in listing] == [investigation_id]


async def test_investigation_scoped_to_one_document(client, user_headers, library):
    response = await client.post(
        "/investigations", headers=user_headers,
        json={"question": "What is the total contract value?", "document_ids": [library["Contract.pdf"]["id"]]},
    )
    report = response.json()
    assert {source["document"] for source in report["sources"]} == {"Contract.pdf"}
    assert report["contradictions"] == []
    assert any("£40,000" in finding["statement"] for finding in report["key_findings"])


async def test_users_cannot_use_or_read_each_others_data(client, user_headers, library):
    intruder = {"X-User-Id": user_headers["X-User-Id"] + "-intruder"}

    # Cannot investigate someone else's document by id.
    response = await client.post(
        "/investigations", headers=intruder,
        json={"question": DEMO_QUESTION, "document_ids": [library["Contract.pdf"]["id"]]},
    )
    assert response.status_code == 404

    # Without ids, the intruder simply has no documents: no evidence, nothing leaked.
    own = (await client.post("/investigations", headers=intruder, json={"question": DEMO_QUESTION})).json()
    assert own["status"] == "insufficient_evidence" and own["sources"] == []
    assert "40,000" not in own["answer_text"]

    # Cannot read someone else's investigation or findings.
    mine = (await client.post("/investigations", headers=user_headers, json={"question": DEMO_QUESTION})).json()
    assert (await client.get(f"/investigations/{mine['investigation_id']}", headers=intruder)).status_code == 404
    assert (await client.get(f"/investigations/{mine['investigation_id']}/findings", headers=intruder)).status_code == 404
    assert (await client.get(f"/investigations/{mine['investigation_id']}", headers=user_headers)).status_code == 200


async def test_invalid_requests_are_rejected(client, user_headers):
    assert (await client.post("/investigations", headers=user_headers, json={"question": ""})).status_code == 422
    assert (await client.post("/investigations", headers=user_headers, json={"question": "   "})).status_code == 400
    too_long = "x" * 5000
    assert (await client.post("/investigations", headers=user_headers, json={"question": too_long})).status_code == 400
    assert (await client.get("/investigations/not-a-uuid", headers=user_headers)).status_code == 422
    assert (await client.get(f"/investigations/{uuid.uuid4()}", headers=user_headers)).status_code == 404
    bad_ids = {"question": "q", "document_ids": [str(uuid.uuid4())]}
    assert (await client.post("/investigations", headers=user_headers, json=bad_ids)).status_code == 404


async def test_chat_matches_the_spec_example_and_remembers_the_conversation(client, user_headers, library):
    first = await client.post("/chat", headers=user_headers, json={"message": "Compare the invoice amount with the contract."})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["risk_level"] == "high"
    assert 0 < body["confidence"] <= 1
    assert "£40,000" in body["answer"] and "£58,000" in body["answer"]
    assert ("Contract.pdf", 4) in pages(body["citations"])
    assert ("Invoice_102.pdf", 1) in pages(body["citations"])
    conversation_id = body["conversation_id"]

    # A follow-up that only makes sense with the earlier turn.
    second = await client.post(
        "/chat", headers=user_headers,
        json={"conversation_id": conversation_id, "message": "What about the payment terms?"},
    )
    assert second.status_code == 200, second.text
    follow_up = second.json()
    assert follow_up["conversation_id"] == conversation_id
    assert "30 days" in follow_up["answer"] and "60 days" in follow_up["answer"]

    async with get_session_factory()() as session:
        messages = (await session.execute(
            select(Message).where(Message.conversation_id == uuid.UUID(conversation_id)).order_by(Message.created_at)
        )).scalars().all()
        investigation = await session.get(Investigation, uuid.UUID(follow_up["investigation_id"]))
    assert [message.role for message in messages] == ["user", "assistant", "user", "assistant"]
    assert investigation.conversation_id == uuid.UUID(conversation_id)


async def test_chat_rejects_unknown_or_foreign_conversations(client, user_headers):
    response = await client.post(
        "/chat", headers=user_headers, json={"conversation_id": str(uuid.uuid4()), "message": "hello"}
    )
    assert response.status_code == 404

    mine = (await client.post("/chat", headers=user_headers, json={"message": "hello"})).json()
    intruder = {"X-User-Id": user_headers["X-User-Id"] + "-intruder"}
    stolen = await client.post("/chat", headers=intruder, json={"conversation_id": mine["conversation_id"], "message": "hello"})
    assert stolen.status_code == 404


async def test_prompt_injection_document_through_the_api(client, user_headers, library, samples_dir):
    await upload(client, user_headers, samples_dir / "adversarial" / "Vendor_Note.txt")
    response = await client.post(
        "/investigations", headers=user_headers,
        json={"question": DEMO_QUESTION + " Also check the vendor note about invoice INV-102."},
    )
    report = response.json()
    assert report["risk_level"] in ("high", "critical")
    assert report["status"] == "review_required"
    assert any("instructions to an AI" in limitation for limitation in report["limitations"])
    assert ("Vendor_Note.txt", 1) in pages(report["sources"])
    assert "do not approve automatically" in report["recommended_action"].lower()


async def test_missing_llm_configuration_gives_a_clear_503(client, user_headers, library, monkeypatch):
    from app.agents import service
    from app.llm.base import LLMConfigurationError

    def broken_graph():
        raise LLMConfigurationError("GEMINI_API_KEY is not set. Add it to your .env file.")

    monkeypatch.setattr(service, "get_graph", broken_graph)
    response = await client.post("/investigations", headers=user_headers, json={"question": DEMO_QUESTION})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"
    assert "GEMINI_API_KEY" in response.json()["error"]["message"]

    listing = (await client.get("/investigations", headers=user_headers)).json()["investigations"]
    assert listing[0]["status"] == "failed"  # the failed attempt is recorded, not lost


async def test_metadata_filters_limit_the_search(client, user_headers, library):
    response = await client.post(
        "/investigations", headers=user_headers,
        json={"question": "What is the total amount?", "document_type": "invoice"},
    )
    assert response.status_code == 201, response.text
    assert {document for document, _ in pages(response.json()["sources"])} == {"Invoice_102.pdf"}

    wrong_year = await client.post(
        "/investigations", headers=user_headers, json={"question": "What is the total amount?", "year": 1999}
    )
    assert wrong_year.json()["status"] == "insufficient_evidence"

    bad_type = await client.post(
        "/investigations", headers=user_headers, json={"question": "q", "document_type": "love_letter"}
    )
    assert bad_type.status_code == 400


async def test_two_files_with_the_same_name_stay_distinguishable(client, user_headers, tmp_path):
    for folder, text in (("a", "Supplier quote. Total amount due: £100."), ("b", "Supplier quote, revised. Total amount due: £250.")):
        (tmp_path / folder).mkdir()
        path = tmp_path / folder / "Quote.txt"
        path.write_text(text, encoding="utf-8")
        await upload(client, user_headers, path)

    report = (await client.post("/investigations", headers=user_headers, json={"question": "Compare the total amounts in the quotes."})).json()
    sources = [source for source in report["sources"] if source["document"] == "Quote.txt"]
    assert len(sources) == 2 and len({source["document_id"] for source in sources}) == 2
    assert [c["type"] for c in report["contradictions"]] == ["amount_mismatch"]

    findings = (await client.get(f"/investigations/{report['investigation_id']}/findings", headers=user_headers)).json()["findings"]
    fact_documents = {f["source_document_id"] for f in findings if f["finding_type"] == "fact"}
    assert fact_documents == {source["document_id"] for source in sources}


async def test_report_is_explicit_when_no_model_answered(client, user_headers, library, monkeypatch):
    from app.agents import service
    from app.agents.graph import build_investigation_graph
    from app.agents.nodes import AgentDeps
    from app.config import get_settings
    from app.database.session import get_session_factory as session_factory
    from app.embeddings.factory import get_embedding_provider
    from app.rag.retriever import PgVectorRetriever
    from tests.helpers import DownLLM

    graph = build_investigation_graph(AgentDeps(
        llm=DownLLM(), retriever=PgVectorRetriever(session_factory(), get_embedding_provider()), settings=get_settings(),
    ))
    monkeypatch.setattr(service, "get_graph", lambda: graph)

    report = (await client.post("/investigations", headers=user_headers, json={"question": DEMO_QUESTION})).json()
    assert report["stats"]["llm_calls"] == 0
    assert report["stats"]["llm_provider"] == "down (unavailable - rules used)"
    assert any("language model was unavailable" in text for text in report["limitations"])
    assert report["risk_level"] == "high"  # the rule-based safety net still catches the mismatch


async def test_health_reports_503_when_the_database_is_unreachable(client, monkeypatch):
    from app.api import health

    async def database_down() -> bool:
        return False

    monkeypatch.setattr(health, "check_db", database_down)
    response = await client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded" and response.json()["database"] == "unreachable"
