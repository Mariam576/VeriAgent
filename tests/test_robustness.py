"""What happens when a run does not end normally: failures, cut-off streams, deleted
documents, old data and many requests at once. Each test is a defect that was found
in review and must stay fixed."""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from sqlalchemy import select, text

from app.config import get_settings
from app.database import repositories
from app.database.models import Finding, Investigation
from app.database.session import get_engine, get_session_factory
from app.schemas.response import InvestigationReport
from app.security.permissions import user_uuid_for

pytestmark = pytest.mark.integration

DEMO_QUESTION = "Review this invoice against the contract and tell me whether it should be approved."


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


async def detail(client, headers, investigation_id) -> dict:
    return (await client.get(f"/investigations/{investigation_id}", headers=headers)).json()


async def start_paused(client, headers) -> dict:
    report = (await client.post("/investigations", headers=headers, json={"question": DEMO_QUESTION})).json()
    assert report["human_review"]["status"] == "pending"
    return report


def fail_saving(monkeypatch, times: int):
    """Make storing the result fail the next ``times`` times (as if the database hiccuped)."""
    original = repositories.save_investigation_result
    remaining = {"count": times}

    async def flaky(*args, **kwargs):
        if remaining["count"] > 0:
            remaining["count"] -= 1
            raise ConnectionError("could not connect to server: postgresql://postgres:hunter2@db/veriagent")
        return await original(*args, **kwargs)

    monkeypatch.setattr(repositories, "save_investigation_result", flaky)


# ============================================= deleted documents and decisions
async def test_a_decision_still_works_after_a_cited_document_was_deleted(client, user_headers, library):
    draft = await start_paused(client, user_headers)
    investigation_id = draft["investigation_id"]
    deleted = await client.delete(f"/documents/{library['Invoice_102.pdf']['id']}", headers=user_headers)
    assert deleted.status_code == 204

    response = await client.post(
        f"/investigations/{investigation_id}/approve", headers=user_headers, json={"decision": "approve"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["human_review"]["status"] == "approved"
    stored = await detail(client, user_headers, investigation_id)
    assert stored["approval_status"] == "approved" and stored["completed_at"] is not None

    # The findings keep their text; only the link to the deleted document is gone.
    async with get_session_factory()() as session:
        findings = (await session.execute(
            select(Finding).where(Finding.investigation_id == uuid.UUID(investigation_id))
        )).scalars().all()
    invoice_facts = [f for f in findings if f.finding_type == "fact" and f.statement.startswith("Invoice_102.pdf")]
    contract_facts = [f for f in findings if f.finding_type == "fact" and f.statement.startswith("Contract.pdf")]
    assert invoice_facts and all(f.source_document_id is None for f in invoice_facts)
    assert contract_facts and all(str(f.source_document_id) == library["Contract.pdf"]["id"] for f in contract_facts)


# ====================================================== failures after a decision
async def test_a_failure_after_the_decision_is_recorded_can_be_resumed_at_once(client, user_headers, library, monkeypatch):
    draft = await start_paused(client, user_headers)
    investigation_id = draft["investigation_id"]
    url = f"/investigations/{investigation_id}"

    fail_saving(monkeypatch, times=1)
    with pytest.raises(ConnectionError):  # the test client re-raises what would be a 500 response
        await client.post(f"{url}/approve", headers=user_headers, json={"decision": "approve", "note": "ok"})

    stored = await detail(client, user_headers, investigation_id)
    assert (stored["status"], stored["approval_status"]) == ("interrupted", "deciding")
    assert stored["error"] == "Something went wrong on the server."  # no connection string, no password
    assert stored["completed_at"] is None
    assert (await client.get("/memory", headers=user_headers)).json()["notes"] == []  # nothing remembered yet

    # The decision is already saved in the checkpoint: it cannot be made a second time ...
    again = await client.post(f"{url}/approve", headers=user_headers, json={"decision": "reject"})
    assert again.status_code == 409 and "resume" in again.json()["error"]["message"]
    # ... but the run can be finished immediately, without waiting for a timeout.
    resumed = await client.post(f"{url}/resume", headers=user_headers)
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["human_review"]["status"] == "approved" and resumed.json()["human_review"]["note"] == "ok"

    stored = await detail(client, user_headers, investigation_id)
    assert (stored["status"], stored["approval_status"], stored["error"]) == ("review_required", "approved", None)
    notes = (await client.get("/memory", headers=user_headers)).json()["notes"]
    assert len(notes) == 1 and "Human decision: approved (ok)." in notes[0]["text"]


async def test_a_resume_that_fails_again_marks_the_investigation_failed(client, user_headers, library, monkeypatch):
    """A broken investigation is not retried forever."""
    draft = await start_paused(client, user_headers)
    url = f"/investigations/{draft['investigation_id']}"
    fail_saving(monkeypatch, times=2)
    with pytest.raises(ConnectionError):
        await client.post(f"{url}/approve", headers=user_headers, json={"decision": "approve"})
    with pytest.raises(ConnectionError):
        await client.post(f"{url}/resume", headers=user_headers)

    stored = await detail(client, user_headers, draft["investigation_id"])
    assert stored["status"] == "failed" and stored["error"] and stored["completed_at"] is not None
    final = await client.post(f"{url}/resume", headers=user_headers)
    assert final.status_code == 409 and "nothing to resume" in final.json()["error"]["message"]
    assert (await client.post(f"{url}/approve", headers=user_headers, json={"decision": "approve"})).status_code == 409


async def test_a_failed_start_says_why_in_safe_words(client, user_headers, library, monkeypatch):
    from app.agents import service
    from app.llm.base import LLMConfigurationError

    async def no_key():
        raise LLMConfigurationError("GEMINI_API_KEY is not set. Add it to your .env file.")

    monkeypatch.setattr(service, "get_graph", no_key)
    assert (await client.post("/investigations", headers=user_headers, json={"question": DEMO_QUESTION})).status_code == 503
    listing = (await client.get("/investigations", headers=user_headers)).json()["investigations"]
    assert listing[0]["status"] == "failed" and "GEMINI_API_KEY is not set" in listing[0]["error"]


# ============================================================ cut-off streams
async def test_a_cut_off_stream_is_recorded_as_interrupted_and_can_be_resumed(client, user_headers, library):
    from app.agents import service

    user = None
    async with get_session_factory()() as session:
        user = await repositories.get_or_create_user(
            session, user_uuid_for(user_headers["X-User-Id"]), user_headers["X-User-Id"]
        )
        events = service.stream_investigation(session, user=user, question=DEMO_QUESTION)
        investigation_id = None
        async for item in events:
            if item["type"] == "started":
                investigation_id = str(item["investigation"].id)
            if item["type"] == "progress" and item["node"] == "extract_evidence":
                break  # the browser tab is closed while the agent is reading the evidence
        await events.aclose()

    stored = await detail(client, user_headers, investigation_id)
    assert stored["status"] == "interrupted" and stored["report"] is None
    assert "cut off" in stored["error"]
    # What it did before it was cut off is in the trace.
    trace = (await client.get(f"/investigations/{investigation_id}/trace", headers=user_headers)).json()
    first_run_nodes = [event["name"] for event in trace["events"] if event["kind"] == "node"]
    assert "analyze_request" in first_run_nodes and "final_report" not in first_run_nodes
    assert str(investigation_id) not in service._active

    # No waiting period: the server knows this run is over.
    resumed = await client.post(f"/investigations/{investigation_id}/resume", headers=user_headers)
    assert resumed.status_code == 200, resumed.text
    report = resumed.json()
    assert report["risk_level"] == "high" and report["human_review"]["status"] == "pending"
    stored = await detail(client, user_headers, investigation_id)
    assert (stored["status"], stored["error"]) == ("review_required", None)

    # The work done before the cut was not repeated, and every model call is in the trace.
    trace = (await client.get(f"/investigations/{investigation_id}/trace", headers=user_headers)).json()
    assert trace["totals"]["runs"] == 2
    second_run_nodes = [event["name"] for event in trace["events"] if event["run"] == 2 and event["kind"] == "node"]
    assert "analyze_request" not in second_run_nodes and "retrieve" not in second_run_nodes[:1]
    assert trace["totals"]["llm_calls"] == report["stats"]["llm_calls"]


async def test_a_stream_closed_after_the_report_is_not_an_interruption(client, user_headers):
    from app.agents import service

    async with get_session_factory()() as session:
        user = await repositories.get_or_create_user(
            session, user_uuid_for(user_headers["X-User-Id"]), user_headers["X-User-Id"]
        )
        events = service.stream_investigation(session, user=user, question="hello")
        investigation_id = None
        async for item in events:
            if item["type"] == "started":
                investigation_id = str(item["investigation"].id)
            if item["type"] == "report":
                break  # the client got what it wanted and stops reading
        await events.aclose()
    stored = await detail(client, user_headers, investigation_id)
    assert stored["status"] == "not_applicable" and stored["completed_at"] is not None and stored["error"] is None


# ================================================================== old data
def test_a_version_1_report_keeps_its_review_flag():
    version_1 = {
        "investigation_id": "x", "question": "q", "status": "review_required", "status_label": "Review Required",
        "risk_level": "high", "confidence": 0.9, "evidence_quality": "moderate", "requires_human_review": True,
        "summary": "s",
    }
    report = InvestigationReport(**version_1)
    assert (report.human_review.required, report.human_review.status) == (True, "not_requested")
    assert report.memory.recalled == [] and report.presentation.detail_level == "standard"
    calm = InvestigationReport(**{**version_1, "requires_human_review": False})
    assert (calm.human_review.required, calm.human_review.status) == (False, "not_required")
    # A Version 2 report is taken as it is.
    decided = InvestigationReport(**{**version_1, "human_review": {"required": True, "status": "approved"}})
    assert decided.human_review.status == "approved"


async def test_migration_backfills_the_review_flag_of_version_1_rows(client, user_headers):
    from app.database.migrations import run_migrations

    await client.get("/investigations", headers=user_headers)  # creates the user
    user_id = user_uuid_for(user_headers["X-User-Id"])
    version_1_report = {"requires_human_review": True, "summary": "old", "status": "review_required"}
    async with get_session_factory()() as session:
        needs_review = Investigation(user_id=user_id, question="old one", status="review_required", report=version_1_report)
        calm = Investigation(user_id=user_id, question="old two", status="no_issues_found", report={"requires_human_review": False})
        session.add_all([needs_review, calm])
        await session.commit()
        ids = (needs_review.id, calm.id)

    async with get_engine().begin() as conn:
        await conn.execute(text("DELETE FROM schema_migrations WHERE name = 'v2_004_backfill_review_flag'"))
        assert await run_migrations(conn) == ["v2_004_backfill_review_flag"]
    async with get_engine().begin() as conn:
        assert await run_migrations(conn) == []  # running it again changes nothing

    async with get_session_factory()() as session:
        statuses = [(await session.get(Investigation, investigation_id)).approval_status for investigation_id in ids]
    assert statuses == ["not_requested", "not_required"]


# ========================================================= many requests at once
async def test_many_requests_at_once_do_not_starve_the_connection_pool(client, user_headers):
    """Each request used to hold one database connection while asking for a second one."""
    started = time.perf_counter()
    responses = await asyncio.gather(
        *(client.get("/memory", headers=user_headers) for _ in range(40)),
        *(client.post("/investigations", headers=user_headers, json={"question": "hello"}) for _ in range(20)),
        *(client.get("/documents", headers=user_headers) for _ in range(10)),
    )
    assert [response.status_code for response in responses[:40]] == [200] * 40
    assert [response.status_code for response in responses[40:60]] == [201] * 20
    assert [response.status_code for response in responses[60:]] == [200] * 10
    assert time.perf_counter() - started < 20  # far below the pool's 30 second timeout


async def test_a_browser_that_disconnects_mid_stream_leaves_a_resumable_investigation(client, user_headers, library, monkeypatch):
    """The real path: Starlette closes the SSE body, which cancels the task that runs the graph."""
    from app.api import sse
    from app.api.investigations import investigation_event
    from app.agents import service
    from app.llm.mock_provider import MockLLM

    original = MockLLM._structured

    async def slow(self, schema, system, user):
        await asyncio.sleep(0.05)  # a model call takes time; the disconnect happens during one
        return await original(self, schema, system, user)

    monkeypatch.setattr(MockLLM, "_structured", slow)
    session = get_session_factory()()
    user = await repositories.get_or_create_user(
        session, user_uuid_for(user_headers["X-User-Id"]), user_headers["X-User-Id"]
    )
    events = service.stream_investigation(session, user=user, question=DEMO_QUESTION)

    async def body():
        try:
            async for item in events:
                yield investigation_event(item)
        finally:
            await events.aclose()
            await session.close()

    stream = sse.sse_body(body())
    investigation_id, seen = None, []
    async for chunk in stream:
        seen.append(chunk)
        if '"investigation_id"' in chunk:
            investigation_id = chunk.split('"investigation_id": "')[1].split('"')[0]
        if "Searching documents..." in chunk:
            break
    await stream.aclose()  # the browser is gone

    stored = await detail(client, user_headers, investigation_id)
    assert stored["status"] == "interrupted" and "cut off" in stored["error"]
    assert investigation_id not in service._active
    calls_before = len(service.get_llm().calls)
    await asyncio.sleep(0.3)
    assert len(service.get_llm().calls) == calls_before  # nothing keeps running (and costing) in the background

    resumed = await client.post(f"/investigations/{investigation_id}/resume", headers=user_headers)
    assert resumed.status_code == 200 and resumed.json()["risk_level"] == "high"
