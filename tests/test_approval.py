"""Checkpointing and human approval: pause, decide, resume, recover.

The first half drives the graph directly with an in-memory checkpointer (no
database needed). The second half goes through the HTTP API with the real
PostgreSQL checkpointer.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import func, select

from app.agents.graph import build_investigation_graph, recursion_limit
from app.agents.nodes import AgentDeps
from app.agents.state import initial_state
from app.config import get_settings
from app.database.models import Finding, Investigation, ToolCall
from app.database.session import get_session_factory
from tests.helpers import ScriptedLLM, sample_library

DEMO_QUESTION = "Review this invoice against the contract and tell me whether it should be approved."


def build(samples_dir, llm=None, *, pause=True, settings=None):
    settings = settings or get_settings()
    llm = llm or ScriptedLLM()
    graph = build_investigation_graph(
        AgentDeps(llm=llm, retriever=sample_library(samples_dir), settings=settings, pause_for_approval=pause),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "inv-1"}, "recursion_limit": recursion_limit(settings.MAX_AGENT_ITERATIONS)}
    return graph, config, llm


def start(question: str = DEMO_QUESTION):
    return initial_state(user_id="user-1", investigation_id="inv-1", question=question)


async def waiting_request(graph, config) -> dict | None:
    snapshot = await graph.aget_state(config)
    requests = [interrupt.value for task in snapshot.tasks for interrupt in task.interrupts]
    return requests[0] if requests else None


# ================================================================ graph level
async def test_graph_pauses_with_a_draft_report_when_a_person_must_decide(samples_dir):
    graph, config, _ = build(samples_dir)
    result = await graph.ainvoke(start(), config)

    assert "report" not in result  # the final report does not exist yet
    request = await waiting_request(graph, config)
    assert request["type"] == "approval_request"
    assert request["options"] == ["approve", "reject", "modify"]
    draft = request["draft_report"]
    assert draft["human_review"] == {
        "required": True, "status": "pending", "reviewer": None, "note": None,
        "decided_at": None, "original_action": None,
    }
    assert draft["risk_level"] == "high" and draft["contradictions"]
    assert "waiting for a decision" in draft["answer_text"]
    assert (await graph.aget_state(config)).next == ("human_approval",)


async def test_approve_finishes_the_report_without_repeating_earlier_work(samples_dir):
    graph, config, llm = build(samples_dir)
    await graph.ainvoke(start(), config)
    calls_before = list(llm.calls)
    draft = (await waiting_request(graph, config))["draft_report"]

    state = await graph.ainvoke(
        Command(resume={"decision": "approve", "note": "Checked with procurement.", "reviewer": "Mariam"}), config
    )

    report = state["report"]
    assert report["human_review"]["status"] == "approved"
    assert report["human_review"]["reviewer"] == "Mariam"
    assert report["human_review"]["note"] == "Checked with procurement."
    assert report["human_review"]["decided_at"]
    assert report["recommended_action"] == draft["recommended_action"]
    assert report["human_review"]["original_action"] is None
    assert "Human Review: Approved by Mariam - Checked with procurement." in report["answer_text"]
    # Resuming continued after the pause: no search, extraction or assessment ran again.
    assert llm.calls == calls_before
    assert report["key_findings"] == draft["key_findings"]
    assert await waiting_request(graph, config) is None


async def test_modify_replaces_the_recommendation_and_keeps_the_original(samples_dir):
    graph, config, _ = build(samples_dir)
    await graph.ainvoke(start(), config)
    draft = (await waiting_request(graph, config))["draft_report"]

    state = await graph.ainvoke(
        Command(resume={"decision": "modify", "modified_action": "Pay £40,000 now and dispute the remaining £18,000."}),
        config,
    )
    report = state["report"]
    assert report["human_review"]["status"] == "modified"
    assert report["recommended_action"] == "Pay £40,000 now and dispute the remaining £18,000."
    assert report["human_review"]["original_action"] == draft["recommended_action"]
    assert report["human_review"]["reviewer"] == "reviewer"  # default when none is given
    # A person changed the recommendation, not the evidence.
    assert report["risk_level"] == draft["risk_level"] and report["contradictions"] == draft["contradictions"]


async def test_reject_is_recorded_with_the_reason(samples_dir):
    graph, config, _ = build(samples_dir)
    await graph.ainvoke(start(), config)
    state = await graph.ainvoke(
        Command(resume={"decision": "reject", "note": "The amendment was signed last week."}), config
    )
    report = state["report"]
    assert report["human_review"]["status"] == "rejected"
    assert report["recommended_action"] == (
        "The reviewer rejected the agent's recommendation. Reason: The amendment was signed last week."
    )
    assert report["human_review"]["original_action"]


@pytest.mark.parametrize(
    "decision",
    [{"decision": "maybe"}, {"decision": "modify"}, {"decision": "modify", "modified_action": "   "}, "approve", 42],
)
async def test_an_invalid_decision_is_refused_and_the_investigation_keeps_waiting(samples_dir, decision):
    graph, config, _ = build(samples_dir)
    await graph.ainvoke(start(), config)

    result = await graph.ainvoke(Command(resume=decision), config)

    assert "report" not in result  # not finished
    request = await waiting_request(graph, config)  # asked again, with the reason
    assert request["error"] and request["draft_report"]["human_review"]["status"] == "pending"
    state = await graph.ainvoke(Command(resume={"decision": "approve"}), config)  # and can still be decided
    assert state["report"]["human_review"]["status"] == "approved"


async def test_reviewer_text_is_cleaned_and_bounded(samples_dir):
    graph, config, _ = build(samples_dir)
    await graph.ainvoke(start(), config)
    state = await graph.ainvoke(
        Command(resume={"decision": "approve", "note": "ok\x00\x07 " + "x" * 5000, "reviewer": "R" * 500}), config
    )
    review = state["report"]["human_review"]
    assert "\x00" not in review["note"] and len(review["note"]) == 1000
    assert len(review["reviewer"]) == 100


async def test_no_pause_when_no_decision_is_needed(samples_dir):
    graph, config, _ = build(samples_dir)
    state = await graph.ainvoke(start("hello"), config)
    assert state["report"]["human_review"] == {
        "required": False, "status": "not_required", "reviewer": None, "note": None,
        "decided_at": None, "original_action": None,
    }
    assert await waiting_request(graph, config) is None


async def test_without_pausing_the_report_only_flags_that_a_person_should_look(samples_dir):
    graph, config, _ = build(samples_dir, pause=False)
    state = await graph.ainvoke(start(), config)
    assert state["report"]["requires_human_review"] is True
    assert state["report"]["human_review"]["status"] == "not_requested"
    assert "Human Review: Required" in state["report"]["answer_text"]


async def test_human_approval_can_be_switched_off_in_the_settings(samples_dir):
    settings = get_settings().model_copy(update={"ENABLE_HUMAN_APPROVAL": False})
    graph, config, _ = build(samples_dir, settings=settings)
    state = await graph.ainvoke(start(), config)
    assert state["report"]["human_review"]["status"] in ("not_required", "not_requested")
    assert await waiting_request(graph, config) is None


def test_pausing_needs_a_checkpointer(samples_dir):
    deps = AgentDeps(llm=ScriptedLLM(), retriever=sample_library(samples_dir), settings=get_settings(), pause_for_approval=True)
    with pytest.raises(ValueError, match="checkpointer"):
        build_investigation_graph(deps)


async def test_a_crashed_run_continues_from_the_last_checkpoint(samples_dir, monkeypatch):
    """Recovery: a node fails once (as if the server stopped); the next run does not start over."""
    from app.agents import nodes

    crash = [RuntimeError("server stopped")]
    original = nodes.baseline_risk

    def crash_once(*args, **kwargs):
        if crash:
            raise crash.pop()
        return original(*args, **kwargs)

    monkeypatch.setattr(nodes, "baseline_risk", crash_once)  # used by the assess_risk node
    graph, config, llm = build(samples_dir)

    with pytest.raises(RuntimeError, match="server stopped"):
        await graph.ainvoke(start(), config)
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("assess_risk",)  # everything before it is saved
    assert snapshot.values["evidence"] and snapshot.values["contradictions"]
    calls_before = list(llm.calls)

    await graph.ainvoke(None, config)  # None = continue from the checkpoint

    new_calls = llm.calls[len(calls_before):]
    assert "RequestAnalysis" not in new_calls and "ExtractedFacts" not in new_calls
    assert "RiskAssessment" in new_calls
    assert (await waiting_request(graph, config))["draft_report"]["risk_level"] == "high"


# ================================================================== API level
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


async def start_paused(client, headers) -> dict:
    response = await client.post("/investigations", headers=headers, json={"question": DEMO_QUESTION})
    assert response.status_code == 201, response.text
    report = response.json()
    assert report["human_review"]["status"] == "pending"
    return report


async def row_counts(investigation_id: str) -> tuple[int, int]:
    async with get_session_factory()() as session:
        findings = await session.scalar(
            select(func.count(Finding.id)).where(Finding.investigation_id == uuid.UUID(investigation_id))
        )
        tool_calls = await session.scalar(
            select(func.count(ToolCall.id)).where(ToolCall.investigation_id == uuid.UUID(investigation_id))
        )
    return int(findings), int(tool_calls)


@pytest.mark.integration
async def test_approve_through_the_api(client, user_headers, library):
    draft = await start_paused(client, user_headers)
    investigation_id = draft["investigation_id"]
    counts_while_paused = await row_counts(investigation_id)

    listing = (await client.get("/investigations", headers=user_headers)).json()["investigations"]
    assert listing[0]["approval_status"] == "pending" and listing[0]["completed_at"] is None

    response = await client.post(
        f"/investigations/{investigation_id}/approve", headers=user_headers,
        json={"decision": "approve", "note": "Amendment confirmed by phone."},
    )
    assert response.status_code == 200, response.text
    final = response.json()
    assert final["investigation_id"] == investigation_id
    assert final["human_review"]["status"] == "approved"
    assert final["human_review"]["reviewer"] == user_headers["X-User-Id"]
    assert final["human_review"]["note"] == "Amendment confirmed by phone."
    assert final["key_findings"] == draft["key_findings"]
    assert final["stats"]["duration_ms"] > draft["stats"]["duration_ms"]  # both runs are counted
    assert final["stats"]["llm_calls"] == draft["stats"]["llm_calls"]  # the decision cost no model call

    detail = (await client.get(f"/investigations/{investigation_id}", headers=user_headers)).json()
    assert detail["approval_status"] == "approved" and detail["completed_at"] is not None
    assert detail["report"]["human_review"]["status"] == "approved"
    assert "Approved by" in detail["report"]["answer_text"]

    # The final save replaced the draft's rows instead of adding to them.
    assert await row_counts(investigation_id) == counts_while_paused

    # The trace shows two runs: the investigation and the decision.
    trace = (await client.get(f"/investigations/{investigation_id}/trace", headers=user_headers)).json()
    assert trace["totals"]["runs"] == 2
    second_run = [event["name"] for event in trace["events"] if event["run"] == 2 and event["kind"] == "node"]
    assert second_run == ["human_approval", "follow_up", "final_report"]

    # A finished investigation leaves no checkpoints behind.
    from app.agents import service

    graph = await service.get_graph()
    snapshot = await graph.aget_state({"configurable": {"thread_id": investigation_id}})
    assert snapshot.values == {}

    # ... and cannot be decided twice.
    again = await client.post(
        f"/investigations/{investigation_id}/approve", headers=user_headers, json={"decision": "reject"}
    )
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "conflict"


@pytest.mark.integration
async def test_modify_and_reject_through_the_api(client, user_headers, library):
    first = await start_paused(client, user_headers)
    modified = await client.post(
        f"/investigations/{first['investigation_id']}/approve", headers=user_headers,
        json={"decision": "modify", "modified_action": "Pay the contract value only.", "note": "Agreed with finance."},
    )
    assert modified.status_code == 200, modified.text
    body = modified.json()
    assert body["recommended_action"] == "Pay the contract value only."
    assert body["human_review"]["original_action"] == first["recommended_action"]

    findings = (await client.get(f"/investigations/{first['investigation_id']}/findings", headers=user_headers)).json()["findings"]
    recommendation = next(f for f in findings if f["finding_type"] == "recommendation")
    assert recommendation["statement"] == "Pay the contract value only."

    second = await start_paused(client, user_headers)
    rejected = await client.post(
        f"/investigations/{second['investigation_id']}/approve", headers=user_headers,
        json={"decision": "reject", "note": "Wrong contract version."},
    )
    assert rejected.json()["human_review"]["status"] == "rejected"
    detail = (await client.get(f"/investigations/{second['investigation_id']}", headers=user_headers)).json()
    assert detail["approval_status"] == "rejected"


@pytest.mark.integration
async def test_invalid_decisions_leave_the_investigation_waiting(client, user_headers, library):
    draft = await start_paused(client, user_headers)
    url = f"/investigations/{draft['investigation_id']}/approve"

    assert (await client.post(url, headers=user_headers, json={"decision": "maybe"})).status_code == 422
    assert (await client.post(url, headers=user_headers, json={})).status_code == 422
    no_action = await client.post(url, headers=user_headers, json={"decision": "modify"})
    assert no_action.status_code == 400 and "modified_action" in no_action.json()["error"]["message"]
    too_long = await client.post(url, headers=user_headers, json={"decision": "approve", "note": "x" * 2000})
    assert too_long.status_code == 422

    intruder = {"X-User-Id": user_headers["X-User-Id"] + "-intruder"}
    assert (await client.post(url, headers=intruder, json={"decision": "approve"})).status_code == 404
    unknown = await client.post(f"/investigations/{uuid.uuid4()}/approve", headers=user_headers, json={"decision": "approve"})
    assert unknown.status_code == 404

    detail = (await client.get(f"/investigations/{draft['investigation_id']}", headers=user_headers)).json()
    assert detail["approval_status"] == "pending"
    assert (await client.post(url, headers=user_headers, json={"decision": "approve"})).status_code == 200


@pytest.mark.integration
async def test_two_decisions_at_the_same_moment_only_one_wins(client, user_headers, library):
    draft = await start_paused(client, user_headers)
    url = f"/investigations/{draft['investigation_id']}/approve"
    responses = await asyncio.gather(
        client.post(url, headers=user_headers, json={"decision": "approve"}),
        client.post(url, headers=user_headers, json={"decision": "reject"}),
    )
    assert sorted(response.status_code for response in responses) == [200, 409]
    winner = next(response for response in responses if response.status_code == 200).json()
    detail = (await client.get(f"/investigations/{draft['investigation_id']}", headers=user_headers)).json()
    assert detail["approval_status"] == winner["human_review"]["status"]


@pytest.mark.integration
async def test_an_investigation_without_findings_to_decide_is_final_at_once(client, user_headers):
    report = (await client.post("/investigations", headers=user_headers, json={"question": "hello"})).json()
    assert report["human_review"]["status"] == "not_required"
    detail = (await client.get(f"/investigations/{report['investigation_id']}", headers=user_headers)).json()
    assert detail["approval_status"] == "not_required" and detail["completed_at"] is not None
    response = await client.post(
        f"/investigations/{report['investigation_id']}/approve", headers=user_headers, json={"decision": "approve"}
    )
    assert response.status_code == 409


@pytest.mark.integration
async def test_chat_reports_that_the_answer_waits_for_a_decision(client, user_headers, library):
    body = (await client.post("/chat", headers=user_headers, json={"message": DEMO_QUESTION})).json()
    assert body["approval_status"] == "pending"
    assert body["answer"].endswith("waiting for a human decision before it is final.")
    approved = await client.post(
        f"/investigations/{body['investigation_id']}/approve", headers=user_headers, json={"decision": "approve"}
    )
    assert approved.status_code == 200


@pytest.mark.integration
async def test_an_interrupted_run_can_be_resumed_from_its_checkpoint(client, user_headers, library, monkeypatch):
    """Simulates a server that stopped in the middle of an investigation."""
    from app.agents import service
    from app.database import repositories
    from app.security.permissions import user_uuid_for

    settings = get_settings()
    graph = await service.get_graph()
    async with get_session_factory()() as session:
        investigation = await repositories.create_investigation(
            session, user_uuid_for(user_headers["X-User-Id"]), DEMO_QUESTION, None
        )
    investigation_id = str(investigation.id)
    config = service._config(investigation.id)
    # Run only the first part of the graph, then "crash" (nothing is stored in the normal tables).
    await graph.ainvoke(
        initial_state(user_id=str(investigation.user_id), investigation_id=investigation_id, question=DEMO_QUESTION),
        config, interrupt_before=["assess_risk"],
    )
    assert (await graph.aget_state(config)).next == ("assess_risk",)
    detail = (await client.get(f"/investigations/{investigation_id}", headers=user_headers)).json()
    assert detail["status"] == "running" and detail["report"] is None

    # Too early: the run might still be alive.
    early = await client.post(f"/investigations/{investigation_id}/resume", headers=user_headers)
    assert early.status_code == 409 and "may still be running" in early.json()["error"]["message"]

    monkeypatch.setattr(settings, "STALE_RUN_SECONDS", 0)
    intruder = {"X-User-Id": user_headers["X-User-Id"] + "-intruder"}
    assert (await client.post(f"/investigations/{investigation_id}/resume", headers=intruder)).status_code == 404

    resumed = await client.post(f"/investigations/{investigation_id}/resume", headers=user_headers)
    assert resumed.status_code == 200, resumed.text
    report = resumed.json()
    assert report["risk_level"] == "high" and report["human_review"]["status"] == "pending"

    # Only the missing nodes ran in the resumed run.
    trace = (await client.get(f"/investigations/{investigation_id}/trace", headers=user_headers)).json()
    nodes = [event["name"] for event in trace["events"] if event["kind"] == "node"]
    assert nodes[0] == "assess_risk" and "retrieve" not in nodes and "extract_evidence" not in nodes

    # It is paused now, not interrupted: resume is refused, approve works.
    assert (await client.post(f"/investigations/{investigation_id}/resume", headers=user_headers)).status_code == 409
    approved = await client.post(
        f"/investigations/{investigation_id}/approve", headers=user_headers, json={"decision": "approve"}
    )
    assert approved.status_code == 200 and approved.json()["human_review"]["status"] == "approved"


@pytest.mark.integration
async def test_resume_with_nothing_saved_marks_the_investigation_failed(client, user_headers, monkeypatch):
    from app.database import repositories
    from app.security.permissions import user_uuid_for

    await client.get("/investigations", headers=user_headers)  # creates the user
    async with get_session_factory()() as session:
        investigation = await repositories.create_investigation(
            session, user_uuid_for(user_headers["X-User-Id"]), "What is the total?", None
        )
    monkeypatch.setattr(get_settings(), "STALE_RUN_SECONDS", 0)
    response = await client.post(f"/investigations/{investigation.id}/resume", headers=user_headers)
    assert response.status_code == 409 and "start it again" in response.json()["error"]["message"]
    async with get_session_factory()() as session:
        stored = await session.get(Investigation, investigation.id)
    assert stored.status == "failed"


@pytest.mark.integration
async def test_a_failed_decision_run_can_be_decided_again(client, user_headers, library, monkeypatch):
    """If delivering the decision fails, the investigation goes back to 'pending' instead of getting stuck."""
    from app.agents import service

    draft = await start_paused(client, user_headers)
    url = f"/investigations/{draft['investigation_id']}/approve"
    graph = await service.get_graph()
    original = graph.astream

    def broken_stream(*args, **kwargs):
        raise ConnectionError("database connection lost")

    monkeypatch.setattr(graph, "astream", broken_stream)
    with pytest.raises(ConnectionError):  # the test client re-raises what would be a 500 response
        await client.post(url, headers=user_headers, json={"decision": "approve"})
    monkeypatch.setattr(graph, "astream", original)

    detail = (await client.get(f"/investigations/{draft['investigation_id']}", headers=user_headers)).json()
    assert detail["approval_status"] == "pending" and detail["status"] == "review_required"
    assert (await client.post(url, headers=user_headers, json={"decision": "approve"})).status_code == 200
