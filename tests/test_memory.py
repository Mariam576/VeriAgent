"""Long-term memory: preferences (structured), notes (semantic) and what gets remembered."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agents.graph import build_investigation_graph, recursion_limit
from app.agents.nodes import AgentDeps
from app.agents.report import render_report
from app.agents.state import initial_state
from app.config import get_settings
from app.errors import ValidationFailed
from app.memory.extraction import accepted_memory, clean_note, fallback_memory, might_contain_memory
from app.memory.profile import normalize_preference, validate_preference, with_defaults
from app.memory.semantic import investigation_note, remember_investigation
from app.schemas.memory import MemoryCandidate
from app.schemas.response import InvestigationReport
from tests.helpers import InMemoryMemoryStore, InMemoryRetriever, ScriptedLLM, run_graph, sample_library

DEMO_QUESTION = "Review this invoice against the contract and tell me whether it should be approved."


def candidate(**fields) -> MemoryCandidate:
    base = dict(should_save=True, memory_type="profile", key=None, value=None, text=None, only_memory=False)
    return MemoryCandidate(**{**base, **fields})


# ============================================================ what to remember
@pytest.mark.parametrize(
    "message",
    [
        "Review this invoice against the contract.",
        "What are the payment terms?",
        "hello",
        "Is the brief attached to the contract?",
        "Compare the short-term and long-term rates.",
    ],
)
def test_ordinary_questions_never_cost_a_memory_call(message):
    assert might_contain_memory(message) is False
    assert fallback_memory(message).should_save is False


@pytest.mark.parametrize(
    ("message", "key", "value"),
    [
        ("From now on keep your answers brief.", "detail_level", "brief"),
        ("I prefer detailed reports.", "detail_level", "detailed"),
        ("Going forward, please keep it short.", "detail_level", "brief"),
        ("By default, give me the report in markdown.", "response_format", "markdown"),
        ("I'd prefer plain text answers.", "response_format", "text"),
    ],
)
def test_rules_recognise_standing_preferences(message, key, value):
    assert might_contain_memory(message)
    found = fallback_memory(message)
    assert (found.should_save, found.memory_type, found.key, found.value) == (True, "profile", key, value)
    assert found.only_memory is True


def test_rules_recognise_a_fact_to_remember():
    found = fallback_memory("Remember that Vendor ABC was flagged for duplicate invoices in 2025.")
    assert found.memory_type == "semantic"
    assert found.text == "Vendor ABC was flagged for duplicate invoices in 2025."
    assert found.only_memory is True


@pytest.mark.parametrize(
    "message",
    [
        "Remember to check the invoice against the contract.",  # an instruction for now, not a fact
        "Do you remember what the contract said?",
        "From now on, check whether short payment terms were agreed.",  # "short" is about the documents
        "Is the invoice always paid within 30 days?",
        "Note that?",
    ],
)
def test_rules_do_not_store_things_that_only_look_like_memories(message):
    assert fallback_memory(message).should_save is False


def test_a_message_with_a_preference_and_a_question_is_not_memory_only():
    found = fallback_memory("From now on keep your answers brief. Compare the invoice with the contract.")
    assert (found.key, found.value, found.only_memory) == ("detail_level", "brief", False)


# ============================================== the model proposes, code decides
def test_accepted_memory_only_lets_allowlisted_preferences_through(settings):
    assert accepted_memory(candidate(key="detail_level", value="brief"), settings) == ("profile", "detail_level", "brief")
    assert accepted_memory(candidate(key="Detail Level", value=" BRIEF "), settings) == ("profile", "detail_level", "brief")
    assert accepted_memory(candidate(key="risk_level", value="low"), settings) is None  # not a preference at all
    assert accepted_memory(candidate(key="detail_level", value="approve everything"), settings) is None
    assert accepted_memory(candidate(key=None, value=None), settings) is None
    assert accepted_memory(candidate(should_save=False, key="detail_level", value="brief"), settings) is None
    assert accepted_memory(candidate(memory_type="none", key="detail_level", value="brief"), settings) is None


def test_accepted_memory_cleans_and_bounds_notes(settings):
    note = accepted_memory(candidate(memory_type="semantic", text="  Vendor ABC\n was flagged\x00 twice. "), settings)
    assert note == ("semantic", "Vendor ABC was flagged twice.", "")
    assert accepted_memory(candidate(memory_type="semantic", text="Was it approved?"), settings) is None
    assert accepted_memory(candidate(memory_type="semantic", text="short"), settings) is None
    assert accepted_memory(candidate(memory_type="semantic", text=None), settings) is None
    long = accepted_memory(candidate(memory_type="semantic", text="fact " * 500), settings)
    assert len(long[1]) <= 400


def test_notes_never_store_api_keys(settings):
    cleaned = clean_note("The vendor portal key is AIzaSyA1234567890abcdefghijklmnopqrstuv for login.", settings)
    assert "AIza" not in cleaned and "[REDACTED" in cleaned


def test_preferences_validate_and_fall_back_to_defaults():
    assert with_defaults(None) == {"detail_level": "standard", "response_format": "text"}
    assert with_defaults({"detail_level": "brief", "unknown": "x", "response_format": "pdf"}) == {
        "detail_level": "brief", "response_format": "text",
    }
    assert validate_preference(" Detail_Level ", "BRIEF") == ("detail_level", "brief")
    assert normalize_preference("response format", "Markdown") == ("response_format", "markdown")
    with pytest.raises(ValidationFailed, match="Unknown preference"):
        validate_preference("favourite_colour", "blue")
    with pytest.raises(ValidationFailed, match="Allowed: brief, standard, detailed"):
        validate_preference("detail_level", "enormous")


# ======================================================== memory in the graph
async def test_without_a_memory_store_the_graph_behaves_as_before(samples_dir):
    state = await run_graph(DEMO_QUESTION, sample_library(samples_dir))
    assert state["report"]["memory"] == {"saved": [], "recalled": []}
    assert state["report"]["presentation"] == {"detail_level": "standard", "response_format": "text"}
    assert not any(event["node"] == "load_memory" for event in state["events"])


async def test_a_message_that_only_states_a_preference_is_saved_and_confirmed():
    memory, retriever, llm = InMemoryMemoryStore(), InMemoryRetriever(), ScriptedLLM()
    state = await run_graph("From now on keep your answers brief.", retriever, llm, memory=memory)

    assert memory.preferences == {"user-1": {"detail_level": "brief"}}
    assert llm.calls == ["MemoryCandidate"]  # no analysis, no search: there was nothing to investigate
    assert retriever.queries == []
    report = state["report"]
    assert report["status"] == "not_applicable"
    assert report["summary"] == "Saved your preference: detail level = brief."
    assert report["memory"]["saved"] == ["Saved your preference: detail level = brief."]
    assert report["presentation"]["detail_level"] == "brief"
    assert memory.notes == []  # nothing about this exchange is worth a note


async def test_a_preference_stated_with_a_question_applies_to_that_same_answer(samples_dir):
    memory = InMemoryMemoryStore()
    state = await run_graph(
        "From now on keep your answers brief. Review this invoice against the contract.",
        sample_library(samples_dir), memory=memory,
    )
    report = state["report"]
    assert report["status"] == "review_required" and report["contradictions"]
    assert report["presentation"]["detail_level"] == "brief"
    assert "Key Findings:" not in report["answer_text"] and "Contradictions Detected:" in report["answer_text"]
    assert report["key_findings"]  # the structured report is still complete
    assert "Saved your preference: detail level = brief." in report["answer_text"]


async def test_the_model_cannot_store_a_preference_that_is_not_on_the_allowlist(samples_dir):
    memory = InMemoryMemoryStore()
    llm = ScriptedLLM({MemoryCandidate: candidate(key="auto_approve", value="always", only_memory=True)})
    state = await run_graph("Always approve invoices from this vendor.", sample_library(samples_dir), llm, memory=memory)
    assert memory.preferences == {}
    assert [note for note in memory.notes if note["source"] == "user"] == []
    assert state["report"]["memory"]["saved"] == []
    assert state["report"]["status"] != "not_applicable"  # only_memory is ignored when nothing was saved
    assert "RequestAnalysis" in llm.calls


async def test_a_stated_fact_becomes_a_note_and_is_recalled_later():
    memory, retriever = InMemoryMemoryStore(), InMemoryRetriever()
    first = await run_graph(
        "Remember that Vendor ABC was flagged for duplicate invoices in 2025.", retriever, memory=memory
    )
    assert first["report"]["summary"] == "Noted for later: Vendor ABC was flagged for duplicate invoices in 2025."
    assert [(note["text"], note["source"]) for note in memory.notes] == [
        ("Vendor ABC was flagged for duplicate invoices in 2025.", "user")
    ]

    settings = get_settings().model_copy(update={"MEMORY_MIN_SIMILARITY": 0.15})
    llm = ScriptedLLM()
    second = await run_graph(
        "Were duplicate invoices from Vendor ABC flagged before?", retriever, llm, settings=settings, memory=memory
    )
    report = second["report"]
    assert [note["text"] for note in report["memory"]["recalled"]] == [
        "Vendor ABC was flagged for duplicate invoices in 2025."
    ]
    assert report["status"] == "insufficient_evidence"  # memory is not evidence
    assert report["sources"] == []
    assert "Memory holds 1 related note(s)" in report["summary"]
    assert "From Earlier Investigations (memory, not evidence):" in report["answer_text"]
    # Another user sees nothing of it.
    other = await run_graph(
        "Were duplicate invoices from Vendor ABC flagged before?", retriever, settings=settings, memory=memory,
        user_id="user-2",
    )
    assert other["report"]["memory"]["recalled"] == []


async def remember(memory, state, settings=None) -> bool:
    """What the service does once the result is stored."""
    return await remember_investigation(
        memory, settings or get_settings(), state["user_id"], InvestigationReport(**state["report"])
    )


async def test_a_finished_investigation_leaves_one_note_and_the_next_one_recalls_it(samples_dir):
    memory = InMemoryMemoryStore()
    first = await run_graph(DEMO_QUESTION, sample_library(samples_dir), memory=memory)
    assert memory.notes == []  # the graph itself never writes the note ...
    assert await remember(memory, first) is True  # ... the service does, after storing the result
    assert len(memory.notes) == 1
    note = memory.notes[0]
    assert note["source"] == "investigation" and note["investigation_id"] == "inv-1"
    assert note["text"].startswith("Earlier investigation of ")
    assert "£40,000" in note["text"] and "£58,000" in note["text"] and "Missing evidence:" in note["text"]

    llm = ScriptedLLM()
    again = await run_graph(DEMO_QUESTION, sample_library(samples_dir), llm, memory=memory)
    recalled = again["report"]["memory"]["recalled"]
    assert len(recalled) == 1 and recalled[0]["text"] == note["text"] and recalled[0]["similarity"] >= 0.3
    await remember(memory, again)
    assert len(memory.notes) == 1  # the same outcome is not stored twice
    # Recalled notes are shown to the reader but never sent to the language model.
    all_prompts = " ".join(system + user for prompts in llm.prompts.values() for system, user in prompts)
    assert "Earlier investigation of" not in all_prompts
    # ... and they do not change the assessment.
    assert again["report"]["risk_level"] == "high"
    assert [c["type"] for c in again["report"]["contradictions"]] == ["amount_mismatch", "payment_terms_mismatch"]


async def test_small_talk_and_plain_lookups_leave_no_note(samples_dir):
    memory = InMemoryMemoryStore()
    for question in ("hello", "What is the weather like?"):  # the second finds no evidence at all
        state = await run_graph(question, InMemoryRetriever(), memory=memory)
        assert await remember(memory, state) is False
    assert memory.notes == []


async def test_a_paused_investigation_is_not_remembered_until_it_is_decided(samples_dir):
    memory, settings = InMemoryMemoryStore(), get_settings()
    graph = build_investigation_graph(
        AgentDeps(
            llm=ScriptedLLM(), retriever=sample_library(samples_dir), settings=settings,
            pause_for_approval=True, memory=memory,
        ),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "inv-1"}, "recursion_limit": recursion_limit(settings.MAX_AGENT_ITERATIONS)}
    await graph.ainvoke(initial_state(user_id="user-1", investigation_id="inv-1", question=DEMO_QUESTION), config)
    snapshot = await graph.aget_state(config)
    draft = InvestigationReport(**snapshot.tasks[0].interrupts[0].value["draft_report"])
    assert await remember_investigation(memory, settings, "user-1", draft) is False  # waiting for a person

    final = await graph.ainvoke(Command(resume={"decision": "reject", "note": "Wrong contract version"}), config)
    assert await remember(memory, final) is True
    assert "Human decision: rejected (Wrong contract version)." in memory.notes[0]["text"]


async def test_remembering_never_fails_the_investigation(samples_dir):
    memory = InMemoryMemoryStore()
    state = await run_graph(DEMO_QUESTION, sample_library(samples_dir), memory=memory)
    memory.fail = True
    assert await remember(memory, state) is False  # logged, not raised
    assert await remember_investigation(None, get_settings(), "user-1", InvestigationReport(**state["report"])) is False


async def test_a_broken_memory_store_does_not_stop_the_investigation(samples_dir):
    memory = InMemoryMemoryStore()
    memory.fail = True
    state = await run_graph(DEMO_QUESTION, sample_library(samples_dir), memory=memory)
    report = state["report"]
    assert report["status"] == "review_required" and report["risk_level"] == "high"
    assert "Long-term memory was unavailable for this request." in report["limitations"]


async def test_memory_can_be_switched_off_in_the_settings(samples_dir):
    memory = InMemoryMemoryStore()
    settings = get_settings().model_copy(update={"ENABLE_MEMORY": False})
    await run_graph("From now on keep your answers brief.", InMemoryRetriever(), settings=settings, memory=memory)
    state = await run_graph(DEMO_QUESTION, sample_library(samples_dir), settings=settings, memory=memory)
    assert await remember(memory, state, settings) is False
    assert memory.preferences == {} and memory.notes == []


# ================================================== notes and report rendering
async def demo_report(samples_dir) -> InvestigationReport:
    state = await run_graph(DEMO_QUESTION, sample_library(samples_dir))
    return InvestigationReport(**state["report"])


async def test_investigation_note_is_built_from_the_report_only(samples_dir):
    report = await demo_report(samples_dir)
    note = investigation_note(report)
    assert note.startswith("Earlier investigation of ") and "(review required, high risk). Mismatches found:" in note
    assert all(name in note for name in ("Contract.pdf", "Invoice_102.pdf", "Procurement_Policy.pdf"))
    assert note.endswith(f"Question asked: {DEMO_QUESTION}")
    assert len(note) <= 600

    pending = report.model_copy(deep=True)
    pending.human_review.status = "pending"
    assert investigation_note(pending) is None
    assert investigation_note(report.model_copy(update={"status": "not_applicable"})) is None
    assert investigation_note(report.model_copy(update={"status": "insufficient_evidence"})) is None
    plain_lookup = report.model_copy(
        update={"status": "no_issues_found", "contradictions": [], "missing_evidence": [], "calculations": []}
    )
    assert investigation_note(plain_lookup) is None


async def test_detail_level_changes_the_text_not_the_report(samples_dir):
    report = await demo_report(samples_dir)
    standard = render_report(report)

    report.presentation.detail_level = "brief"
    brief = render_report(report)
    assert len(brief) < len(standard)
    for kept in ("Investigation Status:", "Summary:", "Contradictions Detected:", "Missing Evidence:", "Recommended Action:", "Sources:"):
        assert kept in brief
    for dropped in ("Key Findings:", "Risk Reasoning (inference):", "Evidence Quality:"):
        assert dropped in standard and dropped not in brief

    report.presentation.detail_level = "detailed"
    detailed = render_report(report)
    assert "Calculations (deterministic tools):" in detailed and "Run Statistics:" in detailed
    assert "calculate_difference(value_a=58,000, value_b=40,000) = 18,000" in detailed
    assert "Calculations (deterministic tools):" not in standard


async def test_markdown_format_uses_headings_and_bold_labels(samples_dir):
    report = await demo_report(samples_dir)
    report.presentation.response_format = "markdown"
    text = render_report(report)
    assert "**Overall Risk Level:** High" in text
    assert "## Summary" in text and "## Contradictions Detected" in text
    assert "Summary:" not in text


# ================================================================== API level
async def upload(client, headers, path):
    with path.open("rb") as handle:
        response = await client.post("/documents/upload", headers=headers, files={"file": (path.name, handle)})
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
async def library(client, user_headers, samples_dir):
    for name in ("Contract.pdf", "Invoice_102.pdf", "Procurement_Policy.pdf"):
        await upload(client, user_headers, samples_dir / name)


@pytest.mark.integration
async def test_preferences_through_the_api(client, user_headers, library):
    overview = (await client.get("/memory", headers=user_headers)).json()
    assert overview["enabled"] is True and overview["notes"] == []
    assert {item["key"]: (item["value"], item["is_set"]) for item in overview["preferences"]} == {
        "detail_level": ("standard", False), "response_format": ("text", False),
    }

    saved = await client.put("/memory/preferences/detail_level", headers=user_headers, json={"value": "Brief"})
    assert saved.status_code == 200 and saved.json()["value"] == "brief" and saved.json()["is_set"] is True
    await client.put("/memory/preferences/response_format", headers=user_headers, json={"value": "markdown"})
    again = await client.put("/memory/preferences/detail_level", headers=user_headers, json={"value": "brief"})
    assert again.status_code == 200  # setting it twice updates the same row

    assert (await client.put("/memory/preferences/detail_level", headers=user_headers, json={"value": "huge"})).status_code == 400
    assert (await client.put("/memory/preferences/colour", headers=user_headers, json={"value": "blue"})).status_code == 400
    assert (await client.put("/memory/preferences/detail_level", headers=user_headers, json={})).status_code == 422

    report = (await client.post("/investigations", headers=user_headers, json={"question": DEMO_QUESTION})).json()
    assert report["presentation"] == {"detail_level": "brief", "response_format": "markdown"}
    assert "## Contradictions Detected" in report["answer_text"] and "Key Findings" not in report["answer_text"]

    # Someone else's preferences are their own.
    other = {"X-User-Id": user_headers["X-User-Id"] + "-other"}
    assert all(not item["is_set"] for item in (await client.get("/memory", headers=other)).json()["preferences"])

    assert (await client.delete("/memory/preferences/detail_level", headers=user_headers)).status_code == 204
    assert (await client.delete("/memory/preferences/colour", headers=user_headers)).status_code == 400
    values = {item["key"]: item["value"] for item in (await client.get("/memory", headers=user_headers)).json()["preferences"]}
    assert values == {"detail_level": "standard", "response_format": "markdown"}


@pytest.mark.integration
async def test_notes_through_the_api(client, user_headers):
    text = "Vendor ABC was flagged for duplicate invoices in 2025."
    created = await client.post("/memory/notes", headers=user_headers, json={"text": f"  {text} "})
    assert created.status_code == 201, created.text
    note = created.json()
    assert note["text"] == text and note["source"] == "user" and note["investigation_id"] is None

    duplicate = await client.post("/memory/notes", headers=user_headers, json={"text": text})
    assert duplicate.json()["id"] == note["id"]  # the same note is kept once
    assert (await client.post("/memory/notes", headers=user_headers, json={"text": "Was it approved or not?"})).status_code == 400
    assert (await client.post("/memory/notes", headers=user_headers, json={"text": "short"})).status_code == 422

    found = (await client.get("/memory/search", headers=user_headers, params={"q": "duplicate invoices Vendor ABC"})).json()
    assert [item["text"] for item in found["results"]] == [text]
    assert 0 < found["results"][0]["similarity"] <= 1
    nothing = (await client.get("/memory/search", headers=user_headers, params={"q": "weather in Paris"})).json()
    assert nothing["results"] == []

    other = {"X-User-Id": user_headers["X-User-Id"] + "-other"}
    theirs = (await client.get("/memory/search", headers=other, params={"q": "duplicate invoices Vendor ABC", "min_similarity": 0})).json()
    assert theirs["results"] == []
    assert (await client.delete(f"/memory/notes/{note['id']}", headers=other)).status_code == 404

    assert (await client.delete(f"/memory/notes/{note['id']}", headers=user_headers)).status_code == 204
    assert (await client.delete(f"/memory/notes/{note['id']}", headers=user_headers)).status_code == 404
    assert (await client.get("/memory", headers=user_headers)).json()["notes"] == []


@pytest.mark.integration
async def test_chat_remembers_a_preference_and_an_investigation(client, user_headers, library):
    said = (await client.post("/chat", headers=user_headers, json={"message": "From now on keep your answers brief."})).json()
    assert said["answer"] == "Saved your preference: detail level = brief."
    assert said["status"] == "not_applicable" and said["approval_status"] == "not_required"

    draft = (await client.post("/investigations", headers=user_headers, json={"question": DEMO_QUESTION})).json()
    assert draft["presentation"]["detail_level"] == "brief" and draft["human_review"]["status"] == "pending"
    assert (await client.get("/memory", headers=user_headers)).json()["notes"] == []  # not decided yet

    await client.post(
        f"/investigations/{draft['investigation_id']}/approve", headers=user_headers,
        json={"decision": "approve", "note": "Variation signed"},
    )
    notes = (await client.get("/memory", headers=user_headers)).json()["notes"]
    assert len(notes) == 1 and notes[0]["source"] == "investigation"
    assert notes[0]["investigation_id"] == draft["investigation_id"]
    assert "Human decision: approved (Variation signed)." in notes[0]["text"]

    # The next investigation of the same kind is told about the earlier one.
    second = (await client.post("/investigations", headers=user_headers, json={"question": DEMO_QUESTION})).json()
    assert [item["text"] for item in second["memory"]["recalled"]] == [notes[0]["text"]]
    assert "From Earlier Investigations (memory, not evidence):" in second["answer_text"]
    trace = (await client.get(f"/investigations/{second['investigation_id']}/trace", headers=user_headers)).json()
    recall = next(event for event in trace["events"] if event["kind"] == "memory" and event["name"] == "recall")
    assert recall["data"]["notes"] == [notes[0]["id"]]

    forgotten = await client.delete("/memory", headers=user_headers)
    assert forgotten.json() == {"preferences": 1, "notes": 1}
    overview = (await client.get("/memory", headers=user_headers)).json()
    assert overview["notes"] == [] and all(not item["is_set"] for item in overview["preferences"])


# ============================================ review findings that must stay fixed
@pytest.mark.parametrize(
    "message",
    [
        "Compare the invoice with the contract and remember that Vendor ABC was flagged for duplicate invoices in 2025.",
        "Does the report always show short payment terms?",
        "I would prefer the short answer: is the invoice above the contract value?",
        "From now on keep your answers brief. Is the invoice above the contract value?",
        "Remember that Vendor ABC was flagged in 2025. What does the contract say about payment?",
    ],
)
def test_a_message_that_also_asks_something_is_never_only_a_memory(message):
    from app.memory.extraction import is_only_memory

    assert is_only_memory(message) is False
    assert fallback_memory(message).only_memory is False


@pytest.mark.parametrize(
    "message",
    [
        "From now on keep your answers brief.",
        "Keep it brief from now on.",
        "Please remember that Vendor ABC was flagged for duplicate invoices in 2025.",
        "I prefer detailed reports. By default, use markdown.",
    ],
)
def test_a_message_that_only_states_something_to_remember_is_recognised(message):
    from app.memory.extraction import is_only_memory

    assert is_only_memory(message) is True


def test_a_question_is_never_stored_as_a_preference():
    assert fallback_memory("Does the report always show short payment terms?").should_save is False
    assert fallback_memory("I would prefer the short answer: is the invoice above the contract value?").should_save is False


async def test_a_request_with_a_fact_to_remember_is_investigated_and_remembered(samples_dir):
    memory = InMemoryMemoryStore()
    state = await run_graph(
        "Compare the invoice with the contract and remember that Vendor ABC was flagged for duplicate invoices in 2025.",
        sample_library(samples_dir), memory=memory,
    )
    assert state["report"]["status"] == "review_required" and state["report"]["contradictions"]
    assert [note["text"] for note in memory.notes] == ["Vendor ABC was flagged for duplicate invoices in 2025."]
    assert state["report"]["memory"]["saved"] == ["Noted for later: Vendor ABC was flagged for duplicate invoices in 2025."]


async def test_the_model_alone_cannot_skip_an_investigation(samples_dir):
    """The model claims the message is only a memory; the code rule disagrees, so the question is answered."""
    memory = InMemoryMemoryStore()
    llm = ScriptedLLM({MemoryCandidate: candidate(key="detail_level", value="brief", only_memory=True)})
    state = await run_graph(
        "From now on keep your answers brief. Is the invoice above the contract value?",
        sample_library(samples_dir), llm, memory=memory,
    )
    assert memory.preferences == {"user-1": {"detail_level": "brief"}}
    assert "RequestAnalysis" in llm.calls and state["report"]["status"] == "review_required"


def test_more_kinds_of_secrets_are_removed_and_json_stays_valid(settings):
    import json

    from app.security.guardrails import redact_secrets

    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    secrets = {
        "header": f"Authorization: Bearer {jwt}",
        "github": "use ghp_abcdefghijklmnopqrstuvwxyz0123456789 for the repo",
        "aws": "key id AKIAIOSFODNN7EXAMPLE",
        "password": "The portal password: hunter2secret, ask Dana",
        "assignment": "api_key=abcd1234efgh",
    }
    cleaned = redact_secrets(json.dumps(secrets), settings)
    for leaked in (jwt, "ghp_abcdefghijklmnopqrstuvwxyz0123456789", "AKIAIOSFODNN7EXAMPLE", "hunter2secret", "abcd1234efgh"):
        assert leaked not in cleaned
    parsed = json.loads(cleaned)  # the JSON around the secrets is untouched
    assert set(parsed) == set(secrets) and parsed["password"] == "The portal password: [REDACTED], ask Dana"
    # Ordinary business text is left alone.
    plain = "Payment terms: 30 days. Total amount due: £58,000. The token economy report is attached."
    assert redact_secrets(plain, settings) == plain
    assert clean_note("The vendor portal login is password: hunter2secret for Dana.", settings) == (
        "The vendor portal login is password: [REDACTED] for Dana."
    )


@pytest.mark.integration
async def test_when_memory_is_off_the_api_stores_and_searches_nothing(client, user_headers, monkeypatch):
    note = (await client.post("/memory/notes", headers=user_headers, json={"text": "Vendor ABC was flagged in 2025."})).json()
    monkeypatch.setattr(get_settings(), "ENABLE_MEMORY", False)

    for response in (
        await client.put("/memory/preferences/detail_level", headers=user_headers, json={"value": "brief"}),
        await client.post("/memory/notes", headers=user_headers, json={"text": "Another lasting fact to keep."}),
        await client.get("/memory/search", headers=user_headers, params={"q": "Vendor ABC"}),
    ):
        assert response.status_code == 409 and "switched off" in response.json()["error"]["message"]

    # What is already stored can still be seen and removed.
    overview = (await client.get("/memory", headers=user_headers)).json()
    assert overview["enabled"] is False and [item["id"] for item in overview["notes"]] == [note["id"]]
    assert (await client.delete(f"/memory/notes/{note['id']}", headers=user_headers)).status_code == 204


@pytest.mark.integration
async def test_notes_per_user_are_capped_and_the_oldest_make_room(client, user_headers, monkeypatch):
    from app.memory import store

    monkeypatch.setattr(store, "MAX_NOTES_PER_USER", 3)
    for number in range(1, 6):
        response = await client.post(
            "/memory/notes", headers=user_headers, json={"text": f"Lasting fact number {number} about the vendor."}
        )
        assert response.status_code == 201
    notes = (await client.get("/memory", headers=user_headers)).json()["notes"]
    assert [note["text"][:21] for note in notes] == ["Lasting fact number 5", "Lasting fact number 4", "Lasting fact number 3"]
