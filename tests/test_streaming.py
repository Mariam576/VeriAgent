"""Streaming: progress events while an investigation runs (Server-Sent Events)."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.api import sse
from app.errors import NotFound
from app.llm.base import LLMConfigurationError

DEMO_QUESTION = "Review this invoice against the contract and tell me whether it should be approved."


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """SSE text -> [(event name, data)]. Comment lines (heartbeats) are skipped."""
    events = []
    for block in text.split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if name is not None:
            events.append((name, data))
    return events


async def collect(body) -> str:
    return "".join([chunk async for chunk in body])


# ================================================================== SSE helper
async def test_sse_body_formats_events_and_ends():
    async def events():
        yield "progress", {"message": "Searching documents..."}
        yield "report", {"status": "ok", "note": "£18,000"}

    text = await collect(sse.sse_body(events()))
    assert text == (
        'event: progress\ndata: {"message": "Searching documents..."}\n\n'
        'event: report\ndata: {"status": "ok", "note": "£18,000"}\n\n'
    )


async def test_sse_body_reports_errors_as_a_safe_event():
    async def not_found():
        yield "progress", {"message": "working"}
        raise NotFound("Investigation not found.")

    async def leaky():
        raise RuntimeError("password=hunter2 at 10.0.0.5")
        yield  # pragma: no cover

    async def bad_key():
        raise LLMConfigurationError("GEMINI_API_KEY is not set.")
        yield  # pragma: no cover

    assert parse_sse(await collect(sse.sse_body(not_found()))) == [
        ("progress", {"message": "working"}),
        ("error", {"code": "not_found", "message": "Investigation not found."}),
    ]
    # An unexpected error tells the client nothing internal.
    assert parse_sse(await collect(sse.sse_body(leaky()))) == [
        ("error", {"code": "internal_error", "message": "Something went wrong on the server."})
    ]
    assert parse_sse(await collect(sse.sse_body(bad_key())))[0][1]["code"] == "llm_not_configured"


async def test_sse_body_sends_heartbeats_while_nothing_happens(monkeypatch):
    monkeypatch.setattr(sse, "HEARTBEAT_SECONDS", 0.02)

    async def slow():
        await asyncio.sleep(0.09)  # e.g. a slow LLM call
        yield "report", {"done": True}

    text = await collect(sse.sse_body(slow()))
    assert text.count(": keep-alive\n\n") >= 2
    assert parse_sse(text) == [("report", {"done": True})]


async def test_closing_the_stream_stops_the_work_behind_it():
    """When the browser disconnects, the investigation task is cancelled and cleans up."""
    state = {"cleaned_up": False, "finished": False}

    async def endless():
        try:
            yield "progress", {"n": 1}
            await asyncio.sleep(60)
            state["finished"] = True
        finally:
            state["cleaned_up"] = True

    body = sse.sse_body(endless())
    assert parse_sse(await body.__anext__()) == [("progress", {"n": 1})]
    await body.aclose()  # what Starlette does when the client goes away
    assert state == {"cleaned_up": True, "finished": False}


# ================================================================== API level
async def upload(client, headers, path):
    with path.open("rb") as handle:
        response = await client.post("/documents/upload", headers=headers, files={"file": (path.name, handle)})
    assert response.status_code == 201, response.text


@pytest.fixture
async def library(client, user_headers, samples_dir):
    for name in ("Contract.pdf", "Invoice_102.pdf", "Procurement_Policy.pdf"):
        await upload(client, user_headers, samples_dir / name)


@pytest.mark.integration
async def test_investigation_stream_reports_progress_and_ends_with_the_report(client, user_headers, library):
    response = await client.post("/investigations/stream", headers=user_headers, json={"question": DEMO_QUESTION})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = parse_sse(response.text)
    names = [name for name, _ in events]

    assert names[0] == "started" and names[-1] == "report"
    assert set(names) == {"started", "progress", "tool", "report"}
    investigation_id = events[0][1]["investigation_id"]

    messages = [data["message"] for name, data in events if name == "progress"]
    for expected in (
        "Analyzing request...", "Searching documents...", "Checking contradictions...",
        "Calculating differences...", "Reviewing findings...", "Waiting for a human decision...",
    ):
        assert expected in messages
    assert any(message.endswith("relevant evidence item(s) found...") for message in messages)
    assert messages.index("Analyzing request...") < messages.index("Searching documents...") < messages.index("Assessing risk...")
    assert not any(a == b for a, b in zip(messages, messages[1:]))  # a stage is not announced twice in a row

    tools = [data for name, data in events if name == "tool"]
    assert {"calculate_difference", "calculate_percentage_change"} <= {tool["tool_name"] for tool in tools}
    assert all(set(tool) == {"tool_name", "status"} for tool in tools)  # status only, no arguments

    report = events[-1][1]
    assert report["investigation_id"] == investigation_id
    assert report["risk_level"] == "high" and report["human_review"]["status"] == "pending"

    # Nothing but safe, fixed sentences: no prompt text, no document text.
    progress_text = " ".join(messages)
    assert "UNTRUSTED" not in progress_text and "£" not in progress_text and "Blue Ridge" not in progress_text

    # The streamed investigation is stored like any other, with its trace.
    detail = (await client.get(f"/investigations/{investigation_id}", headers=user_headers)).json()
    assert detail["approval_status"] == "pending" and detail["report"]["answer_text"] == report["answer_text"]
    trace = (await client.get(f"/investigations/{investigation_id}/trace", headers=user_headers)).json()
    assert trace["totals"]["llm_calls"] == report["stats"]["llm_calls"] > 0


@pytest.mark.integration
async def test_invalid_stream_requests_get_a_normal_error_response(client, user_headers):
    blank = await client.post("/investigations/stream", headers=user_headers, json={"question": "   "})
    assert blank.status_code == 400 and blank.json()["error"]["code"] == "validation_failed"
    assert (await client.post("/investigations/stream", headers=user_headers, json={})).status_code == 422
    unknown = await client.post(
        "/chat/stream", headers=user_headers,
        json={"conversation_id": "00000000-0000-0000-0000-000000000000", "message": "hello"},
    )
    assert unknown.status_code == 404


@pytest.mark.integration
async def test_a_failure_during_the_stream_becomes_an_error_event(client, user_headers, library, monkeypatch):
    from app.agents import service

    async def broken_graph():
        raise LLMConfigurationError("GEMINI_API_KEY is not set. Add it to your .env file.")

    monkeypatch.setattr(service, "get_graph", broken_graph)
    response = await client.post("/investigations/stream", headers=user_headers, json={"question": DEMO_QUESTION})
    assert response.status_code == 200  # the stream had already started
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["started", "error"]
    assert events[1][1]["code"] == "llm_not_configured" and "GEMINI_API_KEY" in events[1][1]["message"]
    listing = (await client.get("/investigations", headers=user_headers)).json()["investigations"]
    assert listing[0]["id"] == events[0][1]["investigation_id"] and listing[0]["status"] == "failed"


@pytest.mark.integration
async def test_chat_stream_ends_with_the_same_answer_as_chat(client, user_headers, library):
    response = await client.post(
        "/chat/stream", headers=user_headers, json={"message": "Compare the invoice amount with the contract."}
    )
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "started" and names[-2:] == ["report", "answer"]
    started, answer = events[0][1], events[-1][1]
    assert answer["conversation_id"] == started["conversation_id"]
    assert answer["investigation_id"] == started["investigation_id"]
    assert "£40,000" in answer["answer"] and "£58,000" in answer["answer"]
    assert answer["risk_level"] == "high" and answer["approval_status"] == "pending"

    # The streamed turn is part of the conversation: a follow-up understands it.
    follow_up = await client.post(
        "/chat", headers=user_headers,
        json={"conversation_id": started["conversation_id"], "message": "What about the payment terms?"},
    )
    assert "30 days" in follow_up.json()["answer"] and "60 days" in follow_up.json()["answer"]
