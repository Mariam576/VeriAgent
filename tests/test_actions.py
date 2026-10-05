"""Follow-up actions through MCP (Version 3, milestone 19).

Most tests talk to the built-in ticket server over real MCP (a child process
on stdin/stdout). GitHub's server is replaced by an in-process MCP server with
the same tool, so its mapping is tested without a network or an account.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import SecretStr
from sqlalchemy import select

from app.actions import client as mcp_client
from app.actions import ticket_server
from app.actions.client import McpEndpoint, ToolOutcome, call_tool, check_arguments, github_endpoint, list_tools, ticket_endpoint
from app.actions.proposals import issue_body, issue_title, propose_actions
from app.actions.runner import UNKNOWN_DETAIL, ActionRunner, ActionTarget, MemoryActionLog, configured_target
from app.agents.graph import build_investigation_graph, recursion_limit
from app.agents.nodes import AgentDeps
from app.agents.state import initial_state
from app.config import Settings, get_settings
from app.schemas.response import ContradictionOut, DocumentIntegrityOut, InvestigationReport
from tests.conftest import RUN_ID
from tests.helpers import ScriptedLLM, sample_library

DEMO = "Review this invoice against the contract and tell me whether it should be approved."


def tickets_in(path) -> list[dict]:
    return ticket_server.read_tickets(path)


# ============================================================ the ticket server
def test_ticket_tools_write_and_read_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIAGENT_TICKETS_FILE", str(tmp_path / "nested" / "tickets.jsonl"))
    first = ticket_server.create_ticket("  Invoice   exceeds contract ", "Body one", "high")
    second = ticket_server.create_ticket("Second", "Body two")
    assert first["status"] == "open" and first["id"].startswith("T-") and first["id"] != second["id"]
    stored = tickets_in(tmp_path / "nested" / "tickets.jsonl")
    assert [(t["title"], t["severity"]) for t in stored] == [("Invoice exceeds contract", "high"), ("Second", "medium")]
    assert [t["title"] for t in ticket_server.list_tickets()] == ["Second", "Invoice exceeds contract"]  # newest first
    with pytest.raises(ToolError, match="needs a title"):
        ticket_server.create_ticket("   ", "body")
    with pytest.raises(ToolError, match="too long"):
        ticket_server.create_ticket("x" * 500, "body")
    (tmp_path / "nested" / "tickets.jsonl").open("a").write("{half a line\n")
    assert len(ticket_server.read_tickets()) == 2  # a damaged line does not hide the others


# ================================================================ the MCP client
async def test_a_real_mcp_round_trip_to_the_ticket_server(tmp_path):
    endpoint = ticket_endpoint(str(tmp_path / "tickets.jsonl"))
    tools = {tool["name"]: tool for tool in await list_tools(endpoint)}
    assert set(tools) == {"create_ticket", "list_tickets"}
    assert tools["create_ticket"]["schema"]["required"] == ["title", "body"]

    outcome = await call_tool(endpoint, [("create_ticket", {"title": "Check invoice INV-102", "body": "Details", "severity": "high"})])
    assert outcome.ok and outcome.tool == "create_ticket" and outcome.data["id"].startswith("T-")
    [ticket] = tickets_in(tmp_path / "tickets.jsonl")
    assert (ticket["id"], ticket["title"], ticket["severity"]) == (outcome.data["id"], "Check invoice INV-102", "high")


async def test_only_listed_tools_with_fitting_arguments_are_called(tmp_path):
    endpoint = ticket_endpoint(str(tmp_path / "tickets.jsonl"))
    # The server has no such tool: nothing is called.
    missing = await call_tool(endpoint, [("delete_everything", {}), ("issue_write", {"title": "x"})])
    assert not missing.ok and "none of the expected tools (delete_everything, issue_write)" in missing.error
    # The arguments do not fit the tool's own schema: refused before the call.
    wrong = await call_tool(endpoint, [("create_ticket", {"title": "No body", "severity": "urgent"})])
    assert not wrong.ok and "do not fit the tool" in wrong.error
    # The tool itself reports a problem.
    refused = await call_tool(endpoint, [("create_ticket", {"title": " ", "body": "b"})])
    assert not refused.ok and "needs a title" in refused.error
    assert tickets_in(tmp_path / "tickets.jsonl") == []


async def test_a_server_that_hangs_or_does_not_exist_is_an_error_not_a_crash():
    asleep = McpEndpoint(name="asleep", transport="stdio", command=sys.executable, args=("-c", "import time; time.sleep(30)"))
    outcome = await call_tool(asleep, [("create_ticket", {})], timeout=1.5)
    assert not outcome.ok and ("did not answer within 1.5 seconds" in outcome.error or "could not be used" in outcome.error)
    absent = McpEndpoint(name="absent", transport="stdio", command="/no/such/program")
    assert not (await call_tool(absent, [("create_ticket", {})], timeout=5)).ok


def test_the_child_process_does_not_inherit_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-that-must-stay-here")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:password@host/db")
    environment = mcp_client._connection(ticket_endpoint(str(tmp_path / "t.jsonl"))).env
    assert "GEMINI_API_KEY" not in environment and "DATABASE_URL" not in environment
    assert environment["VERIAGENT_TICKETS_FILE"] == str(tmp_path / "t.jsonl") and "PYTHONPATH" in environment


def test_argument_check_and_endpoints():
    schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    assert check_arguments(schema, {"title": "ok"}) is None and check_arguments(None, {"anything": 1}) is None
    assert "title" in check_arguments(schema, {})
    endpoint = github_endpoint("https://api.githubcopilot.com/mcp/", "ghp_example")
    assert endpoint.headers == {"Authorization": "Bearer ghp_example", "X-MCP-Toolsets": "issues"}
    assert endpoint.describe() == "https://api.githubcopilot.com/mcp/"  # the token is not part of any description


# ================================================================== proposals
def report(**changes) -> InvestigationReport:
    values = dict(
        investigation_id="inv-1", question="Review the invoice.", status="review_required", status_label="Review Required",
        risk_level="high", confidence=0.9, evidence_quality="moderate", requires_human_review=True,
        summary="The invoice exceeds the contract.", recommended_action="Do not approve automatically.",
        contradictions=[ContradictionOut(
            id="X1", type="amount_mismatch", severity="high", source_a="Contract.pdf", source_b="Invoice.pdf",
            description="Contract.pdf states the total amount as £40,000 but Invoice.pdf states £58,000.",
        )],
        missing_evidence=["Written approval"],
    )
    return InvestigationReport(**{**values, **changes})


def test_an_issue_is_proposed_for_high_risk_results_only():
    [proposal] = propose_actions(report(), server="tickets", target="the local ticket list")
    assert (proposal["id"], proposal["kind"], proposal["server"], proposal["severity"]) == ("A1", "create_issue", "tickets", "high")
    assert proposal["title"] == "VeriAgent: high risk - Contract.pdf states the total amount as £40,000 but Invoice.pdf states £58,000."
    assert "### Contradictions" in proposal["body"] and "Written approval" in proposal["body"]
    assert "Created by VeriAgent after human approval" in proposal["body"]
    assert propose_actions(report(risk_level="medium"), server="tickets", target="t") == []
    assert propose_actions(report(status="not_applicable"), server="tickets", target="t") == []
    assert len(issue_title(report(summary="x" * 400, contradictions=[]))) <= 120


def test_document_text_cannot_plant_links_or_mentions_in_the_issue():
    hostile = report(
        contradictions=[ContradictionOut(
            id="X1", type="party_mismatch", severity="high", source_a="C.txt", source_b="I.txt",
            description="I.txt is issued by @octocat [click](https://evil.example) ![x](https://evil.example/p.png), not Redbridge.",
        )],
        document_integrity=[DocumentIntegrityOut(id="F1", document="I.txt", level="high", score=0.8)],
    )
    body = issue_body(hostile)
    assert "@octocat" not in body and "@​octocat" in body  # no notification to a real user
    assert "](https://evil.example" not in body and "![x]" not in body  # no working link, no image
    assert "### Document integrity (risk indicators, not proof)" in body
    assert len(issue_body(report(summary="long " * 3000))) <= 6000


# ============================================================= target and runner
def test_targets_come_from_settings():
    base = dict(_env_file=None, LLM_PROVIDER="mock")
    assert configured_target(Settings(**base)) is None  # switched off by default
    tickets = configured_target(Settings(**base, ACTION_SERVER="tickets", TICKETS_FILE="data/t.jsonl"))
    assert (tickets.server, tickets.label, tickets.endpoint.transport) == ("tickets", "the local ticket list", "stdio")
    with pytest.raises(ValueError, match="GITHUB_TOKEN and GITHUB_REPOSITORY"):
        configured_target(Settings(**base, ACTION_SERVER="github"))
    with pytest.raises(ValueError):
        configured_target(Settings(**base, ACTION_SERVER="github", GITHUB_TOKEN=SecretStr("t"), GITHUB_REPOSITORY="not a repository"))
    github = configured_target(Settings(**base, ACTION_SERVER="github", GITHUB_TOKEN=SecretStr("t"), GITHUB_REPOSITORY="Mariam576/VeriAgent"))
    assert github.label == "GitHub repository Mariam576/VeriAgent" and github.endpoint.url == "https://api.githubcopilot.com/mcp/"
    names = [name for name, _ in github.calls({"title": "T", "body": "B"})]
    assert names == ["issue_write", "create_issue"]
    assert github.calls({"title": "T", "body": "B"})[0][1] == {"method": "create", "owner": "Mariam576", "repo": "VeriAgent", "title": "T", "body": "B"}


def test_a_link_is_only_accepted_when_it_points_at_the_repository_we_wrote_to():
    target = ActionTarget("github", "GitHub repository acme/books", github_endpoint("https://x/mcp/", "t"), repository="acme/books")
    assert target.read_result(ToolOutcome(ok=True, data={"url": "https://github.com/acme/books/issues/17"})) == ("#17", "https://github.com/acme/books/issues/17")
    assert target.read_result(ToolOutcome(ok=True, text='{"id": 1, "html_url": "https://github.com/acme/books/issues/18"}')) == ("#18", "https://github.com/acme/books/issues/18")
    assert target.read_result(ToolOutcome(ok=True, data={"url": "https://github.com/evil/phish/issues/1"})) == (None, None)
    assert target.read_result(ToolOutcome(ok=True, text="see https://evil.example/login")) == (None, None)
    tickets = ActionTarget("tickets", "the local ticket list", ticket_endpoint("t.jsonl"))
    assert tickets.read_result(ToolOutcome(ok=True, data={"id": "T-1A2B3C4D"})) == ("T-1A2B3C4D", None)
    assert tickets.read_result(ToolOutcome(ok=True, data={"id": "<script>"})) == (None, None)


def fake_github() -> MCPServer:
    """An in-process MCP server with GitHub's current issue tool."""
    server = MCPServer("fake-github")
    server.created = []  # type: ignore[attr-defined]

    @server.tool()
    def issue_write(method: str, owner: str, repo: str, title: str, body: str = "") -> dict:
        """Create or update an issue."""
        if method != "create":
            raise ToolError("only create is supported here")
        server.created.append({"owner": owner, "repo": repo, "title": title, "body": body})  # type: ignore[attr-defined]
        return {"id": 99, "url": f"https://github.com/{owner}/{repo}/issues/{16 + len(server.created)}"}  # type: ignore[attr-defined]

    return server


async def test_an_issue_is_created_through_githubs_tool(monkeypatch):
    server = fake_github()
    monkeypatch.setattr(mcp_client, "_connection", lambda endpoint: server)  # in-process instead of the network
    target = ActionTarget("github", "GitHub repository acme/books", github_endpoint("https://x/mcp/", "t"), repository="acme/books")
    runner = ActionRunner(target, MemoryActionLog())
    [proposal] = runner.propose(report())

    outcome, record = await runner.execute("inv-1", "user-1", proposal)
    assert (outcome["status"], outcome["reference"], outcome["url"]) == ("done", "#17", "https://github.com/acme/books/issues/17")
    assert server.created == [{"owner": "acme", "repo": "books", "title": proposal["title"], "body": proposal["body"]}]
    assert record["tool_name"] == "mcp:github/issue_write" and record["status"] == "ok"
    assert record["arguments"]["title"] == proposal["title"] and "body" not in record["arguments"]  # the length, not the text


async def test_an_action_is_carried_out_at_most_once(monkeypatch):
    server = fake_github()
    monkeypatch.setattr(mcp_client, "_connection", lambda endpoint: server)
    target = ActionTarget("github", "GitHub repository acme/books", github_endpoint("https://x/mcp/", "t"), repository="acme/books")
    log = MemoryActionLog()
    runner = ActionRunner(target, log)
    [proposal] = runner.propose(report())

    first, _ = await runner.execute("inv-1", "user-1", proposal)
    again, record = await runner.execute("inv-1", "user-1", proposal)  # the node ran a second time after a restart
    assert again["status"] == "done" and again["url"] == first["url"] and record is None
    assert len(server.created) == 1

    # Started, but the result was never written down: nobody knows. It is not tried again.
    log.records[("inv-2", "A1")] = {"status": "started"}
    unknown, record = await runner.execute("inv-2", "user-1", proposal)
    assert unknown["status"] == "unknown" and unknown["detail"] == UNKNOWN_DETAIL and record is None
    assert len(server.created) == 1

    class BrokenLog(MemoryActionLog):
        async def start(self, investigation_id, user_id, proposal):
            raise ConnectionError("database is down")

    refused, _ = await ActionRunner(target, BrokenLog()).execute("inv-3", "user-1", proposal)
    assert refused["status"] == "failed" and "could not be recorded" in refused["detail"]
    assert len(server.created) == 1  # no record, no action


async def test_a_failed_action_is_recorded_as_failed(monkeypatch):
    server = MCPServer("empty")  # offers no tools at all
    monkeypatch.setattr(mcp_client, "_connection", lambda endpoint: server)
    log = MemoryActionLog()
    runner = ActionRunner(ActionTarget("tickets", "the local ticket list", ticket_endpoint("t.jsonl")), log)
    [proposal] = runner.propose(report())
    outcome, record = await runner.execute("inv-1", "user-1", proposal)
    assert outcome["status"] == "failed" and "none of the expected tools" in outcome["detail"]
    assert record["status"] == "error" and log.records[("inv-1", "A1")]["status"] == "failed"


async def test_no_answer_after_the_request_was_sent_is_unknown_not_failed(monkeypatch):
    """Found in review: a server that created the ticket and answered too late was reported as
    "failed ... was not completed" - an invitation to create the same issue again by hand."""
    import asyncio

    server = MCPServer("slow-github")
    created: list[str] = []

    @server.tool()
    async def issue_write(method: str, owner: str, repo: str, title: str, body: str = "") -> dict:
        """Create an issue, then take too long to say so."""
        created.append(title)
        await asyncio.sleep(2)
        return {"url": f"https://github.com/{owner}/{repo}/issues/1"}

    monkeypatch.setattr(mcp_client, "_connection", lambda endpoint: server)
    target = ActionTarget("github", "GitHub repository acme/books", github_endpoint("https://x/mcp/", "t"), repository="acme/books")
    log = MemoryActionLog()
    runner = ActionRunner(target, log, timeout=0.3)
    [proposal] = runner.propose(report())

    outcome, record = await runner.execute("inv-1", "user-1", proposal)
    assert created == [proposal["title"]]  # it did happen
    assert outcome["status"] == "unknown" and outcome["reference"] is None
    assert "did not answer within 0.3 seconds" in outcome["detail"] and "may have been carried out" in outcome["detail"]
    assert record["status"] == "error" and record["tool_name"] == "mcp:github/issue_write"
    assert log.records[("inv-1", "A1")]["status"] == "unknown"

    again, _ = await runner.execute("inv-1", "user-1", proposal)  # the node runs again after a restart
    assert again["status"] == "unknown" and created == [proposal["title"]]  # still once


async def test_done_is_only_said_about_something_that_can_be_pointed_at(monkeypatch):
    """Found in review: an answer without an error flag and without an id was recorded as "done"."""
    server = MCPServer("vague-github")

    @server.tool()
    def issue_write(method: str, owner: str, repo: str, title: str, body: str = "") -> dict:
        """Answers politely and creates nothing that it names."""
        return {"message": "Your request has been received."}

    monkeypatch.setattr(mcp_client, "_connection", lambda endpoint: server)
    target = ActionTarget("github", "GitHub repository acme/books", github_endpoint("https://x/mcp/", "t"), repository="acme/books")
    log = MemoryActionLog()
    runner = ActionRunner(target, log)
    [proposal] = runner.propose(report())
    outcome, record = await runner.execute("inv-1", "user-1", proposal)
    assert outcome["status"] == "unknown" and "without naming what it created" in outcome["detail"]
    assert record["status"] == "error" and record["result"] is None
    assert log.records[("inv-1", "A1")]["status"] == "unknown"


async def test_a_failure_before_anything_was_sent_stays_a_plain_failure(tmp_path):
    endpoint = McpEndpoint(name="nowhere", transport="stdio", command=sys.executable, args=("-c", "import sys; sys.exit(3)"))
    outcome = await call_tool(endpoint, [("create_ticket", {"title": "x", "body": "y"})], timeout=10)
    assert not outcome.ok and not outcome.uncertain  # the tool call never left, so nothing can have happened
    wrong = await call_tool(ticket_endpoint(str(tmp_path / "t.jsonl")), [("create_ticket", {"title": "No body"})])
    assert not wrong.ok and not wrong.uncertain


# =============================================================== in the graph
def build(samples_dir, tmp_path, *, pause: bool = True, files: tuple[str, ...] = ()):
    settings = get_settings()
    runner = ActionRunner(
        ActionTarget("tickets", "the local ticket list", ticket_endpoint(str(tmp_path / "tickets.jsonl"))), MemoryActionLog()
    )
    graph = build_investigation_graph(
        AgentDeps(llm=ScriptedLLM(), retriever=sample_library(samples_dir, *files), settings=settings, pause_for_approval=pause, actions=runner),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "inv-1"}, "recursion_limit": recursion_limit(settings.MAX_AGENT_ITERATIONS)}
    return graph, config, tmp_path / "tickets.jsonl"


async def pending(graph, config) -> dict:
    snapshot = await graph.aget_state(config)
    return [interrupt.value for task in snapshot.tasks for interrupt in task.interrupts][0]


async def test_the_issue_is_proposed_with_the_draft_and_created_only_after_approval(samples_dir, tmp_path):
    graph, config, tickets = build(samples_dir, tmp_path)
    await graph.ainvoke(initial_state(user_id="user-1", investigation_id="inv-1", question=DEMO), config)

    request = await pending(graph, config)
    [proposed] = request["draft_report"]["actions"]
    assert (proposed["id"], proposed["status"], proposed["target"]) == ("A1", "proposed", "the local ticket list")
    assert proposed["title"].startswith("VeriAgent: high risk - ") and "£58,000" in proposed["body"]
    assert "proposed, waiting for approval" in request["draft_report"]["answer_text"]
    assert tickets_in(tickets) == []  # nothing has left VeriAgent yet

    final = await graph.ainvoke(Command(resume={"decision": "approve", "approved_actions": ["A1"], "reviewer": "Mariam"}), config)
    [done] = final["report"]["actions"]
    assert done["status"] == "done" and done["reference"].startswith("T-") and done["body"] == proposed["body"]
    [ticket] = tickets_in(tickets)
    assert (ticket["id"], ticket["title"], ticket["body"], ticket["severity"]) == (done["reference"], proposed["title"], proposed["body"], "high")
    assert f"created ({done['reference']})" in final["report"]["answer_text"]
    call = next(call for call in final["tool_calls"] if call["tool_name"] == "mcp:tickets/create_ticket")
    assert call["status"] == "ok" and call["result"]["reference"] == done["reference"]
    assert final["report"]["stats"]["tool_calls"] == len(final["tool_calls"])


@pytest.mark.parametrize(
    "decision",
    [
        {"decision": "approve"},  # approved the recommendation, did not tick the action
        {"decision": "reject", "approved_actions": ["A1"]},  # rejecting rejects what was proposed with it
        {"decision": "modify", "modified_action": "Return the invoice to the supplier."},
    ],
)
async def test_nothing_is_created_unless_the_action_itself_was_approved(samples_dir, tmp_path, decision):
    graph, config, tickets = build(samples_dir, tmp_path)
    await graph.ainvoke(initial_state(user_id="user-1", investigation_id="inv-1", question=DEMO), config)
    final = await graph.ainvoke(Command(resume=decision), config)
    assert [action["status"] for action in final["report"]["actions"]] == ["not_approved"]
    assert "not approved, nothing was created" in final["report"]["answer_text"]
    assert tickets_in(tickets) == [] and all(not call["tool_name"].startswith("mcp:") for call in final["tool_calls"])


async def test_an_action_that_was_not_proposed_cannot_be_approved(samples_dir, tmp_path):
    graph, config, tickets = build(samples_dir, tmp_path)
    await graph.ainvoke(initial_state(user_id="user-1", investigation_id="inv-1", question=DEMO), config)
    await graph.ainvoke(Command(resume={"decision": "approve", "approved_actions": ["A7"]}), config)
    request = await pending(graph, config)  # still waiting, with the reason
    assert "Unknown action id: A7" in request["error"] and tickets_in(tickets) == []
    await graph.ainvoke(Command(resume={"decision": "approve", "approved_actions": "A1"}), config)
    assert "must be a list" in (await pending(graph, config))["error"]


async def test_no_action_without_a_pause_or_without_high_risk(samples_dir, tmp_path):
    # Pausing is off: there is no moment at which a person could approve, so nothing is proposed.
    graph, config, tickets = build(samples_dir, tmp_path, pause=False)
    final = await graph.ainvoke(initial_state(user_id="user-1", investigation_id="inv-1", question=DEMO), config)
    assert final["report"]["actions"] == [] and final["report"]["risk_level"] == "high" and tickets_in(tickets) == []

    # A plain look-up in one document: low risk, nothing to follow up.
    graph, config, tickets = build(samples_dir, tmp_path, files=("Contract.pdf",))
    final = await graph.ainvoke(initial_state(user_id="user-1", investigation_id="inv-1", question="What payment terms does the contract set?"), config)
    assert final["report"]["actions"] == [] and tickets_in(tickets) == []


# ====================================================== API and PostgreSQL
@pytest.fixture
async def ticket_actions(tmp_path, monkeypatch, database):
    """Switch the action server on for one test (the graph is rebuilt with it, and again without it afterwards)."""
    from app.agents.service import reset_graph

    settings = get_settings()
    monkeypatch.setattr(settings, "ACTION_SERVER", "tickets")
    monkeypatch.setattr(settings, "TICKETS_FILE", str(tmp_path / "tickets.jsonl"))
    reset_graph()
    yield tmp_path / "tickets.jsonl"
    reset_graph()


@pytest.mark.integration
async def test_action_configuration_endpoint(client, user_headers, monkeypatch):
    assert (await client.get("/actions/config", headers=user_headers)).json() == {"enabled": False, "server": None, "target": None, "problem": None}
    settings = get_settings()
    monkeypatch.setattr(settings, "ACTION_SERVER", "github")
    body = (await client.get("/actions/config", headers=user_headers)).json()
    assert body["enabled"] is False and "GITHUB_TOKEN" in body["problem"]
    monkeypatch.setattr(settings, "GITHUB_TOKEN", SecretStr("ghp_not_a_real_token_0000000000"))
    monkeypatch.setattr(settings, "GITHUB_REPOSITORY", "Mariam576/VeriAgent")
    body = (await client.get("/actions/config", headers=user_headers)).json()
    assert body == {"enabled": True, "server": "github", "target": "GitHub repository Mariam576/VeriAgent", "problem": None}
    assert "ghp_" not in json.dumps(body)


@pytest.mark.integration
async def test_approved_action_through_the_api(client, user_headers, samples_dir, ticket_actions):
    from app.database.models import ExternalAction, ToolCall
    from app.database.session import get_session_factory

    for name in ("Contract.pdf", "Invoice_102.pdf"):
        with (samples_dir / name).open("rb") as handle:
            assert (await client.post("/documents/upload", headers=user_headers, files={"file": (name, handle)})).status_code == 201
    assert (await client.get("/actions/config", headers=user_headers)).json()["target"] == "the local ticket list"

    draft = (await client.post("/investigations", headers=user_headers, json={"question": DEMO})).json()
    investigation_id = draft["investigation_id"]
    assert draft["human_review"]["status"] == "pending" and [a["status"] for a in draft["actions"]] == ["proposed"]
    assert tickets_in(ticket_actions) == []
    assert (await client.get("/actions", headers=user_headers)).json() == {"actions": []}

    refused = await client.post(
        f"/investigations/{investigation_id}/approve", headers=user_headers, json={"decision": "approve", "approved_actions": ["A9"]},
    )
    assert refused.status_code == 400 and tickets_in(ticket_actions) == []

    response = await client.post(
        f"/investigations/{investigation_id}/approve", headers=user_headers,
        json={"decision": "approve", "note": "Please follow up.", "approved_actions": ["A1"]},
    )
    assert response.status_code == 200, response.text
    report = response.json()
    [action] = report["actions"]
    assert action["status"] == "done" and action["reference"].startswith("T-")
    [ticket] = tickets_in(ticket_actions)
    assert ticket["id"] == action["reference"] and "£58,000" in ticket["body"]

    listed = (await client.get("/actions", headers=user_headers)).json()["actions"]
    assert [(a["investigation_id"], a["status"], a["reference"], a["server"]) for a in listed] == [(investigation_id, "done", action["reference"], "tickets")]
    stranger = {"X-User-Id": f"test-{RUN_ID}-{uuid.uuid4().hex[:8]}"}
    assert (await client.get("/actions", headers=stranger)).json() == {"actions": []}

    async with get_session_factory()() as session:
        calls = (await session.execute(select(ToolCall).where(ToolCall.investigation_id == uuid.UUID(investigation_id)))).scalars().all()
        assert any(call.tool_name == "mcp:tickets/create_ticket" and call.status == "ok" for call in calls)
        rows = (await session.execute(select(ExternalAction).where(ExternalAction.investigation_id == uuid.UUID(investigation_id)))).scalars().all()
        assert len(rows) == 1 and rows[0].finished_at is not None

    trace = (await client.get(f"/investigations/{investigation_id}/trace", headers=user_headers)).json()
    assert any(event["kind"] == "action" and event["name"] == "create_issue" for event in trace["events"])
    stored = (await client.get(f"/investigations/{investigation_id}", headers=user_headers)).json()
    assert stored["report"]["actions"][0]["status"] == "done"


@pytest.mark.integration
async def test_the_database_lets_an_action_start_only_once(client, user_headers, samples_dir):
    """Two attempts to start the same action: the unique (investigation, action) pair lets one through."""
    import asyncio

    from app.actions.runner import PgActionLog
    from app.database.models import Investigation, User
    from app.database.session import get_session_factory

    await client.get("/documents", headers=user_headers)  # creates the user
    async with get_session_factory()() as session:
        user = (await session.execute(select(User).where(User.name == user_headers["X-User-Id"]))).scalar_one()
        investigation = Investigation(user_id=user.id, question="q", status="running")
        session.add(investigation)
        await session.commit()
        investigation_id, user_id = str(investigation.id), str(user.id)

    log = PgActionLog(get_session_factory())
    proposal = {"id": "A1", "kind": "create_issue", "server": "tickets", "target": "the local ticket list", "title": "T", "body": "B"}
    first, second = await asyncio.gather(log.start(investigation_id, user_id, proposal), log.start(investigation_id, user_id, proposal))
    assert sorted([first is None, second is None]) == [False, True]  # exactly one of them may go ahead
    assert (first or second)["status"] == "started"

    await log.finish(investigation_id, "A1", {"status": "done", "reference": "T-1A2B3C4D", "url": None, "detail": None})
    assert await log.start(investigation_id, user_id, proposal) == {"status": "done", "reference": "T-1A2B3C4D", "url": None, "detail": None}
