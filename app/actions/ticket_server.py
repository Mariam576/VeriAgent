"""A minimal MCP server: a ticket list kept in a file.

    python -m app.actions.ticket_server

It exists so the "action after approval" feature can be tried without a GitHub
account. It speaks MCP over stdin/stdout ("stdio" transport): VeriAgent starts
it as a child process for one call and it ends when the call is done.

Concept - an MCP tool:
    A normal Python function with type hints and a docstring. The decorator
    publishes it: its name, its description and the shape of its arguments
    are what an MCP client sees when it asks "which tools do you have?".

Tickets are appended to a JSON-lines file (one ticket per line). The location
comes from VERIAGENT_TICKETS_FILE.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

DEFAULT_TICKETS_FILE = "data/tickets.jsonl"
MAX_TITLE_CHARS = 200
MAX_BODY_CHARS = 20_000

server = MCPServer(
    "veriagent-tickets",
    instructions="A small ticket list. create_ticket adds a ticket; list_tickets shows the newest ones.",
)


def tickets_file() -> Path:
    return Path(os.environ.get("VERIAGENT_TICKETS_FILE") or DEFAULT_TICKETS_FILE)


def read_tickets(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or tickets_file()
    if not path.exists():
        return []
    tickets = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            tickets.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written line must not make the whole list unreadable
    return tickets


@server.tool()
def create_ticket(title: str, body: str, severity: Literal["low", "medium", "high", "critical"] = "medium") -> dict[str, str]:
    """Create a ticket for something a person has to follow up. Returns the new ticket's id."""
    title, body = " ".join(title.split()), body.strip()
    # ToolError = a failure we expected: the client gets this message instead of "the tool crashed".
    if not title:
        raise ToolError("A ticket needs a title.")
    if len(title) > MAX_TITLE_CHARS or len(body) > MAX_BODY_CHARS:
        raise ToolError("The title or body is too long.")
    ticket = {
        "id": f"T-{uuid.uuid4().hex[:8].upper()}",
        "title": title,
        "body": body,
        "severity": severity,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = tickets_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ticket, ensure_ascii=False) + "\n")
    return {"id": ticket["id"], "status": "open"}


@server.tool()
def list_tickets(limit: int = 20) -> list[dict[str, str]]:
    """The newest tickets, newest first (id, title, severity, status, created_at)."""
    newest = read_tickets()[-max(1, min(limit, 100)):][::-1]
    return [{key: ticket.get(key, "") for key in ("id", "title", "severity", "status", "created_at")} for ticket in newest]


if __name__ == "__main__":
    server.run("stdio")
