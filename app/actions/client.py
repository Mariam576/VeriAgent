"""The MCP client side: connect to a server, find the tool, call it.

Every call follows the same careful steps:

    1. connect and ask the server which tools it has       (discovery)
    2. pick the first tool from a short list of names we allow (allowlist)
    3. check our arguments against the tool's own schema    (validation)
    4. call it, with a time limit                           (bounded)
    5. hand back the result as plain data                   (untrusted)

The server is another program. What it returns is treated like document text:
it is shown to people (escaped) and never interpreted as instructions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 4000


@dataclass(frozen=True)
class McpEndpoint:
    """How to reach one MCP server.

    ``stdio``: start ``command`` as a child process and talk over its stdin / stdout.
    ``http``:  talk to ``url`` ("streamable HTTP"), sending ``headers`` (for example a token).
    """

    name: str
    transport: str  # "stdio" | "http"
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        return self.url if self.transport == "http" else " ".join((self.command, *self.args))


@dataclass
class ToolOutcome:
    ok: bool
    tool: str = ""  # the tool that was called
    text: str = ""  # what the server answered, as text (untrusted)
    data: dict[str, Any] | None = None  # the structured answer, when the server gave one
    error: str = ""


class McpError(Exception):
    """The server could not be reached, has no suitable tool, or refused the arguments."""


def _connection(endpoint: McpEndpoint) -> Any:
    if endpoint.transport == "stdio":
        from mcp import StdioServerParameters
        from mcp.client.stdio import get_default_environment

        # The child process gets a minimal environment plus what we pass on purpose.
        # It never inherits API keys or the database password.
        return StdioServerParameters(
            command=endpoint.command or sys.executable, args=list(endpoint.args),
            env={**get_default_environment(), **endpoint.env},
        )
    if endpoint.transport == "http":
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        return streamable_http_client(endpoint.url, http_client=create_mcp_http_client(headers=dict(endpoint.headers)))
    raise McpError(f"Unknown MCP transport: {endpoint.transport!r}")


def check_arguments(schema: dict[str, Any] | None, arguments: dict[str, Any]) -> str | None:
    """Do the arguments fit the tool's schema? Returns what is wrong, or None."""
    if not schema:
        return None
    try:
        import jsonschema

        jsonschema.validate(arguments, schema)
    except ImportError:  # the check is an extra safeguard; the server validates as well
        return None
    except Exception as exc:  # noqa: BLE001 - jsonschema.ValidationError and schema errors alike
        return str(getattr(exc, "message", exc))[:300]
    return None


def _as_text(result: Any) -> str:
    parts = [getattr(block, "text", "") for block in getattr(result, "content", []) or []]
    return "\n".join(part for part in parts if part)[:MAX_RESULT_CHARS]


async def list_tools(endpoint: McpEndpoint, *, timeout: float = 30.0) -> list[dict[str, Any]]:
    """The tools a server offers: name, description and argument schema."""
    from mcp import Client

    try:
        async with asyncio.timeout(timeout):
            async with Client(_connection(endpoint), read_timeout_seconds=timeout) as client:
                listed = await client.list_tools()
    except Exception as exc:  # noqa: BLE001 - any failure to connect is the same to the caller
        raise McpError(f"The MCP server '{endpoint.name}' could not be reached: {type(exc).__name__}") from exc
    return [{"name": tool.name, "description": tool.description or "", "schema": tool.input_schema} for tool in listed.tools]


async def call_tool(
    endpoint: McpEndpoint,
    candidates: list[tuple[str, dict[str, Any]]],
    *,
    timeout: float = 30.0,
) -> ToolOutcome:
    """Call the first of ``candidates`` (tool name, arguments) that the server offers.

    Several candidates exist because servers rename tools between versions
    (GitHub's "create_issue" became "issue_write"). Only names listed here can
    ever be called, whatever else the server offers.
    """
    from mcp import Client

    try:
        async with asyncio.timeout(timeout):
            async with Client(_connection(endpoint), read_timeout_seconds=timeout) as client:
                offered = {tool.name: tool for tool in (await client.list_tools()).tools}
                chosen = next(((name, arguments) for name, arguments in candidates if name in offered), None)
                if chosen is None:
                    wanted = ", ".join(name for name, _ in candidates)
                    return ToolOutcome(ok=False, error=f"The server offers none of the expected tools ({wanted}).")
                name, arguments = chosen
                problem = check_arguments(offered[name].input_schema, arguments)
                if problem:
                    return ToolOutcome(ok=False, tool=name, error=f"The arguments do not fit the tool: {problem}")
                result = await client.call_tool(name, arguments)
    except TimeoutError:
        return ToolOutcome(ok=False, error=f"The MCP server '{endpoint.name}' did not answer within {timeout:g} seconds.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("mcp_call_failed server=%s error=%s", endpoint.name, type(exc).__name__)
        return ToolOutcome(ok=False, error=f"The MCP server '{endpoint.name}' could not be used: {type(exc).__name__}")

    text = _as_text(result)
    data = result.structured_content if isinstance(getattr(result, "structured_content", None), dict) else None
    if data is None and text:
        try:
            parsed = json.loads(text)
            data = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            data = None
    if getattr(result, "is_error", False):
        return ToolOutcome(ok=False, tool=name, text=text, data=data, error=text[:300] or "The tool reported an error.")
    return ToolOutcome(ok=True, tool=name, text=text, data=data)


def ticket_endpoint(tickets_file: str) -> McpEndpoint:
    """The built-in ticket server, started as a child process for each call."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return McpEndpoint(
        name="tickets", transport="stdio", command=sys.executable, args=("-m", "app.actions.ticket_server"),
        env={"VERIAGENT_TICKETS_FILE": os.path.abspath(tickets_file), "PYTHONPATH": root},
    )


def github_endpoint(url: str, token: str) -> McpEndpoint:
    """GitHub's hosted MCP server. The token is a personal access token with permission to write issues."""
    return McpEndpoint(
        name="github", transport="http", url=url,
        # Ask for the issue tools only: the token can do no more than this client could ever call.
        headers={"Authorization": f"Bearer {token}", "X-MCP-Toolsets": "issues"},
    )
