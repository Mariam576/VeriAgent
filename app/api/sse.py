"""Server-Sent Events: send progress to the browser while an investigation runs.

Concept - streaming with SSE:
    A normal HTTP response arrives in one piece when the work is finished. An
    investigation takes many seconds, so instead the server keeps the response
    open and writes small messages into it as things happen:

        event: progress
        data: {"node": "retrieve", "message": "Searching documents..."}

        event: report
        data: {...the full report...}

    Each message is "event:" + "data:" + a blank line. Browsers read this
    format natively (EventSource / fetch), and it is plain HTTP, so no
    WebSocket is needed for one-way updates.

What is streamed: fixed progress sentences, tool names with their status, and
the final report. Never the model's reasoning, prompts or document text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.errors import safe_error
from app.security.guardrails import redact_secrets

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 15.0
_DONE = object()


def error_payload(exc: BaseException) -> tuple[int, dict[str, str]]:
    """HTTP status and safe {"code", "message"} for any error. Internal details never leave the server."""
    return safe_error(exc)


def format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def sse_body(events: AsyncIterator[tuple[str, dict[str, Any]]]) -> AsyncIterator[str]:
    """Turn (event name, data) pairs into SSE text, with heartbeats and safe error reporting.

    The events are produced by a separate task and handed over through a
    queue. That lets this function wake up every few seconds to send a
    heartbeat comment while a slow step (an LLM call) produces nothing, which
    keeps proxies from closing the idle connection.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def produce() -> None:
        try:
            async for item in events:
                await queue.put(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the client as an "error" event
            status, payload = error_payload(exc)
            if status >= 500 and payload["code"] == "internal_error":
                details = "".join(traceback.format_exception(exc))
                logger.error("Unhandled error while streaming:\n%s", redact_secrets(details, get_settings()))
            await queue.put(("error", payload))
        finally:
            await queue.put(_DONE)

    producer = asyncio.create_task(produce())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"  # a comment line; clients ignore it
                continue
            if item is _DONE:
                break
            name, data = item
            yield format_sse(name, data)
    finally:
        # The client went away (or we are done): stop the work that feeds the stream.
        if not producer.done():
            producer.cancel()
        try:
            await producer
        except BaseException:  # noqa: BLE001
            pass


def sse_response(events: AsyncIterator[tuple[str, dict[str, Any]]]) -> StreamingResponse:
    return StreamingResponse(
        sse_body(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # ask proxies not to buffer
    )
