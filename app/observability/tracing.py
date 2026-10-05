"""Vendor-neutral tracing of one investigation.

Concept - observability:
    When an agent gives a strange answer, "what did it actually do?" must be
    answerable after the fact. A trace is the ordered list of everything that
    happened: each node entered and left, each LLM call with its latency and
    token usage, each tool call with arguments and result, each search with
    the chunk ids it returned, and each error.

How it works
    ``start_trace()`` creates a ``RunTrace`` and puts it in a context variable.
    Any code that runs during the investigation can then call ``record(...)``
    without being handed the trace explicitly. When the investigation ends,
    the service stores the events in the ``trace_events`` table, and
    ``GET /investigations/{id}/trace`` returns them.

    Nothing here knows about a tracing vendor. To also send traces to
    LangSmith, set the LANGSMITH_* variables in ``.env``; LangChain picks them
    up by itself and the application code stays unchanged.

What is never recorded: prompts, document text, API keys.
"""

from __future__ import annotations

import math
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

MAX_EVENTS = 2000  # a runaway loop must not be able to fill the database


def json_safe(value: Any) -> Any:
    """Make a value storable as JSON in PostgreSQL (which rejects NaN and Infinity)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


@dataclass
class TraceEvent:
    seq: int
    kind: str  # node | llm | tool | retrieval | memory | error
    name: str
    latency_ms: float | None
    data: dict[str, Any]
    offset_ms: float  # time since the trace started


@dataclass
class RunTrace:
    events: list[TraceEvent] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)
    # Token usage reported by the provider for the call that is in flight.
    pending_usage: dict[str, int] | None = None

    def record(self, kind: str, name: str, latency_ms: float | None = None, **data: Any) -> None:
        if len(self.events) >= MAX_EVENTS:
            return
        self.events.append(
            TraceEvent(
                seq=len(self.events) + 1,
                kind=kind,
                name=name,
                latency_ms=None if latency_ms is None else round(latency_ms, 1),
                data=json_safe(data),
                offset_ms=round((time.perf_counter() - self.started) * 1000, 1),
            )
        )

    def take_usage(self) -> dict[str, int]:
        usage, self.pending_usage = self.pending_usage or {}, None
        return usage

    def totals(self) -> dict[str, Any]:
        llm = [event for event in self.events if event.kind == "llm"]
        return {
            "events": len(self.events),
            "llm_calls": sum(1 for event in llm if event.data.get("status") == "ok"),
            "llm_failures": sum(1 for event in llm if event.data.get("status") != "ok"),
            "llm_latency_ms": round(sum(event.latency_ms or 0 for event in llm), 1),
            "input_tokens": sum(int(event.data.get("input_tokens") or 0) for event in llm),
            "output_tokens": sum(int(event.data.get("output_tokens") or 0) for event in llm),
            "tool_calls": sum(1 for event in self.events if event.kind == "tool"),
            "errors": sum(1 for event in self.events if event.kind == "error"),
        }


_current: ContextVar[RunTrace | None] = ContextVar("veriagent_trace", default=None)


def start_trace() -> RunTrace:
    """Begin a new trace for the current task and everything it awaits or spawns."""
    trace = RunTrace()
    _current.set(trace)
    return trace


def current_trace() -> RunTrace | None:
    return _current.get()


def record(kind: str, name: str, latency_ms: float | None = None, **data: Any) -> None:
    """Add an event to the active trace. Does nothing when no trace is active."""
    trace = _current.get()
    if trace is not None:
        trace.record(kind, name, latency_ms, **data)


def report_usage(input_tokens: int | None, output_tokens: int | None) -> None:
    """Called by an LLM provider with the token counts of the call it just made."""
    trace = _current.get()
    if trace is not None and (input_tokens or output_tokens):
        trace.pending_usage = {"input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0)}
