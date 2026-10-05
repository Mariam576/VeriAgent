"""Observability: what is recorded during an investigation, and what is not."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.llm.base import LLMPing, LLMUnavailableError
from app.llm.gemini_provider import create_gemini_llm
from app.observability import tracing
from app.observability.tracing import json_safe, start_trace
from tests.helpers import DownLLM, run_graph, sample_library

DEMO_QUESTION = "Review this invoice against the contract and tell me whether it should be approved."


async def test_trace_records_nodes_llm_calls_tools_and_searches(samples_dir):
    trace = start_trace()
    await run_graph(DEMO_QUESTION, sample_library(samples_dir))

    kinds = {event.kind for event in trace.events}
    assert {"node", "llm", "tool", "retrieval"} <= kinds
    assert [event.seq for event in trace.events] == list(range(1, len(trace.events) + 1))

    nodes = [event.name for event in trace.events if event.kind == "node"]
    assert nodes[0] == "analyze_request" and nodes[-1] == "final_report"
    assert nodes.count("retrieve") == 2  # the loop is visible in the trace
    assert all(event.latency_ms is not None and event.latency_ms >= 0 for event in trace.events if event.kind == "node")

    llm = [event for event in trace.events if event.kind == "llm"]
    assert {event.name for event in llm} >= {"analyze_request", "extract_evidence", "check_sufficiency", "assess_risk"}
    assert all(event.data["provider"] == "mock" and event.data["status"] == "ok" for event in llm)

    difference = next(event for event in trace.events if event.kind == "tool" and event.name == "calculate_difference"
                      and event.data["arguments"] == {"value_a": 58000.0, "value_b": 40000.0})
    assert difference.data["result"] == {"value": 18000.0} and difference.data["status"] == "ok"

    search = next(event for event in trace.events if event.kind == "retrieval")
    assert search.data["query"] and search.data["chunks"]
    assert {"chunk_id", "document", "page", "score", "matched_by"} <= set(search.data["chunks"][0])

    totals = trace.totals()
    assert totals["llm_calls"] == len(llm) and totals["tool_calls"] >= 3 and totals["errors"] == 0


async def test_trace_never_contains_prompts_or_document_text(samples_dir):
    trace = start_trace()
    await run_graph(DEMO_QUESTION, sample_library(samples_dir))
    dump = json.dumps([event.data for event in trace.events])
    assert "UNTRUSTED DOCUMENT CONTENT" not in dump  # no prompt text
    assert "forty thousand pounds sterling" not in dump  # no document text
    assert "Rules you always follow" not in dump  # no system prompt


async def test_failed_llm_calls_are_traced_with_their_error(samples_dir):
    trace = start_trace()
    await run_graph(DEMO_QUESTION, sample_library(samples_dir), DownLLM())
    failed = [event for event in trace.events if event.kind == "llm" and event.data["status"] == "failed"]
    assert failed and all("LLMUnavailableError" in event.data["error"] for event in failed)
    assert trace.totals()["llm_calls"] == 0 and trace.totals()["llm_failures"] == len(failed)


async def test_recording_without_an_active_trace_is_a_no_op():
    tracing._current.set(None)
    tracing.record("node", "anything", 1.0)  # must not raise
    tracing.report_usage(10, 5)
    assert tracing.current_trace() is None


async def test_token_usage_from_the_provider_reaches_the_trace():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps({"reply": "pong"})}]},
                            "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 7, "totalTokenCount": 127},
        })

    settings = Settings(_env_file=None, GEMINI_API_KEY=SecretStr("AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE123"), GEMINI_MODEL="gemini-test")
    llm = create_gemini_llm(settings, client_args={"transport": httpx.MockTransport(handler)})
    trace = start_trace()
    await llm.structured(LLMPing, system="s", user="u", purpose="ping")
    event = trace.events[-1]
    assert (event.kind, event.name, event.data["status"]) == ("llm", "ping", "ok")
    assert (event.data["input_tokens"], event.data["output_tokens"]) == (120, 7)
    assert trace.totals()["input_tokens"] == 120 and trace.totals()["output_tokens"] == 7
    assert trace.pending_usage is None  # consumed, so it cannot leak into the next call


async def test_provider_error_text_in_the_trace_is_redacted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": 503, "message": "backend down, key AIzaSyA1234567890abcdefghijklmnopqrstu", "status": "UNAVAILABLE"}})

    settings = Settings(_env_file=None, GEMINI_API_KEY=SecretStr("AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE123"), GEMINI_MODEL="gemini-test", LLM_MAX_RETRIES=0)
    llm = create_gemini_llm(settings, client_args={"transport": httpx.MockTransport(handler)})
    trace = start_trace()
    with pytest.raises(LLMUnavailableError):
        await llm.structured(LLMPing, system="s", user="u", purpose="ping")
    assert "AIzaSyA1234567890" not in json.dumps([event.data for event in trace.events])


def test_json_safe_handles_values_postgres_rejects():
    assert json_safe({"a": float("nan"), "b": [float("inf"), 1.5], "c": {1: object}}) == {
        "a": "nan", "b": ["inf", 1.5], "c": {"1": str(object)},
    }
    assert json_safe((1, "x", None, True)) == [1, "x", None, True]


def test_trace_is_bounded():
    trace = start_trace()
    for index in range(tracing.MAX_EVENTS + 50):
        trace.record("node", f"n{index}")
    assert len(trace.events) == tracing.MAX_EVENTS
