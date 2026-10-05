"""The real Gemini code path, with the network replaced by a fake HTTP transport.

Nothing here calls Google. A fake transport answers the HTTP requests that
the Gemini SDK builds, so these tests exercise the same code that runs in
production: request building, response parsing, structured output, tool
calls and error handling.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from app.agents import prompts
from app.config import Settings
from app.llm.base import LLMConfigurationError, LLMPing, LLMUnavailableError
from app.llm.gemini_provider import create_gemini_llm
from app.schemas.investigation import (
    EvidenceExtraction,
    RequestAnalysis,
    RetrievedChunk,
    RiskAssessment,
    SufficiencyAssessment,
)
from app.tools.calculator import CALCULATOR_TOOLS

FAKE_KEY = "AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE123"


class FakeGemini:
    """Records requests and returns canned Gemini API responses."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.headers: list[httpx.Headers] = []
        self.urls: list[str] = []
        self.reply: httpx.Response | None = None

    def text(self, payload: dict | str) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        self.reply = httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        })

    def function_call(self, name: str, args: dict) -> None:
        self.reply = httpx.Response(200, json={
            "candidates": [{
                "content": {"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]},
                "finishReason": "STOP", "index": 0,
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        })

    def error(self, status: int, message: str, code: str) -> None:
        self.reply = httpx.Response(status, json={"error": {"code": status, "message": message, "status": code}})

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(request.headers)
        self.urls.append(str(request.url))
        assert self.reply is not None
        return httpx.Response(self.reply.status_code, content=self.reply.content, headers={"content-type": "application/json"})


@pytest.fixture
def gemini():
    fake = FakeGemini()
    settings = Settings(
        _env_file=None, LLM_PROVIDER="gemini", GEMINI_API_KEY=SecretStr(FAKE_KEY),
        GEMINI_MODEL="gemini-test-model", LLM_MAX_RETRIES=2, LLM_TIMEOUT_SECONDS=5,
    )
    llm = create_gemini_llm(settings, client_args={"transport": httpx.MockTransport(fake.handler)})
    llm._backoff = 0.0  # no waiting between retries in tests
    return llm, fake


async def test_structured_output_round_trip(gemini):
    llm, fake = gemini
    fake.text({"reply": "pong"})
    answer = await llm.structured(LLMPing, system="SYSTEM RULES", user="USER TEXT", purpose="test")
    assert answer == LLMPing(reply="pong")

    request = fake.requests[0]
    assert "gemini-test-model" in fake.urls[0]
    # System and user content travel separately; the key travels in a header, never in the URL.
    assert request["systemInstruction"]["parts"][0]["text"] == "SYSTEM RULES"
    assert request["contents"][0]["parts"][0]["text"] == "USER TEXT"
    assert FAKE_KEY not in fake.urls[0]
    assert fake.headers[0]["x-goog-api-key"] == FAKE_KEY


@pytest.mark.parametrize("schema", [RequestAnalysis, EvidenceExtraction, SufficiencyAssessment, RiskAssessment])
async def test_every_agent_schema_is_sent_with_all_fields_required(gemini, schema):
    """Providers may skip optional fields, so nothing the agent relies on may be optional."""
    llm, fake = gemini
    fake.text("{}")  # the answer does not matter here; only the request does
    with pytest.raises(LLMUnavailableError):
        await llm.structured(schema, system="s", user="u")

    config = fake.requests[0]["generationConfig"]
    sent_schema = config.get("responseJsonSchema") or config.get("responseSchema")
    assert sent_schema["type"].lower() == "object"
    assert set(sent_schema["required"]) == set(sent_schema["properties"])
    assert '"default"' not in json.dumps(sent_schema)  # not a keyword Gemini documents as supported
    if schema is EvidenceExtraction:
        fact = sent_schema.get("$defs", {}).get("ExtractedFact") or sent_schema["properties"]["facts"]["items"]
        assert set(fact["required"]) == {"chunk_ref", "claim", "key", "value_number", "unit", "quote"}


async def test_evidence_extraction_answer_is_parsed_into_typed_facts(gemini):
    llm, fake = gemini
    fake.text({"facts": [{
        "chunk_ref": "C1", "claim": "The invoice total is £58,000.", "key": "total_amount",
        "value_number": 58000, "unit": "GBP", "quote": "Total amount due: £58,000",
    }, {
        "chunk_ref": "C1", "claim": "The invoice number is INV-7.", "key": "other",
        "value_number": None, "unit": None, "quote": "Invoice INV-7",
    }]})
    chunk = RetrievedChunk(chunk_id="1", document_id="d", filename="Invoice.pdf", page=1, ref="C1", text="Invoice INV-7\nTotal amount due: £58,000")
    result = await llm.structured(
        EvidenceExtraction, system=prompts.system_prompt(prompts.EXTRACT_TASK),
        user=prompts.build_extraction_prompt("What is the total?", [chunk]),
    )
    assert [(fact.key, fact.value_number, fact.unit) for fact in result.facts] == [
        ("total_amount", 58000.0, "GBP"), ("other", None, None),
    ]
    # The document text reached the model inside the untrusted wrapper, in the user turn only.
    request = fake.requests[0]
    assert "UNTRUSTED DOCUMENT CONTENT" in request["contents"][0]["parts"][0]["text"]
    assert "Total amount due" not in request["systemInstruction"]["parts"][0]["text"]


async def test_tool_calls_are_parsed_and_tools_are_declared(gemini):
    llm, fake = gemini
    fake.function_call("calculate_ratio", {"numerator": 58000, "denominator": 40000})
    calls = await llm.choose_tools(system="s", user="u", tools=CALCULATOR_TOOLS)
    assert [(call.name, call.args) for call in calls] == [("calculate_ratio", {"numerator": 58000, "denominator": 40000})]

    declared = [d["name"] for tool in fake.requests[0]["tools"] for d in tool["functionDeclarations"]]
    assert declared == ["calculate_difference", "calculate_percentage_change", "calculate_ratio", "sum_values"]


@pytest.mark.parametrize(
    ("status", "message", "code"),
    [
        (400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT"),
        (403, "Permission denied.", "PERMISSION_DENIED"),
        (404, "models/gemini-test-model is not found for API version v1beta.", "NOT_FOUND"),
    ],
)
async def test_wrong_key_or_model_fails_immediately_with_a_clear_message(gemini, status, message, code):
    llm, fake = gemini
    fake.error(status, message, code)
    with pytest.raises(LLMConfigurationError) as raised:
        await llm.structured(LLMPing, system="s", user="u")
    assert len(fake.requests) == 1  # not retried: it can never succeed
    text = str(raised.value)
    assert f"HTTP {status}" in text and message.split(".")[0] in text
    assert "Check the API key and the model name" in text
    assert FAKE_KEY not in text


@pytest.mark.parametrize(("status", "code"), [(429, "RESOURCE_EXHAUSTED"), (500, "INTERNAL"), (503, "UNAVAILABLE")])
async def test_temporary_provider_errors_are_retried_a_bounded_number_of_times(gemini, status, code):
    llm, fake = gemini
    fake.error(status, "Please try again later.", code)
    with pytest.raises(LLMUnavailableError, match="after 3 attempt"):
        await llm.structured(LLMPing, system="s", user="u")
    assert len(fake.requests) == 3  # 1 try + LLM_MAX_RETRIES


async def test_malformed_model_output_is_not_accepted(gemini):
    llm, fake = gemini
    fake.text('{"reply": ')  # truncated JSON
    with pytest.raises(LLMUnavailableError):
        await llm.structured(LLMPing, system="s", user="u")
    assert len(fake.requests) == 3
