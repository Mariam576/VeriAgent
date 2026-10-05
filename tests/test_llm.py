"""The LLM layer: provider factory, retries, timeouts, fallback, LangChain adapter."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import BaseModel, SecretStr

from app.config import Settings
from app.llm import factory
from app.llm.base import (
    BaseLLM,
    FallbackLLM,
    LLMConfigurationError,
    LLMPing,
    LLMUnavailableError,
    ToolCallRequest,
)
from app.llm.gemini_provider import create_gemini_llm
from app.llm.grok_provider import create_grok_llm
from app.llm.langchain_adapter import LangChainLLM
from app.llm.mock_provider import MockLLM


class FlakyLLM(BaseLLM):
    """Fails a set number of times, then succeeds."""

    provider = "flaky"
    model = "test"

    def __init__(self, failures: int, error: Exception | None = None, delay: float = 0.0, **kwargs: Any) -> None:
        super().__init__(backoff_seconds=0.0, **kwargs)
        self.failures = failures
        self.error = error or RuntimeError("boom")
        self.delay = delay
        self.attempts = 0

    async def _structured(self, schema, system, user):
        self.attempts += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.attempts <= self.failures:
            raise self.error
        return schema(reply="ok")

    async def _choose_tools(self, system, user, tools):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error
        return [ToolCallRequest(name="t", args={})]


async def ask(llm: BaseLLM) -> LLMPing:
    return await llm.structured(LLMPing, system="s", user="u", purpose="test")


# ----------------------------------------------------------------- factory
def test_get_llm_returns_mock_in_tests():
    factory.get_llm.cache_clear()
    llm = factory.get_llm()
    assert llm.provider == "mock"


def test_gemini_without_key_is_a_configuration_error():
    settings = Settings(_env_file=None, LLM_PROVIDER="gemini", GEMINI_API_KEY=None, GEMINI_MODEL="some-model")
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        create_gemini_llm(settings)


def test_gemini_without_model_is_a_configuration_error():
    settings = Settings(_env_file=None, LLM_PROVIDER="gemini", GEMINI_API_KEY=SecretStr("k"), GEMINI_MODEL="")
    with pytest.raises(LLMConfigurationError, match="GEMINI_MODEL"):
        create_gemini_llm(settings)


def test_gemini_and_grok_objects_build_without_network():
    settings = Settings(
        _env_file=None,
        GEMINI_API_KEY=SecretStr("fake-key"), GEMINI_MODEL="gemini-test-model",
        GROK_API_KEY=SecretStr("fake-key"), GROK_MODEL="grok-test-model",
        LLM_TEMPERATURE=0.2,
    )
    gemini = create_gemini_llm(settings)
    grok = create_grok_llm(settings)
    assert (gemini.provider, gemini.model) == ("gemini", "gemini-test-model")
    assert (grok.provider, grok.model) == ("grok", "grok-test-model")


def test_switching_provider_is_only_a_setting():
    settings = Settings(_env_file=None, LLM_PROVIDER="grok", GROK_API_KEY=SecretStr("k"), GROK_MODEL="m")
    assert factory._create(settings.LLM_PROVIDER, settings).provider == "grok"


def test_api_keys_are_not_shown_when_settings_are_printed():
    settings = Settings(_env_file=None, GEMINI_API_KEY=SecretStr("super-secret-value"))
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings.model_dump())


# ----------------------------------------------------------------- retries
async def test_succeeds_after_transient_failures():
    llm = FlakyLLM(failures=2, max_retries=2)
    assert (await ask(llm)).reply == "ok"
    assert llm.attempts == 3


async def test_gives_up_after_bounded_retries():
    llm = FlakyLLM(failures=99, max_retries=2)
    with pytest.raises(LLMUnavailableError, match="after 3 attempt"):
        await ask(llm)
    assert llm.attempts == 3  # never retries forever


async def test_configuration_errors_are_not_retried():
    llm = FlakyLLM(failures=99, error=LLMConfigurationError("no key"), max_retries=3)
    with pytest.raises(LLMConfigurationError):
        await ask(llm)
    assert llm.attempts == 1


async def test_timeout_counts_as_a_failure():
    llm = FlakyLLM(failures=0, delay=0.2, max_retries=1, timeout_seconds=0.05)
    with pytest.raises(LLMUnavailableError, match="no answer within"):
        await ask(llm)
    assert llm.attempts == 2


async def test_rate_limiter_spaces_out_calls():
    llm = FlakyLLM(failures=0, requests_per_minute=600)  # one call per 0.1 s
    started = asyncio.get_running_loop().time()
    await ask(llm)
    await ask(llm)
    await ask(llm)
    assert asyncio.get_running_loop().time() - started >= 0.19


# ---------------------------------------------------------------- fallback
async def test_fallback_is_used_and_logged_when_primary_stays_down(caplog):
    primary = FlakyLLM(failures=99, max_retries=0)
    fallback = FlakyLLM(failures=0)
    llm = FallbackLLM(primary, fallback)
    with caplog.at_level("WARNING"):
        assert (await ask(llm)).reply == "ok"
    assert fallback.attempts == 1
    assert any("llm_fallback" in record.message for record in caplog.records)


async def test_fallback_not_used_when_primary_works():
    primary = FlakyLLM(failures=0)
    fallback = FlakyLLM(failures=0)
    await ask(FallbackLLM(primary, fallback))
    assert fallback.attempts == 0


# ------------------------------------------------------------------- mock
async def test_mock_llm_answers_ping_and_rejects_unknown_schemas():
    class Unknown(BaseModel):
        value: int

    llm = MockLLM()
    assert "pong" in (await ask(llm)).reply
    with pytest.raises(LLMUnavailableError):
        await llm.structured(Unknown, system="s", user="u")


# ------------------------------------------------------- LangChain adapter
class ScriptedChatModel(BaseChatModel):
    """A fake LangChain chat model that always answers with a scripted tool call."""

    tool_name: str
    tool_args: dict

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = AIMessage(
            content="",
            tool_calls=[{"name": self.tool_name, "args": self.tool_args, "id": "call-1"}],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


async def test_adapter_returns_validated_pydantic_objects():
    chat_model = ScriptedChatModel(tool_name="LLMPing", tool_args={"reply": "hello"})
    llm = LangChainLLM(chat_model, provider="fake", model="m", structured_method="function_calling")
    answer = await ask(llm)
    assert isinstance(answer, LLMPing)
    assert answer.reply == "hello"


async def test_adapter_rejects_output_that_does_not_fit_the_schema():
    chat_model = ScriptedChatModel(tool_name="LLMPing", tool_args={"wrong_field": 1})
    llm = LangChainLLM(
        chat_model, provider="fake", model="m", structured_method="function_calling",
        max_retries=1, backoff_seconds=0.0,
    )
    with pytest.raises(LLMUnavailableError):
        await ask(llm)


async def test_adapter_returns_tool_call_requests_without_running_them():
    executed = []

    @tool
    def calculate_difference(value_a: float, value_b: float) -> float:
        """Return value_a - value_b."""
        executed.append((value_a, value_b))
        return value_a - value_b

    chat_model = ScriptedChatModel(tool_name="calculate_difference", tool_args={"value_a": 58000, "value_b": 40000})
    llm = LangChainLLM(chat_model, provider="fake", model="m")
    calls = await llm.choose_tools(system="s", user="u", tools=[calculate_difference])
    assert calls == [ToolCallRequest(name="calculate_difference", args={"value_a": 58000, "value_b": 40000}, id="call-1")]
    assert executed == []  # the model only asked; Python decides whether to run it


# ------------------------------------------------- permanent provider errors
class HttpError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_rejected_requests_fail_fast_instead_of_being_retried(status):
    llm = FlakyLLM(failures=99, error=HttpError(status, "API key not valid"), max_retries=3)
    with pytest.raises(LLMConfigurationError, match=f"HTTP {status}"):
        await ask(llm)
    assert llm.attempts == 1


@pytest.mark.parametrize("status", [408, 429, 500, 503])
async def test_temporary_http_errors_are_retried(status):
    llm = FlakyLLM(failures=1, error=HttpError(status, "try later"), max_retries=2)
    assert (await ask(llm)).reply == "ok"
    assert llm.attempts == 2


async def test_status_code_is_found_on_a_wrapped_cause():
    from app.llm.base import is_permanent_error, provider_status_code

    try:
        try:
            raise HttpError(404, "model not found")
        except HttpError as inner:
            raise RuntimeError("SDK wrapper error") from inner
    except RuntimeError as wrapped:
        assert provider_status_code(wrapped) == 404
        assert is_permanent_error(wrapped)
    assert not is_permanent_error(RuntimeError("connection reset by peer"))
    assert is_permanent_error(RuntimeError("API key not valid. Please pass a valid API key."))


async def test_provider_error_text_is_redacted_before_it_is_reported():
    secret_error = RuntimeError("request failed for key AIzaSyA1234567890abcdefghijklmnopqrstu")
    llm = FlakyLLM(failures=99, error=secret_error, max_retries=0)
    with pytest.raises(LLMUnavailableError) as raised:
        await ask(llm)
    assert "AIzaSyA1234567890" not in str(raised.value) and "[REDACTED]" in str(raised.value)
