"""Grok (xAI) provider.

xAI exposes an OpenAI-compatible API, so the standard LangChain OpenAI chat
model works when pointed at GROK_BASE_URL. The same file is a template for
adding OpenAI itself or any other OpenAI-compatible provider.
"""

from __future__ import annotations

from app.config import Settings
from app.llm.base import BaseLLM, LLMConfigurationError
from app.llm.langchain_adapter import LangChainLLM


def create_grok_llm(settings: Settings) -> BaseLLM:
    if settings.GROK_API_KEY is None:
        raise LLMConfigurationError("GROK_API_KEY is not set. Add it to your .env file.")
    if not settings.GROK_MODEL:
        raise LLMConfigurationError("GROK_MODEL is not set. Add a Grok model name to your .env file.")

    from langchain_openai import ChatOpenAI

    options: dict = {
        "model": settings.GROK_MODEL,
        "api_key": settings.GROK_API_KEY.get_secret_value(),
        "base_url": settings.GROK_BASE_URL,
        "timeout": settings.LLM_TIMEOUT_SECONDS,
        "max_retries": 0,
    }
    if settings.LLM_TEMPERATURE is not None:
        options["temperature"] = settings.LLM_TEMPERATURE

    return LangChainLLM(
        ChatOpenAI(**options),
        provider="grok",
        model=settings.GROK_MODEL,
        # Tool-calling based structured output is the most widely supported mode
        # across OpenAI-compatible APIs.
        structured_method="function_calling",
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
        requests_per_minute=settings.LLM_REQUESTS_PER_MINUTE,
    )
