"""Gemini provider (the default)."""

from __future__ import annotations

from app.config import Settings
from app.llm.base import BaseLLM, LLMConfigurationError
from app.llm.langchain_adapter import LangChainLLM


def create_gemini_llm(settings: Settings) -> BaseLLM:
    if settings.GEMINI_API_KEY is None:
        raise LLMConfigurationError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "(free key: https://aistudio.google.com/apikey), or set LLM_PROVIDER=mock to run offline."
        )
    if not settings.GEMINI_MODEL:
        raise LLMConfigurationError(
            "GEMINI_MODEL is not set. Put a model name from "
            "https://ai.google.dev/gemini-api/docs/models into your .env file."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    options: dict = {
        "model": settings.GEMINI_MODEL,
        "google_api_key": settings.GEMINI_API_KEY.get_secret_value(),
        "timeout": settings.LLM_TIMEOUT_SECONDS,
        # Retries are handled once, in BaseLLM, so the SDK must not add its own.
        "max_retries": 0,
    }
    if settings.LLM_TEMPERATURE is not None:
        options["temperature"] = settings.LLM_TEMPERATURE

    return LangChainLLM(
        ChatGoogleGenerativeAI(**options),
        provider="gemini",
        model=settings.GEMINI_MODEL,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
        requests_per_minute=settings.LLM_REQUESTS_PER_MINUTE,
    )
