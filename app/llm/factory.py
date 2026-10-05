"""The single place where an LLM object is created.

The rest of the application does::

    from app.llm.factory import get_llm
    llm = get_llm()

and never imports Gemini or Grok directly. Switching provider is therefore a
change to ``LLM_PROVIDER`` in ``.env`` and nothing else.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import Settings, get_settings
from app.llm.base import BaseLLM, FallbackLLM, LLMConfigurationError

logger = logging.getLogger(__name__)


def _create(provider: str, settings: Settings) -> BaseLLM:
    if provider == "gemini":
        from app.llm.gemini_provider import create_gemini_llm

        return create_gemini_llm(settings)
    if provider == "grok":
        from app.llm.grok_provider import create_grok_llm

        return create_grok_llm(settings)
    if provider == "mock":
        from app.agents.heuristics import create_mock_llm

        return create_mock_llm()
    raise LLMConfigurationError(f"Unsupported LLM provider: {provider!r}")


@lru_cache
def get_llm() -> BaseLLM:
    settings = get_settings()
    primary = _create(settings.LLM_PROVIDER, settings)

    fallback_name = settings.LLM_FALLBACK_PROVIDER
    if fallback_name and fallback_name != settings.LLM_PROVIDER:
        try:
            fallback = _create(fallback_name, settings)
        except LLMConfigurationError as exc:
            logger.warning("Fallback provider %s is not usable and will be ignored: %s", fallback_name, exc)
            return primary
        return FallbackLLM(primary, fallback)
    return primary


@lru_cache
def get_reviewer_llm() -> BaseLLM:
    """The model that reviews extracted statements.

    By default the same model as everywhere else. ``REVIEWER_LLM_PROVIDER`` and
    ``REVIEWER_MODEL`` let a different one do the checking: a second model does
    not share the first one's blind spots.
    """
    settings = get_settings()
    provider = settings.REVIEWER_LLM_PROVIDER or settings.LLM_PROVIDER
    if provider == settings.LLM_PROVIDER and not settings.REVIEWER_MODEL:
        return get_llm()
    if settings.REVIEWER_MODEL:
        settings = settings.model_copy(update={"GEMINI_MODEL" if provider == "gemini" else "GROK_MODEL": settings.REVIEWER_MODEL})
    return _create(provider, settings)
