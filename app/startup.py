"""Checks made once, before the application starts serving.

Concept - fail fast:
    Some mistakes in the settings only show when a certain request arrives:
    the first upload fails because a library is missing, or the first visitor
    finds a site that asks for no password. It is kinder to refuse to start,
    with one sentence that says what to change, than to let that happen later.

    ``configuration_problems`` returns the mistakes that must stop the start;
    ``configuration_warnings`` returns the ones that are only worth a log line.
    Both are plain functions of the settings, so they are easy to test.
"""

from __future__ import annotations

import importlib.util
import logging

from app.config import Settings

logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    """The settings cannot work together. The message says what to change."""


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def configuration_problems(settings: Settings) -> list[str]:
    problems: list[str] = []

    if settings.EMBEDDING_PROVIDER == "local" and not _installed("sentence_transformers"):
        problems.append(
            "EMBEDDING_PROVIDER=local needs the sentence-transformers library, which is not installed "
            "(the small hosting image leaves it out). Set EMBEDDING_PROVIDER=gemini and EMBEDDING_DIM=768, "
            "or build the image with LOCAL_EMBEDDINGS=true."
        )

    if settings.AUTH_MODE == "dev" and settings.TRUST_PROXY_HEADERS:
        # TRUST_PROXY_HEADERS says "this runs behind a hosting platform's proxy", that is: on the internet.
        problems.append(
            "AUTH_MODE=dev accepts any name without a password and is meant for your own PC only, "
            "but TRUST_PROXY_HEADERS=true says this server is reachable from the internet. Set AUTH_MODE=jwt."
        )

    return problems


def configuration_warnings(settings: Settings) -> list[str]:
    warnings: list[str] = []
    providers = {settings.LLM_PROVIDER, settings.REVIEWER_LLM_PROVIDER or settings.LLM_PROVIDER}

    if ("gemini" in providers or settings.EMBEDDING_PROVIDER == "gemini") and not settings.GEMINI_API_KEY:
        warnings.append("GEMINI_API_KEY is empty: every request that needs the Gemini API will fail until it is set.")
    if "grok" in providers and not settings.GROK_API_KEY:
        warnings.append("GROK_API_KEY is empty: every request that needs the Grok API will fail until it is set.")

    if settings.ACTION_SERVER == "github" and not (settings.GITHUB_TOKEN and settings.GITHUB_REPOSITORY.count("/") == 1):
        warnings.append(
            "ACTION_SERVER=github needs GITHUB_TOKEN and GITHUB_REPOSITORY (written as owner/repository). "
            "Until both are set, no follow-up issue is offered."
        )

    if settings.TRUST_PROXY_HEADERS and settings.ALLOW_REGISTRATION and not settings.SIGNUP_CODE:
        warnings.append(
            "Anyone who finds this address can create an account and use your model quota. "
            "Set SIGNUP_CODE, or ALLOW_REGISTRATION=false once your own account exists."
        )
    if settings.TRUST_PROXY_HEADERS and settings.LLM_PROVIDER != "mock" and settings.MAX_INVESTIGATIONS_PER_DAY == 0:
        warnings.append("MAX_INVESTIGATIONS_PER_DAY=0: an account may start any number of investigations per day.")

    return warnings


def check_configuration(settings: Settings) -> None:
    """Log the warnings; raise ``ConfigurationError`` when the application must not start."""
    for warning in configuration_warnings(settings):
        logger.warning("Configuration: %s", warning)
    problems = configuration_problems(settings)
    if problems:
        raise ConfigurationError("VeriAgent cannot start with these settings:\n  - " + "\n  - ".join(problems))
