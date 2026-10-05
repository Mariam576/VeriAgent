"""Provider-independent LLM interface.

The application only ever uses two operations:

``structured(schema, system=..., user=...)``
    Ask the model a question and get back a validated Pydantic object.
    Concept - structured output: instead of parsing free prose ("the risk
    seems high..."), the model must fill in a schema (``level="high"``).
    Decisions are then read from typed fields, never guessed from text.

``choose_tools(system=..., user=..., tools=[...])``
    Show the model a list of tools and let it decide which to call and with
    what arguments. Concept - tool calling: the model only *requests* a call.
    Python validates the request, runs the tool and owns the result.

Every provider (Gemini, Grok, the offline mock) implements the two small
``_structured`` / ``_choose_tools`` hooks. Retries, timeouts, rate limiting
and logging live here once, so they behave the same for all providers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from app.observability import tracing

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for LLM problems."""


class LLMConfigurationError(LLMError):
    """The provider is not set up correctly, or it rejected the request outright.

    Examples: missing or invalid API key, unknown model name, a request the
    provider refuses (HTTP 400/401/403/404). Retrying cannot help, and quietly
    continuing with rule-based fallbacks would hide the real problem, so the
    investigation stops and the user is told what to fix.
    """


class LLMUnavailableError(LLMError):
    """The provider kept failing after the allowed number of retries."""


@dataclass(frozen=True)
class ToolCallRequest:
    """A tool call requested by the model. Not executed yet."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = ""


class LLMPing(BaseModel):
    """Tiny schema used by scripts/check_llm.py to test a provider end to end."""

    reply: str


def _looks_like_rate_limit(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text or "quota" in text


# HTTP statuses that mean "this request will never work": bad key, no permission, unknown model, bad request.
_PERMANENT_STATUS_CODES = {400, 401, 403, 404, 405, 422}
_PERMANENT_MARKERS = (
    "api key not valid", "api_key_invalid", "invalid api key", "incorrect api key",
    "permission_denied", "unauthenticated", "is not found for api version", "model not found",
)


def provider_status_code(exc: BaseException) -> int | None:
    """The HTTP status behind a provider error, wherever the SDK stored it.

    SDK wrappers re-raise their own exception types "from" the original one,
    so the whole chain of causes is searched.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attribute in ("status_code", "code"):
            value = getattr(current, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
                return value
        current = current.__cause__ or current.__context__
    return None


def is_permanent_error(exc: BaseException) -> bool:
    status = provider_status_code(exc)
    if status is not None:
        return status in _PERMANENT_STATUS_CODES
    text = str(exc).lower()
    return any(marker in text for marker in _PERMANENT_MARKERS)


def _safe(text: object, limit: int = 300) -> str:
    """Provider error text, shortened and with anything key-like removed, for logs and messages."""
    from app.config import get_settings
    from app.security.guardrails import redact_secrets

    return redact_secrets(str(text), get_settings())[:limit]


class BaseLLM(ABC):
    provider: str = "base"
    model: str = ""

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        requests_per_minute: int = 0,
        backoff_seconds: float = 1.0,
    ) -> None:
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._backoff = backoff_seconds
        self._rate_lock = asyncio.Lock()
        self._last_call_at = 0.0

    # ------------------------------------------------- provider-specific hooks
    @abstractmethod
    async def _structured(self, schema: type[T], system: str, user: str) -> T:
        """One attempt at a structured answer."""

    @abstractmethod
    async def _choose_tools(self, system: str, user: str, tools: list[Any]) -> list[ToolCallRequest]:
        """One attempt at letting the model pick tools."""

    # ----------------------------------------------------------- public API
    async def structured(self, schema: type[T], *, system: str, user: str, purpose: str = "") -> T:
        result = await self._call_with_retries(lambda: self._structured(schema, system, user), purpose)
        if not isinstance(result, schema):
            result = schema.model_validate(result)
        return result

    async def choose_tools(
        self, *, system: str, user: str, tools: list[Any], purpose: str = ""
    ) -> list[ToolCallRequest]:
        return await self._call_with_retries(lambda: self._choose_tools(system, user, tools), purpose)

    # ------------------------------------------------------------ internals
    async def _respect_rate_limit(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._rate_lock:
            wait = self._last_call_at + self._min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_at = time.monotonic()

    async def _call_with_retries(self, call, purpose: str):
        """Bounded retries with a timeout per attempt. Never retries forever."""
        attempts = self._max_retries + 1
        last_error: BaseException | None = None

        for attempt in range(1, attempts + 1):
            await self._respect_rate_limit()
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(call(), timeout=self._timeout)
                latency = (time.perf_counter() - started) * 1000
                trace = tracing.current_trace()
                usage = trace.take_usage() if trace else {}
                logger.info(
                    "llm_call provider=%s model=%s purpose=%s attempt=%d latency_ms=%.0f input_tokens=%s output_tokens=%s",
                    self.provider, self.model, purpose or "-", attempt, latency,
                    usage.get("input_tokens", "-"), usage.get("output_tokens", "-"),
                )
                tracing.record(
                    "llm", purpose or "-", latency, provider=self.provider, model=self.model,
                    attempt=attempt, status="ok", **usage,
                )
                return result
            except LLMConfigurationError:
                raise
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"no answer within {self._timeout:.0f}s")
            except Exception as exc:  # noqa: BLE001 - any provider error is retried a bounded number of times
                if is_permanent_error(exc):
                    status = provider_status_code(exc)
                    logger.error(
                        "llm_call_rejected provider=%s model=%s purpose=%s status=%s error=%s",
                        self.provider, self.model, purpose or "-", status, _safe(exc),
                    )
                    raise LLMConfigurationError(
                        f"{self.provider} rejected the request"
                        f"{f' (HTTP {status})' if status else ''}: {_safe(exc)} "
                        "Check the API key and the model name in your .env file."
                    ) from exc
                last_error = exc

            logger.warning(
                "llm_call_failed provider=%s purpose=%s attempt=%d/%d error=%s: %s",
                self.provider, purpose or "-", attempt, attempts,
                type(last_error).__name__, _safe(last_error),
            )
            tracing.record(
                "llm", purpose or "-", (time.perf_counter() - started) * 1000, provider=self.provider,
                model=self.model, attempt=attempt, status="failed",
                error=f"{type(last_error).__name__}: {_safe(last_error, 200)}",
            )
            if attempt < attempts:
                delay = self._backoff * (2 ** (attempt - 1))
                if _looks_like_rate_limit(last_error):
                    delay = max(delay, self._backoff * 15)  # give the quota window time to move
                await asyncio.sleep(min(delay, 45.0))

        raise LLMUnavailableError(
            f"{self.provider} failed after {attempts} attempt(s): "
            f"{type(last_error).__name__}: {_safe(last_error)}"
        )


class FallbackLLM(BaseLLM):
    """Try the primary provider; if it stays unavailable, use the fallback.

    The switch is always logged as a warning, never silent (spec section 39).
    """

    def __init__(self, primary: BaseLLM, fallback: BaseLLM) -> None:
        super().__init__()
        self.primary = primary
        self.fallback = fallback
        self.provider = primary.provider
        self.model = primary.model

    async def _structured(self, schema, system, user):  # pragma: no cover - not used directly
        raise NotImplementedError

    async def _choose_tools(self, system, user, tools):  # pragma: no cover - not used directly
        raise NotImplementedError

    async def structured(self, schema: type[T], *, system: str, user: str, purpose: str = "") -> T:
        try:
            return await self.primary.structured(schema, system=system, user=user, purpose=purpose)
        except LLMUnavailableError as exc:
            logger.warning(
                "llm_fallback primary=%s fallback=%s purpose=%s reason=%s",
                self.primary.provider, self.fallback.provider, purpose or "-", exc,
            )
            return await self.fallback.structured(schema, system=system, user=user, purpose=purpose)

    async def choose_tools(
        self, *, system: str, user: str, tools: list[Any], purpose: str = ""
    ) -> list[ToolCallRequest]:
        try:
            return await self.primary.choose_tools(system=system, user=user, tools=tools, purpose=purpose)
        except LLMUnavailableError as exc:
            logger.warning(
                "llm_fallback primary=%s fallback=%s purpose=%s reason=%s",
                self.primary.provider, self.fallback.provider, purpose or "-", exc,
            )
            return await self.fallback.choose_tools(system=system, user=user, tools=tools, purpose=purpose)
