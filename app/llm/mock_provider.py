"""Offline stand-in for a real LLM. No API key, no network, no cost.

It is NOT intelligent. For each schema the application asks for, a small
rule-based "handler" reads the prompt and fills the schema in. That is enough
to run the whole pipeline in tests and to watch an investigation work before
you have a Gemini key. Real reasoning needs ``LLM_PROVIDER=gemini``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from app.llm.base import BaseLLM, LLMError, LLMPing, T, ToolCallRequest

StructuredHandler = Callable[[str, str], BaseModel]
ToolChooser = Callable[[str, str, list[Any]], list[ToolCallRequest]]


class MockLLM(BaseLLM):
    provider = "mock"
    model = "rule-based-mock"

    def __init__(
        self,
        handlers: dict[type[BaseModel], StructuredHandler] | None = None,
        tool_chooser: ToolChooser | None = None,
    ) -> None:
        super().__init__(max_retries=0)
        self._handlers: dict[type[BaseModel], StructuredHandler] = {
            LLMPing: lambda system, user: LLMPing(reply="pong (mock provider, no real model was called)"),
        }
        self._handlers.update(handlers or {})
        self._tool_chooser = tool_chooser
        self.calls: deque[str] = deque(maxlen=500)  # recent schema / operation names, handy in tests

    async def _structured(self, schema: type[T], system: str, user: str) -> T:
        self.calls.append(schema.__name__)
        handler = self._handlers.get(schema)
        if handler is None:
            raise LLMError(f"MockLLM has no handler for schema {schema.__name__}")
        return handler(system, user)  # type: ignore[return-value]

    async def _choose_tools(self, system: str, user: str, tools: list[Any]) -> list[ToolCallRequest]:
        self.calls.append("choose_tools")
        if self._tool_chooser is None:
            return []
        return self._tool_chooser(system, user, tools)
