"""Adapter that turns any LangChain chat model into a ``BaseLLM``.

LangChain already gives every chat model the same two methods:
``with_structured_output(schema)`` and ``bind_tools(tools)``. Gemini and Grok
therefore share this one adapter; a provider file only has to build the right
chat model object.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.base import BaseLLM, T, ToolCallRequest
from app.observability import tracing


def _report_usage(message: Any) -> None:
    """Pass the provider's token counts to the trace, when the provider reports them."""
    usage = getattr(message, "usage_metadata", None) or {}
    tracing.report_usage(usage.get("input_tokens"), usage.get("output_tokens"))


class LangChainLLM(BaseLLM):
    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        provider: str,
        model: str,
        structured_method: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.provider = provider
        self.model = model
        self._chat_model = chat_model
        self._structured_method = structured_method

    @staticmethod
    def _messages(system: str, user: str) -> list:
        # The system message carries our rules. The user message carries the
        # question and the (untrusted) document text. They are never mixed.
        return [SystemMessage(content=system), HumanMessage(content=user)]

    async def _structured(self, schema: type[T], system: str, user: str) -> T:
        options = {"method": self._structured_method} if self._structured_method else {}
        # include_raw=True also returns the raw model message, which carries the token usage.
        runnable = self._chat_model.with_structured_output(schema, include_raw=True, **options)
        outcome = await runnable.ainvoke(self._messages(system, user))
        _report_usage(outcome.get("raw"))
        if outcome.get("parsing_error") is not None:
            raise outcome["parsing_error"]  # output did not fit the schema: counts as a failed attempt
        result = outcome.get("parsed")
        if result is None:
            raise ValueError("The model returned no structured output.")
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)  # raises if the output does not fit the schema

    async def _choose_tools(self, system: str, user: str, tools: list[Any]) -> list[ToolCallRequest]:
        message = await self._chat_model.bind_tools(tools).ainvoke(self._messages(system, user))
        _report_usage(message)
        return [
            ToolCallRequest(name=call["name"], args=dict(call.get("args") or {}), id=call.get("id") or "")
            for call in (getattr(message, "tool_calls", None) or [])
        ]
