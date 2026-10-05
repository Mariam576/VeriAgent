"""Rule-based helpers used by the offline mock LLM and as fallbacks.

Filled in together with the agent (milestone 5).
"""

from __future__ import annotations

from app.llm.mock_provider import MockLLM


def create_mock_llm() -> MockLLM:
    return MockLLM()
