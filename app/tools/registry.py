"""Tool allowlist and the single function that executes tools.

Concept - tool calling safety:
    The model can only *ask* for a tool by name with some arguments. This
    module decides whether the request is carried out:

    1. the tool must be on the allowlist,
    2. the arguments must pass the tool's schema (types, required fields),
    3. numbers must be finite,
    4. errors are caught and recorded instead of crashing the investigation.

    Every call, successful or not, becomes a ``ToolCallRecord`` that is saved
    to the ``tool_calls`` table for auditing.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.schemas.investigation import ToolCallRecord
from app.tools.calculator import CALCULATOR_TOOLS

logger = logging.getLogger(__name__)

# Only tools listed here can ever run. Version 1 has calculator tools only:
# they have no side effects, so no human approval is needed before running them.
ALLOWED_TOOLS: dict[str, BaseTool] = {tool.name: tool for tool in CALCULATOR_TOOLS}


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return False  # calculator tools take numbers only


def execute_tool(name: str, arguments: dict[str, Any]) -> ToolCallRecord:
    """Validate and run one tool call. Never raises."""
    started = time.perf_counter()

    def record(status: str, result: dict | None = None, error: str | None = None) -> ToolCallRecord:
        latency = round((time.perf_counter() - started) * 1000, 3)
        logger.info("tool_call name=%s status=%s latency_ms=%.3f", name, status, latency)
        return ToolCallRecord(
            tool_name=name, arguments=arguments, result=result, status=status, latency_ms=latency, error=error
        )

    tool = ALLOWED_TOOLS.get(name)
    if tool is None:
        return record("rejected", error=f"Tool '{name}' is not on the allowlist.")
    if not isinstance(arguments, dict) or not _all_finite(arguments):
        return record("rejected", error="Arguments must be finite numbers.")

    try:
        value = tool.invoke(arguments)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return record("rejected", error=f"Invalid arguments: {problems}")
    except Exception as exc:  # noqa: BLE001 - a failing tool must not stop the investigation
        return record("error", error=str(exc))

    return record("ok", result={"value": value})
