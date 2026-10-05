"""Deterministic calculator tools.

Concept - why not let the LLM do the maths?
    A language model predicts likely text. "58,000 - 40,000" usually comes out
    right, but nothing guarantees it, and a wrong amount in a financial review
    is unacceptable. Python arithmetic is always right. So the model's job is
    only to decide *which* calculation is needed; Python performs it.

Concept - @tool:
    The decorator turns a normal function into a LangChain tool. The function
    name, the type hints and the docstring are sent to the model as the tool's
    description, so the docstrings below are written for the model to read.

``Decimal`` is used instead of ``float`` so that money sums like 0.1 + 0.2
give 0.3 and not 0.30000000000000004.
"""

from __future__ import annotations

from decimal import Decimal

from langchain_core.tools import tool


def _to_decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _to_float(value: Decimal) -> float:
    return float(round(value, 6))


@tool
def calculate_difference(value_a: float, value_b: float) -> float:
    """Return value_a minus value_b.

    Use for "how much more / less" questions, for example the invoice total
    (value_a) minus the contract total (value_b).
    """
    return _to_float(_to_decimal(value_a) - _to_decimal(value_b))


@tool
def calculate_percentage_change(old: float, new: float) -> float:
    """Return the percentage change from old to new: (new - old) / |old| * 100.

    Use to express how far a new value is above or below a reference value.
    """
    old_value = _to_decimal(old)
    if old_value == 0:
        raise ValueError("Percentage change is undefined when the old value is 0.")
    return _to_float((_to_decimal(new) - old_value) / abs(old_value) * 100)


@tool
def calculate_ratio(numerator: float, denominator: float) -> float:
    """Return numerator divided by denominator."""
    bottom = _to_decimal(denominator)
    if bottom == 0:
        raise ValueError("A ratio is undefined when the denominator is 0.")
    return _to_float(_to_decimal(numerator) / bottom)


@tool
def sum_values(values: list[float]) -> float:
    """Return the sum of a list of numbers, for example several line items."""
    if not values:
        raise ValueError("sum_values needs at least one number.")
    return _to_float(sum((_to_decimal(value) for value in values), Decimal(0)))


CALCULATOR_TOOLS = [calculate_difference, calculate_percentage_change, calculate_ratio, sum_values]
