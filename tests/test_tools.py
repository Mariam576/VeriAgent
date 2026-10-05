"""Calculator tools, the tool allowlist and contradiction detection."""

from __future__ import annotations

import pytest

from app.formatting import format_value, key_label
from app.schemas.investigation import EvidenceItem
from app.tools.calculator import (
    CALCULATOR_TOOLS,
    calculate_difference,
    calculate_percentage_change,
    calculate_ratio,
    sum_values,
)
from app.tools.contradiction import find_contradictions
from app.tools.registry import ALLOWED_TOOLS, execute_tool


def evidence(
    id_: str, filename: str, doc_type: str, key: str, value: float | None, unit: str | None = "GBP", page: int = 1
) -> EvidenceItem:
    return EvidenceItem(
        id=id_, claim=f"{key} is {value}", key=key, value_number=value, unit=unit, quote="",
        chunk_id=f"chunk-{id_}", document_id=f"doc-{filename}", filename=filename,
        document_type=doc_type, page=page,
    )


# -------------------------------------------------------------- calculator
def test_difference_matches_the_spec_example():
    assert calculate_difference.invoke({"value_a": 58000, "value_b": 40000}) == 18000


def test_percentage_change():
    assert calculate_percentage_change.invoke({"old": 40000, "new": 58000}) == 45
    assert calculate_percentage_change.invoke({"old": 200, "new": 150}) == -25


def test_ratio_and_sum():
    assert calculate_ratio.invoke({"numerator": 58000, "denominator": 40000}) == 1.45
    assert sum_values.invoke({"values": [46000, 12000]}) == 58000


def test_decimal_arithmetic_has_no_float_artifacts():
    assert sum_values.invoke({"values": [0.1, 0.2]}) == 0.3
    assert calculate_difference.invoke({"value_a": 1.15, "value_b": 1.05}) == 0.1


def test_division_by_zero_is_an_error_not_a_number():
    with pytest.raises(ValueError):
        calculate_percentage_change.invoke({"old": 0, "new": 5})
    with pytest.raises(ValueError):
        calculate_ratio.invoke({"numerator": 1, "denominator": 0})


def test_tools_describe_themselves_for_the_model():
    for tool in CALCULATOR_TOOLS:
        assert tool.name and tool.description
        assert tool.args_schema.model_json_schema()["properties"]


# ---------------------------------------------------------------- registry
def test_execute_tool_records_a_successful_call():
    record = execute_tool("calculate_difference", {"value_a": 58000, "value_b": 40000})
    assert record.status == "ok"
    assert record.result == {"value": 18000}
    assert record.latency_ms >= 0
    assert record.arguments == {"value_a": 58000, "value_b": 40000}


def test_only_allowlisted_tools_can_run():
    assert set(ALLOWED_TOOLS) == {"calculate_difference", "calculate_percentage_change", "calculate_ratio", "sum_values"}
    record = execute_tool("delete_all_documents", {})
    assert record.status == "rejected"
    assert "allowlist" in record.error


@pytest.mark.parametrize(
    "arguments",
    [
        {"value_a": "58000; DROP TABLE users", "value_b": 1},
        {"value_a": float("nan"), "value_b": 1},
        {"value_a": float("inf"), "value_b": 1},
        {"value_a": True, "value_b": 1},
        {"value_a": 1},
        {"wrong": 1, "names": 2},
    ],
)
def test_invalid_arguments_are_rejected(arguments):
    assert execute_tool("calculate_difference", arguments).status == "rejected"


def test_tool_errors_are_captured_not_raised():
    record = execute_tool("calculate_percentage_change", {"old": 0, "new": 10})
    assert record.status == "error"
    assert "undefined" in record.error
    assert record.result is None


# ----------------------------------------------------------- contradictions
def test_amount_mismatch_between_contract_and_invoice():
    found = find_contradictions([
        evidence("E1", "Invoice_102.pdf", "invoice", "total_amount", 58000),
        evidence("E2", "Contract.pdf", "contract", "total_amount", 40000),
    ])
    assert len(found) == 1
    item = found[0]
    assert item.type == "amount_mismatch"
    assert item.severity == "high"
    # The contract is the reference (value_a) even though the invoice came first.
    assert (item.value_a, item.value_b) == (40000, 58000)
    assert (item.source_a, item.source_b) == ("Contract.pdf", "Invoice_102.pdf")
    assert set(item.evidence_ids) == {"E1", "E2"}
    assert "£40,000" in item.description and "£58,000" in item.description
    assert item.difference is None  # filled in later by the calculator tool


def test_payment_terms_mismatch():
    found = find_contradictions([
        evidence("E1", "Contract.pdf", "contract", "payment_terms_days", 30, "days"),
        evidence("E2", "Invoice_102.pdf", "invoice", "payment_terms_days", 60, "days"),
    ])
    assert [(c.type, c.severity) for c in found] == [("payment_terms_mismatch", "medium")]
    assert "30 days" in found[0].description and "60 days" in found[0].description


def test_agreeing_values_are_not_contradictions():
    assert find_contradictions([
        evidence("E1", "Contract.pdf", "contract", "total_amount", 40000),
        evidence("E2", "Invoice.pdf", "invoice", "total_amount", 40000.00),
        evidence("E3", "Contract.pdf", "contract", "payment_terms_days", 30, "days"),
    ]) == []


def test_facts_without_a_comparison_key_are_ignored():
    assert find_contradictions([
        evidence("E1", "Invoice.pdf", "invoice", "other", 46000),
        evidence("E2", "Invoice.pdf", "invoice", "other", 12000),
        evidence("E3", "Contract.pdf", "contract", "total_amount", None),
    ]) == []


def test_small_differences_get_lower_severity():
    found = find_contradictions([
        evidence("E1", "Contract.pdf", "contract", "total_amount", 40000),
        evidence("E2", "Invoice.pdf", "invoice", "total_amount", 40100),
    ])
    assert found[0].severity == "low"
    found = find_contradictions([
        evidence("E1", "Contract.pdf", "contract", "total_amount", 40000),
        evidence("E2", "Invoice.pdf", "invoice", "total_amount", 42000),
    ])
    assert found[0].severity == "medium"


def test_same_document_with_two_values_is_an_internal_inconsistency():
    found = find_contradictions([
        evidence("E1", "Invoice.pdf", "invoice", "total_amount", 58000, page=1),
        evidence("E2", "Invoice.pdf", "invoice", "total_amount", 85000, page=2),
    ])
    assert [c.type for c in found] == ["internal_inconsistency"]


def test_different_currencies_are_flagged_but_not_subtracted():
    found = find_contradictions([
        evidence("E1", "Contract.pdf", "contract", "total_amount", 40000, "GBP"),
        evidence("E2", "Invoice.pdf", "invoice", "total_amount", 58000, "USD"),
    ])
    assert found[0].type == "unit_mismatch"
    assert "cannot be compared" in found[0].description


def test_supporting_sources_are_grouped_per_value():
    found = find_contradictions([
        evidence("E1", "Contract.pdf", "contract", "payment_terms_days", 30, "days"),
        evidence("E2", "Policy.pdf", "policy", "payment_terms_days", 30, "days"),
        evidence("E3", "Invoice.pdf", "invoice", "payment_terms_days", 60, "days"),
    ])
    assert len(found) == 1
    assert found[0].source_a == "Contract.pdf, Policy.pdf"
    assert found[0].source_b == "Invoice.pdf"
    assert found[0].id == "X1"


# --------------------------------------------------------------- formatting
def test_format_value():
    assert format_value(58000, "GBP") == "£58,000"
    assert format_value(1234.5, "USD") == "$1,234.50"
    assert format_value(-18000, "GBP") == "-£18,000"
    assert format_value(1200, "PKR") == "PKR 1,200"
    assert format_value(30, "days") == "30 days"
    assert format_value(45, "%") == "45%"
    assert format_value(7, None) == "7"
    assert format_value(None, "GBP") == "unknown"
    assert key_label("payment_terms_days") == "payment terms"
    assert key_label("unit_price") == "unit price"
