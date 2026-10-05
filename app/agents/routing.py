"""Conditional edges: the decisions between nodes (spec section 21).

Concept - conditional edge:
    A normal edge always goes from node A to node B. A conditional edge runs a
    small function after node A that looks at the state and returns the name
    of the node to run next. That is how the graph branches and loops.

Routing functions only *read* the state. Anything they need to know (for
example "should we search again?") was written into the state by a node.
"""

from __future__ import annotations

from typing import Literal

from app.agents.nodes import MAX_REVIEW_ROUNDS
from app.agents.state import InvestigationState


def route_after_memory(state: InvestigationState) -> Literal["analyze_request", "final_report"]:
    """Only something to remember?  yes -> confirm it in the report   no -> investigate."""
    return "final_report" if state.get("memory_only") else "analyze_request"


def route_after_analysis(state: InvestigationState) -> Literal["retrieve", "extract_evidence"]:
    """Need RAG?  yes -> search the documents   no -> skip the search."""
    return "retrieve" if state.get("analysis", {}).get("needs_rag") else "extract_evidence"


def route_after_contradictions(state: InvestigationState) -> Literal["calculate", "check_sufficiency"]:
    """Need calculation?  yes when the request involves numbers or a numeric mismatch was found."""
    needs_calculation = state.get("analysis", {}).get("needs_calculation")
    return "calculate" if needs_calculation or state.get("contradictions") else "check_sufficiency"


def route_after_sufficiency(state: InvestigationState) -> Literal["retrieve", "assess_risk"]:
    """Enough evidence?  no -> retrieve again (bounded)   yes, or nothing more to try -> assess risk."""
    return "retrieve" if state.get("retrieve_again") else "assess_risk"


def route_after_review(state: InvestigationState) -> Literal["correct_findings", "human_approval"]:
    """Approved?  no -> correct and review again (bounded)   yes -> human review, then the final report."""
    review = state.get("review", {})
    if review.get("approved", True):
        return "human_approval"
    if state.get("review_rounds", 0) > MAX_REVIEW_ROUNDS:
        return "human_approval"  # loop protection; the remaining fixes are applied by rule
    return "correct_findings"
