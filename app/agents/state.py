"""The LangGraph state for one investigation.

Concept - state:
    The state is the investigation's shared notebook. Every node receives the
    current state, does one job and returns only the keys it wants to change.
    LangGraph merges that update into the state and passes it to the next node.
    No node calls another node directly; they only communicate through state.

Concept - reducers:
    By default a returned key *replaces* the old value. A key annotated with a
    reducer is *combined* with the old value instead:

    * ``messages`` uses ``add_messages``  -> new messages are appended
    * ``events``, ``tool_calls``, ``limitations``, ``errors`` use ``operator.add``
      -> lists are concatenated
    * ``llm_calls`` uses ``operator.add``  -> numbers are summed

    So a node can return ``{"llm_calls": 1}`` and the counter goes up by one.

Everything stored here is plain JSON-like data (dicts, lists, strings,
numbers). That keeps the state easy to log and ready for checkpointing
(pause / resume), which is added in Version 2.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class InvestigationState(TypedDict, total=False):
    # ----------------------------------------------------------- inputs
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    investigation_id: str
    question: str  # exactly what the user typed
    standalone_question: str  # the same request, rewritten to make sense without the conversation
    history: list[dict[str, str]]  # earlier turns of the conversation: {"role", "content"}
    document_ids: list[str]  # optional: restrict the investigation to these documents
    filters: dict[str, Any]  # optional metadata filters: document_type, year

    # --------------------------------------------------- request analysis
    analysis: dict[str, Any]
    search_queries: list[str]
    executed_queries: list[str]

    # ------------------------------------------------------------ evidence
    retrieved_documents: list[dict[str, Any]]
    new_chunk_ids: list[str]  # chunks added by the most recent retrieval round
    retrieval_rounds: int
    evidence: list[dict[str, Any]]
    discarded_facts: int  # extracted statements dropped because they were not grounded
    contradictions: list[dict[str, Any]]
    calculations: list[dict[str, Any]]
    tool_selection_done: bool
    sufficiency: dict[str, Any]
    evidence_count_assessed: int  # how many evidence items the last sufficiency judgement looked at
    missing_evidence: list[str]
    retrieve_again: bool

    # ---------------------------------------------------------- assessment
    risk_draft: dict[str, Any]
    risk_findings: list[dict[str, Any]]
    risk_level: str
    confidence_score: float
    review: dict[str, Any]
    review_rounds: int
    requires_human_review: bool

    # -------------------------------------------------------------- output
    report: dict[str, Any]
    final_answer: str

    # --------------------------------------------------- control and audit
    iteration: int
    iteration_limit_reached: bool
    used_fallback: bool
    events: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    limitations: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]


def initial_state(
    *,
    user_id: str,
    investigation_id: str,
    question: str,
    history: list[dict[str, str]] | None = None,
    document_ids: list[str] | None = None,
    filters: dict[str, Any] | None = None,
) -> InvestigationState:
    from langchain_core.messages import HumanMessage

    return InvestigationState(
        messages=[HumanMessage(content=question)],
        user_id=user_id,
        investigation_id=investigation_id,
        question=question,
        standalone_question=question,
        history=history or [],
        document_ids=document_ids or [],
        filters=filters or {},
        search_queries=[],
        executed_queries=[],
        retrieved_documents=[],
        new_chunk_ids=[],
        retrieval_rounds=0,
        evidence=[],
        discarded_facts=0,
        contradictions=[],
        calculations=[],
        tool_selection_done=False,
        missing_evidence=[],
        retrieve_again=False,
        risk_findings=[],
        review_rounds=0,
        requires_human_review=False,
        iteration=0,
        iteration_limit_reached=False,
        used_fallback=False,
        events=[],
        tool_calls=[],
        limitations=[],
        errors=[],
        llm_calls=0,
    )
