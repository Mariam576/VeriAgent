"""Build the LangGraph investigation graph.

Concept - why LangGraph?
    An investigation is not a straight line: it branches (is a search needed?
    a calculation?) and it loops (not enough evidence -> search again; review
    failed -> correct and review again). LangGraph lets us draw exactly that
    as a graph of nodes and edges over one shared state, with the loop limits
    visible in one place, instead of hiding the control flow inside prompts.

One graph, one agent. A second agent is only worth adding when a step
needs a truly independent point of view; here ordinary nodes are enough.

                         START
                           |
                      load_memory           preferences + related notes from earlier sessions
                           |                (a message that only asks to remember something
                           |                 skips straight to final_report)
                    analyze_request
                     /            \\
           needs RAG                no RAG
               |                      |
           retrieve <-----------+     |
               |                |     |
        extract_evidence <------|-----+
               |                |
      detect_contradictions     |
          /          \\          |
     calculate        |         |  "retrieve again"
          \\          /          |  (bounded by MAX_AGENT_ITERATIONS)
       check_sufficiency -------+
               |
          assess_risk
               |
        review_findings <----+
          /          \\       |
     approved     not approved
         |            |      |
         |     correct_findings
         |
     human_approval          pauses here (interrupt) when a person must decide,
         |                   and continues after POST /investigations/{id}/approve
     follow_up               carries out the follow-up actions the person approved,
         |                   through MCP (creates an issue); does nothing otherwise
     final_report
         |
        END                  (the service then stores the result and, once that
                              succeeded, a note about it in long-term memory)

Concept - checkpointing:
    When the graph is compiled with a checkpointer, LangGraph saves the state
    after every node under a ``thread_id`` (here: the investigation id). That
    is what makes pausing possible: the paused investigation lives in the
    database, not in memory, so it survives a restart and can be resumed by a
    later request.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes import AgentDeps, InvestigationNodes
from app.agents.routing import (
    route_after_analysis,
    route_after_contradictions,
    route_after_memory,
    route_after_review,
    route_after_sufficiency,
)
from app.agents.state import InvestigationState


def build_investigation_graph(
    deps: AgentDeps, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    # Without a checkpointer the graph cannot pause, so it must not try to.
    if checkpointer is None and deps.pause_for_approval:
        raise ValueError("pause_for_approval needs a checkpointer")
    nodes = InvestigationNodes(deps)
    graph = StateGraph(InvestigationState)

    # --- nodes: name -> function
    graph.add_node("load_memory", nodes.load_memory)
    graph.add_node("analyze_request", nodes.analyze_request)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("extract_evidence", nodes.extract_evidence)
    graph.add_node("detect_contradictions", nodes.detect_contradictions)
    graph.add_node("calculate", nodes.calculate)
    graph.add_node("check_sufficiency", nodes.check_sufficiency)
    graph.add_node("assess_risk", nodes.assess_risk)
    graph.add_node("review_findings", nodes.review_findings)
    graph.add_node("correct_findings", nodes.correct_findings)
    graph.add_node("human_approval", nodes.human_approval)
    graph.add_node("follow_up", nodes.follow_up)
    graph.add_node("final_report", nodes.final_report)

    # --- edges: fixed order
    graph.add_edge(START, "load_memory")
    graph.add_edge("retrieve", "extract_evidence")
    graph.add_edge("extract_evidence", "detect_contradictions")
    graph.add_edge("calculate", "check_sufficiency")
    graph.add_edge("assess_risk", "review_findings")
    graph.add_edge("correct_findings", "review_findings")  # loop: review again after correcting
    graph.add_edge("human_approval", "follow_up")
    graph.add_edge("follow_up", "final_report")
    graph.add_edge("final_report", END)

    # --- conditional edges: the router function picks the next node
    graph.add_conditional_edges("load_memory", route_after_memory, ["analyze_request", "final_report"])
    graph.add_conditional_edges("analyze_request", route_after_analysis, ["retrieve", "extract_evidence"])
    graph.add_conditional_edges("detect_contradictions", route_after_contradictions, ["calculate", "check_sufficiency"])
    graph.add_conditional_edges("check_sufficiency", route_after_sufficiency, ["retrieve", "assess_risk"])  # loop
    graph.add_conditional_edges("review_findings", route_after_review, ["correct_findings", "human_approval"])

    return graph.compile(checkpointer=checkpointer)


def recursion_limit(max_agent_iterations: int) -> int:
    """Hard backstop on graph steps, on top of the iteration counter in the state.

    One retrieval loop is 5 nodes; a correction loop is 2. The straight path is
    12 nodes. This leaves room for every allowed loop and nothing more.
    """
    return 15 + 5 * max_agent_iterations + 6
