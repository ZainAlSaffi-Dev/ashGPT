"""LangGraph workflow for the Property Law Exam Assistant.

This module compiles the cognitive nodes (Phase 3) into a conditional state
machine. The router's intent classification determines which reasoning path
the query takes through the graph:

    ratio      → Retrieval → Ratio Extractor → Synthesis
    chronology → Retrieval → Chronology Generator → Synthesis
    summary    → Retrieval → Ratio Extractor → Chronology Generator → Synthesis
    general    → Retrieval → Synthesis (context-only, no specialised reasoning)

Optional ``chat_history`` (prior turns) is passed through ``run_query`` so the
router, retrieval packing, and downstream prompts can resolve follow-up
questions while the current utterance remains ``query``.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    chronology_node,
    ratio_extractor_node,
    retrieval_node,
    router_node,
    synthesis_node,
    verification_node,
)
from src.agent.chat_memory import prepare_chat_history_for_run
from src.agent.state import AgentState
from src.config import USE_VERIFICATION

load_dotenv()

log = logging.getLogger(__name__)


# ── Conditional edge: decides which reasoning path to take ─────────────────────


def _route_after_retrieval(state: AgentState) -> str:
    """Return the next node name based on the router's intent classification.

    This function is used as a conditional edge after the retrieval node.
    LangGraph calls it with the current state and uses the returned string
    to select which node to execute next.
    """
    intent = state.get("intent", "general")
    log.info("Routing: intent=%s", intent)
    return intent


# ── Graph construction ─────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph workflow.

    Returns a compiled graph that can be invoked with:
        result = graph.invoke({"query": "your question here"})
    """

    # 1. Create the graph with our state schema
    graph = StateGraph(AgentState)

    # 2. Register each node by name
    graph.add_node("router", router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("ratio_extractor", ratio_extractor_node)
    graph.add_node("chronology", chronology_node)
    graph.add_node("synthesis", synthesis_node)
    if USE_VERIFICATION:
        graph.add_node("verification", verification_node)

    # 3. Define the fixed edges (always follow this path)
    graph.set_entry_point("router")
    graph.add_edge("router", "retrieval")

    # 4. Define the conditional branch after retrieval
    #    The _route_after_retrieval function returns a string key,
    #    and this dict maps each key to the next node name.
    graph.add_conditional_edges(
        source="retrieval",
        path=_route_after_retrieval,
        path_map={
            "ratio": "ratio_extractor",
            "chronology": "chronology",
            "summary": "ratio_extractor",  # summary runs ratio first, then chronology
            "general": "synthesis",
        },
    )

    # 5. Define edges from reasoning nodes to synthesis (or next step)
    #    For "ratio" intent: ratio_extractor → synthesis
    #    For "summary" intent: ratio_extractor → chronology → synthesis
    #    We handle this by making ratio_extractor conditionally route:
    graph.add_conditional_edges(
        source="ratio_extractor",
        path=lambda state: "chronology" if state.get("intent") == "summary" else "synthesis",
        path_map={
            "chronology": "chronology",
            "synthesis": "synthesis",
        },
    )

    graph.add_edge("chronology", "synthesis")

    # 6. Synthesis → verification (if enabled) → END.
    #    Verification fact-checks cited cases against retrieved sources and
    #    asks the synthesis model to remove or hedge unsupported claims.
    if USE_VERIFICATION:
        graph.add_edge("synthesis", "verification")
        graph.add_edge("verification", END)
    else:
        graph.add_edge("synthesis", END)

    # 7. Compile and return
    compiled = graph.compile()
    log.info("LangGraph workflow compiled successfully")
    return compiled


# ── Convenience runner ─────────────────────────────────────────────────────────

_compiled_graph = None


def get_graph():
    """Return a cached compiled graph instance."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(
    query: str,
    week_filter: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    user_id: str | None = None,
) -> AgentState:
    """Run a query through the full agent pipeline. Returns the final state.

    ``chat_history`` should contain **prior** turns only (each dict has ``role``
    and ``content``); the current user message must be passed only in ``query``.
    ``user_id`` is the tenant namespace — when set, retrieval is restricted to
    that user's chunks. Omit for legacy shared-collection behaviour (eval).
    """
    graph = get_graph()
    initial_state: AgentState = {"query": query}
    if week_filter:
        initial_state["week_filter"] = week_filter
    if user_id:
        initial_state["user_id"] = user_id
    prepared = prepare_chat_history_for_run(chat_history)
    if prepared:
        initial_state["chat_history"] = prepared

    result = graph.invoke(initial_state)
    return result
