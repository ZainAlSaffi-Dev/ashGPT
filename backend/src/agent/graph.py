"""LangGraph workflow for the LawGPT study assistant.

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
    cache_check_node,
    chronology_node,
    ratio_extractor_node,
    retrieval_node,
    router_node,
    synthesis_node,
    verification_node,
)
from src.agent.chat_memory import prepare_chat_memory_for_run
from src.agent.state import AgentState
from src.config import (
    ANSWER_CACHE_TTL_DAYS,
    ESCALATION_MODEL,
    LOW_CONFIDENCE_THRESHOLD,
    SYNTHESIS_MODEL,
    USE_ANSWER_CACHE,
    USE_CONFIDENCE_ESCALATION,
    USE_VERIFICATION,
)

load_dotenv()

log = logging.getLogger(__name__)


# ── Conditional edge: decides which reasoning path to take ─────────────────────


def _route_after_retrieval(state: AgentState) -> str:
    """Return the next node name after retrieval.

    Used as a conditional edge: when the cache lookup is enabled and we want
    to gate downstream LLM calls behind it, retrieval → cache_check. The
    intent-based routing happens *after* the cache miss path. This indirection
    lets us short-circuit straight to END on cache hits without re-running
    any reasoning or verification node.
    """
    if state.get("cache_hit"):
        return "cache_hit"
    intent = state.get("intent", "general")
    log.info("Routing: intent=%s", intent)
    return intent


def _route_after_cache_check(state: AgentState) -> str:
    """Branch to END on cache hit, otherwise to the intent-based reasoning path."""
    if state.get("cache_hit"):
        log.info("Routing: cache_hit → END")
        return "cache_hit"
    intent = state.get("intent", "general")
    log.info("Routing (post-cache miss): intent=%s", intent)
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
    graph.add_node("cache_check", cache_check_node)
    graph.add_node("ratio_extractor", ratio_extractor_node)
    graph.add_node("chronology", chronology_node)
    graph.add_node("synthesis", synthesis_node)
    if USE_VERIFICATION:
        graph.add_node("verification", verification_node)

    # 3. Define the fixed edges (always follow this path)
    graph.set_entry_point("router")
    graph.add_edge("router", "retrieval")
    graph.add_edge("retrieval", "cache_check")

    # 4. After cache_check: on hit, jump straight to END (no reasoning, no
    #    verification, no escalation — answer is cached). On miss, fall
    #    through to the intent-based reasoning branch.
    graph.add_conditional_edges(
        source="cache_check",
        path=_route_after_cache_check,
        path_map={
            "cache_hit": END,
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
    *,
    use_cache: bool | None = None,
    use_escalation: bool | None = None,
) -> AgentState:
    """Run a query through the full agent pipeline. Returns the final state.

    ``chat_history`` should contain **prior** turns only (each dict has ``role``
    and ``content``); the current user message must be passed only in ``query``.
    ``user_id`` is the tenant namespace — when set, retrieval is restricted to
    that user's chunks. Omit for legacy shared-collection behaviour (eval).

    Phase 5 hardening (controlled by ``use_cache`` / ``use_escalation`` —
    default to the config flags):

    * **Semantic cache** — after the first run we hash (user_id, normalised
      query, sorted retrieved chunk ids). On a hit we replay the cached
      answer + sources without re-invoking the LLM. Follow-ups (chat_history)
      bypass the cache to keep multi-turn reasoning faithful.
    * **Confidence-gated escalation** — when verification's
      ``confidence_score`` falls below ``LOW_CONFIDENCE_THRESHOLD``, we
      re-synthesize once with ``ESCALATION_MODEL`` against the same context.
      Records ``escalated_from`` in the result.
    """
    graph = get_graph()
    initial_state: AgentState = {"query": query}
    if week_filter:
        initial_state["week_filter"] = week_filter
    if user_id:
        initial_state["user_id"] = user_id
    prepared, overflow, memory = prepare_chat_memory_for_run(chat_history)
    if prepared:
        initial_state["chat_history"] = prepared
    if memory:
        initial_state["conversation_memory"] = memory
    if overflow["memory_compressed"] or overflow["truncated_messages"]:
        initial_state["chat_history_overflow"] = overflow
        initial_state["memory_telemetry"] = {
            "memory_compressed": overflow["memory_compressed"],
            "compressed_turns": overflow["compressed_turns"],
            "recent_messages": overflow["recent_messages"],
            "memory_fact_count": overflow["memory_fact_count"],
            "memory_summary_chars": overflow["memory_summary_chars"],
            "truncated_messages": overflow["truncated_messages"],
        }

    result = graph.invoke(initial_state)

    cache_on = USE_ANSWER_CACHE if use_cache is None else use_cache
    escalate_on = USE_CONFIDENCE_ESCALATION if use_escalation is None else use_escalation

    # On a cache hit the graph short-circuited at cache_check; no verification
    # report was produced for confidence-gated escalation, and the cached
    # answer already reflects whatever escalation ran when it was first
    # written. Skip escalation + cache rewrite entirely on hits.
    if result.get("cache_hit"):
        return result

    if escalate_on and user_id and not prepared:
        result = _maybe_escalate(result)

    if cache_on and user_id and not prepared:
        # Fire-and-forget cache write on a daemon thread so the D1/SQLite
        # round-trip never blocks the response. ``asyncio.run`` creates a
        # fresh event loop in that thread, isolated from any caller loop
        # (FastAPI threadpool executors, eval harness, Streamlit).
        _spawn_cache_write(result, query, user_id)

    return result


def _spawn_cache_write(result: AgentState, query: str, user_id: str) -> None:
    """Background daemon thread runs the cache put — never blocks return."""
    import asyncio
    import threading

    snapshot = dict(result)

    def _run() -> None:
        try:
            asyncio.run(_cache_write_safe(snapshot, query, user_id))
        except Exception as e:  # pragma: no cover — cache failure must not surface
            log.warning("cache write failed: %s", e)

    threading.Thread(target=_run, name="cache-write", daemon=True).start()


def _chunk_ids_from(result: AgentState) -> list[str]:
    """Pull stable identifiers off retrieved docs for cache keying."""
    ids: list[str] = []
    for d in (result.get("retrieved_texts") or []) + (result.get("retrieved_slides") or []):
        ids.append(d.get("source") or d.get("content", "")[:80])
    return ids


def _maybe_escalate(result: AgentState) -> AgentState:
    """Re-synthesize with the stronger model when confidence is low."""
    report = result.get("verification_report") or {}
    confidence = float(report.get("confidence_score", 1.0))
    if confidence >= LOW_CONFIDENCE_THRESHOLD:
        return result
    log.info(
        "escalation: confidence=%.2f < %.2f, retrying with %s",
        confidence,
        LOW_CONFIDENCE_THRESHOLD,
        ESCALATION_MODEL,
    )
    # Re-call synthesis_node with override model via temporary state field.
    from src.agent.nodes import synthesis_node

    escalated_state = {**result, "_override_synthesis_model": ESCALATION_MODEL}
    try:
        delta = synthesis_node(escalated_state)
    except Exception as e:  # pragma: no cover
        log.warning("escalation synthesis failed (%s); keeping original answer", e)
        return result
    new_answer = (delta or {}).get("final_answer")
    if not new_answer:
        return result
    out = dict(result)
    out["final_answer"] = new_answer
    out["escalated_from"] = SYNTHESIS_MODEL
    out["escalated_to"] = ESCALATION_MODEL
    return out


async def _cache_write_safe(result: AgentState, query: str, user_id: str) -> None:
    from datetime import timedelta

    from src.agent.cache import make_cache_key, put

    chunk_ids = _chunk_ids_from(result)
    cache_key = make_cache_key(user_id, query, chunk_ids)
    payload = {
        "intent": result.get("intent"),
        "sources": [
            {
                "source": d.get("source"),
                "doc_type": d.get("doc_type"),
                "week": d.get("week"),
                "snippet": (d.get("content") or "")[:280],
            }
            for d in (result.get("retrieved_texts") or [])
            + (result.get("retrieved_slides") or [])
        ],
        "verification_report": result.get("verification_report"),
    }
    await put(
        cache_key=cache_key,
        user_id=user_id,
        answer=result.get("final_answer", ""),
        payload=payload,
        ttl=timedelta(days=ANSWER_CACHE_TTL_DAYS),
    )
