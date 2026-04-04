"""Phase 4 tests: LangGraph workflow compilation and conditional routing."""

from __future__ import annotations

import pytest

from src.agent.state import AgentState


# ── Unit tests (no API calls) ─────────────────────────────────────────────────


class TestGraphCompilation:
    def test_graph_compiles(self) -> None:
        from src.agent.graph import build_graph

        graph = build_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self) -> None:
        from src.agent.graph import build_graph

        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        for expected in ("router", "retrieval", "ratio_extractor", "chronology", "synthesis"):
            assert expected in node_names, f"Missing node: {expected}"

    def test_cached_graph_is_singleton(self) -> None:
        from src.agent.graph import get_graph

        g1 = get_graph()
        g2 = get_graph()
        assert g1 is g2


class TestRouteFunction:
    def test_route_ratio(self) -> None:
        from src.agent.graph import _route_after_retrieval

        state: AgentState = {"query": "test", "intent": "ratio"}
        assert _route_after_retrieval(state) == "ratio"

    def test_route_chronology(self) -> None:
        from src.agent.graph import _route_after_retrieval

        state: AgentState = {"query": "test", "intent": "chronology"}
        assert _route_after_retrieval(state) == "chronology"

    def test_route_summary(self) -> None:
        from src.agent.graph import _route_after_retrieval

        state: AgentState = {"query": "test", "intent": "summary"}
        assert _route_after_retrieval(state) == "summary"

    def test_route_general(self) -> None:
        from src.agent.graph import _route_after_retrieval

        state: AgentState = {"query": "test", "intent": "general"}
        assert _route_after_retrieval(state) == "general"

    def test_route_defaults_to_general(self) -> None:
        from src.agent.graph import _route_after_retrieval

        state: AgentState = {"query": "test"}
        assert _route_after_retrieval(state) == "general"


# ── Integration tests: verify correct node paths ─────────────────────────────


class TestEndToEndRouting:
    @pytest.mark.integration
    @pytest.mark.slow
    def test_ratio_path(self) -> None:
        """Ratio intent should follow: router → retrieval → ratio_extractor → synthesis."""
        from src.agent.graph import run_query

        result = run_query("What is the ratio decidendi for adverse possession?", week_filter="week_3")
        assert result["intent"] == "ratio"
        assert result["node_trace"] == ["router", "retrieval", "ratio_extractor", "synthesis"]
        assert result.get("ratio_decidendi"), "Should produce a ratio"
        assert result.get("irac_analysis"), "Should produce an IRAC analysis"
        assert result.get("final_answer"), "Should produce a final answer"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_chronology_path(self) -> None:
        """Chronology intent should follow: router → retrieval → chronology → synthesis."""
        from src.agent.graph import run_query

        result = run_query("Show me the timeline of events for week 3 readings")
        assert result["intent"] == "chronology"
        assert result["node_trace"] == ["router", "retrieval", "chronology", "synthesis"]
        assert result.get("mermaid_diagram"), "Should produce a Mermaid diagram"
        assert result.get("final_answer"), "Should produce a final answer"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_summary_path_runs_both_reasoning_nodes(self) -> None:
        """Summary intent should follow: router → retrieval → ratio_extractor → chronology → synthesis."""
        from src.agent.graph import run_query

        result = run_query("Summarise everything about adverse possession from week 3")
        assert result["intent"] == "summary"
        assert result["node_trace"] == [
            "router", "retrieval", "ratio_extractor", "chronology", "synthesis"
        ]
        assert result.get("irac_analysis"), "Should produce an IRAC analysis"
        assert result.get("mermaid_diagram"), "Should produce a Mermaid diagram"
        assert result.get("final_answer"), "Should produce a final answer"
