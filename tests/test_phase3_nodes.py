"""Phase 3 tests: cognitive nodes (router, retrieval, ratio, chronology, synthesis)."""

from __future__ import annotations

import pytest

from src.agent.state import AgentState


# ── Unit tests (no API calls) ─────────────────────────────────────────────────


class TestNodeHelpers:
    def test_append_trace_empty(self) -> None:
        from src.agent.nodes import _append_trace

        state: AgentState = {"query": "test"}
        result = _append_trace(state, "router")
        assert result == ["router"]

    def test_append_trace_preserves_existing(self) -> None:
        from src.agent.nodes import _append_trace

        state: AgentState = {"query": "test", "node_trace": ["router", "retrieval"]}
        result = _append_trace(state, "ratio_extractor")
        assert result == ["router", "retrieval", "ratio_extractor"]

    def test_append_trace_does_not_mutate_original(self) -> None:
        from src.agent.nodes import _append_trace

        original = ["router"]
        state: AgentState = {"query": "test", "node_trace": original}
        result = _append_trace(state, "retrieval")
        assert original == ["router"]
        assert result == ["router", "retrieval"]

    def test_format_context_empty_state(self) -> None:
        from src.agent.nodes import _format_context

        state: AgentState = {"query": "test"}
        result = _format_context(state)
        assert result == "(No context retrieved.)"

    def test_format_context_with_texts(self) -> None:
        from src.agent.nodes import _format_context

        state: AgentState = {
            "query": "test",
            "retrieved_texts": [
                {
                    "content": "Some legal text",
                    "source": "case.pdf",
                    "week": "week_1",
                    "doc_type": "reading",
                    "image_path": None,
                }
            ],
            "retrieved_slides": [],
        }
        result = _format_context(state)
        assert "RETRIEVED TEXT SOURCES" in result
        assert "Some legal text" in result
        assert "case.pdf" in result

    def test_format_context_with_slides(self) -> None:
        from src.agent.nodes import _format_context

        state: AgentState = {
            "query": "test",
            "retrieved_texts": [],
            "retrieved_slides": [
                {
                    "content": "Slide about estates",
                    "source": "slide_01.png",
                    "week": "week_2",
                    "doc_type": "lecture_slide",
                    "image_path": "data/week_2/lecture/slide_01.png",
                }
            ],
        }
        result = _format_context(state)
        assert "RETRIEVED LECTURE SLIDES" in result
        assert "Slide about estates" in result


# ── Integration tests (require API keys and indexed data) ─────────────────────


class TestRouterNode:
    @pytest.mark.integration
    def test_ratio_intent(self) -> None:
        from src.agent.nodes import router_node

        state: AgentState = {
            "query": "What is the ratio decidendi in Mabo v Queensland?"
        }
        result = router_node(state)
        assert result["intent"] == "ratio"
        assert "router" in result["node_trace"]

    @pytest.mark.integration
    def test_chronology_intent(self) -> None:
        from src.agent.nodes import router_node

        state: AgentState = {
            "query": "Show me the timeline of events for adverse possession"
        }
        result = router_node(state)
        assert result["intent"] == "chronology"

    @pytest.mark.integration
    def test_week_filter_extraction(self) -> None:
        from src.agent.nodes import router_node

        state: AgentState = {
            "query": "Summarise the week 3 readings on adverse possession"
        }
        result = router_node(state)
        assert result["week_filter"] is not None
        assert "3" in result["week_filter"]

    @pytest.mark.integration
    def test_no_week_filter_when_absent(self) -> None:
        from src.agent.nodes import router_node

        state: AgentState = {
            "query": "What is a fee simple estate?"
        }
        result = router_node(state)
        assert result["week_filter"] is None

    @pytest.mark.integration
    def test_router_returns_valid_intent(self) -> None:
        from src.agent.nodes import router_node

        state: AgentState = {"query": "Hello, how are you?"}
        result = router_node(state)
        assert result["intent"] in ("ratio", "chronology", "summary", "general")


class TestRetrievalNode:
    @pytest.mark.integration
    def test_retrieval_returns_both_modalities(self) -> None:
        from src.agent.nodes import retrieval_node

        state: AgentState = {"query": "property law principles"}
        result = retrieval_node(state)
        assert len(result["retrieved_texts"]) > 0
        assert len(result["retrieved_slides"]) > 0
        assert "retrieval" in result["node_trace"]

    @pytest.mark.integration
    def test_retrieval_respects_week_filter(self) -> None:
        from src.agent.nodes import retrieval_node

        state: AgentState = {"query": "legal principles", "week_filter": "week_1"}
        result = retrieval_node(state)
        for doc in result["retrieved_texts"]:
            assert doc["week"] == "week_1"


class TestRatioExtractorNode:
    @pytest.mark.integration
    @pytest.mark.slow
    def test_produces_irac_and_ratio(self) -> None:
        from src.agent.nodes import ratio_extractor_node, retrieval_node

        state: AgentState = {
            "query": "What is the ratio decidendi for adverse possession?",
        }
        state.update(retrieval_node(state))
        result = ratio_extractor_node(state)

        assert result["ratio_decidendi"], "ratio_decidendi should not be empty"
        assert result["irac_analysis"], "irac_analysis should not be empty"
        assert "ratio_extractor" in result["node_trace"]


class TestChronologyNode:
    @pytest.mark.integration
    @pytest.mark.slow
    def test_produces_mermaid_diagram(self) -> None:
        from src.agent.nodes import chronology_node, retrieval_node

        state: AgentState = {
            "query": "Show the timeline for adverse possession from week 3",
            "week_filter": "week_3",
        }
        state.update(retrieval_node(state))
        result = chronology_node(state)

        assert result["mermaid_diagram"], "mermaid_diagram should not be empty"
        assert "graph" in result["mermaid_diagram"].lower()
        assert "chronology" in result["node_trace"]


class TestSynthesisNode:
    @pytest.mark.integration
    @pytest.mark.slow
    def test_produces_final_answer(self) -> None:
        from src.agent.nodes import retrieval_node, synthesis_node

        state: AgentState = {
            "query": "Explain the concept of fee simple",
            "intent": "summary",
        }
        state.update(retrieval_node(state))
        result = synthesis_node(state)

        assert result["final_answer"], "final_answer should not be empty"
        assert len(result["final_answer"]) > 50
        assert "synthesis" in result["node_trace"]
