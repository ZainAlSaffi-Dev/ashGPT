"""Phase 4 tests: LangGraph workflow compilation and conditional routing."""

from __future__ import annotations

from unittest.mock import patch

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

    def test_graph_has_verification_when_enabled(self) -> None:
        """USE_VERIFICATION=True (default) should register a verification node."""
        from src.config import USE_VERIFICATION

        if not USE_VERIFICATION:
            pytest.skip("verification is disabled in config")

        from src.agent.graph import build_graph

        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "verification" in node_names

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
        """Ratio intent: router → retrieval → ratio_extractor → synthesis (→ verification)."""
        from src.agent.graph import run_query
        from src.config import USE_VERIFICATION

        result = run_query("What is the ratio decidendi for adverse possession?", week_filter="week_3")
        assert result["intent"] == "ratio"
        expected = ["router", "retrieval", "ratio_extractor", "synthesis"]
        if USE_VERIFICATION:
            expected.append("verification")
        assert result["node_trace"] == expected
        assert result.get("ratio_decidendi"), "Should produce a ratio"
        assert result.get("irac_analysis"), "Should produce an IRAC analysis"
        assert result.get("final_answer"), "Should produce a final answer"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_chronology_path(self) -> None:
        """Chronology intent: router → retrieval → chronology → synthesis (→ verification)."""
        from src.agent.graph import run_query
        from src.config import USE_VERIFICATION

        result = run_query("Show me the timeline of events for week 3 readings")
        assert result["intent"] == "chronology"
        expected = ["router", "retrieval", "chronology", "synthesis"]
        if USE_VERIFICATION:
            expected.append("verification")
        assert result["node_trace"] == expected
        assert result.get("mermaid_diagram"), "Should produce a Mermaid diagram"
        assert result.get("final_answer"), "Should produce a final answer"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_summary_path_runs_both_reasoning_nodes(self) -> None:
        """Summary intent: router → retrieval → ratio_extractor → chronology → synthesis (→ verification)."""
        from src.agent.graph import run_query
        from src.config import USE_VERIFICATION

        result = run_query("Summarise everything about adverse possession from week 3")
        assert result["intent"] == "summary"
        expected_trace = ["router", "retrieval", "ratio_extractor", "chronology", "synthesis"]
        if USE_VERIFICATION:
            expected_trace.append("verification")
        assert result["node_trace"] == expected_trace
        assert result.get("irac_analysis"), "Should produce an IRAC analysis"
        assert result.get("mermaid_diagram"), "Should produce a Mermaid diagram"
        assert result.get("final_answer"), "Should produce a final answer"


# ── Verification node — unit tests (mock LLM) ─────────────────────────────────


class TestVerificationNode:
    def test_clean_answer_is_passthrough(self) -> None:
        """When every cited case appears in the sources, no rewrite is applied."""
        from src.agent.nodes import verification_node

        sources = [
            {
                "content": "In Perry v Clissold the Privy Council held that ...",
                "source": "perry.pdf",
                "week": "week_3",
                "doc_type": "reading",
                "image_path": None,
            }
        ]
        state: AgentState = {
            "query": "What is Perry v Clissold?",
            "final_answer": "Perry v Clissold establishes that possessory title is good against the world.",
            "retrieved_texts": sources,
            "retrieved_slides": [],
        }

        result = verification_node(state)

        report = result["verification_report"]
        assert report["unsupported_claims"] == []
        assert report["rewrites_applied"] is False
        assert "verification" in result["node_trace"]
        # final_answer should not be present (no rewrite)
        assert "final_answer" not in result

    def test_unsupported_citation_triggers_rewrite(self) -> None:
        """A hallucinated case must be removed from final_answer via the rewrite step."""
        from src.agent import nodes
        from src.agent.nodes import verification_node

        sources = [
            {
                "content": "Buckinghamshire County Council v Moran sets out the test for animus possidendi.",
                "source": "moran.pdf",
                "week": "week_3",
                "doc_type": "reading",
                "image_path": None,
            }
        ]
        draft = (
            "The test for adverse possession was set out in Buckinghamshire County "
            "Council v Moran. It was reaffirmed in Hallucinated v Phantom which "
            "extended the principle to chattels."
        )
        state: AgentState = {
            "query": "What is the test for adverse possession?",
            "final_answer": draft,
            "retrieved_texts": sources,
            "retrieved_slides": [],
        }

        rewritten = (
            "The test for adverse possession was set out in Buckinghamshire County "
            "Council v Moran."
        )

        with patch.object(nodes, "llm_call", return_value=rewritten) as mocked:
            result = verification_node(state)

        mocked.assert_called_once()
        report = result["verification_report"]
        assert "Hallucinated v Phantom" in report["unsupported_claims"]
        assert "Buckinghamshire County Council v Moran" not in report["unsupported_claims"]
        assert report["rewrites_applied"] is True
        assert result["final_answer"] == rewritten
        assert "Hallucinated" not in result["final_answer"]

    def test_rewrite_failure_keeps_draft(self) -> None:
        """If the rewrite LLM call fails, the original answer is preserved."""
        from src.agent import nodes
        from src.agent.nodes import verification_node

        state: AgentState = {
            "query": "test",
            "final_answer": "Cited Phantom v Ghost extensively.",
            "retrieved_texts": [],
            "retrieved_slides": [],
        }

        with patch.object(nodes, "llm_call", side_effect=RuntimeError("boom")):
            result = verification_node(state)

        report = result["verification_report"]
        assert "Phantom v Ghost" in report["unsupported_claims"]
        assert report["rewrites_applied"] is False
        assert "final_answer" not in result  # draft preserved by absence of overwrite


# ── Citation extraction unit tests ────────────────────────────────────────────


class TestExtractCaseCitations:
    def test_plain_citation(self) -> None:
        from src.agent.verification import extract_case_citations

        cites = extract_case_citations("The court in Perry v Clissold held...")
        assert cites == ["Perry v Clissold"]

    def test_italicised_citation(self) -> None:
        from src.agent.verification import extract_case_citations

        cites = extract_case_citations("See *Perry v Clissold* and _Mabo v Queensland_.")
        assert "Perry v Clissold" in cites
        assert "Mabo v Queensland" in cites

    def test_vs_variant(self) -> None:
        from src.agent.verification import extract_case_citations

        cites = extract_case_citations("In Smith vs Jones the rule was applied.")
        assert "Smith vs Jones" in cites or "Smith v Jones" in cites

    def test_dedup_case_insensitive(self) -> None:
        from src.agent.verification import extract_case_citations

        cites = extract_case_citations("Perry v Clissold ... again Perry v Clissold.")
        assert len(cites) == 1

    def test_ignores_lowercase_left(self) -> None:
        from src.agent.verification import extract_case_citations

        # "the v Other" should not match — lowercased determiner on the left.
        cites = extract_case_citations("on the v Other side of things")
        assert cites == []
