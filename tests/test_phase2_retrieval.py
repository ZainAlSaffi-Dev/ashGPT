"""Phase 2 tests: state definition, retrieval tools, and metadata filtering."""

from __future__ import annotations

import pytest

from src.agent.state import AgentState, RetrievedDocument


# ── State definition ──────────────────────────────────────────────────────────


class TestAgentState:
    def test_state_accepts_all_keys(self) -> None:
        state: AgentState = {
            "query": "What is the ratio decidendi in Mabo v Queensland?",
            "week_filter": "week_3",
            "intent": "ratio",
            "retrieved_texts": [],
            "retrieved_slides": [],
            "ratio_decidendi": "",
            "irac_analysis": "",
            "mermaid_diagram": "",
            "chronology_summary": "",
            "final_answer": "",
            "node_trace": [],
        }
        assert state["intent"] == "ratio"

    def test_state_allows_partial(self) -> None:
        """AgentState uses total=False, so partial dicts are valid."""
        state: AgentState = {"query": "test question"}
        assert "query" in state

    def test_retrieved_document_structure(self) -> None:
        doc = RetrievedDocument(
            content="Some legal text",
            source="case.pdf",
            week="week_1",
            doc_type="reading",
            image_path=None,
        )
        assert doc["content"] == "Some legal text"
        assert doc["image_path"] is None

    def test_intent_values(self) -> None:
        """Verify all expected intent values are valid."""
        for intent in ("ratio", "chronology", "summary", "general"):
            state: AgentState = {"intent": intent}
            assert state["intent"] == intent


# ── Retrieval tools — unit tests ──────────────────────────────────────────────


class TestFilterBuilder:
    def test_no_filters(self) -> None:
        from src.agent.tools import _build_filter

        assert _build_filter() is None

    def test_week_only(self) -> None:
        from src.agent.tools import _build_filter

        result = _build_filter(week="week_3")
        assert result == {"week": {"$eq": "week_3"}}

    def test_single_doc_type(self) -> None:
        from src.agent.tools import _build_filter

        result = _build_filter(doc_types=["reading"])
        assert result == {"type": {"$eq": "reading"}}

    def test_multiple_doc_types(self) -> None:
        from src.agent.tools import _build_filter

        result = _build_filter(doc_types=["reading", "tutorial"])
        assert result == {"type": {"$in": ["reading", "tutorial"]}}

    def test_week_and_doc_types_combined(self) -> None:
        from src.agent.tools import _build_filter

        result = _build_filter(week="week_1", doc_types=["reading"])
        assert "$and" in result
        assert len(result["$and"]) == 2


# ── Retrieval tools — integration tests ──────────────────────────────────────


class TestRetrieval:
    @pytest.mark.integration
    def test_retrieve_texts_returns_results(self) -> None:
        from src.agent.tools import retrieve_texts

        results = retrieve_texts("property law", k=3)
        assert len(results) > 0
        assert all(r["doc_type"] in ("reading", "tutorial", "supplementary") for r in results)

    @pytest.mark.integration
    def test_retrieve_texts_excludes_slides(self) -> None:
        from src.agent.tools import retrieve_texts

        results = retrieve_texts("lecture slide content", k=5)
        assert all(r["doc_type"] != "lecture_slide" for r in results)

    @pytest.mark.integration
    def test_retrieve_slides_returns_results(self) -> None:
        from src.agent.tools import retrieve_slides

        results = retrieve_slides("property principles", k=3)
        assert len(results) > 0
        assert all(r["doc_type"] == "lecture_slide" for r in results)

    @pytest.mark.integration
    def test_retrieve_slides_have_image_path(self) -> None:
        from src.agent.tools import retrieve_slides

        results = retrieve_slides("property principles", k=3)
        for r in results:
            assert r["image_path"] is not None
            assert r["image_path"] != ""

    @pytest.mark.integration
    def test_week_filter_restricts_results(self) -> None:
        from src.agent.tools import retrieve_texts

        results = retrieve_texts("legal principle", week="week_1", k=5)
        assert all(r["week"] == "week_1" for r in results)

    @pytest.mark.integration
    def test_retrieve_all_returns_both_modalities(self) -> None:
        from src.agent.tools import retrieve_all

        texts, slides = retrieve_all("property law", k_text=3, k_slides=2)
        assert len(texts) > 0, "No text results returned"
        assert len(slides) > 0, "No slide results returned"

    @pytest.mark.integration
    def test_retrieved_documents_have_content(self) -> None:
        from src.agent.tools import retrieve_texts

        results = retrieve_texts("adverse possession", k=3)
        for r in results:
            assert len(r["content"].strip()) > 0, f"Empty content from {r['source']}"
            assert r["week"], "Missing week metadata"
            assert r["source"], "Missing source metadata"
