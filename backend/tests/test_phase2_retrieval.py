"""Phase 2 tests: state definition, retrieval tools, and metadata filtering."""

from __future__ import annotations

from unittest.mock import patch

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


# ── Cross-encoder reranker — unit tests (mocked) ─────────────────────────────


def _doc(content: str, source: str = "x.pdf") -> RetrievedDocument:
    return RetrievedDocument(
        content=content,
        source=source,
        week="week_1",
        doc_type="reading",
        image_path=None,
    )


class TestCrossEncoderReranker:
    def test_empty_input_returns_empty(self) -> None:
        from src.agent.reranker import CrossEncoderReranker

        r = CrossEncoderReranker()
        assert r.rerank("query", [], top_k=4) == []

    def test_top_k_zero_returns_empty(self) -> None:
        from src.agent.reranker import CrossEncoderReranker

        r = CrossEncoderReranker()
        docs = [_doc("a"), _doc("b")]
        assert r.rerank("q", docs, top_k=0) == []

    def test_top_k_truncates_and_orders_by_score(self) -> None:
        """Higher cross-encoder score → earlier in returned list; truncated to top_k."""
        from src.agent.reranker import CrossEncoderReranker

        r = CrossEncoderReranker()

        class FakeModel:
            def predict(self, pairs):
                # Score each doc by length of its content (deterministic mock)
                return [float(len(p[1])) for p in pairs]

        r._model = FakeModel()

        docs = [
            _doc("aa", source="short.pdf"),
            _doc("aaaaa", source="medium.pdf"),
            _doc("aaaaaaaa", source="long.pdf"),
            _doc("a", source="tiny.pdf"),
        ]

        ranked = r.rerank("query", docs, top_k=2)
        assert len(ranked) == 2
        assert ranked[0]["source"] == "long.pdf"
        assert ranked[1]["source"] == "medium.pdf"

    def test_reranker_falls_back_to_input_order_on_failure(self) -> None:
        """If predict raises, return docs[:top_k] without crashing."""
        from src.agent.reranker import CrossEncoderReranker

        r = CrossEncoderReranker()

        class BrokenModel:
            def predict(self, pairs):
                raise RuntimeError("model exploded")

        r._model = BrokenModel()

        docs = [_doc("a", source="first.pdf"), _doc("b", source="second.pdf")]
        ranked = r.rerank("q", docs, top_k=1)
        assert len(ranked) == 1
        assert ranked[0]["source"] == "first.pdf"

    def test_singleton_get_reranker_returns_same_instance(self) -> None:
        from src.agent import reranker as rer

        rer._singleton = None
        a = rer.get_reranker()
        b = rer.get_reranker()
        assert a is b


class TestRetrieveWithRerankerHook:
    def test_use_reranker_false_skips_reranker(self) -> None:
        """Passing use_reranker=False must call MMR with k=k and skip rerank entirely."""
        from src.agent import tools

        recorded: dict[str, int] = {}

        class FakeStore:
            def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult, filter):
                recorded["k"] = k
                return []

            def similarity_search(self, query, k, filter):
                recorded["k"] = k
                return []

        with patch.object(tools, "_get_vectorstore", return_value=FakeStore()):
            with patch.object(tools, "_maybe_rerank", side_effect=lambda q, d, top_k, use_reranker, timings=None: d) as mr:
                tools.retrieve_texts("q", k=8, use_reranker=False)
                # k passed to MMR should equal user-requested k (no over-fetch)
                assert recorded["k"] == 8
                # _maybe_rerank still called but with use_reranker=False (no-op path)
                mr.assert_called_once()
                assert mr.call_args.kwargs["use_reranker"] is False

    def test_use_reranker_true_overfetches_then_reranks(self) -> None:
        """Passing use_reranker=True must over-fetch RERANKER_FETCH_K_TEXT candidates."""
        from src.agent import tools
        from src.config import RERANKER_FETCH_K_TEXT

        recorded: dict[str, int] = {}

        class FakeStore:
            def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult, filter):
                recorded["k"] = k
                return []

            def similarity_search(self, query, k, filter):
                recorded["k"] = k
                return []

        with patch.object(tools, "_get_vectorstore", return_value=FakeStore()):
            with patch.object(tools, "_maybe_rerank", side_effect=lambda q, d, top_k, use_reranker, timings=None: d[:top_k]) as mr:
                tools.retrieve_texts("q", k=8, use_reranker=True)
                assert recorded["k"] == RERANKER_FETCH_K_TEXT
                mr.assert_called_once()
                assert mr.call_args.kwargs["use_reranker"] is True
                assert mr.call_args.kwargs["top_k"] == 8


@pytest.mark.integration
@pytest.mark.slow
class TestRerankerIntegration:
    def test_real_reranker_against_three_chunks(self) -> None:
        """Sanity-check the real cross-encoder against three hand-built chunks."""
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed (Cohere path is the deployment default)",
        )
        from src.agent.reranker import CrossEncoderReranker

        chunks = [
            _doc(
                "Adverse possession requires factual possession plus the intention "
                "to possess (animus possidendi) for the limitation period.",
                source="ap.pdf",
            ),
            _doc(
                "A fee simple is the largest estate known to law and is freely "
                "transferable inter vivos and on death.",
                source="estates.pdf",
            ),
            _doc(
                "The Torrens system replaced deeds-based registration with a "
                "central register conferring indefeasible title.",
                source="torrens.pdf",
            ),
        ]
        r = CrossEncoderReranker()
        ranked = r.rerank("What are the elements of adverse possession?", chunks, top_k=3)

        assert len(ranked) == 3
        assert ranked[0]["source"] == "ap.pdf"
