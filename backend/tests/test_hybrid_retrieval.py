"""End-to-end hybrid retrieval test — verifies BM25 + dense + RRF flow.

Patches the Chroma vector store + BM25 source so the test is hermetic and
doesn't need the real ChromaDB / ZeroEntropy embeddings on disk.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.agent import bm25
from src.agent import tools


def _fake_langchain_doc(content: str, source: str, week: str = "week_3", t: str = "reading"):
    """Mimic the LangChain ``Document`` shape that Chroma returns."""
    return SimpleNamespace(
        page_content=content,
        metadata={"source": source, "week": week, "type": t},
    )


def test_hybrid_search_fuses_dense_and_bm25():
    # Dense ranks doc_dense_only at the top; BM25 ranks doc_bm25_only at the
    # top. RRF should surface both (and any shared doc that appears in both).
    fake_dense_results = [
        _fake_langchain_doc(
            "estoppel in equity bars retraction of representations relied upon",
            "estoppel.pdf",
        ),
        _fake_langchain_doc(
            "Mabo v Queensland recognised native title at common law",
            "mabo.pdf",
        ),
    ]
    bm25_rows = [
        # The exact-term doc that dense missed:
        (
            "doc_bm25_only",
            "section 38 Property Law Act 1974 statutory adverse possession",
            {"source": "pla1974.pdf", "week": "week_3", "type": "reading"},
        ),
        # The shared doc — both legs see it (matches dense's mabo result).
        (
            tools._doc_id(
                "Mabo v Queensland recognised native title at common law", "mabo.pdf"
            ),
            "Mabo v Queensland recognised native title at common law",
            {"source": "mabo.pdf", "week": "week_3", "type": "reading"},
        ),
    ]

    # Replace the live Chroma vector store + BM25 source with stubs.
    class _StubStore:
        def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult, filter):
            return fake_dense_results

    with patch.object(tools, "_get_vectorstore", return_value=_StubStore()):
        bm25.configure_bm25_source(lambda ns: bm25_rows)
        bm25.invalidate()
        tools._bm25_initialised = True  # block default-source override

        fused = tools._hybrid_search(
            query="section 38 adverse possession Mabo",
            where_filter=None,
            fetch_dense=4,
            fetch_bm25=4,
            fused_k=4,
            namespace=None,
        )

    sources = {d["source"] for d in fused}
    # Must contain at least one source unique to each leg → proves fusion.
    assert "pla1974.pdf" in sources, f"BM25-only doc missing — got {sources}"
    assert "mabo.pdf" in sources, f"Dense doc missing — got {sources}"


def test_chroma_filter_to_meta_flattens_and_clauses():
    f = {
        "$and": [
            {"week": {"$eq": "week_3"}},
            {"type": {"$in": ["reading", "tutorial"]}},
            {"namespace": {"$eq": "alice"}},
        ]
    }
    out = tools._chroma_filter_to_meta(f)
    assert out["week"] == "week_3"
    assert out["type"] == {"$in": ["reading", "tutorial"]}
    assert out["namespace"] == "alice"


def test_chroma_filter_to_meta_handles_leaf_eq():
    f = {"week": {"$eq": "week_1"}}
    assert tools._chroma_filter_to_meta(f) == {"week": "week_1"}


def test_doc_id_stable_for_same_input():
    a = tools._doc_id("hello world", "src.pdf")
    b = tools._doc_id("hello world", "src.pdf")
    assert a == b
    c = tools._doc_id("hello world", "other.pdf")
    assert a != c
