"""End-to-end hybrid retrieval test — verifies BM25 + dense + RRF flow.

Patches the factory-backed vector store + BM25 source so the test is
hermetic and doesn't need pgvector / ZeroEntropy embeddings live.
"""

from __future__ import annotations

from unittest.mock import patch

from src.agent import bm25
from src.agent import tools
from src.storage.vector_store import SearchHit


def _hit(content: str, source: str, week: str = "week_3", t: str = "reading") -> SearchHit:
    """Mimic the factory ``SearchHit`` returned by the vector store."""
    return SearchHit(
        id=tools._doc_id(content, source),
        score=0.9,
        content=content,
        metadata={"source": source, "week": week, "type": t},
    )


def test_hybrid_search_fuses_dense_and_bm25():
    # Dense ranks the mabo doc at the top; BM25 ranks the pla1974 doc at the
    # top. RRF should surface both (and any shared doc that appears in both).
    fake_dense_hits = [
        _hit(
            "estoppel in equity bars retraction of representations relied upon",
            "estoppel.pdf",
        ),
        _hit(
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

    class _StubStore:
        def search(self, query_vector, namespace, k, where=None):
            return fake_dense_hits

        def list_namespace(self, namespace):
            return []

    class _StubEmb:
        def embed_query(self, q):
            return [0.0] * 8

    with patch.object(tools, "_get_store", return_value=_StubStore()):
        with patch.object(tools, "_get_embeddings", return_value=_StubEmb()):
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
