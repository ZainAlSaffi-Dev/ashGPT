"""Unit tests for BM25 retrieval leg + RRF fusion."""

from __future__ import annotations

import pytest

from src.agent.bm25 import (
    BM25Index,
    configure_bm25_source,
    get_bm25_index,
    invalidate,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_basic():
    # 'v' is a legal stopword — stripped. Substantive nouns + digits stay.
    assert tokenize("Mabo v Queensland (No 2) — adverse possession") == [
        "mabo",
        "queensland",
        "no",
        "2",
        "adverse",
        "possession",
    ]


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_tokenize_preserves_neutral_citation():
    toks = tokenize("Held in [2021] HCA 5 that the principle stands.")
    assert "cite:2021_hca_5" in toks
    # Bare year / court / pinpoint NOT emitted as separate tokens — they were
    # consumed so the citation survives IDF as one rare signal, not three
    # commodity tokens.
    assert "2021" not in toks
    assert "hca" not in toks


def test_tokenize_preserves_volume_citation():
    toks = tokenize("(1992) 175 CLR 1 recognised native title.")
    assert "cite:1992_175_clr_1" in toks


def test_tokenize_preserves_section_with_subdivisions():
    toks = tokenize("Refer to s 31(1)(a) of the Property Law Act.")
    assert "sec:31_1_a" in toks
    assert "31" not in toks
    assert "s" not in toks


def test_tokenize_handles_section_range():
    toks = tokenize("See sections 31-33 for the statutory scheme.")
    assert "sec:31-33" in toks


def test_tokenize_strips_legal_stopwords():
    toks = tokenize("The plaintiff v the defendant and the trustee of s 5.")
    assert "v" not in toks
    assert "the" not in toks
    assert "of" not in toks
    assert "and" not in toks
    assert "plaintiff" in toks
    assert "defendant" in toks
    assert "trustee" in toks
    assert "sec:5" in toks


def test_citation_token_distinguishes_two_citations():
    toks = tokenize("Compare [2021] HCA 5 with [2019] HCA 11.")
    assert "cite:2021_hca_5" in toks
    assert "cite:2019_hca_11" in toks


def test_bm25_ranks_citation_match_highest():
    idx = BM25Index(
        [
            ("d1", "The decision in [2021] HCA 5 established the test.", {}),
            ("d2", "An unrelated discussion of adverse possession.", {}),
            ("d3", "Williams v Bowen has nothing on point.", {}),
        ]
    )
    hits = idx.search("[2021] HCA 5", k=3)
    assert hits[0][0] == "d1"


def test_bm25_ranks_section_match_highest():
    idx = BM25Index(
        [
            ("d1", "s 31(1)(a) requires written notice.", {}),
            ("d2", "Section 12 deals with mortgages.", {}),
            ("d3", "General principles of contract law apply.", {}),
        ]
    )
    hits = idx.search("section 31(1)(a)", k=3)
    assert hits[0][0] == "d1"


def test_bm25_ranks_term_match_highest():
    idx = BM25Index(
        [
            ("d1", "adverse possession requires continuous open use of land", {"week": "week_3"}),
            ("d2", "estoppel in equity prevents asserting strict legal rights", {"week": "week_2"}),
            ("d3", "tenancy in common vs joint tenancy distinctions", {"week": "week_1"}),
        ]
    )
    hits = idx.search("adverse possession", k=3)
    assert hits[0][0] == "d1"
    assert hits[0][1] > 0


def test_bm25_filters_by_metadata():
    idx = BM25Index(
        [
            ("a", "adverse possession week three", {"week": "week_3"}),
            ("b", "adverse possession week one", {"week": "week_1"}),
        ]
    )
    hits = idx.search("adverse possession", k=3, where={"week": {"$eq": "week_3"}})
    assert [h[0] for h in hits] == ["a"]


def test_bm25_returns_empty_for_no_match():
    idx = BM25Index([("d1", "adverse possession", {})])
    assert idx.search("kangaroo zoo gibberish", k=3) == []


def test_bm25_get_content():
    idx = BM25Index([("a", "hello world", {"k": "v"})])
    out = idx.get_content("a")
    assert out == ("hello world", {"k": "v"})
    assert idx.get_content("missing") is None


def test_bm25_empty_corpus():
    idx = BM25Index([])
    assert idx.search("anything", k=3) == []


def test_rrf_promotes_docs_present_in_multiple_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    # 'a' and 'b' both appear in both lists at ranks (1,2) and (2,1) → tied RRF
    # score; stable sort preserves dict-insertion order so 'a' wins. The hard
    # invariant is that both shared docs outrank the single-list docs (c, d).
    top2 = {fused[0][0], fused[1][0]}
    assert top2 == {"a", "b"}
    assert fused[2][0] in {"c", "d"}
    assert fused[3][0] in {"c", "d"}
    assert {f[0] for f in fused} == {"a", "b", "c", "d"}


def test_rrf_weights_can_favour_one_leg():
    # Weight dense leg 10x → top-ranked in dense wins even if absent from bm25.
    fused = reciprocal_rank_fusion(
        [["dense_top"], ["bm25_top"]], weights=[10.0, 1.0]
    )
    assert fused[0][0] == "dense_top"


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_per_namespace_index_cache():
    rows_by_ns = {
        "alice": [("a1", "adverse possession alice notes", {})],
        "bob": [("b1", "estoppel bob notes", {})],
    }

    def source(ns):
        return rows_by_ns.get(ns or "", [])

    configure_bm25_source(source)
    invalidate()
    alice = get_bm25_index("alice")
    bob = get_bm25_index("bob")
    assert [h[0] for h in alice.search("adverse possession", k=2)] == ["a1"]
    assert [h[0] for h in bob.search("estoppel", k=2)] == ["b1"]
    # Cached — second call returns same instance.
    assert get_bm25_index("alice") is alice


def test_invalidate_rebuilds_index():
    state: dict[str, list[tuple[str, str, dict]]] = {"u": [("a", "first", {})]}

    def source(ns):
        return state["u"]

    configure_bm25_source(source)
    invalidate()
    first = get_bm25_index("u")
    assert len(first) == 1

    state["u"].append(("b", "second", {}))
    # Without invalidate, cached index still shows 1 doc.
    assert len(get_bm25_index("u")) == 1
    invalidate("u")
    assert len(get_bm25_index("u")) == 2


def test_doc_id_alignment_after_round_trip():
    """End-to-end ranking example with realistic legal text."""
    corpus = [
        ("ratio_mabo", "The court in Mabo v Queensland recognised native title at common law.", {"source": "mabo.pdf"}),
        ("ratio_smith", "Adverse possession requires factual possession with intent to possess.", {"source": "smith.pdf"}),
        ("filler", "The library is open from 9am to 5pm on weekdays.", {"source": "admin.txt"}),
    ]
    idx = BM25Index(corpus)
    hits = idx.search("adverse possession factual", k=3)
    assert hits[0][0] == "ratio_smith"
    # Lexically unrelated filler must not surface.
    assert "filler" not in {h[0] for h in hits}
