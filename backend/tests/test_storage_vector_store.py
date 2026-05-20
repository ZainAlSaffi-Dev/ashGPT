"""InMemoryVectorStore tests — exercise the protocol surface."""

from __future__ import annotations

import math

import pytest

from src.storage.vector_store import (
    InMemoryVectorStore,
    SearchHit,
    VectorItem,
    _cosine,
    _match_where,
    make_vector_store,
)


def _embed(text: str, dim: int = 4) -> list[float]:
    """Toy deterministic embedding: char-based bag-of-codes hashed into dim bins."""
    v = [0.0] * dim
    for c in text:
        v[ord(c) % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def test_cosine_sanity():
    assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        _cosine([1], [1, 2])


def test_match_where_handles_eq_and_in():
    meta = {"week": "week_1", "doc_type": "note"}
    assert _match_where(meta, {"week": "week_1"})
    assert not _match_where(meta, {"week": "week_2"})
    assert _match_where(meta, {"week": {"$in": ["week_1", "week_2"]}})
    assert not _match_where(meta, {"week": {"$in": ["week_3"]}})


def test_inmemory_namespace_isolation():
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorItem("a", _embed("foo"), "foo", {"week": "week_1"}, namespace="alice"),
            VectorItem("b", _embed("bar"), "bar", {"week": "week_1"}, namespace="bob"),
            VectorItem("c", _embed("baz"), "baz", {"week": "week_2"}, namespace="alice"),
        ]
    )
    hits_alice = store.search(_embed("foo"), namespace="alice", k=5)
    hits_bob = store.search(_embed("foo"), namespace="bob", k=5)
    ids_alice = {h.id for h in hits_alice}
    ids_bob = {h.id for h in hits_bob}
    assert ids_alice == {"a", "c"}
    assert ids_bob == {"b"}


def test_inmemory_where_filter():
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorItem("a", _embed("x"), "x", {"week": "week_1"}, namespace="u"),
            VectorItem("b", _embed("x"), "x", {"week": "week_2"}, namespace="u"),
        ]
    )
    hits = store.search(_embed("x"), namespace="u", k=5, where={"week": "week_1"})
    assert [h.id for h in hits] == ["a"]


def test_inmemory_delete_and_delete_namespace():
    store = InMemoryVectorStore()
    items = [
        VectorItem("a", _embed("x"), "x", namespace="u"),
        VectorItem("b", _embed("y"), "y", namespace="u"),
        VectorItem("c", _embed("z"), "z", namespace="v"),
    ]
    store.upsert(items)
    store.delete(["a"], namespace="u")
    assert {h.id for h in store.search(_embed("x"), "u", k=5)} == {"b"}
    store.delete_namespace("u")
    assert store.search(_embed("x"), "u", k=5) == []
    assert {h.id for h in store.search(_embed("z"), "v", k=5)} == {"c"}


def test_make_vector_store_memory_factory():
    s = make_vector_store(backend="memory")
    assert isinstance(s, InMemoryVectorStore)


def test_make_vector_store_unknown_backend():
    with pytest.raises(ValueError):
        make_vector_store(backend="???")


def test_search_returns_hits_in_score_order():
    """Example I/O — verifies search ranks by cosine similarity desc."""
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorItem("near", _embed("hello"), "hello", namespace="u"),
            VectorItem("far", _embed("zzz"), "zzz", namespace="u"),
        ]
    )
    hits = store.search(_embed("hello"), namespace="u", k=2)
    assert isinstance(hits[0], SearchHit)
    assert hits[0].id == "near"
    assert hits[0].score >= hits[1].score
