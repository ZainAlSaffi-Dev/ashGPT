"""InMemoryVectorStore tests — exercise the protocol surface."""

from __future__ import annotations

import math

import pytest

from src.storage.vector_store import (
    InMemoryVectorStore,
    PgVectorStore,
    SearchHit,
    VectorItem,
    VectorStoreUnavailable,
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


def test_inmemory_project_file_filters_and_metadata_update():
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorItem(
                "a",
                _embed("x"),
                "x",
                {"project_id": "p1", "folder_id": "old", "file_id": "f1"},
                namespace="u",
            ),
            VectorItem(
                "b",
                _embed("x"),
                "x",
                {"project_id": "p2", "folder_id": "other", "file_id": "f2"},
                namespace="u",
            ),
        ]
    )
    hits = store.search(_embed("x"), namespace="u", k=5, where={"project_id": "p1"})
    assert [h.id for h in hits] == ["a"]
    hits = store.search(_embed("x"), namespace="u", k=5, where={"file_id": {"$in": ["f2"]}})
    assert [h.id for h in hits] == ["b"]
    assert [it.id for it in store.list_namespace("u", where={"project_id": "p1"})] == ["a"]

    store.update_metadata(["a"], namespace="u", patch={"folder_id": "new"})
    assert store.search(_embed("x"), namespace="u", k=5, where={"folder_id": "old"}) == []
    assert [h.id for h in store.search(_embed("x"), namespace="u", k=5, where={"folder_id": "new"})] == ["a"]


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


class _FakeRows:
    def all(self):
        return [("hit", "Frazer v Walker content", {"source": "frazer.pdf"}, 0.91)]


class _FakeConnection:
    def __init__(self, engine):
        self._engine = engine

    def execute(self, *_args, **_kwargs):
        from sqlalchemy.exc import OperationalError

        self._engine.calls += 1
        if self._engine.failures_remaining:
            self._engine.failures_remaining -= 1
            raise OperationalError(
                "SELECT secret_vector",
                {"emb": "[very long embedding vector]"},
                Exception("SSL error: unexpected eof while reading"),
            )
        return _FakeRows()


class _FakeBegin:
    def __init__(self, engine):
        self._engine = engine

    def __enter__(self):
        return _FakeConnection(self._engine)

    def __exit__(self, *_args):
        return False


class _FakeEngine:
    def __init__(self, failures_remaining):
        self.calls = 0
        self.disposals = 0
        self.failures_remaining = failures_remaining

    def begin(self):
        return _FakeBegin(self)

    def dispose(self):
        self.disposals += 1


def _pg_store_with_engine(engine) -> PgVectorStore:
    store = PgVectorStore.__new__(PgVectorStore)
    store._engine = engine
    store._table = "vectors"
    return store


def test_pgvector_search_retries_transient_ssl_eof_once():
    engine = _FakeEngine(failures_remaining=1)
    store = _pg_store_with_engine(engine)

    hits = store.search([0.1] * 4, namespace="u", k=1, where={"project_id": "p"})

    assert engine.calls == 2
    assert engine.disposals == 1
    assert hits[0].id == "hit"


def test_pgvector_search_raises_safe_error_after_retry_exhausted():
    engine = _FakeEngine(failures_remaining=2)
    store = _pg_store_with_engine(engine)

    with pytest.raises(VectorStoreUnavailable) as exc:
        store.search([0.1] * 4, namespace="u", k=1, where={"project_id": "p"})

    detail = str(exc.value)
    assert "retry in a moment" in detail
    assert "SELECT" not in detail
    assert "embedding" not in detail
    assert engine.calls == 2
    assert engine.disposals == 1
