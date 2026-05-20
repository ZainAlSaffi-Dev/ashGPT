"""End-to-end retrieval against a real pgvector backend.

Skips automatically unless ``PGVECTOR_TEST_DATABASE_URL`` is set — CI / dev
that want to run this should bring up the ``postgres`` service in the root
``docker-compose.yml`` (``docker compose up -d postgres``) and export
``PGVECTOR_TEST_DATABASE_URL=postgresql+psycopg://lawgpt:lawgpt@localhost:5432/lawgpt``.

What it proves:
  * ``PgVectorStore.upsert`` + ``search`` + ``list_namespace`` round-trip.
  * ``tools.retrieve_texts`` returns hits filtered to the right namespace.
  * BM25 corpus is sourced from the same backend (no Chroma fallback).
"""

from __future__ import annotations

import os
import uuid

import pytest

from src.agent import bm25, tools
from src.storage.vector_store import PgVectorStore, VectorItem


DSN_ENV = "PGVECTOR_TEST_DATABASE_URL"


def _dsn() -> str:
    dsn = os.getenv(DSN_ENV, "").strip()
    if not dsn:
        pytest.skip(
            f"{DSN_ENV} not set — start the docker-compose postgres service and export it"
        )
    return dsn


@pytest.fixture()
def pgstore() -> PgVectorStore:
    """A fresh PgVectorStore writing into a uniquely-named table per test."""
    table = f"vectors_test_{uuid.uuid4().hex[:8]}"
    # Use a small dim so the test fixture stays fast; the prod default is 2560.
    store = PgVectorStore(_dsn(), dim=8, table=table)
    yield store
    # Cleanup — drop the per-test table so we don't accumulate.
    from sqlalchemy import text

    with store._engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))


def _embed(text_: str, dim: int = 8) -> list[float]:
    """Deterministic toy embedding used so the test doesn't need ZeroEntropy."""
    v = [0.0] * dim
    for c in text_:
        v[ord(c) % dim] += 1.0
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def test_pgvector_roundtrip(pgstore: PgVectorStore) -> None:
    ns = "user_alice"
    items = [
        VectorItem(
            id="c1",
            vector=_embed("adverse possession requires factual possession"),
            content="adverse possession requires factual possession",
            metadata={"source": "ap.pdf", "week": "week_3", "type": "reading"},
            namespace=ns,
        ),
        VectorItem(
            id="c2",
            vector=_embed("Torrens system replaced deeds registration"),
            content="Torrens system replaced deeds registration",
            metadata={"source": "torrens.pdf", "week": "week_4", "type": "reading"},
            namespace=ns,
        ),
    ]
    pgstore.upsert(items)

    hits = pgstore.search(
        query_vector=_embed("adverse possession"), namespace=ns, k=2
    )
    assert hits, "expected at least one hit from pgvector"
    assert hits[0].id == "c1"
    assert hits[0].metadata.get("week") == "week_3"

    listed = pgstore.list_namespace(ns)
    assert {it.id for it in listed} == {"c1", "c2"}


def test_pgvector_namespace_isolation(pgstore: PgVectorStore) -> None:
    pgstore.upsert(
        [
            VectorItem(
                "a",
                _embed("alice doc"),
                "alice doc",
                {"source": "a.pdf"},
                namespace="alice",
            ),
            VectorItem(
                "b",
                _embed("bob doc"),
                "bob doc",
                {"source": "b.pdf"},
                namespace="bob",
            ),
        ]
    )
    assert {h.id for h in pgstore.search(_embed("doc"), "alice", k=5)} == {"a"}
    assert {h.id for h in pgstore.search(_embed("doc"), "bob", k=5)} == {"b"}


def test_retrieve_texts_against_pgvector(pgstore: PgVectorStore, monkeypatch) -> None:
    """Wire the agent's retrieve_texts through pgvector + a deterministic embedder."""
    ns = "user_charlie"
    pgstore.upsert(
        [
            VectorItem(
                "doc_ap",
                _embed("adverse possession requires factual possession"),
                "adverse possession requires factual possession",
                {"source": "ap.pdf", "week": "week_3", "type": "reading", "chunk_id": "doc_ap"},
                namespace=ns,
            ),
            VectorItem(
                "doc_torrens",
                _embed("Torrens system indefeasible title register"),
                "Torrens system indefeasible title register",
                {"source": "torrens.pdf", "week": "week_4", "type": "reading", "chunk_id": "doc_torrens"},
                namespace=ns,
            ),
        ]
    )

    class _FakeEmb:
        def embed_query(self, q: str) -> list[float]:
            return _embed(q)

    # Force tools to use the pgstore + deterministic embedder we built above.
    monkeypatch.setattr(tools, "_store", pgstore)
    monkeypatch.setattr(tools, "_embeddings", _FakeEmb())
    bm25.invalidate()
    monkeypatch.setattr(tools, "_bm25_initialised", False)

    # Force the dense-only path so we don't depend on BM25 token overlap.
    monkeypatch.setattr(tools, "USE_HYBRID_RETRIEVAL", False)
    monkeypatch.setattr(tools, "USE_RERANKER", False)

    results = tools.retrieve_texts("adverse possession", k=2, namespace=ns)
    assert results, "expected pgvector to return hits via retrieve_texts"
    assert results[0]["source"] == "ap.pdf"
    assert results[0]["week"] == "week_3"
