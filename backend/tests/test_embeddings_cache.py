"""ZeroEntropy embed_query memoisation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Skip cleanly when the embedding env var is missing on the dev box; the
# wrapper would otherwise refuse to instantiate.
os.environ.setdefault("ZEMBED_API_KEY", "test-key")

from src.embeddings import ZeroEntropyEmbeddings


def _fake_results(dim: int = 4):
    return {"results": [{"embedding": [0.1] * dim}]}


def test_embed_query_uses_cache_on_repeat_call():
    emb = ZeroEntropyEmbeddings(dimensions=4)
    with patch("src.embeddings.requests.post") as post:
        fake = type("R", (), {"ok": True, "json": lambda self: _fake_results(4)})()
        post.return_value = fake
        v1 = emb.embed_query("adverse possession")
        v2 = emb.embed_query("adverse possession")
        assert v1 == v2
        # second call must be served from cache → no extra HTTP roundtrip
        assert post.call_count == 1


def test_embed_query_distinct_strings_miss_cache():
    emb = ZeroEntropyEmbeddings(dimensions=4)
    with patch("src.embeddings.requests.post") as post:
        fake = type("R", (), {"ok": True, "json": lambda self: _fake_results(4)})()
        post.return_value = fake
        emb.embed_query("q1")
        emb.embed_query("q2")
        assert post.call_count == 2


def test_embed_query_cache_bounded():
    emb = ZeroEntropyEmbeddings(dimensions=4)
    from src.embeddings import _QUERY_CACHE_MAX

    with patch("src.embeddings.requests.post") as post:
        fake = type("R", (), {"ok": True, "json": lambda self: _fake_results(4)})()
        post.return_value = fake
        for i in range(_QUERY_CACHE_MAX + 5):
            emb.embed_query(f"q{i}")
    assert len(emb._query_cache) == _QUERY_CACHE_MAX
