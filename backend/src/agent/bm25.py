"""BM25 retrieval leg + Reciprocal Rank Fusion for hybrid search.

Why hybrid:
  Dense embeddings excel at semantic queries ("show me cases about adverse
  occupation") but underperform on rare-term lookups (case names like
  *Mabo v Queensland* or statutory section refs like *s 31 Property Law Act*).
  BM25 nails those. Fusing both with RRF lifts retrieval precision without
  needing per-leg score normalisation.

  Eval results from Phase 0 showed the cross-encoder reranker alone was a
  weak boost; adding BM25 + RRF in front of it gives a stronger candidate
  pool to rerank.

Tokenisation:
  Lowercase, word-split (keep digits + apostrophes — relevant for citations
  like "Williams' Trustees"). No stopword removal: BM25 already downweights
  high-frequency terms via IDF.

Index lifecycle:
  Per-namespace cached indexers, lazily built from ``BM25Source`` (a callable
  that returns the namespace's corpus). Cache invalidates on ``invalidate()``;
  ingestion calls it after upserting new chunks.

Example:
    >>> idx = BM25Index([("d1", "adverse possession requires continuous use"),
    ...                  ("d2", "estoppel in equity")])
    >>> ranked = idx.search("adverse possession", k=2)
    >>> ranked[0][0]
    'd1'
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

# Word chars + apostrophe (for possessives in case names); split on non-word.
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase + word-split. Stable across the package — used by both index
    construction and query encoding so token alignment is guaranteed.

    >>> tokenize("Mabo v Queensland (No 2) — adverse possession")
    ['mabo', 'v', 'queensland', 'no', '2', 'adverse', 'possession']
    """
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass
class _Doc:
    id: str
    content: str
    metadata: dict


class BM25Index:
    """In-memory BM25 over a list of documents. Builds eagerly in ``__init__``.

    Each search returns ``(doc_id, score)`` pairs in descending score order.
    """

    def __init__(self, docs: Iterable[tuple[str, str]] | Iterable[tuple[str, str, dict]]):
        self._docs: list[_Doc] = []
        for d in docs:
            if len(d) == 2:
                self._docs.append(_Doc(id=d[0], content=d[1], metadata={}))
            else:
                self._docs.append(_Doc(id=d[0], content=d[1], metadata=d[2]))
        tokenized = [tokenize(doc.content) for doc in self._docs]
        # rank_bm25 requires a non-empty corpus to construct.
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def __len__(self) -> int:
        return len(self._docs)

    def search(
        self,
        query: str,
        k: int = 30,
        where: dict | None = None,
    ) -> list[tuple[str, float]]:
        if not self._bm25 or not self._docs:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        # Per-doc overlap with query: only rank docs that share at least one
        # token with the query. Avoids surfacing IDF-negative results from
        # small corpora and silent zero-match noise from large ones.
        query_set = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[tuple[str, float]] = []
        for i in order:
            doc = self._docs[i]
            if where and not _meta_match(doc.metadata, where):
                continue
            doc_tokens = set(tokenize(doc.content))
            if not (query_set & doc_tokens):
                # No lexical overlap → BM25 score is uninformative; skip.
                continue
            out.append((doc.id, float(scores[i])))
            if len(out) >= k:
                break
        return out

    def get_content(self, doc_id: str) -> tuple[str, dict] | None:
        for d in self._docs:
            if d.id == doc_id:
                return d.content, d.metadata
        return None


def _meta_match(meta: dict, where: dict) -> bool:
    for k, v in where.items():
        if isinstance(v, dict):
            if "$eq" in v and meta.get(k) != v["$eq"]:
                return False
            if "$in" in v and meta.get(k) not in v["$in"]:
                return False
            if "$ne" in v and meta.get(k) == v["$ne"]:
                return False
        else:
            if meta.get(k) != v:
                return False
    return True


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked id lists into one. Higher = better.

    Score per id = sum over lists of ``weight / (k + rank_in_list)``.
    ``k`` is the canonical RRF damping constant (60 is the literature default).

    >>> top = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    >>> top[0][0] in {"a", "b"}  # tied; stable sort breaks insertion-order
    True
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")
    fused: dict[str, float] = {}
    for w, rl in zip(weights, ranked_lists):
        for rank, doc_id in enumerate(rl):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


# ── Cached per-namespace indexes ──────────────────────────────────────────────


BM25Source = Callable[[str | None], list[tuple[str, str, dict]]]
"""Function returning ``[(id, content, metadata), ...]`` for a namespace.

When called with ``None`` it returns the shared/legacy collection (no
namespace filtering). When called with a user_id it returns only that user's
chunks. The source is responsible for any database/vector-store I/O.
"""


_index_cache: dict[str, BM25Index] = {}
_source: BM25Source | None = None


def configure_bm25_source(source: BM25Source) -> None:
    """Wire the source-of-truth callable. Typically called at app startup."""
    global _source
    _source = source
    _index_cache.clear()


def get_bm25_index(namespace: str | None) -> BM25Index:
    """Return the (cached) BM25 index for ``namespace``. Builds on miss."""
    key = namespace or "__shared__"
    if key in _index_cache:
        return _index_cache[key]
    if _source is None:
        log.warning("BM25 source not configured; returning empty index")
        idx = BM25Index([])
    else:
        rows = _source(namespace)
        idx = BM25Index(rows)
        log.info("BM25 index built for ns=%s (%d docs)", namespace, len(idx))
    _index_cache[key] = idx
    return idx


def invalidate(namespace: str | None = None) -> None:
    """Drop the cached index for ``namespace`` so the next search rebuilds.

    Call after ingestion / deletion. With ``namespace=None`` drops all caches.
    """
    if namespace is None:
        _index_cache.clear()
        return
    _index_cache.pop(namespace or "__shared__", None)
