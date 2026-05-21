"""BM25 retrieval leg + Reciprocal Rank Fusion for hybrid search.

Tokeniser is law-aware: collapses neutral citations (``[2021] HCA 5`` →
``cite:2021_hca_5``) and statutory section refs (``s 31(1)(a)`` →
``sec:31_1_a``) into single tokens so they survive IDF as rare signals.
Strips a small legal stopword list (``v``, ``s``, ``the``…) that otherwise
dominate token counts in legal corpora.

RRF fusion weights are tunable per call — defaults (see
``config.RRF_WEIGHT_DENSE`` / ``RRF_WEIGHT_BM25``) favour the dense leg
since rerank-on-dense outperformed equal-weight hybrid in our eval.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)


# Tokens that dominate legal text but carry no retrieval signal. Single-letter
# abbreviations ("v", "s", "r") appear in nearly every legal paragraph and
# would otherwise drown out IDF; the rest are commodity English stopwords that
# inflate doc length without distinguishing content.
_LEGAL_STOPWORDS = frozenset(
    {
        "v", "s", "r",
        "the", "a", "an", "of", "and", "or", "in", "on", "at", "to",
        "is", "was", "be", "by", "for", "with", "as", "it",
    }
)

# Australian neutral citation: "[2021] HCA 5", "(2020) 270 CLR 372",
# "[1992] HCA 23; (1992) 175 CLR 1". Year, optional volume, court/reporter,
# pinpoint — folded into a single rare token.
_CITATION_RE = re.compile(
    r"[\[\(](\d{4})[\]\)]\s*(?:(\d+)\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(\d+[A-Za-z]?)",
)

# Statutory section ref: "s 31", "s 31(1)(a)", "ss 31-33", "section 31(2)".
# Subdivisions survive as part of the token so "s 31(1)(a)" ≠ "s 31(2)".
_SECTION_RE = re.compile(
    r"\b(?:s|ss|section|sections)\s+(\d+[A-Za-z]?(?:\s*\(\s*[0-9A-Za-z]+\s*\))*(?:\s*-\s*\d+[A-Za-z]?)?)",
    re.IGNORECASE,
)

# Word chars + apostrophe (for possessives like "Williams'"); split on non-word.
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _normalise_section(raw: str) -> str:
    """``31(1)(a)`` → ``sec:31_1_a``; ``31-33`` → ``sec:31-33``."""
    parts = re.findall(r"[0-9A-Za-z]+(?:-\d+[A-Za-z]?)?", raw)
    return "sec:" + "_".join(p.lower() for p in parts) if parts else ""


def _extract_citation_tokens(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []

    def sub(m: re.Match) -> str:
        year, vol, court, pinpoint = m.group(1), m.group(2), m.group(3), m.group(4)
        court_slug = re.sub(r"\s+", "_", court).lower()
        bits = [year, court_slug, pinpoint.lower()]
        if vol:
            bits.insert(1, vol)
        tokens.append("cite:" + "_".join(bits))
        return " "  # consume so the digits/letters don't re-tokenise as bare words

    stripped = _CITATION_RE.sub(sub, text)
    return stripped, tokens


def _extract_section_tokens(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []

    def sub(m: re.Match) -> str:
        norm = _normalise_section(m.group(1))
        if norm:
            tokens.append(norm)
        return " "

    stripped = _SECTION_RE.sub(sub, text)
    return stripped, tokens


def tokenize(text: str) -> list[str]:
    """Law-aware tokeniser. Stable across index + query so tokens align.

    >>> tokenize("Mabo v Queensland (No 2) — adverse possession")
    ['mabo', 'queensland', 'no', '2', 'adverse', 'possession']
    >>> sorted(tokenize("Held in [2021] HCA 5 that s 31(1)(a) applies."))
    ['applies', 'cite:2021_hca_5', 'held', 'sec:31_1_a', 'that']
    """
    if not text:
        return []
    text, cite_tokens = _extract_citation_tokens(text)
    text, sec_tokens = _extract_section_tokens(text)
    word_tokens = [
        m.group(0).lower()
        for m in _TOKEN_RE.finditer(text)
        if m.group(0).lower() not in _LEGAL_STOPWORDS
    ]
    return [*cite_tokens, *sec_tokens, *word_tokens]


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


BM25Source = Callable[..., list[tuple[str, str, dict]]]
"""Function returning ``[(id, content, metadata), ...]`` for a namespace.

When called with ``None`` it returns the shared/legacy collection (no
namespace filtering). When called with a user_id it returns only that user's
chunks. Newer sources may accept ``where=...`` so scoped project/folder/file
indexes can avoid enumerating the whole namespace before filtering.
"""


_index_cache: dict[str, BM25Index] = {}
_source: BM25Source | None = None


def configure_bm25_source(source: BM25Source) -> None:
    """Wire the source-of-truth callable. Typically called at app startup."""
    global _source
    _source = source
    _index_cache.clear()


def get_bm25_index(
    namespace: str | None,
    scope_hash: str | None = None,
    where: dict | None = None,
) -> BM25Index:
    """Return the (cached) BM25 index for ``namespace``. Builds on miss."""
    key = f"{namespace or '__shared__'}:{scope_hash or 'all'}"
    if key in _index_cache:
        return _index_cache[key]
    if _source is None:
        log.warning("BM25 source not configured; returning empty index")
        idx = BM25Index([])
    else:
        try:
            rows = _source(namespace, where=where)
        except TypeError:
            rows = _source(namespace)
        if where:
            rows = [(doc_id, content, meta) for doc_id, content, meta in rows if _meta_match(meta, where)]
        idx = BM25Index(rows)
        log.info("BM25 index built for ns=%s scope=%s (%d docs)", namespace, scope_hash, len(idx))
    _index_cache[key] = idx
    return idx


def invalidate(namespace: str | None = None) -> None:
    """Drop the cached index for ``namespace`` so the next search rebuilds.

    Call after ingestion / deletion. With ``namespace=None`` drops all caches.
    """
    if namespace is None:
        _index_cache.clear()
        return
    prefix = f"{namespace or '__shared__'}:"
    for key in [k for k in _index_cache if k.startswith(prefix)]:
        _index_cache.pop(key, None)
