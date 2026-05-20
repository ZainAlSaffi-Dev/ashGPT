"""Retrieval tools for querying the indexed knowledge base.

These functions provide the 'senses' of the agent — they reach into the
configured vector store (Chroma / pgvector / Vectorize) via the
``src.storage.vector_store`` factory and return relevant text chunks and
slide descriptions, optionally filtered by week or document type.

Two retrieval modes:

  * ``USE_HYBRID_RETRIEVAL=True`` (default) — runs BM25 + dense in parallel
    (`ThreadPoolExecutor`) and fuses with Reciprocal Rank Fusion before
    reranking. Higher precision on legal text (case names, statutory refs
    benefit from BM25 term-match).
  * ``USE_HYBRID_RETRIEVAL=False`` — legacy dense-only path.

The dense leg uses cosine ANN over the configured backend. MMR is no
longer applied client-side; the cross-encoder/Cohere reranker downstream
handles candidate diversity instead.
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from src.agent.bm25 import (
    configure_bm25_source,
    get_bm25_index,
    reciprocal_rank_fusion,
)
from src.agent.state import RetrievedDocument
from src.config import (
    BM25_FETCH_K_SLIDES,
    BM25_FETCH_K_TEXT,
    HYBRID_FUSED_K_SLIDES,
    HYBRID_FUSED_K_TEXT,
    RERANKER_FETCH_K_SLIDES,
    RERANKER_FETCH_K_TEXT,
    RETRIEVAL_STRATEGY,
    RRF_K,
    RRF_WEIGHT_BM25,
    RRF_WEIGHT_DENSE,
    USE_HYBRID_RETRIEVAL,
    USE_RERANKER,
)
from src.embeddings import ZeroEntropyEmbeddings
from src.storage.vector_store import SearchHit, VectorStore, make_vector_store

load_dotenv()

log = logging.getLogger(__name__)

# ── Singleton vector store (factory-backed) ────────────────────────────────────

_store: VectorStore | None = None
_embeddings: ZeroEntropyEmbeddings | None = None


def _get_store() -> VectorStore:
    """Lazily build and cache the configured vector store.

    Honours ``VECTOR_BACKEND`` env (pgvector in prod, chroma in local dev).
    Fails loud if the backend is misconfigured — silent ``None`` returns
    used to mask retrieval outages and produce ungrounded answers.
    """
    global _store
    if _store is None:
        _store = make_vector_store()
        log.info("vector store initialised: %s", type(_store).__name__)
    return _store


def _get_embeddings() -> ZeroEntropyEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = ZeroEntropyEmbeddings()
    return _embeddings


def _reset_singletons_for_tests() -> None:
    """Drop the cached store + embeddings (tests that swap backends)."""
    global _store, _embeddings, _bm25_initialised
    _store = None
    _embeddings = None
    _bm25_initialised = False


# ── Metadata filter builders ──────────────────────────────────────────────────


def _build_filter(
    week: str | None = None,
    doc_types: list[str] | None = None,
    namespace: str | None = None,
) -> dict | None:
    """Build a ChromaDB where-filter from optional week, type, and namespace constraints.

    Namespace is the user's tenant id. When ``None`` no namespace filter is
    applied (backwards-compat with the legacy shared collection used by the
    coursework eval); when set, only chunks tagged with ``metadata.namespace``
    matching are returned.
    """
    conditions: list[dict] = []

    if week:
        conditions.append({"week": {"$eq": week}})

    if doc_types and len(doc_types) == 1:
        conditions.append({"type": {"$eq": doc_types[0]}})
    elif doc_types and len(doc_types) > 1:
        conditions.append({"type": {"$in": doc_types}})

    if namespace:
        conditions.append({"namespace": {"$eq": namespace}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── Stable doc identity across legs ───────────────────────────────────────────


def _doc_id(content: str, source: str) -> str:
    """Stable hash used to align dense and BM25 hits for RRF fusion.

    Built from source + leading content so both legs agree without a Chroma
    native id roundtrip. Collisions inside the same source are unlikely given
    the 200-char window; if needed Phase 2 ingestion stores ``chunk_id`` in
    metadata and we read that directly.
    """
    key = f"{source}::{(content or '')[:200]}".encode()
    return hashlib.md5(key).hexdigest()  # noqa: S324 — identity hash, not security


# ── BM25 corpus source (reads from Chroma) ────────────────────────────────────


_bm25_initialised = False


def _bm25_source(namespace: str | None) -> list[tuple[str, str, dict]]:
    """Pull every chunk for ``namespace`` from the configured store → BM25 rows.

    Goes through the ``VectorStore.list_namespace`` adapter so the BM25
    corpus tracks whichever backend ingestion writes to (pgvector in prod,
    chroma in local dev).
    """
    try:
        items = _get_store().list_namespace(namespace or "")
    except NotImplementedError:
        log.warning("vector store does not support enumeration; BM25 corpus empty")
        return []
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("bm25 source: list_namespace failed (%s)", exc)
        return []
    rows: list[tuple[str, str, dict]] = []
    for it in items:
        meta = it.metadata or {}
        rid = meta.get("chunk_id") or _doc_id(it.content, meta.get("source", ""))
        rows.append((rid, it.content, meta))
    return rows


def _ensure_bm25_configured() -> None:
    """Wire the default Chroma-backed BM25 source if no source is set yet."""
    global _bm25_initialised
    from src.agent import bm25 as bm25_mod

    if _bm25_initialised:
        return
    # Respect any source the caller has already configured (e.g. tests or
    # Phase 2 ingestion may install their own).
    if bm25_mod._source is None:
        configure_bm25_source(_bm25_source)
    _bm25_initialised = True


# ── Search dispatch ───────────────────────────────────────────────────────────


def _hit_to_retrieved(hit: SearchHit) -> RetrievedDocument:
    """Map a factory ``SearchHit`` into the agent's ``RetrievedDocument``."""
    meta = hit.metadata or {}
    return RetrievedDocument(
        content=hit.content,
        source=meta.get("source", ""),
        week=meta.get("week", ""),
        doc_type=meta.get("type", ""),
        image_path=meta.get("image_path"),
    )


def _dense_search(
    query_vector: list[float],
    k: int,
    factory_where: dict,
    namespace: str | None,
) -> list[SearchHit]:
    """Cosine ANN over the configured vector store.

    The factory always filters by ``namespace`` at the SQL/HTTP layer (it's a
    column on pgvector, a separate vectorize param). Strip any ``namespace``
    key out of the metadata where-dict before forwarding — pgvector doesn't
    store namespace inside the JSONB metadata blob.
    """
    where = {k_: v for k_, v in (factory_where or {}).items() if k_ != "namespace"}
    return _get_store().search(
        query_vector=query_vector,
        namespace=namespace or "",
        k=k,
        where=where or None,
    )


# ── Hybrid (BM25 + dense) retrieval with RRF fusion ───────────────────────────


def _hybrid_search(
    query: str,
    where_filter: dict | None,
    fetch_dense: int,
    fetch_bm25: int,
    fused_k: int,
    namespace: str | None,
    timings: dict | None = None,
) -> list[RetrievedDocument]:
    """Dense ANN + BM25 fused via RRF, returning the top-``fused_k`` docs.

    Dense and BM25 legs run concurrently in a 2-worker thread pool (both are
    blocking I/O: dense = Postgres round-trip, BM25 = in-memory but uses the
    GIL-released ``rank_bm25`` scorer). Query embedding is computed once
    before the split so both legs reuse it. RRF aligns the two ranked id
    lists via the same ``_doc_id`` scheme used by ingestion.
    """
    _ensure_bm25_configured()

    factory_where = _chroma_filter_to_meta(where_filter) if where_filter else {}

    # Embed once before the split — both legs reuse this vector.
    t_embed = time.perf_counter()
    qv = _get_embeddings().embed_query(query)
    embed_ms = (time.perf_counter() - t_embed) * 1000

    def _dense_leg() -> tuple[list[str], dict[str, RetrievedDocument], float]:
        t0 = time.perf_counter()
        hits = _dense_search(
            qv,
            k=fetch_dense,
            factory_where=factory_where,
            namespace=namespace,
        )
        ms = (time.perf_counter() - t0) * 1000
        ranked: list[str] = []
        by_id: dict[str, RetrievedDocument] = {}
        for h in hits:
            meta = h.metadata or {}
            rid = meta.get("chunk_id") or _doc_id(h.content, meta.get("source", ""))
            ranked.append(rid)
            if rid not in by_id:
                by_id[rid] = _hit_to_retrieved(h)
        return ranked, by_id, ms

    def _bm25_leg() -> tuple[list[str], dict[str, RetrievedDocument], float]:
        bm25_index = get_bm25_index(namespace)
        t0 = time.perf_counter()
        hits = bm25_index.search(query, k=fetch_bm25, where=factory_where)
        ms = (time.perf_counter() - t0) * 1000
        ranked: list[str] = []
        by_id: dict[str, RetrievedDocument] = {}
        for doc_id, _score in hits:
            ranked.append(doc_id)
            got = bm25_index.get_content(doc_id)
            if got is None:
                continue
            content, meta = got
            by_id[doc_id] = RetrievedDocument(
                content=content,
                source=meta.get("source", ""),
                week=meta.get("week", ""),
                doc_type=meta.get("type", ""),
                image_path=meta.get("image_path"),
            )
        return ranked, by_id, ms

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid") as ex:
        fd = ex.submit(_dense_leg)
        fb = ex.submit(_bm25_leg)
        dense_ranked, dense_by_id, dense_ms = fd.result()
        bm25_ranked, bm25_by_id, bm25_ms = fb.result()

    by_id: dict[str, RetrievedDocument] = {**bm25_by_id, **dense_by_id}

    fused = reciprocal_rank_fusion(
        [dense_ranked, bm25_ranked],
        k=RRF_K,
        weights=[RRF_WEIGHT_DENSE, RRF_WEIGHT_BM25],
    )
    top: list[RetrievedDocument] = []
    for doc_id, _score in fused[:fused_k]:
        if doc_id in by_id:
            top.append(by_id[doc_id])
    log.info(
        "hybrid: dense=%d bm25=%d fused=%d (ns=%s) embed_ms=%.1f dense_ms=%.1f bm25_ms=%.1f",
        len(dense_ranked),
        len(bm25_ranked),
        len(top),
        namespace,
        embed_ms,
        dense_ms,
        bm25_ms,
    )
    if timings is not None:
        timings["embed_ms"] = round(embed_ms, 1)
        timings["dense_ms"] = round(dense_ms, 1)
        timings["bm25_ms"] = round(bm25_ms, 1)
    return top


def _chroma_filter_to_meta(f: dict) -> dict:
    """Flatten the intermediate ``{"$and": [...]}`` filter shape into the flat
    ``{key: value}`` form consumed by both ``bm25._meta_match`` and the
    factory's ``PgVectorStore`` JSONB where-builder.

    Supported inputs: leaf ``{"k": {"$eq": v}}`` or ``{"k": {"$in": [...]}}``
    and ``{"$and": [...]}`` of such leaves. Unknown ops are dropped.
    """
    out: dict = {}
    if "$and" in f:
        for cond in f["$and"]:
            out.update(_chroma_filter_to_meta(cond))
        return out
    for k, v in f.items():
        if isinstance(v, dict):
            if "$eq" in v:
                out[k] = v["$eq"]
            elif "$in" in v:
                out[k] = {"$in": v["$in"]}
    return out


# ── Retrieval functions ───────────────────────────────────────────────────────


def _hits_to_retrieved(hits: list[SearchHit]) -> list[RetrievedDocument]:
    return [_hit_to_retrieved(h) for h in hits]


def _maybe_rerank(
    query: str,
    docs: list[RetrievedDocument],
    top_k: int,
    use_reranker: bool,
    timings: dict | None = None,
) -> list[RetrievedDocument]:
    """Apply the cross-encoder reranker when enabled, else truncate to top_k."""
    if not use_reranker:
        return docs[:top_k]
    try:
        from src.agent.reranker import get_reranker

        reranker = get_reranker()
        t0 = time.perf_counter()
        out = reranker.rerank(query, docs, top_k=top_k)
        if timings is not None:
            timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
    except Exception as e:  # pragma: no cover — graceful fallback
        log.warning("Reranker unavailable (%s); falling back to candidate order", e)
        return docs[:top_k]


def retrieve_texts(
    query: str,
    week: str | None = None,
    k: int = 6,
    strategy: str | None = None,
    use_reranker: bool | None = None,
    namespace: str | None = None,
    timings: dict | None = None,
) -> list[RetrievedDocument]:
    """Retrieve text chunks (readings, tutorials, supplementary) from the KB.

    Excludes lecture slides so that text and image modalities remain separate
    in the state, allowing independent processing by downstream nodes.

    Args:
        strategy: Override the global RETRIEVAL_STRATEGY for ablation.
        use_reranker: Override the global USE_RERANKER (eval ablation hook).
            When True, MMR fetches ``RERANKER_FETCH_K_TEXT`` candidates and a
            cross-encoder reduces them to ``k``.
        namespace: User tenant id; when set, restricts to chunks tagged with
            ``metadata.namespace`` matching. ``None`` searches all chunks
            (legacy/shared collection).
    """
    doc_types = ["reading", "tutorial", "supplementary", "note"]
    where_filter = _build_filter(week=week, doc_types=doc_types, namespace=namespace)

    rerank_on = USE_RERANKER if use_reranker is None else bool(use_reranker)
    fetch_k = RERANKER_FETCH_K_TEXT if rerank_on else k

    leg_timings: dict[str, float] = {}
    t_total = time.perf_counter()
    if USE_HYBRID_RETRIEVAL:
        docs = _hybrid_search(
            query,
            where_filter=where_filter,
            fetch_dense=fetch_k,
            fetch_bm25=BM25_FETCH_K_TEXT,
            fused_k=HYBRID_FUSED_K_TEXT,
            namespace=namespace,
            timings=leg_timings,
        )
        log.info(
            "retrieve_texts [hybrid, rerank=%s]: %d fused (week=%s, ns=%s)",
            rerank_on,
            len(docs),
            week,
            namespace,
        )
    else:
        t_d = time.perf_counter()
        factory_where = _chroma_filter_to_meta(where_filter) if where_filter else {}
        qv = _get_embeddings().embed_query(query)
        hits = _dense_search(
            qv, k=fetch_k, factory_where=factory_where, namespace=namespace
        )
        leg_timings["dense_ms"] = round((time.perf_counter() - t_d) * 1000, 1)
        log.info(
            "retrieve_texts [%s, rerank=%s]: %d candidates (week=%s)",
            strategy or RETRIEVAL_STRATEGY,
            rerank_on,
            len(hits),
            week,
        )
        docs = _hits_to_retrieved(hits)
    out = _maybe_rerank(query, docs, top_k=k, use_reranker=rerank_on, timings=leg_timings)
    leg_timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)
    if timings is not None:
        for k_, v in leg_timings.items():
            timings[f"text_{k_}"] = v
    return out


def retrieve_slides(
    query: str,
    week: str | None = None,
    k: int = 4,
    strategy: str | None = None,
    use_reranker: bool | None = None,
    namespace: str | None = None,
    timings: dict | None = None,
) -> list[RetrievedDocument]:
    """Retrieve lecture slide descriptions from the KB.

    Returns slides whose VLM-generated descriptions are semantically similar
    to the query. Each result carries an image_path for reference.

    Args:
        strategy: Override the global RETRIEVAL_STRATEGY for ablation.
        use_reranker: Override the global USE_RERANKER (eval ablation hook).
        namespace: User tenant id (see ``retrieve_texts``).
    """
    where_filter = _build_filter(
        week=week, doc_types=["lecture_slide", "slide"], namespace=namespace
    )

    rerank_on = USE_RERANKER if use_reranker is None else bool(use_reranker)
    fetch_k = RERANKER_FETCH_K_SLIDES if rerank_on else k

    leg_timings: dict[str, float] = {}
    t_total = time.perf_counter()
    if USE_HYBRID_RETRIEVAL:
        docs = _hybrid_search(
            query,
            where_filter=where_filter,
            fetch_dense=fetch_k,
            fetch_bm25=BM25_FETCH_K_SLIDES,
            fused_k=HYBRID_FUSED_K_SLIDES,
            namespace=namespace,
            timings=leg_timings,
        )
        log.info(
            "retrieve_slides [hybrid, rerank=%s]: %d fused (week=%s, ns=%s)",
            rerank_on,
            len(docs),
            week,
            namespace,
        )
    else:
        t_d = time.perf_counter()
        factory_where = _chroma_filter_to_meta(where_filter) if where_filter else {}
        qv = _get_embeddings().embed_query(query)
        hits = _dense_search(
            qv, k=fetch_k, factory_where=factory_where, namespace=namespace
        )
        leg_timings["dense_ms"] = round((time.perf_counter() - t_d) * 1000, 1)
        log.info(
            "retrieve_slides [%s, rerank=%s]: %d candidates (week=%s)",
            strategy or RETRIEVAL_STRATEGY,
            rerank_on,
            len(hits),
            week,
        )
        docs = _hits_to_retrieved(hits)
    out = _maybe_rerank(query, docs, top_k=k, use_reranker=rerank_on, timings=leg_timings)
    leg_timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)
    if timings is not None:
        for k_, v in leg_timings.items():
            timings[f"slides_{k_}"] = v
    return out


def retrieve_all(
    query: str,
    week: str | None = None,
    k_text: int = 6,
    k_slides: int = 4,
    use_reranker: bool | None = None,
    namespace: str | None = None,
    timings: dict | None = None,
) -> tuple[list[RetrievedDocument], list[RetrievedDocument]]:
    """Retrieve text and slide chunks concurrently.

    The two legs are independent (different metadata filters, different
    candidate pools) and each performs blocking I/O (ZeroEntropy embed,
    vector store query, BM25 in-memory search, Cohere rerank REST call).
    Running them in a 2-worker thread pool halves the wall-clock spent in
    retrieval on the typical hot path. Inside each leg, dense and BM25 are
    parallelised again so the total retrieval graph is two levels deep.
    Embeddings are deduped by a thread-safe LRU on the ZeroEntropyEmbeddings
    singleton (see src/embeddings.py).
    """
    t_dict: dict[str, float] = {}
    s_dict: dict[str, float] = {}
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieve") as ex:
        ft = ex.submit(
            retrieve_texts,
            query,
            week=week,
            k=k_text,
            use_reranker=use_reranker,
            namespace=namespace,
            timings=t_dict,
        )
        fs = ex.submit(
            retrieve_slides,
            query,
            week=week,
            k=k_slides,
            use_reranker=use_reranker,
            namespace=namespace,
            timings=s_dict,
        )
        texts = ft.result()
        slides = fs.result()
    parallel_ms = round((time.perf_counter() - t_start) * 1000, 1)
    if timings is not None:
        timings.update(t_dict)
        timings.update(s_dict)
        timings["retrieve_all_parallel_ms"] = parallel_ms
    return texts, slides
