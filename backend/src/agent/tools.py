"""Retrieval tools for querying the Property Law ChromaDB knowledge base.

These functions provide the 'senses' of the agent — they reach into the
pre-indexed vector store and return relevant text chunks and slide
descriptions, optionally filtered by week or document type.

Two retrieval modes:

  * ``USE_HYBRID_RETRIEVAL=True`` (default) — runs BM25 + dense in parallel
    and fuses with Reciprocal Rank Fusion before reranking. Higher precision
    on legal text (case names, statutory refs benefit from BM25 term-match).
  * ``USE_HYBRID_RETRIEVAL=False`` — legacy dense-only MMR path.
"""

from __future__ import annotations

import hashlib
import logging
import time

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma

from src.agent.bm25 import (
    configure_bm25_source,
    get_bm25_index,
    reciprocal_rank_fusion,
)
from src.agent.state import RetrievedDocument
from src.config import (
    BM25_FETCH_K_SLIDES,
    BM25_FETCH_K_TEXT,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    HYBRID_FUSED_K_SLIDES,
    HYBRID_FUSED_K_TEXT,
    MMR_FETCH_K,
    MMR_LAMBDA,
    RERANKER_FETCH_K_SLIDES,
    RERANKER_FETCH_K_TEXT,
    RETRIEVAL_STRATEGY,
    RRF_K,
    USE_HYBRID_RETRIEVAL,
    USE_RERANKER,
)
from src.embeddings import ZeroEntropyEmbeddings

load_dotenv()

log = logging.getLogger(__name__)

# ── Singleton vector store ─────────────────────────────────────────────────────

_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    """Lazily initialise and cache the ChromaDB vector store."""
    global _vectorstore
    if _vectorstore is None:
        persistent_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _vectorstore = Chroma(
            client=persistent_client,
            collection_name=CHROMA_COLLECTION,
            embedding_function=ZeroEntropyEmbeddings(),
        )
        log.info("ChromaDB vector store loaded from %s", CHROMA_DIR)
    return _vectorstore


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
    """Pull all chunks for ``namespace`` from Chroma → BM25 corpus rows."""
    persistent_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = persistent_client.get_or_create_collection(CHROMA_COLLECTION)
    where = {"namespace": {"$eq": namespace}} if namespace else None
    try:
        data = collection.get(where=where, include=["documents", "metadatas"])
    except Exception:
        # New empty namespace.
        return []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    rows: list[tuple[str, str, dict]] = []
    for content, meta in zip(docs, metas):
        meta = meta or {}
        rid = meta.get("chunk_id") or _doc_id(content, meta.get("source", ""))
        rows.append((rid, content, meta))
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


def _search(
    store: Chroma,
    query: str,
    k: int,
    where_filter: dict | None,
    strategy: str | None = None,
) -> list:
    """Run either similarity or MMR search based on the configured strategy."""
    strat = strategy or RETRIEVAL_STRATEGY

    if strat == "mmr":
        return store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=MMR_FETCH_K,
            lambda_mult=MMR_LAMBDA,
            filter=where_filter,
        )
    return store.similarity_search(query, k=k, filter=where_filter)


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
    """Dense MMR + BM25 fused via RRF, returning the top-``fused_k`` docs.

    Both legs use the same ``_doc_id`` scheme so RRF can align them. Metadata
    (source, week, doc_type, image_path) comes from whichever leg saw the doc
    first — content from the same lookup.
    """
    _ensure_bm25_configured()

    # Dense leg (MMR over a larger pool — RRF benefits from candidate breadth).
    store = _get_vectorstore()
    t0 = time.perf_counter()
    dense_results = store.max_marginal_relevance_search(
        query,
        k=fetch_dense,
        fetch_k=max(MMR_FETCH_K, fetch_dense * 2),
        lambda_mult=MMR_LAMBDA,
        filter=where_filter,
    )
    dense_ms = (time.perf_counter() - t0) * 1000

    by_id: dict[str, RetrievedDocument] = {}
    dense_ranked: list[str] = []
    for doc in dense_results:
        meta = doc.metadata or {}
        rid = meta.get("chunk_id") or _doc_id(doc.page_content, meta.get("source", ""))
        dense_ranked.append(rid)
        if rid not in by_id:
            by_id[rid] = RetrievedDocument(
                content=doc.page_content,
                source=meta.get("source", ""),
                week=meta.get("week", ""),
                doc_type=meta.get("type", ""),
                image_path=meta.get("image_path"),
            )

    # BM25 leg (lexical) — uses same metadata filter dialect translated to
    # the in-memory matcher (week, doc_type, namespace).
    bm25_where: dict = {}
    if where_filter:
        bm25_where = _chroma_filter_to_meta(where_filter)
    bm25_index = get_bm25_index(namespace)
    t1 = time.perf_counter()
    bm25_hits = bm25_index.search(query, k=fetch_bm25, where=bm25_where)
    bm25_ms = (time.perf_counter() - t1) * 1000
    bm25_ranked: list[str] = []
    for doc_id, _score in bm25_hits:
        bm25_ranked.append(doc_id)
        if doc_id in by_id:
            continue
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

    fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=RRF_K)
    top: list[RetrievedDocument] = []
    for doc_id, _score in fused[:fused_k]:
        if doc_id in by_id:
            top.append(by_id[doc_id])
    log.info(
        "hybrid: dense=%d bm25=%d fused=%d (ns=%s) dense_ms=%.1f bm25_ms=%.1f",
        len(dense_ranked),
        len(bm25_ranked),
        len(top),
        namespace,
        dense_ms,
        bm25_ms,
    )
    if timings is not None:
        timings["dense_ms"] = round(dense_ms, 1)
        timings["bm25_ms"] = round(bm25_ms, 1)
    return top


def _chroma_filter_to_meta(f: dict) -> dict:
    """Flatten the Chroma where-filter dict into a flat ``{key: value}`` form
    that ``bm25._meta_match`` understands.

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


def _raw_to_retrieved(results: list) -> list[RetrievedDocument]:
    """Convert LangChain Document objects to our RetrievedDocument dicts."""
    docs: list[RetrievedDocument] = []
    for doc in results:
        meta = doc.metadata
        docs.append(
            RetrievedDocument(
                content=doc.page_content,
                source=meta.get("source", ""),
                week=meta.get("week", ""),
                doc_type=meta.get("type", ""),
                image_path=meta.get("image_path"),
            )
        )
    return docs


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
        log.warning("Reranker unavailable (%s); falling back to MMR order", e)
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
        store = _get_vectorstore()
        results = _search(store, query, k=fetch_k, where_filter=where_filter, strategy=strategy)
        leg_timings["dense_ms"] = round((time.perf_counter() - t_d) * 1000, 1)
        log.info(
            "retrieve_texts [%s, rerank=%s]: %d candidates (week=%s)",
            strategy or RETRIEVAL_STRATEGY,
            rerank_on,
            len(results),
            week,
        )
        docs = _raw_to_retrieved(results)
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
        store = _get_vectorstore()
        results = _search(store, query, k=fetch_k, where_filter=where_filter, strategy=strategy)
        leg_timings["dense_ms"] = round((time.perf_counter() - t_d) * 1000, 1)
        log.info(
            "retrieve_slides [%s, rerank=%s]: %d candidates (week=%s)",
            strategy or RETRIEVAL_STRATEGY,
            rerank_on,
            len(results),
            week,
        )
        docs = _raw_to_retrieved(results)
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
    """Convenience wrapper that retrieves both text and slides in one call."""
    texts = retrieve_texts(
        query,
        week=week,
        k=k_text,
        use_reranker=use_reranker,
        namespace=namespace,
        timings=timings,
    )
    slides = retrieve_slides(
        query,
        week=week,
        k=k_slides,
        use_reranker=use_reranker,
        namespace=namespace,
        timings=timings,
    )
    return texts, slides
