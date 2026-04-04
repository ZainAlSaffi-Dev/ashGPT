"""Retrieval tools for querying the Property Law ChromaDB knowledge base.

These functions provide the 'senses' of the agent — they reach into the
pre-indexed vector store and return relevant text chunks and slide
descriptions, optionally filtered by week or document type.
"""

from __future__ import annotations

import logging

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma

from src.agent.state import RetrievedDocument
from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    MMR_FETCH_K,
    MMR_LAMBDA,
    RETRIEVAL_STRATEGY,
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
) -> dict | None:
    """Build a ChromaDB where-filter from optional week and type constraints."""
    conditions: list[dict] = []

    if week:
        conditions.append({"week": {"$eq": week}})

    if doc_types and len(doc_types) == 1:
        conditions.append({"type": {"$eq": doc_types[0]}})
    elif doc_types and len(doc_types) > 1:
        conditions.append({"type": {"$in": doc_types}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


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


def retrieve_texts(
    query: str,
    week: str | None = None,
    k: int = 6,
    strategy: str | None = None,
) -> list[RetrievedDocument]:
    """Retrieve text chunks (readings, tutorials, supplementary) from the KB.

    Excludes lecture slides so that text and image modalities remain separate
    in the state, allowing independent processing by downstream nodes.

    Args:
        strategy: Override the global RETRIEVAL_STRATEGY for ablation.
    """
    store = _get_vectorstore()
    doc_types = ["reading", "tutorial", "supplementary"]
    where_filter = _build_filter(week=week, doc_types=doc_types)

    results = _search(store, query, k=k, where_filter=where_filter, strategy=strategy)
    log.info("retrieve_texts [%s]: %d results (week=%s)", strategy or RETRIEVAL_STRATEGY, len(results), week)
    return _raw_to_retrieved(results)


def retrieve_slides(
    query: str,
    week: str | None = None,
    k: int = 4,
    strategy: str | None = None,
) -> list[RetrievedDocument]:
    """Retrieve lecture slide descriptions from the KB.

    Returns slides whose VLM-generated descriptions are semantically similar
    to the query. Each result carries an image_path for reference.

    Args:
        strategy: Override the global RETRIEVAL_STRATEGY for ablation.
    """
    store = _get_vectorstore()
    where_filter = _build_filter(week=week, doc_types=["lecture_slide"])

    results = _search(store, query, k=k, where_filter=where_filter, strategy=strategy)
    log.info("retrieve_slides [%s]: %d results (week=%s)", strategy or RETRIEVAL_STRATEGY, len(results), week)
    return _raw_to_retrieved(results)


def retrieve_all(
    query: str,
    week: str | None = None,
    k_text: int = 6,
    k_slides: int = 4,
) -> tuple[list[RetrievedDocument], list[RetrievedDocument]]:
    """Convenience wrapper that retrieves both text and slides in one call."""
    texts = retrieve_texts(query, week=week, k=k_text)
    slides = retrieve_slides(query, week=week, k=k_slides)
    return texts, slides
