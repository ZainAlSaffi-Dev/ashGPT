"""Rerankers for retrieved documents.

Two backends share one ``Reranker`` interface:

  * ``CohereReranker``        — Cohere Rerank v3 / v3.5 REST API. Default in
                                prod (better quality than the local cross-
                                encoder, no torch in the container). Set
                                ``COHERE_API_KEY`` to enable.
  * ``CrossEncoderReranker``  — local ``ms-marco-MiniLM-L-6-v2`` via
                                sentence-transformers (fallback / offline).

The hybrid retrieval (Phase 1.5) supplies a fused candidate pool; this stage
re-scores against the query and keeps the top-k. With BM25+dense+rerank we
get the most precise top results the eval suite has seen.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

from src.agent.state import RetrievedDocument
from src.config import RERANKER_MODEL, get_settings

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]: ...


_singleton: Optional["Reranker"] = None


def get_reranker() -> Reranker:
    """Return the configured singleton reranker (Cohere if key set, else local)."""
    global _singleton
    if _singleton is None:
        s = get_settings()
        if s.cohere_api_key:
            _singleton = CohereReranker(api_key=s.cohere_api_key, model=s.cohere_rerank_model)
            log.info("Reranker: Cohere %s", s.cohere_rerank_model)
        else:
            _singleton = CrossEncoderReranker(model_name=RERANKER_MODEL)
            log.info("Reranker: local %s", RERANKER_MODEL)
    return _singleton


def reset_reranker() -> None:
    """Drop the cached reranker — used by tests when toggling backends."""
    global _singleton
    _singleton = None


class CrossEncoderReranker:
    """Lazy cross-encoder wrapper.

    The underlying ``CrossEncoder`` model is only constructed on the first
    call to :meth:`rerank`, so importing this module is cheap.
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install it with `pip install sentence-transformers>=3.0.0`."
                ) from e
            log.info("Loading cross-encoder reranker: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """Score each document against the query, return top_k by score.

        Args:
            query: The user's question (or packed retrieval query).
            documents: RetrievedDocument dicts from the MMR step.
            top_k: How many documents to keep (unchanged behaviour upstream).

        Returns:
            documents sorted by cross-encoder score, truncated to top_k.
            Empty input → empty output.
        """
        if not documents:
            return []
        if top_k <= 0:
            return []

        self._ensure_loaded()

        pairs: list[tuple[str, str]] = [
            (query, doc.get("content", "")) for doc in documents
        ]

        try:
            scores = self._model.predict(pairs)
        except Exception as e:  # pragma: no cover — same defensive shape as llm.py
            log.warning("Reranker failed (%s); returning original order", e)
            return documents[:top_k]

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        return [doc for doc, _ in scored[:top_k]]


class CohereReranker:
    """Cohere Rerank API. https://docs.cohere.com/reference/rerank"""

    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        import cohere

        # v6 SDK exposes ClientV2; legacy Client still works but emits a
        # deprecation. Stick with v2 for forward-compat.
        self._client = cohere.ClientV2(api_key=api_key)
        self._model = model

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        if not documents or top_k <= 0:
            return []
        passages = [d.get("content", "") for d in documents]
        try:
            res = self._client.rerank(
                model=self._model,
                query=query,
                documents=passages,
                top_n=min(top_k, len(passages)),
            )
        except Exception as e:  # pragma: no cover
            log.warning("Cohere rerank failed (%s); returning original order", e)
            return documents[:top_k]
        out: list[RetrievedDocument] = []
        for item in res.results:
            idx = getattr(item, "index", None)
            if idx is None or idx >= len(documents):
                continue
            out.append(documents[idx])
        return out[:top_k]
