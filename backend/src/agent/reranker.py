"""Cross-encoder reranker for retrieved documents.

The retrieval node uses MMR for diversity. MMR optimises for marginal
relevance, not raw query-document similarity, so the resulting top-k can
include chunks that are diverse but only weakly related to the question.

This module wraps a small pretrained cross encoder (``ms-marco-MiniLM-L-6-v2``)
that re-scores each (query, chunk) pair and returns the top-k by score.
The model is loaded lazily and cached as a module-level singleton — the
first call pays the load cost (~80 MB download on first run); subsequent
calls reuse the cached instance.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agent.state import RetrievedDocument
from src.config import RERANKER_MODEL

log = logging.getLogger(__name__)


_singleton: Optional["CrossEncoderReranker"] = None


def get_reranker() -> "CrossEncoderReranker":
    """Return the module-level cached reranker instance (lazy init)."""
    global _singleton
    if _singleton is None:
        _singleton = CrossEncoderReranker(model_name=RERANKER_MODEL)
    return _singleton


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
