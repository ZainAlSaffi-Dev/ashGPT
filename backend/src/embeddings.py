"""Custom LangChain embedding wrapper for ZeroEntropy's zembed-1 model.

API reference: https://docs.zeroentropy.dev/api-reference/models/embed
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import List

import requests
from langchain_core.embeddings import Embeddings

from src.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

# Hybrid retrieval calls embed_query twice per turn (once for the text leg,
# once for the slides leg) with the same query string. Holding a tiny LRU on
# the embeddings instance collapses those into one ZeroEntropy round-trip.
_QUERY_CACHE_MAX = 16

ZEROENTROPY_EMBED_URL = "https://api.zeroentropy.dev/v1/models/embed"


class ZeroEntropyEmbeddings(Embeddings):
    """LangChain-compatible wrapper around ZeroEntropy's embedding API.

    Distinguishes between document and query input types, which allows the
    model to optimise embeddings for asymmetric retrieval (short queries
    matched against longer document passages).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
    ) -> None:
        self.api_key = api_key or os.getenv("ZEMBED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ZEMBED_API_KEY not found. Set it in .env or pass api_key directly."
            )
        self.model = model
        self.dimensions = dimensions
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._query_cache: "OrderedDict[tuple[str, int], list[float]]" = OrderedDict()

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Call the ZeroEntropy embed endpoint."""
        payload = {
            "model": self.model,
            "input_type": input_type,
            "input": texts,
            "dimensions": self.dimensions,
        }
        resp = requests.post(
            ZEROENTROPY_EMBED_URL,
            json=payload,
            headers=self._headers,
            timeout=120,
        )
        if not resp.ok:
            raise RuntimeError(
                f"ZeroEntropy API error {resp.status_code}: {resp.text}"
            )
        results = resp.json()["results"]
        return [r["embedding"] for r in results]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts (used during indexing)."""
        if not texts:
            return []
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (used during retrieval).

        Memoised via a small per-instance LRU so the dense leg of hybrid
        retrieval doesn't pay ZeroEntropy twice when ``retrieve_texts`` and
        ``retrieve_slides`` are called back-to-back with the same query.
        """
        key = (text, self.dimensions)
        cached = self._query_cache.get(key)
        if cached is not None:
            self._query_cache.move_to_end(key)
            return cached
        vec = self._embed([text], input_type="query")[0]
        self._query_cache[key] = vec
        if len(self._query_cache) > _QUERY_CACHE_MAX:
            self._query_cache.popitem(last=False)
        return vec
