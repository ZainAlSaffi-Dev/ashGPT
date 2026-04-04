"""Custom LangChain embedding wrapper for ZeroEntropy's zembed-1 model.

API reference: https://docs.zeroentropy.dev/api-reference/models/embed
"""

from __future__ import annotations

import os
from typing import List

import requests
from langchain_core.embeddings import Embeddings

from src.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

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
        """Embed a single query string (used during retrieval)."""
        return self._embed([text], input_type="query")[0]
