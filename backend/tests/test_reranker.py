"""Reranker selection + Cohere adapter (mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agent import reranker
from src.agent.state import RetrievedDocument


def test_get_reranker_picks_cohere_when_key_present(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "ck_test")
    from src.config import reload_settings

    reload_settings()
    reranker.reset_reranker()

    with patch("cohere.ClientV2") as mock_client:
        r = reranker.get_reranker()
        assert isinstance(r, reranker.CohereReranker)
        mock_client.assert_called_once_with(api_key="ck_test")


def test_get_reranker_falls_back_to_local(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    from src.config import reload_settings

    reload_settings()
    reranker.reset_reranker()
    r = reranker.get_reranker()
    assert isinstance(r, reranker.CrossEncoderReranker)


def test_cohere_rerank_returns_top_k_in_score_order(monkeypatch):
    docs: list[RetrievedDocument] = [
        {"content": "irrelevant filler", "source": "a", "week": "", "doc_type": "", "image_path": None},
        {"content": "adverse possession is the key topic", "source": "b", "week": "", "doc_type": "", "image_path": None},
        {"content": "another mention of adverse possession briefly", "source": "c", "week": "", "doc_type": "", "image_path": None},
    ]
    fake_results = SimpleNamespace(
        results=[
            SimpleNamespace(index=1, relevance_score=0.95),
            SimpleNamespace(index=2, relevance_score=0.55),
            SimpleNamespace(index=0, relevance_score=0.10),
        ]
    )
    rr = reranker.CohereReranker.__new__(reranker.CohereReranker)
    rr._client = MagicMock()
    rr._client.rerank.return_value = fake_results
    rr._model = "rerank-v3.5"

    out = rr.rerank("adverse possession", docs, top_k=2)
    assert [d["source"] for d in out] == ["b", "c"]


def test_cohere_rerank_handles_empty_docs():
    rr = reranker.CohereReranker.__new__(reranker.CohereReranker)
    rr._client = MagicMock()
    rr._model = "rerank-v3.5"
    assert rr.rerank("q", [], top_k=5) == []


def test_cohere_rerank_falls_back_on_api_failure():
    docs: list[RetrievedDocument] = [
        {"content": "x", "source": "a", "week": "", "doc_type": "", "image_path": None},
        {"content": "y", "source": "b", "week": "", "doc_type": "", "image_path": None},
    ]
    rr = reranker.CohereReranker.__new__(reranker.CohereReranker)
    rr._client = MagicMock()
    rr._client.rerank.side_effect = RuntimeError("api down")
    rr._model = "rerank-v3.5"

    # Falls back to original order, truncated.
    out = rr.rerank("q", docs, top_k=1)
    assert [d["source"] for d in out] == ["a"]
