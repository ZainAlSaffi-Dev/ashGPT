"""Chunker tests."""

from __future__ import annotations

from src.ingestion.chunker import chunk_text


def test_short_text_returns_single_chunk():
    out = chunk_text("Adverse possession requires factual possession with intent to possess.")
    assert out == ["Adverse possession requires factual possession with intent to possess."]


def test_long_text_splits_into_overlapping_chunks():
    text = (
        "Adverse possession requires factual possession with intent to possess. "
        "The leading case is Williams v Greenway. The court held that the defendant "
        "had open and continuous use. " * 60
    )
    chunks = chunk_text(text, chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    # Overlap → adjacent chunks share some content.
    overlap_seen = any(chunks[i][-40:] in chunks[i + 1] for i in range(len(chunks) - 1))
    assert overlap_seen or len(chunks) == 1


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_custom_separators_respected():
    text = "para one is somewhat long\n---\npara two is also long\n---\npara three closes"
    out = chunk_text(text, chunk_size=30, chunk_overlap=0, separators=["\n---\n"])
    assert len(out) >= 2
    assert any("para one" in c for c in out)
    assert any("para three" in c for c in out)
