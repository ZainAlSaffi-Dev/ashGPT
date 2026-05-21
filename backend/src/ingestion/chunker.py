"""Text chunker for legal-document RAG.

LangChain's ``RecursiveCharacterTextSplitter`` with the legal-tuned defaults
from ``src.config`` (1500 / 300) — preserves whole paragraphs of *ratio* and
keeps citation runs intact.

Example:
    >>> chunks = chunk_text("para one. " * 200, chunk_size=200, chunk_overlap=40)
    >>> all(len(c) <= 200 for c in chunks)
    True
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


@lru_cache(maxsize=16)
def _splitter(
    chunk_size: int,
    chunk_overlap: int,
    separators: tuple[str, ...] | None,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators) if separators else None,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: Sequence[str] | None = None,
) -> list[str]:
    """Recursive character splitter — defaults tuned for legal prose."""
    if not text.strip():
        return []
    return _splitter(
        chunk_size,
        chunk_overlap,
        tuple(separators) if separators else None,
    ).split_text(text)
