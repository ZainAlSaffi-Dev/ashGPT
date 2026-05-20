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

from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: Sequence[str] | None = None,
) -> list[str]:
    """Recursive character splitter — defaults tuned for legal prose."""
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators) if separators else None,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)
