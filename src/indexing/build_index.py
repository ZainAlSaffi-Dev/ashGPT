"""
Multimodal ingestion pipeline for the Property Law knowledge base.

Processes week-by-week folders containing:
  - PDF readings and tutorials  → text chunks embedded via text-embedding-3-large
  - JPEG lecture slides          → VLM-generated descriptions embedded alongside

All documents are upserted into a single ChromaDB collection with rich metadata
(week, type, source, image_path) to support filtered retrieval in later phases.

Usage:
    python -m src.indexing.build_index
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import chromadb
import fitz  # pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    SLIDE_DESCRIPTION_PROMPT,
    VLM_MODEL,
)
from src.embeddings import ZeroEntropyEmbeddings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _week_sort_key(path: Path) -> int:
    """Extract numeric week index for sorting (e.g. 'week_3' → 3)."""
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _sanitise_id(raw: str) -> str:
    """Create a deterministic, ChromaDB-safe document ID."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", raw)


def _classify_pdf(filename: str) -> str:
    """Classify a PDF as 'reading', 'tutorial', or 'supplementary' by filename."""
    lower = filename.lower()
    if "reading" in lower:
        return "reading"
    if "tutorial" in lower:
        return "tutorial"
    return "supplementary"


# ── PDF Text Extraction ───────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


# ── VLM Slide Description ─────────────────────────────────────────────────────


MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def describe_slide(image_path: Path, client: genai.Client) -> str:
    """Send a lecture slide image to the Gemini VLM and return its description."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime = MIME_TYPES.get(image_path.suffix.lower(), "image/png")
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
    response = client.models.generate_content(
        model=VLM_MODEL,
        contents=[SLIDE_DESCRIPTION_PROMPT, image_part],
    )
    return response.text


# ── Ingestion Routines ─────────────────────────────────────────────────────────


def _ingest_pdfs(
    pdf_dir: Path,
    week: str,
    doc_type: str,
    splitter: RecursiveCharacterTextSplitter,
    collection: Chroma,
) -> int:
    """Chunk and upsert all PDFs from a directory. Returns count of chunks added."""
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        return 0

    total_chunks = 0
    for pdf_path in pdfs:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            log.warning("Skipping empty PDF: %s", pdf_path.name)
            continue

        chunks = splitter.split_text(text)
        ids = [
            _sanitise_id(f"{week}_{doc_type}_{pdf_path.stem}_chunk_{i}")
            for i in range(len(chunks))
        ]
        metadatas = [
            {
                "week": week,
                "type": doc_type,
                "source": pdf_path.name,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        collection.add_texts(texts=chunks, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)
        log.info(
            "  ├─ %s: %d chunks indexed", pdf_path.name, len(chunks)
        )

    return total_chunks


def _ingest_slides(
    slide_dir: Path,
    week: str,
    collection: Chroma,
    client: genai.Client,
) -> int:
    """Describe and upsert all slide images from a directory. Returns count added."""
    slides = sorted(
        f for f in slide_dir.iterdir()
        if f.suffix.lower() in MIME_TYPES
    )
    if not slides:
        return 0

    count = 0
    for slide_path in slides:
        try:
            description = describe_slide(slide_path, client)
        except Exception:
            log.exception("VLM failed for %s — skipping", slide_path.name)
            continue

        if not description or not description.strip():
            log.warning("  ├─ %s: empty VLM description — skipping", slide_path.name)
            continue

        doc_id = _sanitise_id(f"{week}_lecture_slide_{slide_path.stem}")
        metadata = {
            "week": week,
            "type": "lecture_slide",
            "source": slide_path.name,
            "image_path": str(slide_path),
        }

        try:
            collection.add_texts(texts=[description], ids=[doc_id], metadatas=[metadata])
        except Exception:
            log.exception(
                "  ├─ %s: embedding failed (desc length=%d) — skipping",
                slide_path.name,
                len(description),
            )
            continue

        count += 1
        log.info("  ├─ %s: slide description indexed", slide_path.name)

    return count


# ── Main Pipeline ──────────────────────────────────────────────────────────────


def build_index() -> None:
    """Run the full multimodal ingestion pipeline."""
    client = genai.Client()  # reads GOOGLE_API_KEY from env

    week_dirs = sorted(DATA_DIR.iterdir(), key=_week_sort_key)
    week_dirs = [d for d in week_dirs if d.is_dir() and d.name.startswith("week")]
    if not week_dirs:
        log.error("No week_* directories found in %s", DATA_DIR)
        sys.exit(1)

    log.info("Found %d week(s) to index: %s", len(week_dirs), [d.name for d in week_dirs])

    embeddings = ZeroEntropyEmbeddings()
    persistent_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = Chroma(
        client=persistent_client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total_text_chunks = 0
    total_slides = 0

    # ── Index any root-level supplementary PDFs (e.g. data/*.pdf) ──────────
    root_pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if root_pdfs:
        log.info("─── Processing root-level PDFs ───")
        for pdf_path in root_pdfs:
            doc_type = _classify_pdf(pdf_path.name)
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                log.warning("  ├─ Skipping empty PDF: %s", pdf_path.name)
                continue
            chunks = splitter.split_text(text)
            ids = [
                _sanitise_id(f"root_{doc_type}_{pdf_path.stem}_chunk_{i}")
                for i in range(len(chunks))
            ]
            metadatas = [
                {
                    "week": "all",
                    "type": doc_type,
                    "source": pdf_path.name,
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]
            collection.add_texts(texts=chunks, ids=ids, metadatas=metadatas)
            total_text_chunks += len(chunks)
            log.info("  ├─ %s: %d chunks indexed", pdf_path.name, len(chunks))

    # ── Index each week ────────────────────────────────────────────────────
    for week_dir in week_dirs:
        week = week_dir.name
        log.info("─── Processing %s ───", week)

        # PDFs in subdirectories (readings/, tutorial/)
        for subdir_name, doc_type in [("readings", "reading"), ("tutorial", "tutorial")]:
            subdir = week_dir / subdir_name
            if subdir.is_dir():
                n = _ingest_pdfs(subdir, week, doc_type, splitter, collection)
                total_text_chunks += n

        # PDFs directly in the week directory (flat layout)
        week_pdfs = sorted(week_dir.glob("*.pdf"))
        for pdf_path in week_pdfs:
            doc_type = _classify_pdf(pdf_path.name)
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                log.warning("  ├─ Skipping empty PDF: %s", pdf_path.name)
                continue
            chunks = splitter.split_text(text)
            ids = [
                _sanitise_id(f"{week}_{doc_type}_{pdf_path.stem}_chunk_{i}")
                for i in range(len(chunks))
            ]
            metadatas = [
                {
                    "week": week,
                    "type": doc_type,
                    "source": pdf_path.name,
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]
            collection.add_texts(texts=chunks, ids=ids, metadatas=metadatas)
            total_text_chunks += len(chunks)
            log.info("  ├─ %s: %d chunks indexed", pdf_path.name, len(chunks))

        # Slide images in lecture/
        lecture_dir = week_dir / "lecture"
        if lecture_dir.is_dir():
            n = _ingest_slides(lecture_dir, week, collection, client)
            total_slides += n

    log.info("═══ Indexing complete ═══")
    log.info("  Text chunks indexed: %d", total_text_chunks)
    log.info("  Slide descriptions indexed: %d", total_slides)
    log.info("  ChromaDB persisted to: %s", CHROMA_DIR.resolve())


if __name__ == "__main__":
    build_index()
