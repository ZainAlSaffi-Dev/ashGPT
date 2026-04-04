"""Centralised configuration for the ashGPT Property Law Exam Assistant."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

# ── Models ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "zembed-1"
EMBEDDING_DIMENSIONS = 2560
VLM_MODEL = "gemini-2.5-pro"
REASONING_MODEL = "gemini-2.5-pro"  # Swap independently for ablation

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "property_law_kb"

# ── Chunking (tuned for legal text — preserves full paragraphs of ratio) ──────
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# ── VLM prompt for lecture slide description ──────────────────────────────────
SLIDE_DESCRIPTION_PROMPT = (
    "You are a legal education assistant. Describe this property law lecture "
    "slide in detail. Include all visible text verbatim, any diagrams or "
    "flowcharts, legal principles, case names, statutory references, and "
    "definitions. Preserve the structure (headings, bullet points) as closely "
    "as possible."
)
