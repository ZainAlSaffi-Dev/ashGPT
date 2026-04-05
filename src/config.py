"""Centralised configuration for the ashGPT Property Law Exam Assistant."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "zembed-1"
EMBEDDING_DIMENSIONS = 2560

# ── Per-Node Model Assignments ─────────────────────────────────────────────────
# Prefix determines provider: "gemini-" → Google, "gpt-" → OpenAI, "claude-" → Anthropic
VLM_MODEL = "gemini-2.5-pro"                # Slide description during indexing

ROUTER_MODEL = "gemini-3.1-flash-lite-preview"                  # Lightweight: intent classification
RATIO_EXTRACTOR_MODEL = "gpt-5.3-chat-latest"     # Mid-tier: IRAC extraction
CHRONOLOGY_MODEL = "gemini-3-flash-preview"              # Lightweight: Mermaid generation
SYNTHESIS_MODEL = "gpt-5.4-mini"                 # Strong: final grounded answer

# ── Evaluation ─────────────────────────────────────────────────────────────────
BASELINE_MODEL = "gpt-5.4-mini"             # Plain LLM baseline (no retrieval)

JUDGE_DRAFT_MODEL = "gemini-3-flash-preview"  # Stage 1: initial judgment (different provider to avoid bias)
JUDGE_CRITIQUE_MODEL = "gpt-5.4"            # Stage 2: critiques and finalises score

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "property_law_kb"

# ── Chunking (tuned for legal text — preserves full paragraphs of ratio) ──────
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_STRATEGY = "mmr"
MMR_LAMBDA = 0.5
MMR_FETCH_K = 20

# ── VLM prompt for lecture slide description ──────────────────────────────────
SLIDE_DESCRIPTION_PROMPT = (
    "You are a legal education assistant. Describe this property law lecture "
    "slide in detail. Include all visible text verbatim, any diagrams or "
    "flowcharts, legal principles, case names, statutory references, and "
    "definitions. Preserve the structure (headings, bullet points) as closely "
    "as possible."
)
