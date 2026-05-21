"""Centralised configuration for LawGPT.

Two layers:
  1. Pipeline constants (model IDs, chunk sizes, retrieval params) — module-level.
  2. Deployment settings (DB URL, storage backend, Clerk keys) — env-driven via
     pydantic-settings ``Settings`` class. Imported lazily so tests/CLIs that
     touch only pipeline constants do not pay env-parsing cost.
"""

import os
from functools import lru_cache
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / "data"  # legacy eval / coursework artefacts only

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "zembed-1"
EMBEDDING_DIMENSIONS = 2560

# ── Per-Node Model Assignments ─────────────────────────────────────────────────
# Prefix determines provider: "gemini-" → Google, "gpt-" → OpenAI
VLM_MODEL = "gemini-2.5-pro"                # Slide description during indexing

ROUTER_MODEL = "gemini-3.1-flash-lite"                  # Lightweight: intent classification
RATIO_EXTRACTOR_MODEL = "gpt-5.5"     # Mid-tier: IRAC extraction
CHRONOLOGY_MODEL = "gemini-3-flash-preview"              # Lightweight: Mermaid generation
SYNTHESIS_MODEL = "gpt-5.5"                 # Strong: final grounded answer

# ── Evaluation ─────────────────────────────────────────────────────────────────
BASELINE_MODEL = "gpt-5.4-mini"             # Plain LLM baseline (no retrieval)

JUDGE_DRAFT_MODEL = "gemini-3-flash-preview"  # Stage 1: initial judgment (different provider to avoid bias)
JUDGE_CRITIQUE_MODEL = "gpt-5.4"            # Stage 2: critiques and finalises score

# ── Chunking (tuned for legal text — preserves full paragraphs of ratio) ──────
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_STRATEGY = "mmr"
MMR_LAMBDA = 0.5
MMR_FETCH_K = 20

# ── Hybrid retrieval (BM25 + dense fused via Reciprocal Rank Fusion) ──────────
# When enabled, retrieval runs two legs (lexical BM25 + dense MMR) and fuses
# their ranked lists via RRF before reranking. Boosts precision on legal text
# where exact case names + statutory references benefit from BM25's term-match,
# while semantic queries still surface via dense.
USE_HYBRID_RETRIEVAL = True
BM25_FETCH_K_TEXT = 30        # BM25 returns top-K candidates per leg
BM25_FETCH_K_SLIDES = 12
RRF_K = 60                    # RRF constant; 60 is the canonical default
HYBRID_FUSED_K_TEXT = 16      # RRF reduces to this many before reranking
HYBRID_FUSED_K_SLIDES = 8
# Dense weighted higher than BM25: eval (n=10) showed dense+rerank beat 1:1
# hybrid on precision@k. BM25 still carries signal for case names / section
# refs (now preserved as single tokens) so its weight stays non-trivial.
RRF_WEIGHT_DENSE = 0.7
RRF_WEIGHT_BM25 = 0.3

# ── Cross-encoder reranker (post-MMR / post-fusion) ───────────────────────────
# When enabled, MMR over-fetches RERANKER_FETCH_K_* candidates and the cross
# encoder reduces to the original retrieval k. Diversity (MMR) is preserved as
# the candidate pool; precision is lifted by re-scoring against the query.
USE_RERANKER = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_FETCH_K_TEXT = 16     # MMR returns 16 text chunks → rerank to 8
RERANKER_FETCH_K_SLIDES = 8    # MMR returns 8 slide chunks  → rerank to 4

# ── Verification node (fact-check cited cases against retrieved sources) ──────
USE_VERIFICATION = True

# ── Confidence-gated escalation (Phase 5) ─────────────────────────────────────
# After verification, compute a confidence score in [0,1] from the claim
# support rate. If it falls below LOW_CONFIDENCE_THRESHOLD and the request was
# served by the cheaper SYNTHESIS_MODEL, automatically re-run synthesis with
# ESCALATION_MODEL (stronger). Burns one extra paid call only when needed.
USE_CONFIDENCE_ESCALATION = True
LOW_CONFIDENCE_THRESHOLD = 0.7
ESCALATION_MODEL = "gpt-5.4"  # Stronger than the default mini synthesis

# ── Semantic answer cache (Phase 5) ───────────────────────────────────────────
# Cache (user_id, normalised query, sorted chunk-id set) → final answer for 7d.
USE_ANSWER_CACHE = True
ANSWER_CACHE_TTL_DAYS = 7

# ── Multi-turn chat (history + follow-up rewriter) ────────────────────────────
CHAT_HISTORY_MAX_MESSAGES = 24
CHAT_HISTORY_MAX_CHARS_PER_MESSAGE = 3500
CHAT_HISTORY_MAX_ASSISTANT_TAIL_CHARS = 1200
CHAT_MEMORY_MAX_SUMMARY_CHARS = 1800
CHAT_MEMORY_MAX_FACTS = 24
CHAT_MEMORY_MAX_FACT_CHARS = 260

# Coreference-resolved retrieval query. When enabled and ``chat_history`` is
# non-empty, retrieval_node calls a small router-tier model to expand
# pronouns / abbreviated case names into a self-contained search query
# before embedding. Adds one LLM hop (~200-400ms) — gate on follow-ups only,
# never on first turns.
USE_QUERY_REWRITER = True

# ── Eval: optional deep retrieval pool (recall-vs-pool); larger = more judge API calls
EVAL_RETRIEVAL_POOL_K_TEXT = 20
EVAL_RETRIEVAL_POOL_K_SLIDES = 10

# ── VLM prompt for image / slide / diagram description ───────────────────────
SLIDE_DESCRIPTION_PROMPT = (
    "You are a legal study assistant. Describe this image in detail. Include "
    "all visible text verbatim, any diagrams, flowcharts, or tables, legal "
    "principles, case names, statutory references, and definitions. Preserve "
    "the structure (headings, bullet points) as closely as possible so the "
    "description can be searched as if it were the original document."
)


# ── Deployment settings (env-driven) ──────────────────────────────────────────
# Loaded lazily via ``get_settings()`` so unit tests that only touch pipeline
# constants do not require env vars.


class _DefaultSettings:
    """Plain attribute container so tests can monkeypatch without a real env."""

    # Storage backend selection: "local" (pgvector + filesystem + sqlite) or
    # "cloudflare" (Vectorize + R2 + D1).
    storage_backend: str = "local"

    # SQL database URL. Examples:
    #   sqlite+aiosqlite:///./lawgpt.db        (local dev, no docker)
    #   postgresql+psycopg://user:pass@host/db (local docker compose)
    database_url: str = "sqlite+aiosqlite:///./lawgpt.db"

    # Vector store: "pgvector" (Postgres + pgvector, dev + prod default),
    # "vectorize" (Cloudflare Vectorize REST API), or "memory" (tests).
    vector_backend: str = "pgvector"
    vectorize_index_name: str = "lawgpt-vectors"
    vectorize_account_id: str = ""
    vectorize_api_token: str = ""

    # Blob storage: "local" (filesystem) or "r2".
    blob_backend: str = "local"
    blob_local_root: str = str(BACKEND_ROOT / "uploads")
    r2_bucket: str = "lawgpt-uploads"
    r2_account_id: str = ""
    r2_access_key: str = ""
    r2_secret_key: str = ""
    r2_endpoint_url: str = ""  # https://<account>.r2.cloudflarestorage.com

    # Clerk auth
    clerk_issuer: str = ""
    clerk_publishable_key: str = ""
    clerk_secret_key: str = ""
    clerk_jwks_url: str = ""  # derived from issuer if empty

    # Dev-mode bypass: when set, treat header `X-Dev-User: <uid>` as
    # authenticated. Used in local docker compose and tests.
    dev_auth_user: str | None = None

    # CORS origins (comma-separated list in env, list at runtime)
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Cohere (Phase 5 reranker swap)
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v3.5"


def _load_settings() -> _DefaultSettings:
    s = _DefaultSettings()
    # Override from environment; tolerant of missing values.
    for key in (
        "storage_backend",
        "database_url",
        "vector_backend",
        "vectorize_index_name",
        "vectorize_account_id",
        "vectorize_api_token",
        "blob_backend",
        "blob_local_root",
        "r2_bucket",
        "r2_account_id",
        "r2_access_key",
        "r2_secret_key",
        "r2_endpoint_url",
        "clerk_issuer",
        "clerk_publishable_key",
        "clerk_secret_key",
        "clerk_jwks_url",
        "dev_auth_user",
        "cohere_api_key",
        "cohere_rerank_model",
    ):
        env_val = os.environ.get(key.upper())
        if env_val is not None and env_val != "":
            setattr(s, key, env_val)
    cors = os.environ.get("CORS_ORIGINS")
    if cors:
        s.cors_origins = [o.strip() for o in cors.split(",") if o.strip()]

    # R2 / S3 credential names — the Worker forwards the S3-standard names
    # (``R2_ACCESS_KEY_ID`` / ``R2_SECRET_ACCESS_KEY``) but the legacy
    # backend settings use ``R2_ACCESS_KEY`` / ``R2_SECRET_KEY``. Accept
    # both so the container can authenticate against R2 regardless of
    # which scheme the operator put in ``wrangler secret put``.
    if not s.r2_access_key:
        s.r2_access_key = os.environ.get("R2_ACCESS_KEY_ID", "") or s.r2_access_key
    if not s.r2_secret_key:
        s.r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "") or s.r2_secret_key
    # Synthesise the endpoint URL from the account id when only the account
    # id was supplied (CF Containers don't ship the full URL by default).
    if not s.r2_endpoint_url and s.r2_account_id:
        s.r2_endpoint_url = f"https://{s.r2_account_id}.r2.cloudflarestorage.com"

    if not s.clerk_jwks_url and s.clerk_issuer:
        s.clerk_jwks_url = s.clerk_issuer.rstrip("/") + "/.well-known/jwks.json"
    return s


@lru_cache(maxsize=1)
def get_settings() -> _DefaultSettings:
    """Return the singleton settings object. Patch via ``reload_settings`` in tests."""
    return _load_settings()


def reload_settings() -> _DefaultSettings:
    """Force a re-read from env (used by tests after monkeypatching env vars)."""
    get_settings.cache_clear()
    return get_settings()
