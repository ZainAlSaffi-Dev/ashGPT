-- Enable pgvector for local dev parity with Cloudflare Vectorize semantics.
-- Schema migrations live in backend/src/storage/migrations/ (added Phase 1).
CREATE EXTENSION IF NOT EXISTS vector;
