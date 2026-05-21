# LawGPT — Architecture (Production)

Status as of `2026-05-21`. Replaces the earlier coursework-era doc.

## High-level

LawGPT is a hosted multi-tenant study assistant for legal study. Each
user uploads their own corpus (PDF / DOCX / images / notes) which is
ingested into namespace-isolated pgvector storage. Queries flow through
a LangGraph pipeline that retrieves from the user's corpus and
synthesises grounded answers with IRAC + optional Mermaid timelines.

```
Browser (Next.js / Pages)
        │ HTTPS
        ▼
Cloudflare Worker (lawgpt-edge)
   • Clerk JWT verify          (jose + remote JWKS)
   • R2 presigned PUT URLs     (aws4fetch)
   • CORS                      (allowlist by origin)
   • forwards X-User-Id        to container
        │ DO RPC
        ▼
Cloudflare Container (LawgptBackend DO)
   FastAPI ▸ LangGraph pipeline
        │
        ├── Neon Postgres (pgvector)  ← vectors, namespace = user_id
        ├── D1 (lawgpt-prod)          ← users, files, sessions, messages
        └── R2 (lawgpt-uploads)       ← raw blobs
```

## Layers

| Layer | Tech | Repo path | Deploys via |
|-------|------|-----------|-------------|
| Frontend | Next.js 15 App Router on CF Pages (edge runtime) | `frontend/` | `.github/workflows/deploy.yml` → `wrangler pages deploy` |
| Edge Worker | Cloudflare Worker (TypeScript) | `infra/worker/` | same workflow → `wrangler deploy` |
| Backend container | FastAPI + LangGraph in a CF Container DO | `backend/` + `backend/Containerfile` | bundled by worker deploy |
| Vector store | Neon Postgres + pgvector | n/a — managed | n/a |
| Relational | Cloudflare D1 (`lawgpt-prod`, id `02f7936a-d1f7-44bc-8b2e-0488dcdc1ecb`) | `backend/src/storage/db.py` | `wrangler d1 migrations` (manual on schema change) |
| Blob | Cloudflare R2 (`lawgpt-uploads`) | `backend/src/storage/blob.py` | bound in `infra/wrangler.toml` |
| Auth | Clerk **production** on `clerk.ashgpt.xyz` (`pk_live_…`) | Worker + FastAPI dep | wrangler.toml + dashboard secrets |

## Request flow — chat (`POST /chat`)

1. Browser hits Worker `/chat` with Clerk `Authorization: Bearer …`.
2. Worker verifies JWT, sets `X-User-Id: <clerk sub>`, forwards to the DO container via `BACKEND.idFromName('global').fetch`.
3. `backend/api/routes_chat.py` validates / creates a session (D1 `sessions`), loads prior turns (D1 `messages`), persists the new user message.
4. `backend/src/agent/graph.run_query` seeds AgentState with the prepared `chat_history`, invokes the LangGraph state machine:
   - `router_node` (with transcript) → intent ∈ `{ratio, chronology, summary, general}`.
   - `retrieval_node` calls `retrieve_all` — `retrieve_texts` ⫼ `retrieve_slides` in a 2-worker pool. Each leg in turn parallelises dense (pgvector ANN) ⫼ BM25 (in-memory per-namespace).
   - Optional `cache_check` short-circuits to a stored answer.
   - Intent-conditional branches: `ratio_extractor` (IRAC) and / or `chronology` (Mermaid).
   - `synthesis` produces the final grounded answer.
   - `verification` rewrites unsupported citations; confidence-gated escalation may rerun synthesis with a stronger model.
5. SSE events stream back through the container → Worker → browser: `node`, `sources`, `irac`, `mermaid`, `verification`, `answer_chunk`, `done`.
6. Assistant message persisted into D1 with retrieved chunk ids, latency, verification report.

## Request flow — upload + ingestion

1. Browser asks `POST /uploads/presign` with `{name, mime, doc_type, week?}`. Worker mints an S3-style PUT URL via aws4fetch (15 min expiry); container records the `File` row in D1 and returns `file_id`.
2. Browser PUTs the file directly to R2.
3. Browser `POST /uploads/{file_id}/process`. Container `ingest_file` (`backend/src/ingestion/pipeline.py`):
   - Resolve blob → local path via R2 (or `LocalBlobStore` in dev).
   - `extract.extract(path, mime)` dispatches per MIME: PDF (per-page PyMuPDF), DOCX, plain text, image (Gemini VLM → text + `image_path` metadata).
   - `chunk_text` splits with the legal-tuned splitter.
   - `ZeroEntropyEmbeddings.embed_documents` (2560-dim).
   - `make_vector_store()` → `PgVectorStore.upsert` under `namespace=user_id`.
   - `ChunkMeta` rows persisted in D1 for citation lookup.
   - `bm25.invalidate(user_id)` so the next query rebuilds the in-memory BM25 corpus.

## Retrieval

Code lives in `backend/src/agent/tools.py`. The pipeline:

- Query embedded once via `ZeroEntropyEmbeddings.embed_query` (LRU-cached).
- Dense leg: `make_vector_store().search(query_vector, namespace, k, where=…)` — cosine ANN over `vectors` table.
- BM25 leg: per-namespace in-memory index built from `VectorStore.list_namespace(...)`. Legal-tuned tokeniser (citation + section folding, narrow stopword set) lives in `backend/src/agent/bm25.py`.
- Both legs ranked independently then fused via Reciprocal Rank Fusion (`reciprocal_rank_fusion`) with `RRF_WEIGHT_DENSE=0.7`, `RRF_WEIGHT_BM25=0.3`.
- Cross-encoder / Cohere reranker reduces fused candidates to final `k`.
- Image-bearing chunks (any with `metadata.image_path`) are split into the `retrieve_slides` channel; the rest stay on `retrieve_texts`.
- Strict namespace isolation: pgvector filters on the `namespace` column at the SQL layer; tools strip `namespace` out of the metadata JSONB where to avoid double-filtering.

## State

`AgentState` (`backend/src/agent/state.py`) holds:

- `query`, `intent`, `week_filter`, `chat_history`
- `retrieved_texts`, `retrieved_slides` (each a list of `RetrievedDocument` TypedDicts)
- `ratio_decidendi`, `irac_analysis`, `mermaid_diagram`, `chronology_summary`
- `final_answer`, `verification_report`
- `node_trace`, `timings`
- `cache_hit`, `_override_synthesis_model` (escalation)

## Frontend

- `frontend/src/app/(app)/layout.tsx` — Clerk-gated shell, edge runtime, sidebar + main pane.
- `(app)/chat/page.tsx` — useChat-driven chat surface. **No session history hydration yet** (gap).
- `(app)/library/page.tsx` — file list + upload via react-dropzone.
- `(app)/exam/page.tsx` — exam generator (scope_type / num_mcq / etc., state still local).
- `lib/api.ts` — wrappers around backend REST endpoints. `listSessions`, `listMessages` exist but are unused by the chat UI.
- `lib/useChat.ts` — owns turns + sessionId + node trace. **sessionId is state-only** (gap).
- `lib/streaming.ts` — SSE parse → handlers.

## Deploys

- Frontend: GitHub Action `deploy-frontend` runs `@cloudflare/next-on-pages` then `wrangler pages deploy --project-name lawgpt`.
- Worker + container: `deploy-worker` job runs `wrangler deploy` from `infra/worker/` — bundles the container image (referenced by `infra/wrangler.toml [[containers]] image = "../backend/Containerfile"`).
- Secrets via `wrangler secret put` (DATABASE_URL, GOOGLE_API_KEY, OPENAI_API_KEY, ZEMBED_API_KEY, COHERE_API_KEY, CLERK_SECRET_KEY, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY).
- Public vars live in `[vars]` of `infra/wrangler.toml` (CLERK_ISSUER, CORS_ORIGINS, R2_BUCKET, R2_ACCOUNT_ID).

## Out-of-band scripts / tooling

- `backend/src/eval/run_evals.py` — eval harness over the legacy coursework cases. Used to compare retrieval modes (dense/hybrid × reranker on/off). Property-law-specific; **not yet generalised**.
- `docker-compose.yml` at repo root — local dev (`postgres` for pgvector, `backend` container, optional `frontend` profile).

## Known gaps (tracked in `POLISH_PLAN.md`)

- Frontend sessionId is state-only — tab swap or reload starts a new session and orphans prior turns in D1.
- `listMessages` / `listSessions` exist but no UI consumes them.
- Build-on-graph router has full transcript, but the **retrieval-side rewrite** only attaches the most recent assistant excerpt — pronouns referring more than one hop back ("the case from earlier") may miss.
- User message persisted before the graph runs → an orphan row if the graph fails.
- Verification record doesn't capture which synthesis model ran when escalation kicked in.
- No multi-turn integration test exercises session_id continuity.
