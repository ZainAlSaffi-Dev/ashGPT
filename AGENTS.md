# ashGPT — session handoff

This file is auto-loaded by Codex at the start of every session in this repo. Read it before doing anything else, then keep it current as the project evolves (see **Maintenance protocol** at the bottom).

---

## What this is

**ashGPT** — a personal study assistant for a law student. Upload notes / cases / statutes / past papers → ask grounded questions → get answers with citations, IRAC analysis, chronology diagrams, and exam practice. The primary user is the repo owner's partner (UQ Law / similar).

Project was forked from a Streamlit prototype called ashGPT and productized into a hosted Next.js + FastAPI app. Some legacy references to "LawGPT" still appear in old commits / docs — branding is **ashGPT** going forward.

**Live domain:** `https://ashgpt.xyz` (frontend) / `https://api.ashgpt.xyz` (edge API). Both proxied through Cloudflare.

---

## Stack at a glance

| Layer | Tech | Where |
|---|---|---|
| Frontend | Next.js 15 (App Router) + Tailwind + Clerk + TanStack Query + framer-motion + react-markdown | `frontend/` |
| Hosting (FE) | Cloudflare Pages (project slug: `ashgpt`) via `@cloudflare/next-on-pages` | dashboard + `frontend/wrangler.toml` |
| Edge proxy | Cloudflare Worker (`lawgpt-edge`) — auth check, R2 presign, BACKEND_ORIGIN proxy | `infra/worker/` |
| Backend | FastAPI + SQLAlchemy async + LangGraph agent | `backend/` |
| Hosting (BE) | Cloudflare Containers (`LawgptBackend` Durable Object, bound from the edge worker — see `[[containers]]` in `infra/wrangler.toml`) | `infra/wrangler.toml` |
| Auth | Clerk **production** instance on `clerk.ashgpt.xyz` (`pk_live_…`) | env vars in Pages + worker |
| Vector store | pgvector on managed Postgres | `backend/src/storage/vector_store.py` |
| Blob store | Cloudflare R2 | `backend/src/storage/blob.py` |
| LLM | Anthropic Codex (synthesis + verification + IRAC) — accuracy over cost | `backend/src/llm.py` |
| Embeddings + rerank | Cohere | `backend/src/embeddings.py`, `backend/src/agent/reranker.py` |
| Parsing | unstructured, pypdf, python-docx, pillow OCR | `backend/src/ingestion/` |

---

## Repo layout

```
ashGPT/
├── AGENTS.md                 ← THIS FILE
├── frontend/                 ← Next.js app
│   ├── src/
│   │   ├── app/              ← App Router routes
│   │   │   ├── (app)/        ← signed-in routes (chat, library, exam, settings)
│   │   │   ├── layout.tsx    ← ClerkProvider wrapper
│   │   │   └── page.tsx      ← landing
│   │   ├── components/       ← React UI
│   │   │   ├── ChatMessage.tsx       ← assistant bubble + citation popover orchestration
│   │   │   ├── ChatMessageBody.tsx   ← memoized markdown render (skips re-walk on popover toggle)
│   │   │   ├── CitationBadge.tsx     ← Gemini-style numbered chip
│   │   │   ├── CitationPopover.tsx   ← portal'd hover/pinned popover + mobile bottom sheet
│   │   │   ├── citation-context.tsx  ← React Context shared by chips ↔ ChatMessage
│   │   │   ├── SourcePanel.tsx       ← collapsed source list (clickable rows + library link)
│   │   │   ├── ChatSurface.tsx       ← chat layout / stream wiring
│   │   │   ├── Sidebar.tsx, Dropzone.tsx, MermaidRenderer.tsx, …
│   │   └── lib/
│   │       ├── rehype-citations.ts   ← rehype plugin emitting per-chip [S#] elements
│   │       ├── citation-highlight.ts ← word-level n-gram overlap for snippet <mark>
│   │       ├── streaming.ts          ← SSE parser (eventsource-parser)
│   │       ├── useChat.ts            ← chat state machine + streaming consumer
│   │       ├── queries.ts            ← TanStack Query hooks
│   │       ├── api.ts                ← typed fetch wrappers + resolveApiBase
│   │       └── types.ts              ← wire types mirroring backend schemas
│   ├── wrangler.toml         ← CF Pages config (project name: ashgpt)
│   └── package.json
├── backend/                  ← FastAPI
│   ├── api/
│   │   ├── main.py           ← app factory, CORS, router mount
│   │   ├── routes_chat.py    ← SSE streaming chat endpoint
│   │   ├── routes_sessions.py← chat history CRUD
│   │   ├── routes_files.py   ← upload / list / delete
│   │   ├── routes_exam.py    ← exam generation + grading
│   │   ├── schemas.py        ← Pydantic wire models
│   │   └── deps.py           ← Clerk auth + DB session
│   ├── src/
│   │   ├── agent/            ← LangGraph: nodes, graph, tools, reranker
│   │   ├── storage/          ← SQLAlchemy models (db.py), R2 blob, vector store
│   │   ├── ingestion/        ← chunker, extractor, pipeline
│   │   ├── eval/             ← golden-set scoring
│   │   ├── llm.py            ← Codex client + escalation
│   │   ├── embeddings.py     ← Cohere embed + cache
│   │   └── config.py         ← env-driven Settings
│   └── tests/                ← pytest
├── infra/worker/             ← Cloudflare Worker (edge proxy)
└── .github/workflows/deploy.yml  ← CI: Pages deploy + Worker deploy
```

---

## Core flows

### Chat (the central one)

1. User submits a query → `frontend/src/lib/useChat.ts:send` POSTs to `/chat/stream` via SSE.
2. Edge worker checks the Clerk JWT, forwards to FastAPI (`BACKEND_ORIGIN`).
3. `backend/api/routes_chat.py` runs the LangGraph pipeline:
   - `query_rewriter` → multi-vector embed → pgvector + BM25 hybrid retrieval → Cohere rerank
   - Intent classifier (`ratio` | `chronology` | `summary` | `general`)
   - Branching nodes: ratio → IRAC; chronology → mermaid; otherwise synthesis only
   - Verification node checks citations against retrieved chunks
4. SSE events streamed back: `node`, `sources`, `irac`, `mermaid`, `verification`, `history_overflow`, `answer_chunk`, `done`, `error`.
5. Frontend `useChat` patches `ChatTurn` state per event. `ChatMessage` renders. Citation chips `[S#]` are turned into clickable badges by `rehype-citations.ts`.
6. Final assistant message persisted with full `sources` JSON, `irac`, `mermaid`, `verification`, `latency_ms`, `intent`, `retrieved_chunk_ids` (see `backend/api/routes_chat.py:150`).

### Citation popover

- `rehype-citations.ts` walks the hast tree of streamed markdown and replaces each `[S#]` with `<cite data-source-index="N" data-cite-occurrence="N-K" data-cite-context="…preceding prose…">`. The occurrence id is a per-render counter so duplicate `[S1]`s remain individually addressable.
- `ChatMessageBody.tsx` is `React.memo`'d on `bodyMd`; chips read active state from `CitationContext` so popover hover/pin doesn't re-walk the hast tree.
- `CitationPopover.tsx` is **portaled to `document.body`** — the parent `ChatMessage` `motion.div` has a transform, which (per CSS containing-block rules) would otherwise re-root `position:fixed` descendants and push the popover offscreen. This was a real bug, don't undo it.
- `citation-highlight.ts` finds the longest n-gram overlap between the chip's preceding prose and the cited chunk snippet → wraps it in `<mark>` inside the popover.
- Modes: `hover` (250ms open / 120ms close, transient) vs `pinned` (click-to-pin, dismissed by outside-click / Esc / X). Outside-click ignores citation-chip targets so toggle doesn't flicker.
- Mobile (<640 px) pinned mode renders as a bottom sheet with backdrop instead of a floating popover.

### Reloaded chats (rehydration)

When the user opens an old chat, `frontend/src/app/(app)/chat/[sessionId]/page.tsx` calls `listMessages(sessionId)`. The backend returns the full `sources` array, IRAC text, and mermaid diagram for every assistant message, so citation chips work immediately on reload without rerunning retrieval. **Old rows (pre-rehydration commit) have `sources = NULL`** — chips on those don't open. A backfill script would need to rerun retrieval per turn.

### Library

- Drag-drop in `frontend/src/components/Dropzone.tsx` → `POST /files/presign` returns an R2 PUT URL → client uploads directly to R2 → backend kicks off chunk + embed via `backend/src/ingestion/pipeline.py`.
- `useFiles` query polls every 4s while any file is `uploaded`/`processing`/`queued`.
- Source rows in chat are clickable → opens pinned popover + external-link icon deep-links to `/library?file=<name>` which auto-scrolls + highlights that file.

### Exam

`backend/api/routes_exam.py` — generates MCQs + short-answer questions from a chosen scope (file / week / all / past_paper). Grading uses Codex with rubric prompting.

---

## Branding + domains

- **Public name:** ashGPT (always lowercase `a`, capital `GPT`).
- **Custom domains** (via Cloudflare): `ashgpt.xyz`, `www.ashgpt.xyz` → Pages project `ashgpt`. `api.ashgpt.xyz` → worker `lawgpt-edge`. Domain registered on Namecheap; nameservers moved to Cloudflare.
- The CF Pages project slug `ashgpt` and the worker slug `lawgpt-edge` are immutable. Don't try to rename the worker — just keep the slug and rely on the custom domain.

---

## Deploy

**CI** (`.github/workflows/deploy.yml`): on push to `main` it runs `@cloudflare/next-on-pages` then `wrangler pages deploy .vercel/output/static --project-name=ashgpt --branch=main`, then deploys the worker. Required GH secrets: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `NEXT_PUBLIC_API_BASE` (= `https://api.ashgpt.xyz`). Worker routes/custom domains are dashboard-managed, not `wrangler.toml`-managed, so CI does not need zone route permissions.

**Pages dashboard build config** (also valid for manual setup):
- Root dir: `frontend`
- Build command: `pnpm dlx @cloudflare/next-on-pages@1`
- Build output: `.vercel/output/static`
- Compatibility flags (Production + Preview): `nodejs_compat`
- Node: 20

**Manual local deploy** (from repo root):

```bash
cd frontend
pnpm install --frozen-lockfile
NEXT_PUBLIC_API_BASE=https://api.ashgpt.xyz \
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_live_Y2xlcmsuYXNoZ3B0Lnh5eiQ \
pnpm dlx @cloudflare/next-on-pages@1
npx wrangler pages deploy .vercel/output/static --project-name=ashgpt --branch=main
```

**Backend deploy** — backend runs as a Cloudflare Container bound to the `lawgpt-edge` worker via Durable Object. `cd infra && wrangler deploy` ships both the worker and rebuilds the container from `backend/Containerfile`. Env vars + secrets are managed through `infra/wrangler.toml` `[vars]` and `wrangler secret put`.

---

## Environment variables

### Frontend (Pages project env)

Public-build vars (`NEXT_PUBLIC_*`) live in `frontend/wrangler.toml`'s `[vars]` block and are inlined into client JS by Next at build time. The Pages dashboard now defers to wrangler.toml for these — the dashboard fields are read-only when a `[vars]` block exists.

- `NEXT_PUBLIC_API_BASE` → `https://api.ashgpt.xyz` (wrangler.toml)
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` → `pk_live_Y2xlcmsuYXNoZ3B0Lnh5eiQ` (wrangler.toml)
- `NEXT_PUBLIC_CLERK_PROXY_URL` → `https://ashgpt.xyz/__clerk/` (wrangler.toml). Clerk production domain is configured to use this proxy URL; browser Clerk traffic should stay same-origin through `ashgpt.xyz/__clerk`.
- `CLERK_SECRET_KEY` → **secret** `sk_live_…` (Pages dashboard → encrypted env vars, **not** in wrangler.toml)
- `NODE_VERSION` → `20` (frontend/wrangler.toml). Keep Cloudflare Pages on Node 20; dashboard default images may drift ahead and produce different Next/Pages function output than CI.
- Pages deploys must use `wrangler pages deploy .vercel/output/static` (or the GitHub Action). Running plain `wrangler deploy` from `frontend/` is a Worker deploy and fails with "Missing entry-point to Worker script or to assets directory"; do not "fix" that by adding `main` or `[assets]` to the Pages config.

### Worker (`lawgpt-edge`, `infra/wrangler.toml`)

- `[vars]` (public): `CLERK_ISSUER = "https://clerk.ashgpt.xyz"`, `CLERK_PROXY_URL = "https://ashgpt.xyz/__clerk"`, `CLERK_FAPI = "https://frontend-api.clerk.dev"`, `CLERK_JWKS_URL = "https://ashgpt.xyz/__clerk/.well-known/jwks.json"`, `CORS_ORIGINS`, `R2_BUCKET`, `R2_ACCOUNT_ID`.
- Secrets (set via `wrangler secret put`, never committed): `BACKEND_ORIGIN`, `CLERK_SECRET_KEY` (`sk_live_…`), `DATABASE_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`.

### Backend (Cloudflare Container env)

Backend container picks up the same `[vars]` block as the worker (single `infra/wrangler.toml`), so `CLERK_ISSUER` / `CORS_ORIGINS` / `R2_*` are set once. Backend-specific env reads:

- `CLERK_ISSUER` (used to derive JWKS URL — no separate `CLERK_JWKS_URL` needed).
- Secrets (via `wrangler secret put`): `DATABASE_URL` (Postgres + pgvector), `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`.

---

## Working preferences (carry these into every session)

- **Accuracy over cost.** Pick the best model, then engineer around cost via gating / escalation, not blanket downgrades.
- **Tests + push per change.** Every meaningful change gets a test + example I/O alongside it, and is committed + pushed before moving on. Multiple small commits over one big one when scopes are separable.
- **Auto mode is the default.** Don't pause for clarifying questions on routine work — make the reasonable call. Surface a question only when the choice is genuinely irreversible.
- **lean-ctx is mandatory.** Always use `ctx_read` / `ctx_shell` / `ctx_search` / `ctx_tree` MCP tools over native `Read` / `bash` / `Grep` / `ls`. `ctx_read` modes: `auto` (default), `full`, `map`, `signatures`, `diff`, `aggressive`, `entropy`, `task`, `reference`, `lines:N-M`. Native `Edit`/`Write` stay as-is.
- **Caveman mode (when active).** Drop articles / filler / pleasantries / hedging in chat replies. Fragments OK. Code / commits / security warnings / multi-step sequences stay normal prose. Off only on "stop caveman" or "normal mode".
- **Memory** at `/Users/zer0/.Codex/projects/-Users-zer0-Documents-projects-ashGPT/memory/`. Persistent across sessions. Use it for user/feedback/project/reference facts; don't duplicate things derivable from the repo.

---

## Frontend gotchas you'll hit again

1. **`position: fixed` inside a transformed ancestor** — framer-motion's `transform` becomes the containing block. Always **portal** popovers / tooltips / modals to `document.body`.
2. **TanStack Query staleTime** — `useMessages` is `staleTime: 60s` because past messages are immutable. Don't refetch on focus. `useSessions` is default-fresh.
3. **Streaming + memoization** — `ChatMessageBody` is `React.memo`'d on `bodyMd`. Chips read active state from `CitationContext`, not closures, so popover toggle doesn't re-walk the hast tree.
4. **typedRoutes is on** — cast dynamic strings with `as Route` when needed.
5. **`unified` is a transitive dep**, not a direct one. Don't import types from it — the Pages build's stricter pnpm install will fail typecheck. Type rehype transformers as plain `(tree: Root) => void`.
6. **Edit vs Write** — `Write` requires you to `Read` the file first in the same session. `Edit` requires the same. Cheap mistake to forget after writing new files in earlier turns.
7. **Chat auth is streaming, not `request()`** — JSON API calls use `withAuth` + one-shot 401 token replay. Keep `/chat` streaming on the same token-wait/retry behavior; grabbing `getToken()` once can race right after sign-in/sign-up and produce edge 401s.
8. **Chat streams can outlive route state** — always pass an `AbortSignal` into `/chat` streaming and abort on unmount/session switch. A first-message stream that completes after the user leaves `/chat` can otherwise `router.replace` them back to `/chat/<id>`.
9. **No `keepPreviousData` for per-session messages** — it can briefly render chat A's turns under chat B's URL during rapid session switching. Seed/cache only the exact `['messages', sessionId]` key.
10. **Protected-route sign-in redirects must stay same-origin** — unauthenticated document navigations to `/chat` routes should redirect to `/?redirect_url=...`, not Clerk's account portal host. The landing page sanitizes that value before handing it to Clerk modal buttons.
11. **Landing redirect must not wait on `getToken()`** — after modal sign-in, navigate as soon as Clerk reports `isSignedIn`; token readiness is handled by authed query gates. Waiting for a token on `/` can leave the user on the spinner until manual refresh.
12. **Route transitions inside the app shell must be enter-only** — `AppShell` scrolls inside a custom `<main>`, so an exiting `AnimatePresence` route left in normal flow stacks the old page above the new one and makes workspace pages look blank until you scroll.
13. **Uploads should stay on the API origin** — browser → R2 direct PUTs are fragile because bucket CORS and signed `Content-Type` must match exactly. Production uploads now use an authenticated Worker `/uploads/blob` URL; keep Dropzone on `withAuth`, clean up the placeholder file row when blob upload fails, and disable subject uploads until a `?folder=` param is validated.
14. **Client-only giants still affect Pages Functions** — `@cloudflare/next-on-pages` can hoist dynamic client imports into the generated Pages Function modules. Mermaid is loaded from jsDelivr in `MermaidRenderer` to keep the Function bundle under Cloudflare's publish limits; don't switch it back to `import('mermaid')` without checking `_worker.js` sizes.
15. **Dashboard Pages deploys can fail after successful asset upload** — `Failed to publish your Function. Got error: Unknown internal error occurred.` usually means the generated Pages Function publish failed, not that Next failed. First compare module sizes in `.vercel/output/static/_worker.js`; GitHub Actions direct upload with Wrangler 4 is still the source-of-truth deploy path.

## Backend gotchas

1. **No proper Alembic yet.** `init_db` runs `create_all` then idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for post-v1 columns on Postgres. SQLite (tests) skips the ALTERs because `create_all` already produces the right schema on a fresh DB. When you add new columns, append to `_apply_inline_migrations` in `backend/src/storage/db.py`.
2. **Secrets** never go into Codex at all. No API keys, tokens, passwords, connection strings in chat or in code committed to the repo.
3. **Conversation memory is not source evidence.** `conversation_memory` is rebuilt per session from older persisted turns and may resolve shorthand, jurisdiction, goals, corrections, and authorities already discussed, but answer facts still need retrieved `[S#]` citations or `[external]`.
4. **Scoped retrieval is two-legged.** Project/folder/file scope must be applied to both pgvector metadata filters and BM25 cache/search. BM25 cache keys are now `user_id:scope_hash`; invalidating a user must clear all keys with that prefix.
5. **Worker routes are dashboard-managed.** `infra/wrangler.toml` intentionally does not declare `ashgpt.xyz/__clerk` routes; otherwise GitHub Actions needs zone-level `Workers Routes: Edit/Write` for `ashgpt.xyz` plus `Zone: Read` and fails with Cloudflare code 10000 when the token is account-only.
6. **Upload rows are created before bytes land.** `/uploads/presign` registers the file, then the browser/Worker writes the blob, then `/process` indexes it. A missing blob must remain retryable, a failed blob transfer should delete the placeholder row, and zero extracted chunks should surface as a clear failed upload rather than a ready-but-useless file.

---

## Common commands

```bash
# Frontend dev
cd frontend && pnpm dev

# Frontend tests / typecheck
cd frontend && npx tsc --noEmit && npx vitest run

# Backend tests
cd backend && python -m pytest -q

# Backend dev (assumes Postgres + R2 envs set)
cd backend && uvicorn api.main:app --reload --port 8000

# Manual deploy (Pages)
cd frontend && pnpm dlx @cloudflare/next-on-pages@1 && npx wrangler pages deploy .vercel/output/static --project-name=ashgpt --branch=main
```

---

## Recent surgery (May 2026)

In chronological order, most recent last. Helps a fresh session understand the current state vs. what's in git history.

- Citation UX rebuild: per-chip occurrence ids, Gemini-style numbered badges, hover (250ms) vs pinned modes, mobile bottom sheet, snippet overlap `<mark>` highlight, react-portal'd popover, `React.memo`'d markdown body.
- SourcePanel: rows clickable → opens pinned popover anchored on row; per-row external-link icon → `/library?file=<name>` deep link (library page reads `?file=` via `useSearchParams`, scrolls + highlights).
- Chat history rehydration: backend now persists + returns `sources`, `irac`, `mermaid` per assistant message; frontend hydrates them into `ChatTurn` on reload. Schema migration is idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` (Postgres only).
- Caching: `useMessages` got `staleTime: 60s` + `gcTime: 10m`. `ChatSurface` invalidates `['messages', sessionId]` + `['sessions']` on busy→idle.
- Branding: LawGPT → ashGPT in user-visible strings; `wrangler.toml` project slug `lawgpt` → `ashgpt`; `NEXT_PUBLIC_API_BASE` → `https://api.ashgpt.xyz`. CI updated.
- Bug fix: `import type { Plugin } from 'unified'` in rehype-citations.ts broke the CF Pages typecheck (transitive dep, not in package.json). Replaced with a plain transformer type.
- Auth production fix: Clerk Frontend API is proxied through `https://ashgpt.xyz/__clerk`, sign-in/sign-up redirect to `/chat`, Worker JWKS discovery uses the proxy path, and chat streaming now waits for a Clerk token plus retries once after a 401.
- Production auth verification: `https://ashgpt.xyz` now boots Clerk cleanly, sign-in modal opens on the production domain, and phone chat requests passed Worker auth; avoid testing Clerk on `*.ashgpt.pages.dev` previews unless that origin is explicitly added/configured in Clerk.
- Performance/stability pass: chat streams now abort on unmount/session switch, first-session completion seeds exact message/session caches before URL promotion, per-session message queries no longer reuse previous-session data, cold history rehydrates when messages arrive, and the first-run tour lazy-loads after the app shell paints.
- Protected-route redirect fix: unauthenticated browser visits to signed-in pages now redirect to the same-origin landing page with a sanitized `redirect_url`, avoiding the broken `accounts.ashgpt.xyz` account-portal hop while preserving deep-link intent after modal sign-in.
- Future-session planning: added focused handoff prompts for a next-generation in-chat memory system and a browser-driven frontend animation/refresh/endpoint QA pass.
- In-chat memory v1: backend now keeps the latest 24 messages verbatim, compresses older persisted turns into per-session `conversation_memory`, feeds that into router/retrieval/synthesis with explicit grounding boundaries, and emits `memory` SSE telemetry when compression occurs.
- Project-scoped library design: added `docs/project_scoped_library_system_design.md` mapping current user/session/upload persistence and the target project/folder retrieval architecture.
- Project-scoped library v1: added project/folder persistence and CRUD, scoped file upload/list/move, scoped sessions/messages/chat snapshots, project/folder/file retrieval filters across dense + BM25, source metadata rehydration, and a basic frontend subject/folder library selector with scoped upload/chat plumbing.
- Project-scoped library hardening: fixed landing post-login navigation, scoped cache invalidation, legacy-session scope mismatch rejection, recursive folder scope cleanup, delete cache invalidation, and removed CI-owned Cloudflare routes so main deploys can proceed with account-level tokens.
- Project workspace revamp: `/library` is now a subject overview, subjects open `/library/[projectId]` workspaces with folder-scoped files/uploads/chat links, and the sidebar shows nested subject links.
- Workspace navigation polish: route transitions are enter-only and the app shell resets its internal scroll container on pathname changes so subject workspace pages open at the top instead of stacking below the previous page.
- Project workspace sessions: subject pages now show a scoped recent-chats panel with resume links and a new subject-chat action backed by `useSessions({ projectId })`.
- Upload hardening: production uploads now stream through authenticated Worker `/uploads/blob` into R2 instead of direct browser → R2 CORS PUTs; Dropzone waits for Clerk tokens, validates/rejects unsupported files visibly, cleans up failed placeholder rows, blocks invalid folder scopes, and backend tests cover scoped upload/process retry.
- Subject-chat and Pages deploy hardening: subject workspace chats now pre-create scoped sessions, `/chat/[sessionId]` rehydrates session scope before messages exist, global history hides subject chats, Mermaid loads client-side from CDN to shrink Cloudflare Pages Function output, frontend deploys use pnpm as the single package-manager path, and tests cover scoped session persistence.

---

## Optimization handoff (next session)

The next workstream should focus on speed, stability, and deterministic navigation after auth. Use `docs/optimization_handoff.md` as the starting prompt.

Current state:

- Production auth is no longer the main blocker. The landing page leaves `Loading...`, Clerk proxy calls to `https://ashgpt.xyz/__clerk` return OK, and `/chat` requests from a signed-in phone reached the Worker as authenticated.
- A visible chat "401" after this point is more likely to be an upstream provider error emitted inside the SSE stream than a browser/Clerk auth failure. Check backend stream logs before changing auth again.
- The preview URL `898be570.ashgpt.pages.dev` showed Clerk 400s because Clerk could not attribute that origin to the production instance. Test user flows on `https://ashgpt.xyz` unless preview auth is intentionally configured.
- Cache/history behavior today: valid authenticated requests can populate TanStack Query and persisted chat history; failed chat streams may appear temporarily in UI but are not reliable durable history.

Optimization targets:

- Eliminate random refreshes, wrong-page redirects, and accidental route changes while sending chat or immediately after sign-in/sign-up.
- Make guided tour/onboarding feel instant; lazy-load heavy tutorial code only after the main chat shell is interactive.
- Make chat history loading fast and predictable. Audit `useSessions`, `useMessages`, local optimistic state, invalidations, stale times, and focus/refetch behavior.
- Stress test core flows with browser automation: fresh visit, sign-in, sign-up redirect, send chat, stream interruption, reload during stream, back/forward navigation, session switch, history reload, and mobile viewport.
- Do not add Redis by default. First measure client/server bottlenecks and Cloudflare/container constraints; add new infrastructure only if the profile proves in-process/browser/query caching is insufficient.

---

## In-chat memory

Current model: `backend/api/routes_chat.py` loads all persisted rows for a session. `backend/src/agent/graph.py` calls `prepare_chat_memory_for_run`, keeps the newest `CHAT_HISTORY_MAX_MESSAGES = 24` rows as verbatim `chat_history`, and compresses older rows into `conversation_memory`.

The compressed memory contains deterministic summary text plus typed facts: jurisdiction, shorthand, study goals, source constraints, authorities discussed, and corrections. Router, retrieval, ratio, chronology, synthesis, and the query rewriter receive this memory. Retrieval packing uses memory anchors so >24-turn follow-ups can still resolve terms like "AP" or "that case".

Grounding rule: compressed memory is session context only. It must never be cited as legal authority. Prompts explicitly tell the model that every legal proposition still needs retrieved `[S#]` support or an `[external]` marker.

Telemetry: long chats emit the existing `history_overflow` event plus a `memory` SSE event containing `memory_compressed`, `compressed_turns`, `recent_messages`, `memory_fact_count`, `memory_summary_chars`, and `truncated_messages`.

Design note: see `docs/in_chat_memory_adr.md`. No schema change was added for v1; memory is rebuilt from per-session rows to avoid stale persisted summaries.

---

## Project-scoped library design

Use `docs/project_scoped_library_system_design.md` as the starting point for the next major library/session/retrieval workstream. The current app persists users, sessions, messages, files, chunks, blobs, and vectors per `user_id`, but it does not yet have first-class `project_id` / `folder_id` scope.

The intended architecture is `user -> project -> folder/file -> session/message -> retrieval scope`. Vector namespace stays `user_id` for hard tenant isolation; project/folder/file filtering should be metadata filters stamped onto files, chunks, vector rows, sessions, messages, and answer-cache keys. Explicit empty project/folder/file scopes must not fall back to the whole library.

---

## Frontend animation QA handoff

Run this as a dedicated browser session after the current performance/routing work is deployed. Use Chrome or Browser tooling with real screenshots and interaction timing; avoid relying only on static code review.

Prompt for the next session:

```text
You are working in /Users/zer0/Documents/projects/ashGPT. Read AGENTS.md first.

Focus only on frontend animation smoothness, refresh behavior, endpoint health, and browser-observed user-flow reliability. Do not redesign product UI or backend agent behavior during this pass.

Primary objective: verify the app feels fluid and deterministic in a real browser: animations do not jank, route transitions do not flicker, refreshes land on the same logical page, and all visible flows hit healthy endpoints.

Required browser matrix:
- Production custom domain `https://ashgpt.xyz` first. Use preview URLs only for deployment isolation; do not use pages.dev for Clerk auth conclusions unless that origin is configured.
- Desktop viewport and mobile viewport.
- Signed-out and signed-in states if an auth session is available. Do not ask for or handle credentials; stop at credential boundaries unless the user takes over.

Flows to test:
1. Fresh visit to `/`, Clerk boot, sign-in/sign-up modal open/close animations.
2. Signed-out deep link to `/chat` and `/chat/<id>`: should stay same-origin and preserve `redirect_url`.
3. App shell navigation: `/chat`, `/library`, `/exam`, `/settings`, sidebar open/close, mobile drawer, back/forward.
4. Chat idle refresh and session-history reload. Confirm no wrong-session message flash.
5. First chat send, existing chat send, stream node trace, stream completion URL promotion, and reload after completion.
6. New chat button while idle and while busy. Busy state should prevent route jumps.
7. Citation chips, popover hover/pin, source panel row interactions, mobile bottom sheet.
8. Library upload UI, file list polling states, delete dialog animations, and `/library?file=` deep-link highlight.
9. Exam page empty-library and ready-library states.
10. Guided tour/onboarding: confirm lazy load does not block chat shell; modal animation is smooth.

Endpoint checks:
- Browser/network checks for duplicate `sessions`, `messages`, `files`, `users/me`, and `/chat` requests on load, focus, route switch, and reload.
- Confirm `/chat` SSE errors are surfaced as stream errors, not mistaken for page-level auth failures.
- Confirm no repeated Clerk proxy failures under `https://ashgpt.xyz/__clerk`.

Quality bar:
- Capture before/after notes with exact URLs, viewport size, and route-change count where relevant.
- Record animation defects with screenshots or short descriptions tied to component names/files.
- Prefer small, targeted fixes: animation duration/easing, `AnimatePresence` mode, layout stability, scroll behavior, cache invalidation, and query gating.
- Run `cd frontend && pnpm typecheck && pnpm test && pnpm build` after changes.

Deliverables:
- Browser QA report with pass/fail per flow.
- Code fixes and tests for any confirmed regressions.
- Updated AGENTS.md if a new animation/routing gotcha is discovered.
```

---

## Maintenance protocol (for future sessions)

When you make a meaningful change, update this file:

- **Surgery log** above gets a new line per substantive shipped change (one sentence, the why + the touched concept, not file diffs — git log already has those).
- **Gotchas** sections grow when you discover something that bit you and would bite the next person.
- **Stack / repo layout / env vars** stay current when anything moves or is renamed.
- Keep it tight. If a line stops being load-bearing (the gotcha was fixed at the root, the surgery is now ancient history), delete it.

Don't write speculative TODOs here. Use issues, plans, or memory for in-flight work.
