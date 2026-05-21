# ashGPT — session handoff

This file is auto-loaded by Claude Code at the start of every session in this repo. Read it before doing anything else, then keep it current as the project evolves (see **Maintenance protocol** at the bottom).

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
| Hosting (BE) | GCP Cloud Run | configured outside repo |
| Auth | Clerk **production** instance on `clerk.ashgpt.xyz` (`pk_live_…`) | env vars in Pages + worker |
| Vector store | pgvector on managed Postgres | `backend/src/storage/vector_store.py` |
| Blob store | Cloudflare R2 | `backend/src/storage/blob.py` |
| LLM | Anthropic Claude (synthesis + verification + IRAC) — accuracy over cost | `backend/src/llm.py` |
| Embeddings + rerank | Cohere | `backend/src/embeddings.py`, `backend/src/agent/reranker.py` |
| Parsing | unstructured, pypdf, python-docx, pillow OCR | `backend/src/ingestion/` |

---

## Repo layout

```
ashGPT/
├── CLAUDE.md                 ← THIS FILE
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
│   │   ├── llm.py            ← Claude client + escalation
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

`backend/api/routes_exam.py` — generates MCQs + short-answer questions from a chosen scope (file / week / all / past_paper). Grading uses Claude with rubric prompting.

---

## Branding + domains

- **Public name:** ashGPT (always lowercase `a`, capital `GPT`).
- **Custom domains** (via Cloudflare): `ashgpt.xyz`, `www.ashgpt.xyz` → Pages project `ashgpt`. `api.ashgpt.xyz` → worker `lawgpt-edge`. Domain registered on Namecheap; nameservers moved to Cloudflare.
- The CF Pages project slug `ashgpt` and the worker slug `lawgpt-edge` are immutable. Don't try to rename the worker — just keep the slug and rely on the custom domain.

---

## Deploy

**CI** (`.github/workflows/deploy.yml`): on push to `main` it runs `@cloudflare/next-on-pages` then `wrangler pages deploy .vercel/output/static --project-name=ashgpt --branch=main`, then deploys the worker. Required GH secrets: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `NEXT_PUBLIC_API_BASE` (= `https://api.ashgpt.xyz`).

**Pages dashboard build config** (also valid for manual setup):
- Root dir: `frontend`
- Build command: `npx @cloudflare/next-on-pages@1`
- Build output: `.vercel/output/static`
- Compatibility flags (Production + Preview): `nodejs_compat`
- Node: 20

**Manual local deploy** (from repo root):

```bash
cd frontend
npm ci
NEXT_PUBLIC_API_BASE=https://api.ashgpt.xyz \
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_live_Y2xlcmsuYXNoZ3B0Lnh5eiQ \
npx @cloudflare/next-on-pages@1
npx wrangler pages deploy .vercel/output/static --project-name=ashgpt --branch=main
```

**Backend deploy** — out of repo, runs on GCP Cloud Run. Push to GitHub and trigger the Cloud Run deploy manually.

---

## Environment variables

### Frontend (Pages project env)

Public-build vars (`NEXT_PUBLIC_*`) live in `frontend/wrangler.toml`'s `[vars]` block and are inlined into client JS by Next at build time. The Pages dashboard now defers to wrangler.toml for these — the dashboard fields are read-only when a `[vars]` block exists.

- `NEXT_PUBLIC_API_BASE` → `https://api.ashgpt.xyz` (wrangler.toml)
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` → `pk_live_Y2xlcmsuYXNoZ3B0Lnh5eiQ` (wrangler.toml)
- `CLERK_SECRET_KEY` → **secret** `sk_live_…` (Pages dashboard → encrypted env vars, **not** in wrangler.toml)
- `NODE_VERSION` → `20` (dashboard)

### Worker (`lawgpt-edge`, `infra/wrangler.toml`)

- `[vars]` (public): `CLERK_ISSUER = "https://clerk.ashgpt.xyz"`, `CORS_ORIGINS`, `R2_BUCKET`, `R2_ACCOUNT_ID`. JWKS URL is derived as `<CLERK_ISSUER>/.well-known/jwks.json`.
- Secrets (set via `wrangler secret put`, never committed): `BACKEND_ORIGIN`, `CLERK_SECRET_KEY` (`sk_live_…`), `DATABASE_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`.

### Backend (Cloud Run env)
- `CORS_ORIGINS=https://ashgpt.xyz,https://www.ashgpt.xyz,https://ashgpt.pages.dev`
- `DATABASE_URL` (Postgres + pgvector)
- `ANTHROPIC_API_KEY`, `COHERE_API_KEY`
- `CLERK_JWKS_URL`, `CLERK_SECRET_KEY`
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`

---

## Working preferences (carry these into every session)

- **Accuracy over cost.** Pick the best model, then engineer around cost via gating / escalation, not blanket downgrades.
- **Tests + push per change.** Every meaningful change gets a test + example I/O alongside it, and is committed + pushed before moving on. Multiple small commits over one big one when scopes are separable.
- **Auto mode is the default.** Don't pause for clarifying questions on routine work — make the reasonable call. Surface a question only when the choice is genuinely irreversible.
- **lean-ctx is mandatory.** Always use `ctx_read` / `ctx_shell` / `ctx_search` / `ctx_tree` MCP tools over native `Read` / `bash` / `Grep` / `ls`. `ctx_read` modes: `auto` (default), `full`, `map`, `signatures`, `diff`, `aggressive`, `entropy`, `task`, `reference`, `lines:N-M`. Native `Edit`/`Write` stay as-is.
- **Caveman mode (when active).** Drop articles / filler / pleasantries / hedging in chat replies. Fragments OK. Code / commits / security warnings / multi-step sequences stay normal prose. Off only on "stop caveman" or "normal mode".
- **Memory** at `/Users/zer0/.claude/projects/-Users-zer0-Documents-projects-ashGPT/memory/`. Persistent across sessions. Use it for user/feedback/project/reference facts; don't duplicate things derivable from the repo.

---

## Frontend gotchas you'll hit again

1. **`position: fixed` inside a transformed ancestor** — framer-motion's `transform` becomes the containing block. Always **portal** popovers / tooltips / modals to `document.body`.
2. **TanStack Query staleTime** — `useMessages` is `staleTime: 60s` because past messages are immutable. Don't refetch on focus. `useSessions` is default-fresh.
3. **Streaming + memoization** — `ChatMessageBody` is `React.memo`'d on `bodyMd`. Chips read active state from `CitationContext`, not closures, so popover toggle doesn't re-walk the hast tree.
4. **typedRoutes is on** — cast dynamic strings with `as Route` when needed.
5. **`unified` is a transitive dep**, not a direct one. Don't import types from it — the Pages build's stricter pnpm install will fail typecheck. Type rehype transformers as plain `(tree: Root) => void`.
6. **Edit vs Write** — `Write` requires you to `Read` the file first in the same session. `Edit` requires the same. Cheap mistake to forget after writing new files in earlier turns.

## Backend gotchas

1. **No proper Alembic yet.** `init_db` runs `create_all` then idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for post-v1 columns on Postgres. SQLite (tests) skips the ALTERs because `create_all` already produces the right schema on a fresh DB. When you add new columns, append to `_apply_inline_migrations` in `backend/src/storage/db.py`.
2. **Secrets** never go into Claude Code at all. No API keys, tokens, passwords, connection strings in chat or in code committed to the repo.

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
cd frontend && npx @cloudflare/next-on-pages@1 && npx wrangler pages deploy .vercel/output/static --project-name=ashgpt --branch=main
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

---

## Maintenance protocol (for future sessions)

When you make a meaningful change, update this file:

- **Surgery log** above gets a new line per substantive shipped change (one sentence, the why + the touched concept, not file diffs — git log already has those).
- **Gotchas** sections grow when you discover something that bit you and would bite the next person.
- **Stack / repo layout / env vars** stay current when anything moves or is renamed.
- Keep it tight. If a line stops being load-bearing (the gotcha was fixed at the root, the surgery is now ancient history), delete it.

Don't write speculative TODOs here. Use issues, plans, or memory for in-flight work.
