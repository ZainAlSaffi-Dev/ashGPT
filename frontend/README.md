# frontend/

Next.js 15 (App Router) + Tailwind + Clerk + TanStack Query app for LawGPT.

## Setup

```bash
cd frontend
pnpm install
cp .env.local.example .env.local         # fill Clerk keys
pnpm dev                                  # http://localhost:3000
```

The dev server expects the FastAPI backend on `http://localhost:8000` (override via `NEXT_PUBLIC_API_BASE`). `next.config.mjs` rewrites `/api/*` to that host so the React code can call the API without CORS issues.

### Dev auth bypass

For local testing without Clerk, set `NEXT_PUBLIC_DEV_USER=usr_demo` and start the backend with `DEV_AUTH_USER=usr_demo`. All requests will carry `X-Dev-User: usr_demo` and the backend treats them as that user.

## Routes

- `/` — landing → redirect to `/chat` when signed in
- `/(app)/chat` — streaming chat with IRAC accordion, Mermaid render, source panel, week filter
- `/(app)/library` — drag-drop upload (PDF/DOCX/MD/TXT/PNG/JPG), file list with status + delete
- `/(app)/exam` — generate + take + grade MCQ / short-answer / past-paper (Phase 4)
- `/(app)/settings` — token usage + model preferences (Phase 5)

## Tests

```bash
pnpm test         # vitest run
pnpm test:watch   # vitest watch
pnpm typecheck    # tsc --noEmit
```

Tests cover the SSE parser (`src/lib/streaming.ts`) and pure utils (`src/lib/utils.ts`). Component + integration tests come post-Phase-4 once the agent endpoints settle.
