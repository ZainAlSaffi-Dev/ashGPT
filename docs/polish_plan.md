# LawGPT — Polish Plan

Multi-stage plan to move LawGPT from "works end-to-end" to "production
grade". Companion docs: `architecture.md` (ground truth) and
`chat_history_notes.md` (chat deep-dive).

Each stage ships independently with its own commit set + tests. Mark a
stage **DONE** here when its acceptance criteria are met.

---

## Stage 1 — Conversation history correctness (backend)

**Goal:** the backend never loses a turn, always records who wrote the
final answer, and survives graph failures cleanly.

**Scope**
- `backend/api/routes_chat.py`
  - Wrap user-message persistence so we either commit the full pair
    (user + assistant) or roll the user row back / mark `status='failed'`
    when the graph throws.
  - After streaming completes, write `synthesis_model` and `escalated`
    (true when `_override_synthesis_model` was set) into the assistant
    message's `verification` JSON blob.
- `backend/src/agent/chat_memory.py`
  - Add `chat_history_overflow` reporting — when turns are dropped at the
    24-turn cap or chars trimmed at 3500, emit a structured note so the
    UI / logs can warn.
  - Surface the overflow via `AgentState.chat_history_overflow`.
- `backend/src/agent/state.py`
  - Add `chat_history_overflow: dict | None` and `rewritten_query: str | None`.
- `backend/tests/test_multi_turn_chat.py` (NEW)
  - Mock `run_query` to capture invocations; walk 3 turns through one
    session, assert message ordering, session_id stability, and that a
    graph failure on turn 2 doesn't orphan a user row.

**Acceptance**
- `pytest backend/tests/test_multi_turn_chat.py` green.
- Manual: kill the container mid-stream → no orphan messages in D1.
- `MessageOut` (no schema change required; lives inside `verification`)
  carries `synthesis_model` and `escalated` for every assistant turn.

---

## Stage 2 — Session continuity (frontend)

**Goal:** the URL is the source of truth for the current session;
reload / share / back-button all just work; the sidebar lists past
conversations.

**Scope**
- `frontend/src/app/(app)/chat/[sessionId]/page.tsx` (NEW)
  - Dynamic Next.js route. Reads `params.sessionId`, hydrates messages
    via `listMessages` (use `@tanstack/react-query` since it's installed
    but unused).
  - The current `chat/page.tsx` becomes `/chat` (new chat shell) and
    redirects to `/chat/<id>` once the first `done` event arrives via
    `router.replace`.
- `frontend/src/lib/useChat.ts`
  - Accept `initialSessionId?: string` and `initialTurns?: ChatTurn[]`.
  - Drop the `reset()` clear path — "new chat" should
    `router.push('/chat')` instead.
- `frontend/src/components/Sidebar.tsx`
  - Render `listSessions()` under a "History" group with active marker
    via `usePathname()`.
  - "+ New chat" button at the top.
- `frontend/src/lib/api.ts`
  - Already has the endpoints; just verify they accept the Clerk token.
- `frontend/src/app/(app)/layout.tsx`
  - Wrap children in a `QueryClientProvider` so sessions / messages get
    cached across navigation.
- Optional polish: persist the last-seen session id in `localStorage` so
  `/chat` (no param) can redirect to it on next visit.

**Acceptance**
- Open `/chat`, ask a question, watch URL become `/chat/<id>`.
- Reload the URL → all turns rehydrate, next question continues the
  same session.
- Click another session in the sidebar → switches without losing scroll
  position on the active one (router-level state, not just unmount).
- Frontend `vitest` covers the new dynamic route's hydration path.

---

## Stage 3 — UX polish

**Goal:** ChatGPT/Gemini fluidity. Smooth streaming, no jank.

**Scope (incremental, each its own PR-ish commit)**
- Streaming cursor (`▍`) on the in-flight assistant bubble.
- Scroll-lock: auto-scroll to bottom while at bottom; if the user
  scrolls up mid-stream, pause until they scroll back down.
- `framer-motion`: 150-200ms fade+slide entry on each new message.
- Textarea: `field-sizing: content` (or measured `useLayoutEffect`)
  auto-grow; Enter sends, Shift+Enter newline; draft preserved in
  `useChat` store across re-renders (already covered by Stage 2 since
  `useChat` survives navigation when lifted to provider).
- Markdown: enable `rehype-highlight` (install if absent) for code
  blocks; double-check `remark-gfm` covers tables and check-lists.
- Sidebar prefetch via `<Link prefetch>` (default in Next 15) — verify.
- Library page: skeleton loader; consider `@tanstack/react-virtual`
  only when a user has 100+ files.

**Acceptance**
- Local lighthouse / manual: send a long question, scroll up midway,
  observe stream continues but page doesn't yank back down.
- No layout shift when a new turn arrives.
- Code blocks in answers render with syntax highlight.

---

## Stage 4 — Follow-up question quality

**Goal:** "explain that further", "what about the second case", "go
back to my first question" all retrieve the right context.

**Scope**
- `backend/src/agent/nodes.py`
  - Add `query_rewriter_node` (router-tier model) inserted between the
    router and retrieval **only when** the router classified a
    follow-up (intent != "general" *and* `chat_history` non-empty).
  - The rewriter produces a self-contained search query: resolves
    pronouns, expands abbreviated case names from prior turns, drops
    chit-chat. Output written to `state.rewritten_query`.
- `backend/src/agent/graph.py`
  - Wire the conditional edge: `router → query_rewriter → retrieval`
    when `chat_history` is non-empty; else `router → retrieval`.
- `backend/src/agent/tools.py`
  - `retrieve_all` accepts an explicit `query_override` param so the
    rewritten query drives the embed call.
- `backend/src/agent/chat_memory.py`
  - When LLM rewrite is unavailable / disabled, fall back to a smarter
    deterministic packer: include the last 3 turns (capped to e.g.
    2000 chars total) instead of only the most recent assistant excerpt.
- `backend/src/config.py`
  - `USE_QUERY_REWRITER = True` flag for eval ablation.
- Tests:
  - `tests/test_query_rewriter.py` (NEW) — mock the rewriter LLM call,
    verify "explain it" with prior turn about *Mabo* becomes
    "Explain the ratio decidendi in Mabo v Queensland (No 2)".

**Acceptance**
- Manual: 3-turn conversation about a single case, third turn says
  "and what about the dissent?" — sources include the correct case.
- Eval harness: when re-run on a small multi-turn fixture, recall@k
  on follow-ups improves vs. the deterministic packer.

---

## Stage 5 — Production hygiene

**Goal:** prod won't fall over and we can debug when it does.

**Scope**
- Keep-warm cron in `infra/wrangler.toml` — every 5 min hit a
  container `/internal/warm` endpoint so the DO doesn't cold-start
  10 minutes after the last user.
- Sentry-lite: log `error` events with `console.error` from the Worker
  (already piped to Workers Logs) — make sure every catch block emits
  a structured one-line JSON.
- Backend: add a `/health` endpoint that touches pgvector with a
  `SELECT 1` so the warm cron also validates the data layer.
- Add a Playwright (or just Vitest + msw) smoke that runs against a
  staging deploy: login, ask a question, see streamed answer.
- Document deploy hygiene in `docs/deploy.md` (NEW) once Stage 5 lands.

**Acceptance**
- Cron logs visible in CF dashboard for the warm endpoint.
- `/health` returns `{"ok": true, "vector_store": "pgvector"}` with a
  round-trip latency line in logs.
- README updated to point at `docs/`.

---

## Cross-cutting (do alongside any stage)

- Eval harness rebrand: drop coursework cases or replace with a small
  general-law fixture (Stage 4 needs at least one multi-turn case).
- Type contracts: confirm `MessageOut` reflects every new field added
  (e.g. `synthesis_model`).
- Memory rules: keep `commit + push per meaningful change` (per repo
  memory) — each stage = several small commits.

---

## Sequence of attack

Default order: **Stage 1 → Stage 2 → Stage 4 → Stage 3 → Stage 5**.
Reasoning: correctness before UX; the dynamic-route + history hydration
in Stage 2 is the highest-visible win; coreference rewriter (Stage 4)
matters more than scroll polish (Stage 3); ops hygiene last.

This document is the source of truth for *what* — not *how*. Each
stage gets a focused per-task plan in `/Users/zer0/.claude/plans/` when
implementation starts.
