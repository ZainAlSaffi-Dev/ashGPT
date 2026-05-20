# Chat History + Follow-up Notes

Deep dive into how multi-turn chat works today and where it must improve.
Pair with `polish_plan.md` (Stage 1 + Stage 4 cover this surface).

## What we have

### Storage
- D1 `sessions` (`backend/src/storage/db.py`): `id, user_id, title, created_at, updated_at`.
- D1 `messages`: `id, session_id, user_id, role, content, intent, retrieved_chunk_ids, latency_ms, tokens_in, tokens_out, verification, created_at`.
- Messages are ordered by `created_at ASC` in `list_messages`.

### Backend wiring
- `routes_chat.py` accepts `ChatRequest(query, session_id, week_filter)`:
  - `_ensure_session(session_id, user, db)` either fetches the existing row or creates a new one with title = first 80 chars of the query.
  - `_load_history(session_id)` queries `messages` ordered by `created_at ASC` and returns `[{role, content}, …]`.
  - The new user message is persisted **before** `run_query` runs.
  - After streaming finishes the assistant message is persisted with `intent`, `retrieved_chunk_ids`, `latency_ms`, `verification`.
- `graph.run_query` calls `prepare_chat_history_for_run(raw)` → normalises roles, truncates each turn to `CHAT_HISTORY_MAX_CHARS_PER_MESSAGE` (3500), caps to the last `CHAT_HISTORY_MAX_MESSAGES` (24), seeds `AgentState.chat_history`.
- `router_node` (`backend/src/agent/nodes.py`) sees the full transcript when classifying intent — already wired.
- `retrieval_node` calls `build_retrieval_query(current_query, chat_history)` (`backend/src/agent/chat_memory.py`) which appends the last assistant answer (capped at `CHAT_HISTORY_MAX_ASSISTANT_TAIL_CHARS = 1200`) to the embed query so dense retrieval picks up the referent.
- Synthesis and ratio / chronology nodes receive `format_transcript_for_llm(history)` as a prefix block.

### Frontend wiring
- `useChat` (`frontend/src/lib/useChat.ts`) keeps `sessionId` in React state. Captured from the `done` SSE event. Passed back as `session_id` on the next `send`.
- `reset()` clears `turns` and `sessionId` → "New chat".
- `lib/api.ts` exposes `listSessions`, `listMessages`, `createSession`, `deleteSession`. **None of them are called from the chat page yet.**
- Sidebar (`components/Sidebar.tsx`) is a static nav — no session list.

## What works

- Single-tab linear conversation: `sessionId` is captured after turn 1 and reused on turn 2+. D1 persists everything. Refresh via `/chat` re-renders an empty surface but the D1 history is still there.
- Router intent classification uses the transcript, so "explain it again" rounds-trip correctly.
- Retrieval rewrite ensures the dense leg sees enough of the prior assistant answer to anchor pronouns like "the case" when the referent appeared in the immediately preceding turn.
- Per-message latency and verification are recorded for audit.

## Where it breaks (and what to fix)

| Symptom | Root cause | Plan stage |
|---------|------------|------------|
| Tab swap / reload starts a new session | `sessionId` is React state only; no URL or localStorage persistence | Stage 2 |
| Sidebar has no past sessions | `listSessions` never called | Stage 2 |
| Chat page opens empty even with history | No `listMessages` hydration on mount | Stage 2 |
| "The first case I asked about" misses if turn ≥ 3 | `build_retrieval_query` only packs the last assistant answer (1200 chars) | Stage 4 |
| Pronouns ("it", "that doctrine") survive into the retrieval query unresolved | No LLM coreference rewrite, only the last-answer excerpt | Stage 4 |
| Failed graph leaves orphan user message | User message persisted before `run_query` runs | Stage 1 |
| Confidence-gated escalation re-runs synthesis but the DB doesn't capture which model wrote the final answer | `verification` JSON has no `synthesis_model` / `escalated` fields | Stage 1 |
| Long sessions silently lose old turns at the 24-turn cap | `prepare_chat_history_for_run` drops without surfacing | Stage 1 + Stage 4 (token-budget cap) |
| Multi-turn flow has no integration test | Only single-shot tests against mocked graph | Stage 1 |

## Target behaviour after polish

- Address bar shows `/chat/<sessionId>` after the first turn (Next.js `useRouter().replace`).
- Reload restores the conversation by hydrating `listMessages(sessionId)` into `turns` before the next send.
- Sidebar lists recent sessions (paginated optional) — clicking switches `useRouter().push('/chat/' + id)`.
- "New chat" → `useRouter().push('/chat')`.
- A failed graph rolls back the user message (or marks it `status='failed'`).
- The retrieval rewrite uses up to the last *N* turns and an LLM coreference pass (cheap router-tier model) to produce a self-contained query — logged onto `AgentState.rewritten_query` for inspection.
- Truncation reasons surface as a `chat_history_overflow` field that the UI can warn on ("Older turns trimmed to fit context").
- `verification` records `synthesis_model` and `escalated: bool`.
- Multi-turn integration test (mocked graph) walks 3 turns through the same session_id and asserts ordering + history-loading.

## Surface area to touch (Stage 1 + Stage 4)

- `backend/api/routes_chat.py` — wrap user-message persistence in `try/except` with rollback on graph failure; record `synthesis_model` + `escalated` into the assistant `verification` blob.
- `backend/src/agent/chat_memory.py` — multi-turn assistant context, optional LLM rewrite (gated on a flag), `chat_history_overflow` surfaced.
- `backend/src/agent/graph.py` — new node `query_rewriter_node` (router-tier model) between router and retrieval when the router signals "follow-up".
- `backend/src/agent/state.py` — add `rewritten_query`, `chat_history_overflow`.
- `backend/api/schemas.py` — extend `MessageOut` with `synthesis_model`, `escalated`.
- `backend/tests/test_chat_memory.py` + new `backend/tests/test_multi_turn_chat.py`.
- `frontend/src/app/(app)/chat/[sessionId]/page.tsx` (NEW) — dynamic route.
- `frontend/src/lib/useChat.ts` — accept `initialSessionId` + hydrate on mount.
- `frontend/src/components/Sidebar.tsx` — render session list + active marker.

## Out of scope here

- Pagination of session history (200+ session users) — defer.
- Cross-device sync of in-flight drafts — defer.
- Server-sent typing indicators — defer.
