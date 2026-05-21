# Session, Memory, and Persistence Test Plan

## Product Contract

ashGPT should behave like a durable ChatGPT-style workspace:

- signing in on any device shows the user's subjects, files, sessions, and messages;
- browser cache may make the first paint fast, but SQL remains the source of truth;
- every chat belongs to the signed-in user and may optionally carry a project/folder/file scope snapshot;
- reloading `/chat/{sessionId}` rehydrates persisted messages and citations without rerunning retrieval;
- long-chat memory is rebuilt from persisted session messages, not from browser state.

## Current Storage Model

- `users` are keyed by Clerk `sub`.
- `sessions` persist `user_id`, title, timestamps, optional `project_id`, optional `folder_id`, and scope JSON.
- `messages` persist `user_id`, `session_id`, role/content, sources, IRAC, mermaid, verification, latency, retrieved chunks, and scope JSON.
- `files`, `chunks`, and vector metadata persist project/folder/file identifiers for scoped retrieval.
- Frontend TanStack cache persists selected query families in localStorage for speed only; it is cleared after sign-out and is not trusted for tenant isolation.
- In-chat memory v1 is recomputed from server messages each request: latest 24 messages verbatim, older turns compressed into deterministic `conversation_memory`.

## Backend Tests

Add or keep coverage for:

- User A cannot list, open, delete, or chat against User B sessions/messages.
- Session created with `project_id`/`folder_id` returns through `GET /sessions?project_id=...`.
- `GET /sessions/{id}/messages` returns messages ordered by creation time with source metadata intact.
- Reload simulation: create session, persist user + assistant messages, close DB session, open a new DB session, list sessions/messages and assert same data.
- Scope mismatch: sending to an existing scoped session with a conflicting project/folder/file scope is rejected.
- Long-chat memory: create more than 24 persisted messages, send a follow-up, assert the stream emits `memory` telemetry and the graph receives recent transcript plus compressed older context.
- Grounding boundary: compressed memory may influence query rewriting but final legal propositions still require retrieved sources or `[external]`.

## Frontend Tests

Add or keep coverage for:

- `useMessages` uses exact `['messages', sessionId]` keys and never uses previous-session placeholder data.
- `useSessions({ projectId })` keys are project-scoped, so ChatGPT-style sidebar sections do not mix subjects.
- Route switch `/chat/a -> /chat/b` never renders chat A turns under chat B.
- Sign-out clears persisted query cache so another user on the same browser cannot see prior sessions.
- Workspace route navigation resets the app-shell scroll container and does not stack outgoing pages above incoming pages.

## Browser Stress Matrix

Run on production custom domain first:

| Flow | Expected result |
|---|---|
| Sign in, create subject, open subject | Workspace opens at top with project name visible. |
| Create scoped chat from subject | URL becomes `/chat?project=...`, first successful send promotes to `/chat/{id}`. |
| Reload active chat | Same messages, sources, and scope rehydrate from backend. |
| Switch sessions quickly | No previous-session message flash. |
| Sign out then sign in again | Sessions/files return from backend for the same Clerk user. |
| Sign out then different user signs in | No cached projects/sessions/files from the previous user. |
| Long chat over 24 messages | Memory telemetry appears; shorthand from old turns still resolves. |
| Project with no files | Scoped chat reports empty scope, with no fallback to all documents. |

## Best-Way-Forward Notes

- Keep Postgres as the durable source of sessions/messages; do not rely on localStorage for correctness.
- Keep original documents in R2 and extracted chunks/vectors in Postgres/pgvector.
- Add pagination/search to sessions before the sidebar gets large, but keep the first implementation server-driven.
- If memory quality needs improvement, add a versioned per-session memory summary table with correction replay. Do not introduce cross-session profile memory until privacy and opt-out semantics are explicit.
- Preserve tenant isolation at every layer: SQL filters by `user_id`, vector namespace stays `user_id`, browser cache is cleared on sign-out, and API responses never trust client-provided user ids.
