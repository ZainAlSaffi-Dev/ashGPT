# ADR: In-chat Memory Compression

## Status

Accepted, first production version.

## Current Flow

1. `backend/api/routes_chat.py` loads all persisted `messages` for the session before the new user row is saved.
2. `backend/src/agent/graph.py:run_query` normalises that transcript.
3. The last `CHAT_HISTORY_MAX_MESSAGES` rows are passed as verbatim `chat_history`.
4. Older rows are deterministically compressed into `conversation_memory`.
5. Router, retrieval, ratio, chronology, and synthesis prompts receive both the compressed memory and recent transcript.
6. Retrieval query packing uses compressed memory anchors for earlier shorthand, authorities, study goals, and jurisdiction.
7. The final assistant row still persists the answer, sources, IRAC, diagram, verification, and latency exactly as before.

## Decision

Use a hybrid model:

- Recent window: keep the latest 24 messages verbatim.
- Durable session memory: compress older messages into a short summary plus typed facts.
- Fact types: jurisdiction, shorthand, study goal, source constraint, authority discussed, correction.
- Telemetry: expose `history_overflow` for compatibility and a `memory` SSE event with compression stats.
- Storage: no schema change yet. Memory is rebuilt from per-session persisted messages on each request, preserving user/session isolation and avoiding stale stored summaries.

## Grounding Contract

Compressed memory is not legal source evidence. It can resolve:

- pronouns and coreference;
- student goals and preferences;
- current jurisdiction or course context;
- user-defined shorthand;
- authorities already discussed;
- explicit corrections.

Every legal proposition in an answer must still be grounded in retrieved uploaded sources with `[S#]` citations or marked `[external]`. Prompt blocks label memory as session context only.

## Rejected Options

- Rolling window only: loses early constraints and shorthand after long chats.
- Persisted LLM summaries only: risks stale or hallucinated memory without a correction contract.
- Vector retrieval over prior turns: adds infrastructure and ranking complexity before profiling proves it is needed.
- Cross-session profile memory: useful later, but privacy and namespace semantics need a separate design.

## Known Limits

The first compressor is deterministic and conservative. It recognises common legal-study memory patterns but will not capture every nuanced preference. If profiling shows repeated loss of older context, the next step is a persisted per-session summary row with explicit versioning and correction replay, not Redis or a separate vector store by default.
