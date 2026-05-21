# Project-Scoped Library, Sessions, and Retrieval Design

## Goal

Make ashGPT behave like a durable study workspace:

- a user signs in and sees the same library, chats, and indexed notes every time;
- notes are grouped into project/subject folders such as `LAWS1100 Torts` or `Equity`;
- chats can be scoped to one project, folder, file set, week, or the whole library;
- retrieval is fast, tenant-safe, and never leaks another user or another project by accident;
- upload and ingestion survive refreshes, navigation, retries, and later logins.

This is the most important product boundary: `user -> project -> folders/files -> sessions/messages -> retrieval scope`.

## Current System Map

### Users

Every protected backend route depends on `current_user`.

Flow:

1. Cloudflare Worker verifies Clerk and forwards `X-User-Id`, or local/dev routes verify the JWT/dev header directly.
2. `backend/api/deps.py` resolves Clerk `sub`.
3. `backend/src/storage/db.py:get_or_create_user` upserts a `users` row by `clerk_id`.
4. Domain rows store internal `users.id`, not Clerk `sub`.

Current persistence answer: users do not lose files or sessions when they log in again, as long as Clerk returns the same `sub`. The one caveat is production Worker auth currently forwards only user id, so `users.email` may remain null.

### Sessions and History

Current tables:

- `sessions`: `id`, `user_id`, `title`, timestamps.
- `messages`: `session_id`, `user_id`, role/content, sources, IRAC, mermaid, verification, latency, retrieved ids.

Current flow:

1. `POST /chat` ensures or creates a user-scoped session.
2. It loads prior messages for that session.
3. It persists the current user message before running the graph.
4. On success, it persists the assistant message with source metadata.
5. On graph failure, it deletes the just-written user message to avoid orphan history.

Current limitation: sessions do not belong to a project/folder. A chat always retrieves from the whole user namespace, optionally filtered by `week_filter`.

### Uploads and Library

Current tables:

- `files`: `user_id`, `name`, `mime`, `blob_key`, `status`, `doc_type`, `week`, `chunk_count`.
- `chunks`: `file_id`, `user_id`, text content, source, doc_type, week, page, image path.
- `vectors` table behind pgvector: vector, namespace, content, JSON metadata.

Current upload flow:

1. `POST /uploads/presign` creates a `files` row with `status="uploaded"`.
2. Browser uploads directly to R2 or local backend.
3. `POST /uploads/{file_id}/process` validates blob existence, extracts text, chunks, embeds, upserts vectors, writes `chunks`, marks file `ready`.
4. `GET /files` returns all files for the signed-in user.

Current persistence answer: original documents are stored in R2/local blob storage under a user-derived key, not in the SQL DB. The DB stores metadata, extracted chunks, and vector embeddings. Users should not need to re-upload after login.

### Retrieval

Current retrieval scope:

- vector namespace = internal `users.id`;
- filters = `week`, `doc_types`;
- chat API exposes only `week_filter`;
- ingestion metadata includes `file_id`, but retrieval does not expose or filter by it yet.

Current limitation: exact file/folder/project retrieval is not first-class. This is the main thing to fix.

## Storage Size Model

User profile rows are tiny. Document storage is split:

- original files: R2 object storage;
- extracted text chunks: Postgres `chunks.content`;
- embeddings: pgvector `vectors.embedding`;
- citation/source metadata: JSON on messages and vectors.

Approximate per chunk:

- text chunk: usually 1.5-4 KB depending on legal text density;
- embedding: `EMBEDDING_DIMENSIONS = 2560`, usually about 10 KB raw float data plus row/index overhead;
- metadata and indexes: variable, often another 1-3 KB.

Rule of thumb: plan for 15-25 KB database/vector storage per chunk before index overhead. A large subject with 2,000 chunks could be around 30-50 MB in Postgres/vector storage, while the original PDFs live in R2. This is manageable, but project scoping and cache invalidation must be deliberate.

## Proposed Data Model

Use `project` as the top-level product concept. The UI can call it “Subject” if that is friendlier.

### New Tables

`projects`

- `id`
- `user_id`
- `name`
- `slug`
- `description`
- `color`
- `archived_at`
- `created_at`
- `updated_at`

`folders`

- `id`
- `user_id`
- `project_id`
- `parent_id` nullable for nested folders
- `name`
- `sort_order`
- `created_at`
- `updated_at`

`ingestion_jobs`

- `id`
- `user_id`
- `file_id`
- `status`: `queued | processing | ready | failed`
- `attempts`
- `error`
- `locked_at`
- `created_at`
- `updated_at`

### Existing Table Changes

`files`

- add `project_id`
- add `folder_id`
- add `uploaded_by`
- add `content_hash`
- add `last_indexed_at`

`chunks`

- add `project_id`
- add `folder_id`
- keep `file_id`, `user_id`, `page`, `chunk_index`

`sessions`

- add `project_id`
- add `folder_id` nullable
- add `scope` JSON snapshot, e.g. `{ "type": "project", "project_id": "..." }`

`messages`

- add `scope` JSON snapshot
- extend persisted sources with `chunk_id`, `file_id`, `file_name`, `page`, `project_id`, `folder_id`

`answer_cache`

- add or fold into cache key: `scope_hash`
- cache key becomes `(user_id, normalised_query, scope_hash, sorted_chunk_ids)`

`vectors.metadata`

Stamp every vector row with:

- `chunk_id`
- `file_id`
- `file_name`
- `project_id`
- `folder_id`
- `week`
- `doc_type`
- `source`
- `page`

No separate vector namespace per project. Keep namespace as `user_id` for hard tenant isolation and use metadata filters for project/folder/file scope.

## API Design

### Projects

- `GET /projects`
- `POST /projects`
- `GET /projects/{project_id}`
- `PATCH /projects/{project_id}`
- `DELETE /projects/{project_id}` soft-archive first

### Folders

- `GET /projects/{project_id}/folders`
- `POST /projects/{project_id}/folders`
- `PATCH /folders/{folder_id}`
- `DELETE /folders/{folder_id}`

Folder delete should be blocked if non-empty unless `?recursive=true` is explicit.

### Files

- `GET /files?project_id=&folder_id=&status=`
- `POST /uploads/presign` with `project_id`, `folder_id`, `doc_type`, `week`
- `POST /uploads/{file_id}/process` idempotent
- `PATCH /files/{file_id}` to move folders, retag doc type/week, rename display name
- `DELETE /files/{file_id}`

File move must update:

- `files.project_id/folder_id`;
- `chunks.project_id/folder_id`;
- vector metadata for all chunk ids;
- BM25 cache for affected user/project scopes.

### Sessions and Chat

- `GET /sessions?project_id=`
- `POST /sessions` with optional `project_id`, `folder_id`, `scope`
- `GET /sessions/{session_id}/messages`
- `POST /chat` with:

```json
{
  "query": "Explain the duty element",
  "session_id": "optional",
  "scope": {
    "type": "project",
    "project_id": "..."
  }
}
```

Supported scope types:

- `all`
- `project`
- `folder`
- `files`
- `week`
- `doc_type`

Backward compatibility: keep `week_filter`, but internally translate it into `scope`.

## Retrieval Contract

Retrieval must accept a `RetrievalScope` object from chat/exam/session state.

Example:

```python
RetrievalScope(
    project_id="...",
    folder_id=None,
    file_ids=[],
    week=None,
    doc_types=None,
)
```

Rules:

- Always pass `namespace=user.id`.
- Never retrieve without namespace in production.
- Apply project/folder/file filters in both dense and BM25 legs.
- If an explicit project/folder/file scope returns no chunks, return “no material in this scope” instead of silently falling back to all user documents.
- Persist the scope snapshot on every message so reloads and citations stay understandable later.

## Cache Strategy

### Frontend

TanStack query keys should include scope:

- `['projects']`
- `['folders', projectId]`
- `['files', { projectId, folderId, status }]`
- `['sessions', { projectId }]`
- `['messages', sessionId]`

Do not use one global `['files']` cache after scoped library ships. That would create stale cross-project UI.

### Backend

BM25 cache should be scoped by user plus filter hash:

- current: namespace only;
- proposed: `bm25:{user_id}:{scope_hash}`.

Answer cache should include `scope_hash`. Otherwise the same wording in different projects could replay the wrong answer.

Upload/process cache behavior:

- keep R2 as source of original document truth;
- keep Postgres chunks/vector metadata as search truth;
- use `content_hash` to detect duplicate uploads inside the same project;
- invalidate only the affected user/project BM25 caches on ingestion, move, delete, and retag.

## Robust Upload Design

Current upload works, but processing is too tied to the client surviving long enough to call `process`.

Target behavior:

1. `POST /uploads/presign` creates a file row and ingestion job.
2. Browser uploads to R2.
3. Client calls `POST /uploads/{file_id}/process`, but this endpoint is idempotent and simply starts or resumes the job.
4. If client leaves, `GET /files` still shows `uploaded` or `queued`, and the UI can resume by calling `process` again.
5. Job lock prevents double ingestion.
6. Reprocessing a failed file clears old chunks/vectors only after new ingestion succeeds, or uses a staged chunk set.

This gives the “I logged back in and my notes are still there” feeling even if the previous tab died mid-ingestion.

## Navigation and Refresh Fixes

Known current risks:

- First chat on `/chat` has no durable session id until the SSE `done` event. Reload before `done` loses the optimistic turn.
- Sidebar links can navigate mid-stream. The hook aborts the stream, but the user gets no explicit “stream cancelled” state.
- Library upload progress is local component state. Navigating away loses progress display.
- `useFiles` is currently global and has `refetchOnMount: false`, so finished background ingestion can remain visually stale for up to five minutes if no invalidation fires.

Design fixes:

- Create the session before streaming when the user sends the first query, then stream into that session. URL promotion can happen immediately after session creation, not after final answer.
- Add a global in-flight upload/process store keyed by `file_id`, or derive upload/process progress from `GET /files` and `ingestion_jobs`.
- Scope query keys by project/folder.
- Add route-change guard while chat is busy: either confirm cancellation or explicitly show “stream cancelled” after navigation.
- Add a lightweight endpoint health telemetry event for chat: `session_created`, `stream_started`, `stream_aborted`, `message_persisted`.

## Endpoint Stress-Test Matrix

| Flow | Expected endpoints | Pass condition |
|---|---|---|
| Signed-out `/chat/abc` | Clerk proxy only, no app API before auth | same-origin redirect to `/?redirect_url=...` |
| Signed-in cold `/chat` | `GET /users/me`, `GET /sessions`, scoped `GET /files` | empty composer interactive, no redirect loop |
| First send | `POST /sessions`, `POST /chat` | URL becomes `/chat/{id}` before or during stream; reload can recover |
| Existing chat send | `GET /sessions/{id}/messages`, `POST /chat` | stays on same session; one assistant row persisted |
| Switch chats mid-stream | abort old `POST /chat`, fetch new messages | new chat never shows old optimistic turns |
| Sidebar new chat mid-stream | abort old stream or confirm cancel | no late `router.replace` back to old session |
| Upload success | presign, R2 PUT, process, `GET /files` | row reaches `ready`, chunks > 0 |
| Navigate away mid-upload | upload/process may continue or resume | library later shows accurate status |
| Delete file | `DELETE /files/{id}`, `GET /files` | DB chunks and vectors removed for that file |
| Move file to folder | `PATCH /files/{id}` | retrieval from old folder excludes it; new folder includes it |
| Project-scoped chat | `POST /chat` with project scope | retrieved sources all have that project id |
| File-scoped chat | `POST /chat` with file ids | no fallback to all docs if file has no chunks |
| Exam file/project scope | `POST /exam/generate` with scope | questions only use scoped chunks |
| Sign out/in as other user | fresh scoped queries | no previous user cache/data visible |

## Implementation Phases

Implementation note (2026-05-21): v1 has shipped the core backend contract and a basic frontend scope selector. `project_id` / `folder_id` remain nullable for legacy rows. BM25 cache keys are now scoped as `user_id:scope_hash`, while vector namespace remains `user_id`. File moves update SQL file rows, chunk metadata, and pgvector/in-memory vector metadata; Cloudflare Vectorize would need a re-upsert path if it is used again.

### Phase 1: Scope Schema

- Add `Project` and `Folder` models.
- Add `project_id`, `folder_id`, and `scope` columns to files/chunks/sessions/messages.
- Add inline Postgres migrations in `backend/src/storage/db.py`.
- Add schemas and CRUD routes.
- Tests: tenant isolation, folder CRUD, file move, non-empty folder delete.

### Phase 2: Scoped Upload and Ingestion

- Extend presign request with `project_id` and `folder_id`.
- Stamp file/chunk/vector metadata with project/folder.
- Add `ingestion_jobs` and idempotent processing.
- Fix file delete to explicitly load chunk ids before vector cleanup.
- Tests: upload persistence, retry, delete cleanup, project/folder metadata on chunks/vectors.

### Phase 3: Scoped Retrieval and Chat

- Add `RetrievalScope` to backend schemas/state.
- Pass scope through `run_query -> retrieval_node -> retrieve_all`.
- Extend `_build_filter` for `project_id`, `folder_id`, `file_id`.
- Persist scope snapshots on messages.
- Tests: project-scoped chat, folder-scoped chat, no fallback when explicit scope is empty, citation metadata rehydration.

### Phase 4: Frontend Project Library

- Add project sidebar/list in Library.
- Add folder tree and move/rename actions.
- Scope query keys by project/folder.
- Add project-scoped chat entry point.
- Tests: cache isolation between projects, upload resume, project switch, folder/file deep links.

### Phase 5: Endpoint QA and Hardening

- Browser matrix: desktop/mobile, signed-out/signed-in, cold load/reload, mid-stream nav, mid-upload nav.
- Network assertions: no duplicate session/message/file fetch storms, no wrong-session flash, no stale file status after process.
- Add observability fields: request id, session id, project id, file id, ingestion job id, retrieval scope hash.

## Non-Negotiables

- User namespace isolation is mandatory at every SQL/vector/blob boundary.
- Project/folder/file scope must be explicit and testable.
- Explicit empty scope must not fall back to the full library.
- Chat history and citations must reload without rerunning retrieval.
- Upload state must be recoverable after refresh.
- Query caches must include project/folder scope.
- Original documents stay in R2; extracted chunks and vectors stay in Postgres/pgvector.
