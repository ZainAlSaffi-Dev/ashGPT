# Project-Scoped Library QA Report

Date: 2026-05-21

## Coverage Run

- Backend full suite: `uv run python -m pytest -q`
  - Result: 213 passed, 26 skipped.
- Frontend typecheck: `npm run typecheck -- --pretty false`
  - Result: passed.
- Frontend unit tests: `npm test -- --run`
  - Result: 33 passed.
- Frontend production build: `npm run build`
  - Result: passed.

## Endpoint Matrix

| Flow | Status | Notes |
|---|---|---|
| Signed-out `/chat/abc` | Not browser-tested | Existing middleware/auth behavior unchanged in this slice. |
| Signed-in cold `/chat` | Unit-covered | Scoped chat body/type plumbing compiles; browser auth not exercised locally. |
| First send | Unit-covered | Existing stream/session tests pass; scoped `done.scope` remains optional for compatibility. |
| Existing chat send | Unit-covered | Multi-turn backend tests pass after scoped `run_query` compatibility guard. |
| Switch chats mid-stream | Unit-covered | Existing `useChat` abort/late-promotion tests pass. |
| Sidebar new chat mid-stream | Not browser-tested | No sidebar redesign in this slice. |
| Upload success | API-covered | Project/folder presign/list/delete behavior covered by `test_projects_api.py`. |
| Navigate away mid-upload | Not browser-tested | Backend process endpoint is idempotent for ready rows; UI still needs deeper persistent upload-store work. |
| Delete file | Existing tests pass | Delete now explicitly loads chunk ids before vector cleanup. |
| Move file to folder | Unit/API-covered | `PATCH /files/{id}` updates file, chunk metadata, vector metadata where backend supports it, and invalidates BM25. |
| Project-scoped chat | Backend covered by filter tests | Dense/BM25 filter builders cover project/folder/file scope; browser send plumbing compiles. |
| File-scoped chat | Unit-covered | Explicit empty `files` scope returns empty retrieval instead of widening. |
| Exam file/project scope | Partially covered | Legacy file/week/past-paper scopes now use `RetrievalScope` and no fallback; project/folder exam UI/API remains future work. |
| Sign out/in as other user | API-covered | Project CRUD cross-user access returns empty/404 via `X-User-Id` test path. |

## Residual Risks

- Browser matrix on `https://ashgpt.xyz` still needs a live authenticated pass for route flicker, upload navigation, and mobile folder selection.
- Cloudflare Vectorize cannot update metadata in place; pgvector and memory stores can. If `VECTOR_BACKEND=vectorize` is revived, file move should re-upsert affected vectors.
- Existing legacy files with `project_id = NULL` remain visible only in all-library scope until a default-project backfill is added.
- GitHub Actions Worker deploy will keep failing until `CLOUDFLARE_API_TOKEN` gets zone-level `Workers Routes: Edit/Write` for `ashgpt.xyz`.
