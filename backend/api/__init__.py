"""FastAPI HTTP layer.

Routes mounted in ``api.main``:

  * POST /chat                       — Server-Sent Events stream
  * GET  /sessions                   — list user's sessions
  * POST /sessions                   — create a session
  * GET  /sessions/{id}/messages     — list messages
  * POST /uploads/presign            — get a presigned PUT url + file_id
  * POST /uploads/local/{key}        — local-blob receive (dev only)
  * POST /uploads/{file_id}/process  — kick ingestion
  * GET  /files                      — list user's files
  * DELETE /files/{file_id}          — remove file + chunks + vectors
"""
