-- D1 schema mirroring backend/src/storage/db.py (SQLite-flavoured DDL).
-- Run once after `wrangler d1 create lawgpt-prod`:
--   wrangler d1 execute lawgpt-prod --file=infra/d1-schema.sql

CREATE TABLE IF NOT EXISTS users (
    id           TEXT PRIMARY KEY,
    clerk_id     TEXT NOT NULL UNIQUE,
    email        TEXT,
    token_budget INTEGER NOT NULL DEFAULT 1000000,
    tokens_used  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS users_clerk_id_idx ON users (clerk_id);

CREATE TABLE IF NOT EXISTS files (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    mime        TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    blob_key    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'uploaded',
    error       TEXT,
    doc_type    TEXT NOT NULL DEFAULT 'note',
    week        TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS files_user_idx ON files (user_id);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    file_id      TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    user_id      TEXT NOT NULL,
    content      TEXT NOT NULL,
    page         INTEGER,
    chunk_index  INTEGER NOT NULL DEFAULT 0,
    source       TEXT NOT NULL DEFAULT '',
    doc_type     TEXT NOT NULL DEFAULT 'note',
    week         TEXT,
    image_path   TEXT,
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS chunks_file_idx ON chunks (file_id);
CREATE INDEX IF NOT EXISTS chunks_user_idx ON chunks (user_id);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT NOT NULL DEFAULT 'New chat',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions (user_id);

CREATE TABLE IF NOT EXISTS messages (
    id                   TEXT PRIMARY KEY,
    session_id           TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id              TEXT NOT NULL,
    role                 TEXT NOT NULL,
    content              TEXT NOT NULL,
    retrieved_chunk_ids  TEXT,  -- JSON array
    intent               TEXT,
    latency_ms           INTEGER,
    tokens_in            INTEGER,
    tokens_out           INTEGER,
    verification         TEXT,  -- JSON object
    created_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id);

CREATE TABLE IF NOT EXISTS exams (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    scope_type  TEXT NOT NULL,
    scope_value TEXT,
    num_mcq     INTEGER NOT NULL DEFAULT 0,
    num_short   INTEGER NOT NULL DEFAULT 0,
    difficulty  TEXT NOT NULL DEFAULT 'medium',
    questions   TEXT NOT NULL,  -- JSON object
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS exams_user_idx ON exams (user_id);

CREATE TABLE IF NOT EXISTS attempts (
    id           TEXT PRIMARY KEY,
    exam_id      TEXT NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    user_id      TEXT NOT NULL,
    answers      TEXT NOT NULL,  -- JSON object
    results      TEXT,            -- JSON object
    score        REAL,
    submitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS attempts_exam_idx ON attempts (exam_id);

CREATE TABLE IF NOT EXISTS answer_cache (
    cache_key   TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    answer      TEXT NOT NULL,
    payload     TEXT NOT NULL,  -- JSON object
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS answer_cache_user_idx ON answer_cache (user_id);
