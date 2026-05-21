"""Async SQLAlchemy models + session factory for LawGPT.

Schema is multi-tenant by design: every domain row carries ``user_id`` so the
same DB can serve many users with row-level isolation enforced at the query
layer. Phase 5 will add per-user token budgets here.

Example:
    >>> import asyncio
    >>> from src.storage.db import get_engine, init_db, get_session, User
    >>> async def example():
    ...     engine = get_engine("sqlite+aiosqlite:///:memory:")
    ...     await init_db(engine)
    ...     async with get_session(engine) as s:
    ...         user = User(clerk_id="usr_demo", email="a@b.c")
    ...         s.add(user)
    ...         await s.commit()
    ...         return user.id
    >>> asyncio.run(example())  # doctest: +SKIP
    1
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    clerk_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    token_budget: Mapped[int] = mapped_column(Integer, default=1_000_000)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    onboarded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    files: Mapped[list["File"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sessions: Mapped[list["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    projects: Mapped[list["Project"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    slug: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    color: Mapped[str | None] = mapped_column(String(32), nullable=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    user: Mapped[User] = relationship(back_populates="projects")
    folders: Mapped[list["Folder"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    files: Mapped[list["File"]] = relationship(back_populates="project")
    sessions: Mapped[list["Session"]] = relationship(back_populates="project")


class Folder(Base):
    __tablename__ = "folders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str] = mapped_column(String(32), ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("folders.id", ondelete="CASCADE"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    project: Mapped[Project] = relationship(back_populates="folders")
    parent: Mapped["Folder | None"] = relationship(remote_side="Folder.id", back_populates="children")
    children: Mapped[list["Folder"]] = relationship(back_populates="parent", cascade="all, delete-orphan")
    files: Mapped[list["File"]] = relationship(back_populates="folder")


class File(Base):
    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(512))
    mime: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    blob_key: Mapped[str] = mapped_column(String(512))  # R2 object key or local relative path
    project_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True)
    folder_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("folders.id", ondelete="SET NULL"), nullable=True, index=True)
    uploaded_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")  # uploaded | processing | ready | failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_type: Mapped[str] = mapped_column(String(32), default="note")  # note | past_paper | slide | reading
    week: Mapped[str | None] = mapped_column(String(32), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped[User] = relationship(back_populates="files")
    project: Mapped[Project | None] = relationship(back_populates="files")
    folder: Mapped[Folder | None] = relationship(back_populates="files")
    chunks: Mapped[list["ChunkMeta"]] = relationship(back_populates="file", cascade="all, delete-orphan")


class ChunkMeta(Base):
    """Metadata copy of vectors. Vector itself lives in the vector store."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    file_id: Mapped[str] = mapped_column(String(32), ForeignKey("files.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    project_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    folder_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    content: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(512), default="")
    doc_type: Mapped[str] = mapped_column(String(32), default="note")
    week: Mapped[str | None] = mapped_column(String(32), nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    file: Mapped[File] = relationship(back_populates="chunks")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True)
    folder_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("folders.id", ondelete="SET NULL"), nullable=True, index=True)
    scope: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    title: Mapped[str] = mapped_column(String(255), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    user: Mapped[User] = relationship(back_populates="sessions")
    project: Mapped[Project | None] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(32), ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    scope: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    retrieved_chunk_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    # Full SourceHit list (source/doc_type/week/snippet) so reloaded
    # conversations can rehydrate the citation popovers without re-running
    # retrieval. retrieved_chunk_ids stays for cheaper joins / analytics.
    sources: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    irac: Mapped[str | None] = mapped_column(Text, nullable=True)
    mermaid: Mapped[str | None] = mapped_column(Text, nullable=True)
    intent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verification: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[Session] = relationship(back_populates="messages")


class Exam(Base):
    __tablename__ = "exams"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    scope_type: Mapped[str] = mapped_column(String(32))  # file | week | all | past_paper
    scope_value: Mapped[str | None] = mapped_column(String(255), nullable=True)
    scope: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    num_mcq: Mapped[int] = mapped_column(Integer, default=0)
    num_short: Mapped[int] = mapped_column(Integer, default=0)
    difficulty: Mapped[str] = mapped_column(String(16), default="medium")
    questions: Mapped[dict] = mapped_column(JSON)  # full generated exam payload
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    exam_id: Mapped[str] = mapped_column(String(32), ForeignKey("exams.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    answers: Mapped[dict] = mapped_column(JSON)
    results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    score: Mapped[float | None] = mapped_column(nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AnswerCache(Base):
    """Phase 5: semantic answer cache. Hash key on (user_id, query_norm, chunk_ids)."""

    __tablename__ = "answer_cache"

    cache_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    scope_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    answer: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    file_id: Mapped[str] = mapped_column(String(32), ForeignKey("files.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


# ── Engine + session helpers ──────────────────────────────────────────────────

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _normalize_async_url(url: str) -> str:
    """Coerce plain Postgres URLs to the async psycopg3 driver scheme.

    ``sqlalchemy.create_async_engine("postgresql://...")`` picks psycopg2
    which isn't installed (we ship psycopg[binary]). Rewrite to
    ``postgresql+psycopg://`` so the async psycopg3 dialect is used.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def get_engine(database_url: str | None = None) -> AsyncEngine:
    """Return the singleton engine. Call ``reset_engine`` between tests."""
    global _engine, _sessionmaker
    if _engine is None:
        if database_url is None:
            from src.config import get_settings

            database_url = get_settings().database_url
        database_url = _normalize_async_url(database_url)
        _engine = create_async_engine(database_url, echo=False, future=True)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def reset_engine() -> None:
    """Drop cached engine so the next ``get_engine`` rebuilds (for tests)."""
    global _engine, _sessionmaker
    _engine = None
    _sessionmaker = None


@asynccontextmanager
async def get_session(engine: AsyncEngine | None = None) -> AsyncIterator[AsyncSession]:
    """Yield an async session against the configured engine."""
    if engine is None:
        engine = get_engine()
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with _sessionmaker() as s:
        try:
            yield s
        except Exception:
            await s.rollback()
            raise


async def init_db(engine: AsyncEngine | None = None) -> None:
    """Create all tables and apply lightweight idempotent migrations.

    ``create_all`` only creates *missing* tables — it doesn't ALTER existing
    ones. For columns added after the first deploy we issue
    ``ADD COLUMN IF NOT EXISTS`` so production rows pick up new fields
    without an out-of-band migration step. Once we have Alembic wired up
    these inline migrations should move into versioned scripts.
    """
    if engine is None:
        engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _apply_inline_migrations(conn)


async def _apply_inline_migrations(conn) -> None:  # type: ignore[no-untyped-def]
    """Run ``ADD COLUMN IF NOT EXISTS`` for columns added post-v1.

    Postgres supports the ``IF NOT EXISTS`` clause; SQLite (used by tests)
    does not, but its ``create_all`` already produces the up-to-date schema
    on a fresh DB, so we just skip the ALTERs there.
    """
    from sqlalchemy import text

    dialect = conn.dialect.name
    if dialect != "postgresql":
        return

    statements = (
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS sources JSONB",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS irac TEXT",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS mermaid TEXT",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS scope JSONB",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarded_at TIMESTAMPTZ",
        "ALTER TABLE files ADD COLUMN IF NOT EXISTS project_id VARCHAR(32)",
        "ALTER TABLE files ADD COLUMN IF NOT EXISTS folder_id VARCHAR(32)",
        "ALTER TABLE files ADD COLUMN IF NOT EXISTS uploaded_by VARCHAR(32)",
        "ALTER TABLE files ADD COLUMN IF NOT EXISTS content_hash VARCHAR(128)",
        "ALTER TABLE files ADD COLUMN IF NOT EXISTS last_indexed_at TIMESTAMPTZ",
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS project_id VARCHAR(32)",
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS folder_id VARCHAR(32)",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS project_id VARCHAR(32)",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS folder_id VARCHAR(32)",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS scope JSONB",
        "ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS scope_hash VARCHAR(64)",
        "ALTER TABLE exams ADD COLUMN IF NOT EXISTS scope JSONB",
        "CREATE INDEX IF NOT EXISTS projects_user_archived_idx ON projects (user_id, archived_at)",
        "CREATE INDEX IF NOT EXISTS projects_user_slug_idx ON projects (user_id, slug)",
        "CREATE INDEX IF NOT EXISTS folders_user_project_idx ON folders (user_id, project_id)",
        "CREATE INDEX IF NOT EXISTS files_user_scope_idx ON files (user_id, project_id, folder_id)",
        "CREATE INDEX IF NOT EXISTS chunks_user_scope_idx ON chunks (user_id, project_id, folder_id)",
        "CREATE INDEX IF NOT EXISTS sessions_user_project_idx ON sessions (user_id, project_id)",
        "CREATE INDEX IF NOT EXISTS ingestion_jobs_user_file_idx ON ingestion_jobs (user_id, file_id)",
    )
    for stmt in statements:
        await conn.execute(text(stmt))


async def get_or_create_user(session: AsyncSession, clerk_id: str, email: str | None = None) -> User:
    """Upsert a user by clerk_id. Returns the persisted ``User`` row."""
    from sqlalchemy import select

    result = await session.execute(select(User).where(User.clerk_id == clerk_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(clerk_id=clerk_id, email=email)
        session.add(user)
        await session.commit()
        await session.refresh(user)
    return user
