"""Pydantic request/response schemas for the API layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────────────


class RetrievalScopeIn(BaseModel):
    type: Literal["all", "project", "folder", "files", "week", "doc_type"] = "all"
    project_id: str | None = None
    folder_id: str | None = None
    file_ids: list[str] | None = None
    week: str | None = None
    doc_types: list[str] | None = None


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    week_filter: str | None = None
    scope: RetrievalScopeIn | None = None


class SessionOut(BaseModel):
    id: str
    title: str
    project_id: str | None = None
    folder_id: str | None = None
    scope: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class CreateSessionRequest(BaseModel):
    title: str | None = None
    project_id: str | None = None
    folder_id: str | None = None
    scope: RetrievalScopeIn | None = None


class SourceHitOut(BaseModel):
    chunk_id: str | None = None
    file_id: str | None = None
    file_name: str | None = None
    project_id: str | None = None
    folder_id: str | None = None
    page: int | None = None
    source: str | None = None
    doc_type: str | None = None
    week: str | None = None
    snippet: str = ""


class MessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    scope: dict[str, Any] | None = None
    intent: str | None = None
    retrieved_chunk_ids: list[str] | None = None
    # Full per-citation rehydration payload so the frontend can wire up
    # the source popovers on reloaded conversations without re-running
    # retrieval.
    sources: list[SourceHitOut] | None = None
    irac: str | None = None
    mermaid: str | None = None
    latency_ms: int | None = None
    verification: dict[str, Any] | None = None
    created_at: datetime


# ── Files / uploads ───────────────────────────────────────────────────────────


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    color: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    color: str | None = None
    archived: bool | None = None


class ProjectOut(BaseModel):
    id: str
    name: str
    slug: str
    description: str | None = None
    color: str | None = None
    archived_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class FolderCreate(BaseModel):
    name: str
    parent_id: str | None = None
    sort_order: int = 0


class FolderUpdate(BaseModel):
    name: str | None = None
    parent_id: str | None = None
    sort_order: int | None = None


class FolderOut(BaseModel):
    id: str
    project_id: str
    parent_id: str | None = None
    name: str
    sort_order: int
    created_at: datetime
    updated_at: datetime


class PresignRequest(BaseModel):
    name: str = Field(..., description="Original filename (used to derive extension)")
    mime: str
    # Free-form so callers can categorise (case, statute, note, past_paper,
    # transcript, slide, …) without the backend imposing a fixed taxonomy.
    doc_type: str = "document"
    week: str | None = None
    project_id: str | None = None
    folder_id: str | None = None


class PresignResponse(BaseModel):
    file_id: str
    upload_url: str
    blob_key: str
    method: Literal["PUT", "POST"]


class FileOut(BaseModel):
    id: str
    name: str
    mime: str
    size_bytes: int
    project_id: str | None = None
    folder_id: str | None = None
    status: str
    error: str | None = None
    doc_type: str
    week: str | None = None
    chunk_count: int
    created_at: datetime


class FileUpdateRequest(BaseModel):
    name: str | None = None
    project_id: str | None = None
    folder_id: str | None = None
    doc_type: str | None = None
    week: str | None = None


class ProcessResponse(BaseModel):
    file_id: str
    status: str
    chunk_count: int
    job_id: str | None = None


# ── Exam (Phase 4) ────────────────────────────────────────────────────────────


class ExamGenerateRequest(BaseModel):
    scope_type: Literal["file", "week", "all", "past_paper"] = "all"
    scope_value: str | None = None
    num_mcq: int = 5
    num_short: int = 2
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class ExamSubmitRequest(BaseModel):
    answers: dict[str, Any]  # {question_id: chosen_idx | text}


# ── Users ─────────────────────────────────────────────────────────────────────


class UserMe(BaseModel):
    id: str
    clerk_id: str
    email: str | None = None
    onboarded_at: datetime | None = None
    created_at: datetime
