"""Pydantic request/response schemas for the API layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    week_filter: str | None = None


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class CreateSessionRequest(BaseModel):
    title: str | None = None


class MessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    intent: str | None = None
    retrieved_chunk_ids: list[str] | None = None
    latency_ms: int | None = None
    verification: dict[str, Any] | None = None
    created_at: datetime


# ── Files / uploads ───────────────────────────────────────────────────────────


class PresignRequest(BaseModel):
    name: str = Field(..., description="Original filename (used to derive extension)")
    mime: str
    # Free-form so callers can categorise (case, statute, note, past_paper,
    # transcript, slide, …) without the backend imposing a fixed taxonomy.
    doc_type: str = "document"
    week: str | None = None


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
    status: str
    error: str | None = None
    doc_type: str
    week: str | None = None
    chunk_count: int
    created_at: datetime


class ProcessResponse(BaseModel):
    file_id: str
    status: str
    chunk_count: int


# ── Exam (Phase 4) ────────────────────────────────────────────────────────────


class ExamGenerateRequest(BaseModel):
    scope_type: Literal["file", "week", "all", "past_paper"] = "all"
    scope_value: str | None = None
    num_mcq: int = 5
    num_short: int = 2
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class ExamSubmitRequest(BaseModel):
    answers: dict[str, Any]  # {question_id: chosen_idx | text}
