"""Exam endpoints — generate, submit, fetch result."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.exam_generation import generate_exam
from src.agent.exam_grading import grade_attempt
from src.storage.db import Attempt, Exam, User

from .deps import current_user, db_session
from .schemas import ExamGenerateRequest, ExamSubmitRequest

router = APIRouter(prefix="/exam", tags=["exam"])


@router.post("/generate")
async def generate(
    body: ExamGenerateRequest,
    user: Annotated[User, Depends(current_user)],
) -> dict:
    try:
        exam = await generate_exam(
            user_id=user.id,
            scope_type=body.scope_type,
            scope_value=body.scope_value,
            num_mcq=body.num_mcq,
            num_short=body.num_short,
            difficulty=body.difficulty,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {
        "id": exam.id,
        "mcq": (exam.questions or {}).get("mcq", []),
        "short": (exam.questions or {}).get("short", []),
    }


@router.post("/{exam_id}/submit")
async def submit(
    exam_id: Annotated[str, Path()],
    body: ExamSubmitRequest,
    user: Annotated[User, Depends(current_user)],
) -> dict:
    try:
        attempt = await grade_attempt(
            exam_id=exam_id, user_id=user.id, answers=body.answers
        )
    except LookupError:
        raise HTTPException(404, "exam not found")
    return {
        "id": attempt.id,
        "score": attempt.score,
        "per_question": (attempt.results or {}).get("per_question", []),
    }


@router.get("/{exam_id}/result")
async def latest_result(
    exam_id: Annotated[str, Path()],
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> dict:
    row = (
        await db.execute(
            select(Attempt)
            .where(Attempt.exam_id == exam_id, Attempt.user_id == user.id)
            .order_by(Attempt.submitted_at.desc())
        )
    ).scalars().first()
    if row is None:
        raise HTTPException(404, "no attempt yet")
    return {
        "id": row.id,
        "score": row.score,
        "per_question": (row.results or {}).get("per_question", []),
    }
