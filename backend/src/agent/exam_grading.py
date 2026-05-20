"""Exam grading.

  * **MCQ**          deterministic ``answers[qid] == correct_idx``.
  * **Short answer** structured-output LLM judge against the question's
    ``grading_rubric``; returns 0–10 score + rubric hits + feedback. Cheap
    Gemini model used by default; can be overridden.

Returns an ``ExamResult`` payload plus the persisted ``Attempt`` row.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.config import JUDGE_DRAFT_MODEL
from src.llm import llm_call
from src.storage.db import Attempt, Exam, get_engine, get_session

log = logging.getLogger(__name__)


class _Grade(BaseModel):
    score: float = Field(ge=0, le=10)
    rubric_hits: list[str] = Field(default_factory=list)
    feedback: str = ""


_PROMPT = """You are grading a law-student short-answer response against a rubric.

Question:
{question}

Model answer (for your reference, do not just compare verbatim — student answers can be valid in different wording):
{model_answer}

Rubric criteria the student must hit (each is binary — covered or not):
{rubric}

Student response:
{student}

Reply with **JSON only** in exactly this shape:
{{"score": <0.0–10.0>, "rubric_hits": ["criterion 1", ...], "feedback": "<2–3 sentences>"}}

Scoring guide: score = 10 * (criteria_hit / total_criteria). Round to one decimal.
Feedback: name what's missing or incorrect — be concise."""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(t: str) -> str:
    return _FENCE.sub("", t.strip()).strip()


def grade_short_answer(
    question: str,
    model_answer: str,
    rubric: list[str],
    student: str,
    model: str | None = None,
) -> _Grade:
    """Send one short-answer item to the judge. Returns parsed grade."""
    if not student.strip():
        return _Grade(score=0.0, rubric_hits=[], feedback="No response provided.")
    rubric_str = "\n".join(f"- {c}" for c in rubric)
    prompt = _PROMPT.format(
        question=question, model_answer=model_answer, rubric=rubric_str, student=student
    )
    text = llm_call(prompt, model=model or JUDGE_DRAFT_MODEL, temperature=0.0)
    try:
        return _Grade.model_validate(json.loads(_strip_fences(text)))
    except (ValidationError, json.JSONDecodeError) as e:
        log.warning("short-answer grade parse failed (%s); defaulting", e)
        return _Grade(
            score=0.0, rubric_hits=[], feedback=f"Grading failed: {str(e)[:120]}"
        )


async def grade_attempt(
    exam_id: str,
    user_id: str,
    answers: dict[str, Any],
    judge_model: str | None = None,
) -> Attempt:
    """Grade ``answers`` against the persisted exam. Returns the saved attempt.

    ``answers`` keys are ``mcq_<i>`` and ``short_<i>`` matching the indexes
    used in the generated exam.
    """
    from sqlalchemy import select

    async with get_session(get_engine()) as session:
        exam = (
            await session.execute(
                select(Exam).where(Exam.id == exam_id, Exam.user_id == user_id)
            )
        ).scalar_one_or_none()
        if exam is None:
            raise LookupError("exam not found")

        questions = exam.questions or {}
        per: list[dict[str, Any]] = []
        total_score = 0.0
        total_weight = 0.0

        # MCQ: 10 points each
        for i, q in enumerate(questions.get("mcq") or []):
            qid = f"mcq_{i}"
            chosen = answers.get(qid)
            correct = q.get("correct_idx")
            try:
                chosen_int = int(chosen) if chosen is not None else None
            except (TypeError, ValueError):
                chosen_int = None
            hit = chosen_int is not None and chosen_int == correct
            score = 10.0 if hit else 0.0
            per.append(
                {
                    "question_id": qid,
                    "score": score,
                    "rubric_hits": [],
                    "feedback": (
                        q.get("explanation", "")
                        if hit
                        else f"Correct answer: option {correct}. {q.get('explanation','')}"
                    ),
                }
            )
            total_score += score
            total_weight += 10.0

        # Short answer: 10 points each, LLM-judged
        for i, q in enumerate(questions.get("short") or []):
            qid = f"short_{i}"
            student = str(answers.get(qid) or "")
            grade = grade_short_answer(
                question=q["question"],
                model_answer=q.get("model_answer", ""),
                rubric=q.get("grading_rubric") or [],
                student=student,
                model=judge_model,
            )
            per.append(
                {
                    "question_id": qid,
                    "score": grade.score,
                    "rubric_hits": grade.rubric_hits,
                    "feedback": grade.feedback,
                }
            )
            total_score += grade.score
            total_weight += 10.0

        final_score = round((total_score / total_weight) * 100, 1) if total_weight else 0.0

        attempt = Attempt(
            exam_id=exam_id,
            user_id=user_id,
            answers=answers,
            results={"per_question": per},
            score=final_score,
        )
        session.add(attempt)
        await session.commit()
        await session.refresh(attempt)
    return attempt
