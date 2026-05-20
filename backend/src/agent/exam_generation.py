"""Exam generation — MCQ + short-answer items from a user's notes.

Pipeline:

  1. Retrieve a diverse pool of chunks from the user's namespace, optionally
     scoped to one file or one week. Uses the existing hybrid retrieval so
     citation-heavy passages get fair representation alongside narrative.
  2. Compose a prompt that asks the synthesis model to emit a JSON object
     conforming to ``GeneratedExam``.
  3. Parse + validate via Pydantic; retry once on parse failure with a
     stricter "fix your JSON" reprompt.
  4. Persist the exam to the ``exams`` table.

Returns the persisted ``Exam`` row.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.agent.tools import retrieve_texts
from src.config import SYNTHESIS_MODEL
from src.llm import llm_call
from src.storage.db import Exam, get_engine, get_session

log = logging.getLogger(__name__)


class MCQItem(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_idx: int = Field(ge=0, le=3)
    explanation: str
    source_chunks: list[str] = Field(default_factory=list)


class ShortAnswerItem(BaseModel):
    question: str
    model_answer: str
    grading_rubric: list[str] = Field(min_length=2, max_length=6)
    source_chunks: list[str] = Field(default_factory=list)


class GeneratedExam(BaseModel):
    mcq: list[MCQItem]
    short: list[ShortAnswerItem]

    @field_validator("mcq", "short", mode="before")
    @classmethod
    def _coerce_none_to_empty(cls, v):
        return v or []


# ── Prompt template ───────────────────────────────────────────────────────────


_EXAM_PROMPT = """You are a senior law tutor writing a practice exam for a property-law student.

Your task: produce exactly {num_mcq} multiple-choice items and {num_short} short-answer items at "{difficulty}" difficulty. ALL items must be grounded in the supplied context — do not invent legal rules or cases. Cite at least one supporting chunk id in ``source_chunks`` for each item.

Difficulty calibration:
  - easy   = single-step recall / definition
  - medium = apply a rule to a brief hypothetical
  - hard   = multi-step analysis with a distractor option that catches a common misconception

Multiple-choice rules:
  - Exactly 4 options.
  - ``correct_idx`` ∈ [0, 3].
  - Plausible distractors that reflect a real student misconception or adjacent doctrine.
  - ``explanation`` cites the source chunk and explains briefly why the correct option is right.

Short-answer rules:
  - The ``model_answer`` is a 3–6 sentence ideal response that uses IRAC where appropriate.
  - The ``grading_rubric`` lists 3–5 atomic criteria a student response must hit (each criterion ≤ 20 words).
  - ``source_chunks`` lists the chunk ids that support the model answer.

Respond with **JSON only** (no markdown, no prose) matching exactly this shape:
{{
  "mcq": [
    {{
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "correct_idx": 0,
      "explanation": "...",
      "source_chunks": ["chunk_id_or_source_name"]
    }}
  ],
  "short": [
    {{
      "question": "...",
      "model_answer": "...",
      "grading_rubric": ["criterion 1", "criterion 2", "criterion 3"],
      "source_chunks": ["chunk_id_or_source_name"]
    }}
  ]
}}

Context chunks (each starts with [source_id]):
{context}
"""


_FIXUP_PROMPT = """Your previous reply was not valid JSON. The parser said:

    {error}

Reply ONLY with corrected JSON in the same shape. No prose, no markdown fences."""


def _format_context(docs: list) -> str:
    parts: list[str] = []
    for d in docs:
        sid = d.get("source") or "unknown"
        parts.append(f"[{sid}]\n{d.get('content','')}")
    return "\n\n".join(parts)


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _JSON_FENCE_RE.sub("", text.strip()).strip()


def _parse_or_raise(text: str) -> GeneratedExam:
    raw = _strip_fences(text)
    data = json.loads(raw)
    return GeneratedExam.model_validate(data)


async def generate_exam(
    user_id: str,
    scope_type: Literal["file", "week", "all", "past_paper"] = "all",
    scope_value: str | None = None,
    num_mcq: int = 5,
    num_short: int = 2,
    difficulty: Literal["easy", "medium", "hard"] = "medium",
    model: str | None = None,
) -> Exam:
    week_filter = scope_value if scope_type == "week" else None

    # Pull a wide pool from the user's notes. Hybrid retrieval enabled by
    # default surfaces both lexical case-name hits and semantic passages.
    docs = retrieve_texts(
        f"key principles overview difficulty {difficulty}",
        week=week_filter,
        k=12,
        namespace=user_id,
    )
    if scope_type == "file" and scope_value:
        docs = [d for d in docs if d.get("source") == scope_value][:12] or docs
    if scope_type == "past_paper":
        docs = [d for d in docs if d.get("doc_type") == "past_paper"][:12] or docs

    if not docs:
        raise ValueError("no chunks available for the requested scope")

    context = _format_context(docs)
    prompt = _EXAM_PROMPT.format(
        num_mcq=num_mcq, num_short=num_short, difficulty=difficulty, context=context
    )
    model = model or SYNTHESIS_MODEL

    text = llm_call(prompt, model=model, temperature=0.4)
    try:
        exam = _parse_or_raise(text)
    except (ValidationError, json.JSONDecodeError) as e:
        log.info("exam JSON parse failed (%s), retrying with fixup prompt", e)
        fixup = llm_call(
            _FIXUP_PROMPT.format(error=str(e)[:500]) + "\n\nPrevious reply:\n" + text,
            model=model,
            temperature=0.0,
        )
        exam = _parse_or_raise(fixup)

    if len(exam.mcq) != num_mcq:
        log.warning(
            "exam returned %d MCQs (requested %d) — keeping anyway", len(exam.mcq), num_mcq
        )
    if len(exam.short) != num_short:
        log.warning(
            "exam returned %d short (requested %d) — keeping anyway", len(exam.short), num_short
        )

    row = Exam(
        user_id=user_id,
        scope_type=scope_type,
        scope_value=scope_value,
        num_mcq=len(exam.mcq),
        num_short=len(exam.short),
        difficulty=difficulty,
        questions=exam.model_dump(),
    )
    async with get_session(get_engine()) as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row
