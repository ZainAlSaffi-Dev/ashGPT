"""Exam generation — JSON parsing + Pydantic validation + retry."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import pytest_asyncio
from pydantic import ValidationError

from src.agent.exam_generation import (
    GeneratedExam,
    MCQItem,
    ShortAnswerItem,
    _parse_or_raise,
    _strip_fences,
    generate_exam,
)


# ── Pure parse / validate ────────────────────────────────────────────────────


def test_mcq_correct_idx_constrained():
    with pytest.raises(ValidationError):
        MCQItem(
            question="Q",
            options=["a", "b", "c", "d"],
            correct_idx=4,  # out of range
            explanation="why",
        )


def test_mcq_requires_exactly_four_options():
    with pytest.raises(ValidationError):
        MCQItem(
            question="Q", options=["a", "b", "c"], correct_idx=0, explanation=""
        )


def test_short_answer_rubric_length():
    with pytest.raises(ValidationError):
        ShortAnswerItem(question="Q", model_answer="ma", grading_rubric=["only one"])


def test_strip_fences_removes_markdown_blocks():
    assert _strip_fences("```json\n{\"x\":1}\n```") == '{"x":1}'
    assert _strip_fences("```\n{\"x\":1}\n```") == '{"x":1}'
    assert _strip_fences('{"x":1}') == '{"x":1}'


def test_parse_or_raise_round_trip():
    payload = {
        "mcq": [
            {
                "question": "What is adverse possession?",
                "options": ["a", "b", "c", "d"],
                "correct_idx": 1,
                "explanation": "see chunk",
                "source_chunks": ["s1"],
            }
        ],
        "short": [
            {
                "question": "Define ratio decidendi.",
                "model_answer": "The principle of law on which the case is decided.",
                "grading_rubric": [
                    "Identifies the principle of law",
                    "Distinguishes from obiter dicta",
                ],
                "source_chunks": ["s1"],
            }
        ],
    }
    exam = _parse_or_raise(json.dumps(payload))
    assert isinstance(exam, GeneratedExam)
    assert exam.mcq[0].correct_idx == 1


# ── End-to-end with mocked LLM + retrieval + DB ───────────────────────────────


@pytest_asyncio.fixture
async def db_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    from src.config import reload_settings
    from src.storage import db as db_mod

    reload_settings()
    db_mod.reset_engine()
    await db_mod.init_db(db_mod.get_engine())
    yield
    db_mod.reset_engine()


def _valid_exam_json() -> str:
    return json.dumps(
        {
            "mcq": [
                {
                    "question": "What is the essential element of adverse possession?",
                    "options": [
                        "Continuous occupation",
                        "Payment of rates",
                        "Tenant relationship",
                        "Statutory grant",
                    ],
                    "correct_idx": 0,
                    "explanation": "Per Smith v Jones, factual possession is key.",
                    "source_chunks": ["smith.pdf"],
                }
            ],
            "short": [
                {
                    "question": "Explain ratio decidendi.",
                    "model_answer": "It is the legal principle on which the case is decided, distinct from obiter dicta.",
                    "grading_rubric": [
                        "Identifies the principle of law",
                        "Distinguishes from obiter dicta",
                        "Provides an example",
                    ],
                    "source_chunks": ["case.pdf"],
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_generate_exam_persists(db_env):
    fake_docs = [
        {
            "content": "Smith v Jones held that adverse possession requires continuous occupation.",
            "source": "smith.pdf",
            "week": "week_3",
            "doc_type": "reading",
            "image_path": None,
        }
    ]
    with patch("src.agent.exam_generation.retrieve_texts", return_value=fake_docs):
        with patch("src.agent.exam_generation.llm_call", return_value=_valid_exam_json()):
            exam = await generate_exam(
                user_id="usr_demo",
                scope_type="all",
                num_mcq=1,
                num_short=1,
                difficulty="medium",
            )
    assert exam.id
    assert exam.num_mcq == 1
    assert exam.num_short == 1
    assert exam.questions["mcq"][0]["correct_idx"] == 0


@pytest.mark.asyncio
async def test_generate_exam_retries_on_bad_json(db_env):
    """First LLM call returns garbage; fixup call returns valid JSON."""
    fake_docs = [
        {
            "content": "Adverse possession context.",
            "source": "smith.pdf",
            "week": None,
            "doc_type": "note",
            "image_path": None,
        }
    ]
    calls = {"n": 0}

    def fake_llm(prompt, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json {{ )"
        return _valid_exam_json()

    with patch("src.agent.exam_generation.retrieve_texts", return_value=fake_docs):
        with patch("src.agent.exam_generation.llm_call", side_effect=fake_llm):
            exam = await generate_exam(user_id="usr_demo", num_mcq=1, num_short=1)
    assert calls["n"] == 2
    assert exam.num_mcq == 1


@pytest.mark.asyncio
async def test_generate_exam_raises_when_no_chunks(db_env):
    with patch("src.agent.exam_generation.retrieve_texts", return_value=[]):
        with pytest.raises(ValueError):
            await generate_exam(user_id="usr_demo")
