"""Exam grading — deterministic MCQ + LLM-judged short answer."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import pytest_asyncio

from src.agent.exam_grading import grade_attempt, grade_short_answer
from src.storage import db as db_mod
from src.storage.db import Exam, User


@pytest_asyncio.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    from src.config import reload_settings

    reload_settings()
    db_mod.reset_engine()
    await db_mod.init_db(db_mod.get_engine())

    async with db_mod.get_session() as session:
        u = User(clerk_id="usr_demo")
        session.add(u)
        await session.commit()
        await session.refresh(u)
        exam = Exam(
            user_id=u.id,
            scope_type="all",
            scope_value=None,
            num_mcq=2,
            num_short=1,
            difficulty="medium",
            questions={
                "mcq": [
                    {
                        "question": "What is adverse possession?",
                        "options": ["A", "B", "C", "D"],
                        "correct_idx": 1,
                        "explanation": "B is right because…",
                        "source_chunks": ["s"],
                    },
                    {
                        "question": "What is estoppel?",
                        "options": ["W", "X", "Y", "Z"],
                        "correct_idx": 0,
                        "explanation": "W is right because…",
                        "source_chunks": ["s"],
                    },
                ],
                "short": [
                    {
                        "question": "Define ratio decidendi.",
                        "model_answer": "The legal principle on which a case is decided.",
                        "grading_rubric": [
                            "Identifies the principle of law",
                            "Distinguishes from obiter dicta",
                        ],
                        "source_chunks": ["s"],
                    }
                ],
            },
        )
        session.add(exam)
        await session.commit()
        await session.refresh(exam)
        yield {"user_id": u.id, "exam_id": exam.id}
    db_mod.reset_engine()


def test_grade_short_answer_empty_response_zero():
    g = grade_short_answer(
        question="Q", model_answer="ma", rubric=["a", "b"], student=""
    )
    assert g.score == 0.0
    assert g.feedback


def test_grade_short_answer_parses_judge_json():
    judge_payload = {"score": 7.5, "rubric_hits": ["a"], "feedback": "Solid but missing b"}
    with patch("src.agent.exam_grading.llm_call", return_value=json.dumps(judge_payload)):
        g = grade_short_answer(
            question="Q", model_answer="ma", rubric=["a", "b"], student="some answer"
        )
    assert g.score == 7.5
    assert g.rubric_hits == ["a"]


def test_grade_short_answer_handles_bad_json():
    with patch("src.agent.exam_grading.llm_call", return_value="garbage {{"):
        g = grade_short_answer(
            question="Q", model_answer="ma", rubric=["a", "b"], student="answer"
        )
    assert g.score == 0.0
    assert "failed" in g.feedback.lower()


@pytest.mark.asyncio
async def test_grade_attempt_all_correct(env):
    judge = json.dumps({"score": 10.0, "rubric_hits": ["a", "b"], "feedback": "Excellent"})
    with patch("src.agent.exam_grading.llm_call", return_value=judge):
        attempt = await grade_attempt(
            exam_id=env["exam_id"],
            user_id=env["user_id"],
            answers={"mcq_0": 1, "mcq_1": 0, "short_0": "It is the legal principle on which a case is decided."},
        )
    assert attempt.score == 100.0
    per = attempt.results["per_question"]
    assert all(q["score"] == 10.0 for q in per)


@pytest.mark.asyncio
async def test_grade_attempt_mcq_wrong(env):
    judge = json.dumps({"score": 0.0, "rubric_hits": [], "feedback": "Off-topic"})
    with patch("src.agent.exam_grading.llm_call", return_value=judge):
        attempt = await grade_attempt(
            exam_id=env["exam_id"],
            user_id=env["user_id"],
            answers={"mcq_0": 0, "mcq_1": 0, "short_0": "off topic"},  # mcq_0 wrong
        )
    # 0 (wrong) + 10 (right) + 0 (judge) = 10/30 → 33.3
    assert attempt.score == pytest.approx(33.3, abs=0.1)
    per = attempt.results["per_question"]
    assert per[0]["score"] == 0.0
    assert per[1]["score"] == 10.0


@pytest.mark.asyncio
async def test_grade_attempt_missing_exam_raises(env):
    with patch("src.agent.exam_grading.llm_call", return_value='{"score":0}'):
        with pytest.raises(LookupError):
            await grade_attempt(
                exam_id="does_not_exist",
                user_id=env["user_id"],
                answers={},
            )


@pytest.mark.asyncio
async def test_grade_attempt_mcq_coerces_string_answers(env):
    """Frontend may send "1" as a string — accept it."""
    judge = json.dumps({"score": 5.0, "rubric_hits": [], "feedback": ""})
    with patch("src.agent.exam_grading.llm_call", return_value=judge):
        attempt = await grade_attempt(
            exam_id=env["exam_id"],
            user_id=env["user_id"],
            answers={"mcq_0": "1", "mcq_1": "0", "short_0": "x"},
        )
    per = attempt.results["per_question"]
    assert per[0]["score"] == 10.0
    assert per[1]["score"] == 10.0
