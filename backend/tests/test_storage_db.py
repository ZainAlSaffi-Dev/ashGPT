"""Async SQLAlchemy model tests against an in-memory SQLite."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from src.storage.db import (
    AnswerCache,
    Attempt,
    ChunkMeta,
    Exam,
    File,
    Message,
    Session,
    User,
    get_or_create_user,
    init_db,
)
from sqlalchemy.ext.asyncio import async_sessionmaker


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    await init_db(engine)
    Maker = async_sessionmaker(engine, expire_on_commit=False)
    async with Maker() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_get_or_create_user_is_idempotent(db):
    u1 = await get_or_create_user(db, clerk_id="usr_abc", email="a@b.c")
    u2 = await get_or_create_user(db, clerk_id="usr_abc")
    assert u1.id == u2.id
    assert u1.clerk_id == "usr_abc"
    assert u1.email == "a@b.c"
    assert u1.token_budget == 1_000_000
    assert u1.tokens_used == 0


@pytest.mark.asyncio
async def test_file_chunk_message_cascade(db):
    user = await get_or_create_user(db, clerk_id="usr_cascade")

    f = File(user_id=user.id, name="notes.pdf", mime="application/pdf", blob_key="usr/abc/notes.pdf")
    db.add(f)
    await db.commit()
    await db.refresh(f)

    ch = ChunkMeta(file_id=f.id, user_id=user.id, content="adverse possession requires...")
    db.add(ch)
    s = Session(user_id=user.id, title="test")
    db.add(s)
    await db.commit()
    await db.refresh(s)

    msg = Message(
        session_id=s.id,
        user_id=user.id,
        role="user",
        content="What is adverse possession?",
    )
    db.add(msg)
    await db.commit()

    # Deleting the user cascades through sessions + messages + files + chunks.
    await db.delete(user)
    await db.commit()

    from sqlalchemy import select

    files_left = (await db.execute(select(File))).scalars().all()
    chunks_left = (await db.execute(select(ChunkMeta))).scalars().all()
    sessions_left = (await db.execute(select(Session))).scalars().all()
    messages_left = (await db.execute(select(Message))).scalars().all()
    assert files_left == []
    assert chunks_left == []
    assert sessions_left == []
    assert messages_left == []


@pytest.mark.asyncio
async def test_exam_attempt_roundtrip(db):
    user = await get_or_create_user(db, clerk_id="usr_exam")
    exam = Exam(
        user_id=user.id,
        scope_type="week",
        scope_value="week_1",
        num_mcq=3,
        num_short=1,
        questions={"mcq": [{"question": "Q?", "options": ["a", "b", "c", "d"], "correct_idx": 1}]},
    )
    db.add(exam)
    await db.commit()
    await db.refresh(exam)

    attempt = Attempt(exam_id=exam.id, user_id=user.id, answers={"q0": 1}, score=10.0)
    db.add(attempt)
    await db.commit()
    await db.refresh(attempt)

    assert attempt.exam_id == exam.id
    assert attempt.score == 10.0
    assert exam.questions["mcq"][0]["correct_idx"] == 1


@pytest.mark.asyncio
async def test_answer_cache_row(db):
    from datetime import datetime, timezone, timedelta

    row = AnswerCache(
        cache_key="x" * 64,
        user_id="usr_demo",
        answer="cached",
        payload={"sources": []},
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    )
    db.add(row)
    await db.commit()
    assert row.cache_key.startswith("xxx")
