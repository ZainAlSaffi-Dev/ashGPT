"""Chat session CRUD."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.db import Message, Session, User

from .deps import current_user, db_session
from .schemas import CreateSessionRequest, MessageOut, SessionOut

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> list[SessionOut]:
    rows = (
        await db.execute(
            select(Session).where(Session.user_id == user.id).order_by(Session.updated_at.desc())
        )
    ).scalars().all()
    return [
        SessionOut(id=s.id, title=s.title, created_at=s.created_at, updated_at=s.updated_at)
        for s in rows
    ]


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: CreateSessionRequest,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> SessionOut:
    session = Session(user_id=user.id, title=body.title or "New chat")
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return SessionOut(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/{session_id}/messages", response_model=list[MessageOut])
async def list_messages(
    session_id: Annotated[str, Path()],
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> list[MessageOut]:
    sess = (
        await db.execute(
            select(Session).where(Session.id == session_id, Session.user_id == user.id)
        )
    ).scalar_one_or_none()
    if sess is None:
        raise HTTPException(404, "session not found")
    rows = (
        await db.execute(
            select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc())
        )
    ).scalars().all()
    return [
        MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            intent=m.intent,
            retrieved_chunk_ids=m.retrieved_chunk_ids,
            sources=m.sources,
            irac=m.irac,
            mermaid=m.mermaid,
            latency_ms=m.latency_ms,
            verification=m.verification,
            created_at=m.created_at,
        )
        for m in rows
    ]


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: Annotated[str, Path()],
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> None:
    sess = (
        await db.execute(
            select(Session).where(Session.id == session_id, Session.user_id == user.id)
        )
    ).scalar_one_or_none()
    if sess is None:
        raise HTTPException(404, "session not found")
    await db.delete(sess)
    await db.commit()
