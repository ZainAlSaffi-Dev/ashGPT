"""Chat session CRUD."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.db import Folder, Message, Project, Session, User

from .deps import current_user, db_session
from .schemas import CreateSessionRequest, MessageOut, RetrievalScopeIn, SessionOut

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _scope_snapshot(scope: RetrievalScopeIn | None, project_id: str | None, folder_id: str | None) -> dict | None:
    if scope:
        return scope.model_dump(exclude_none=True)
    if project_id:
        out = {"type": "project", "project_id": project_id}
        if folder_id:
            out = {"type": "folder", "project_id": project_id, "folder_id": folder_id}
        return out
    return None


async def _validate_scope(db: AsyncSession, user: User, project_id: str | None, folder_id: str | None) -> None:
    if project_id:
        project = (
            await db.execute(select(Project).where(Project.id == project_id, Project.user_id == user.id))
        ).scalar_one_or_none()
        if project is None:
            raise HTTPException(404, "project not found")
    if folder_id:
        folder = (
            await db.execute(select(Folder).where(Folder.id == folder_id, Folder.user_id == user.id))
        ).scalar_one_or_none()
        if folder is None:
            raise HTTPException(404, "folder not found")
        if folder.project_id != project_id:
            raise HTTPException(400, "folder does not belong to project")


def _session_out(session: Session) -> SessionOut:
    return SessionOut(
        id=session.id,
        title=session.title,
        project_id=session.project_id,
        folder_id=session.folder_id,
        scope=session.scope,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
    project_id: Annotated[str | None, Query()] = None,
) -> list[SessionOut]:
    if project_id:
        await _validate_scope(db, user, project_id, None)
    query = select(Session).where(Session.user_id == user.id)
    if project_id:
        query = query.where(Session.project_id == project_id)
    rows = (
        await db.execute(
            query.order_by(Session.updated_at.desc())
        )
    ).scalars().all()
    return [_session_out(s) for s in rows]


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: CreateSessionRequest,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> SessionOut:
    project_id = body.scope.project_id if body.scope and body.scope.project_id else body.project_id
    folder_id = body.scope.folder_id if body.scope and body.scope.folder_id else body.folder_id
    await _validate_scope(db, user, project_id, folder_id)
    session = Session(
        user_id=user.id,
        title=body.title or "New chat",
        project_id=project_id,
        folder_id=folder_id,
        scope=_scope_snapshot(body.scope, project_id, folder_id),
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return _session_out(session)


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
            scope=m.scope,
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
