"""Project and folder CRUD routes."""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.db import ChunkMeta, File as FileRow
from src.storage.db import Folder, Project, User, _utcnow

from .deps import current_user, db_session
from .schemas import FolderCreate, FolderOut, FolderUpdate, ProjectCreate, ProjectOut, ProjectUpdate

router = APIRouter(tags=["projects"])


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"


def _project_out(project: Project) -> ProjectOut:
    return ProjectOut(
        id=project.id,
        name=project.name,
        slug=project.slug,
        description=project.description,
        color=project.color,
        archived_at=project.archived_at,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def _folder_out(folder: Folder) -> FolderOut:
    return FolderOut(
        id=folder.id,
        project_id=folder.project_id,
        parent_id=folder.parent_id,
        name=folder.name,
        sort_order=folder.sort_order,
        created_at=folder.created_at,
        updated_at=folder.updated_at,
    )


async def _require_project(db: AsyncSession, user: User, project_id: str) -> Project:
    project = (
        await db.execute(
            select(Project).where(Project.id == project_id, Project.user_id == user.id)
        )
    ).scalar_one_or_none()
    if project is None:
        raise HTTPException(404, "project not found")
    return project


async def _require_folder(
    db: AsyncSession, user: User, folder_id: str, project_id: str | None = None
) -> Folder:
    query = select(Folder).where(Folder.id == folder_id, Folder.user_id == user.id)
    if project_id is not None:
        query = query.where(Folder.project_id == project_id)
    folder = (await db.execute(query)).scalar_one_or_none()
    if folder is None:
        raise HTTPException(404, "folder not found")
    return folder


async def _unique_slug(db: AsyncSession, user: User, name: str, exclude_id: str | None = None) -> str:
    base = _slugify(name)
    slug = base
    i = 2
    while True:
        query = select(Project.id).where(Project.user_id == user.id, Project.slug == slug)
        if exclude_id:
            query = query.where(Project.id != exclude_id)
        exists = (await db.execute(query)).scalar_one_or_none()
        if exists is None:
            return slug
        slug = f"{base}-{i}"
        i += 1


async def _descendant_folder_ids(db: AsyncSession, user: User, folder_id: str) -> list[str]:
    descendants: list[str] = []
    frontier = [folder_id]
    while frontier:
        child_ids = (
            await db.execute(
                select(Folder.id).where(
                    Folder.user_id == user.id,
                    Folder.parent_id.in_(frontier),
                )
            )
        ).scalars().all()
        if not child_ids:
            break
        descendants.extend(child_ids)
        frontier = list(child_ids)
    return descendants


@router.get("/projects", response_model=list[ProjectOut])
async def list_projects(
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
    include_archived: bool = False,
) -> list[ProjectOut]:
    query = select(Project).where(Project.user_id == user.id)
    if not include_archived:
        query = query.where(Project.archived_at.is_(None))
    rows = (await db.execute(query.order_by(Project.updated_at.desc()))).scalars().all()
    return [_project_out(row) for row in rows]


@router.post("/projects", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreate,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> ProjectOut:
    project = Project(
        user_id=user.id,
        name=body.name,
        slug=await _unique_slug(db, user, body.name),
        description=body.description,
        color=body.color,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return _project_out(project)


@router.get("/projects/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: Annotated[str, Path()],
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> ProjectOut:
    return _project_out(await _require_project(db, user, project_id))


@router.patch("/projects/{project_id}", response_model=ProjectOut)
async def update_project(
    project_id: Annotated[str, Path()],
    body: ProjectUpdate,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> ProjectOut:
    project = await _require_project(db, user, project_id)
    if body.name is not None:
        project.name = body.name
        project.slug = await _unique_slug(db, user, body.name, exclude_id=project.id)
    if body.description is not None:
        project.description = body.description
    if body.color is not None:
        project.color = body.color
    if body.archived is not None:
        project.archived_at = _utcnow() if body.archived else None
    await db.commit()
    await db.refresh(project)
    return _project_out(project)


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive_project(
    project_id: Annotated[str, Path()],
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> None:
    project = await _require_project(db, user, project_id)
    project.archived_at = _utcnow()
    await db.commit()


@router.get("/projects/{project_id}/folders", response_model=list[FolderOut])
async def list_folders(
    project_id: Annotated[str, Path()],
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> list[FolderOut]:
    await _require_project(db, user, project_id)
    rows = (
        await db.execute(
            select(Folder)
            .where(Folder.user_id == user.id, Folder.project_id == project_id)
            .order_by(Folder.sort_order.asc(), Folder.created_at.asc())
        )
    ).scalars().all()
    return [_folder_out(row) for row in rows]


@router.post("/projects/{project_id}/folders", response_model=FolderOut, status_code=status.HTTP_201_CREATED)
async def create_folder(
    project_id: Annotated[str, Path()],
    body: FolderCreate,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> FolderOut:
    await _require_project(db, user, project_id)
    if body.parent_id:
        await _require_folder(db, user, body.parent_id, project_id=project_id)
    folder = Folder(
        user_id=user.id,
        project_id=project_id,
        parent_id=body.parent_id,
        name=body.name,
        sort_order=body.sort_order,
    )
    db.add(folder)
    await db.commit()
    await db.refresh(folder)
    return _folder_out(folder)


@router.patch("/folders/{folder_id}", response_model=FolderOut)
async def update_folder(
    folder_id: Annotated[str, Path()],
    body: FolderUpdate,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> FolderOut:
    folder = await _require_folder(db, user, folder_id)
    if body.name is not None:
        folder.name = body.name
    if "parent_id" in body.model_fields_set:
        if body.parent_id is None:
            folder.parent_id = None
        elif body.parent_id == folder.id:
            raise HTTPException(400, "folder cannot be its own parent")
        else:
            descendants = await _descendant_folder_ids(db, user, folder.id)
            if body.parent_id in descendants:
                raise HTTPException(400, "folder cannot be moved inside its descendants")
            await _require_folder(db, user, body.parent_id, project_id=folder.project_id)
            folder.parent_id = body.parent_id
    if body.sort_order is not None:
        folder.sort_order = body.sort_order
    await db.commit()
    await db.refresh(folder)
    return _folder_out(folder)


@router.delete("/folders/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder(
    folder_id: Annotated[str, Path()],
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
    recursive: Annotated[bool, Query()] = False,
) -> None:
    folder = await _require_folder(db, user, folder_id)
    child_count = (
        await db.execute(
            select(func.count()).select_from(Folder).where(Folder.user_id == user.id, Folder.parent_id == folder.id)
        )
    ).scalar_one()
    file_count = (
        await db.execute(
            select(func.count()).select_from(FileRow).where(FileRow.user_id == user.id, FileRow.folder_id == folder.id)
        )
    ).scalar_one()
    if (child_count or file_count) and not recursive:
        raise HTTPException(409, "folder is not empty")
    if recursive:
        folder_ids = [folder.id, *await _descendant_folder_ids(db, user, folder.id)]
        affected_file_ids = (
            await db.execute(
                select(FileRow.id).where(
                    FileRow.user_id == user.id,
                    FileRow.folder_id.in_(folder_ids),
                )
            )
        ).scalars().all()
        if affected_file_ids:
            chunk_ids = (
                await db.execute(
                    select(ChunkMeta.id).where(
                        ChunkMeta.user_id == user.id,
                        ChunkMeta.file_id.in_(affected_file_ids),
                    )
                )
            ).scalars().all()
            await db.execute(
                ChunkMeta.__table__.update()
                .where(
                    ChunkMeta.user_id == user.id,
                    ChunkMeta.file_id.in_(affected_file_ids),
                )
                .values(folder_id=None)
            )
            if chunk_ids:
                try:
                    from src.storage.vector_store import make_vector_store

                    make_vector_store().update_metadata(
                        chunk_ids,
                        namespace=user.id,
                        patch={"folder_id": None},
                    )
                except Exception as e:  # pragma: no cover
                    raise HTTPException(500, "vector metadata update failed") from e
        await db.execute(
            FileRow.__table__.update()
            .where(FileRow.user_id == user.id, FileRow.folder_id.in_(folder_ids))
            .values(folder_id=None)
        )
        for descendant_id in reversed(folder_ids[1:]):
            child = await _require_folder(db, user, descendant_id)
            await db.delete(child)
    await db.delete(folder)
    await db.commit()
    if recursive:
        try:
            from src.agent import bm25
            from src.agent.cache import invalidate_user

            bm25.invalidate(user.id)
            await invalidate_user(user.id)
        except Exception:  # pragma: no cover
            pass
