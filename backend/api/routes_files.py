"""File library + upload routes.

The processing step (chunk + embed + upsert) is delegated to the ingestion
pipeline in ``src.ingestion`` (Phase 2). Phase 1 wires the endpoints and the
DB rows; until Phase 2 lands the ``/process`` call returns ``status=queued``
with ``chunk_count=0`` rather than actually ingesting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path as PathParam, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.storage.blob import LocalBlobStore, make_blob_store
from src.storage.db import File as FileRow
from src.storage.db import User

from .deps import current_user, db_session
from .schemas import FileOut, PresignRequest, PresignResponse, ProcessResponse

router = APIRouter(tags=["files"])
log = logging.getLogger(__name__)


@router.post("/uploads/presign", response_model=PresignResponse)
async def presign(
    body: PresignRequest,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> PresignResponse:
    blob = make_blob_store()
    url, key = blob.presign_put(user_id=user.id, name=body.name, mime=body.mime)
    row = FileRow(
        user_id=user.id,
        name=body.name,
        mime=body.mime,
        blob_key=key,
        status="uploaded",
        doc_type=body.doc_type,
        week=body.week,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    method = "POST" if isinstance(blob, LocalBlobStore) else "PUT"
    return PresignResponse(file_id=row.id, upload_url=url, blob_key=key, method=method)


@router.post("/uploads/local/{key:path}", status_code=status.HTTP_204_NO_CONTENT)
async def upload_local(
    key: str,
    request: Request,
    user: Annotated[User, Depends(current_user)],
) -> None:
    """Local-only receive endpoint. The presign URL for ``LocalBlobStore``
    points here. Multipart form not required — body is the raw file bytes.
    """
    if get_settings().blob_backend != "local":
        raise HTTPException(404, "endpoint disabled when blob_backend != local")
    # Authorisation check on the path: the key always starts with user_id/...
    if not key.startswith(f"{user.id}/"):
        raise HTTPException(403, "key does not belong to user")
    blob = make_blob_store()
    if not isinstance(blob, LocalBlobStore):
        raise HTTPException(500, "local blob store not configured")
    body = await request.body()
    dest = Path(blob.open_path(key))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)


@router.post("/uploads/{file_id}/process", response_model=ProcessResponse)
async def process_upload(
    file_id: Annotated[str, PathParam()],
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> ProcessResponse:
    row = (
        await db.execute(select(FileRow).where(FileRow.id == file_id, FileRow.user_id == user.id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "file not found")

    blob = make_blob_store()
    if not blob.exists(row.blob_key):
        row.status = "failed"
        row.error = "blob missing — upload not completed"
        await db.commit()
        raise HTTPException(409, row.error)

    row.status = "processing"
    await db.commit()

    try:
        # Lazy import so Phase 1 tests that don't touch ingestion don't pay
        # the cost of loading torch/PyMuPDF.
        from src.ingestion.pipeline import ingest_file

        chunk_count = await ingest_file(
            file_id=row.id,
            user_id=row.user_id,
            blob_key=row.blob_key,
            mime=row.mime,
            week=row.week,
            doc_type=row.doc_type,
        )
        row.status = "ready"
        row.chunk_count = chunk_count
        row.error = None
    except ImportError:
        # Phase 2 not yet installed — leave row in 'processing' state.
        row.status = "queued"
        chunk_count = 0
    except Exception as e:
        log.exception("ingestion failed for file %s", file_id)
        row.status = "failed"
        row.error = str(e)[:1000]
        chunk_count = 0

    await db.commit()
    return ProcessResponse(file_id=row.id, status=row.status, chunk_count=chunk_count)


@router.get("/files", response_model=list[FileOut])
async def list_files(
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> list[FileOut]:
    rows = (
        await db.execute(
            select(FileRow).where(FileRow.user_id == user.id).order_by(FileRow.created_at.desc())
        )
    ).scalars().all()
    return [
        FileOut(
            id=r.id,
            name=r.name,
            mime=r.mime,
            size_bytes=r.size_bytes,
            status=r.status,
            error=r.error,
            doc_type=r.doc_type,
            week=r.week,
            chunk_count=r.chunk_count,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.delete("/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_id: Annotated[str, PathParam()],
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> None:
    row = (
        await db.execute(select(FileRow).where(FileRow.id == file_id, FileRow.user_id == user.id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "file not found")

    # Best-effort: drop vectors first, then blob, then DB row (cascades chunks).
    try:
        from src.storage.vector_store import make_vector_store

        vs = make_vector_store()
        chunk_ids = [c.id for c in row.chunks] if row.chunks else []
        if chunk_ids:
            vs.delete(chunk_ids, namespace=user.id)
    except Exception as e:  # pragma: no cover
        log.warning("vector cleanup failed for file %s: %s", file_id, e)

    try:
        blob = make_blob_store()
        blob.delete(row.blob_key)
    except Exception as e:  # pragma: no cover
        log.warning("blob delete failed for %s: %s", row.blob_key, e)

    await db.delete(row)
    await db.commit()
