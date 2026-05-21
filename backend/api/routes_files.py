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

from fastapi import APIRouter, Depends, File, HTTPException, Path as PathParam, Query, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.storage.blob import LocalBlobStore, make_blob_store
from src.storage.db import ChunkMeta, File as FileRow
from src.storage.db import Folder, IngestionJob, Project, User, _utcnow

from .deps import current_user, db_session
from .schemas import FileOut, FileUpdateRequest, PresignRequest, PresignResponse, ProcessResponse

router = APIRouter(tags=["files"])
log = logging.getLogger(__name__)


async def _validate_scope(
    db: AsyncSession,
    user: User,
    project_id: str | None,
    folder_id: str | None,
) -> None:
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
        if project_id and folder.project_id != project_id:
            raise HTTPException(400, "folder does not belong to project")
        if not project_id:
            raise HTTPException(400, "folder scope requires project_id")


def _file_out(row: FileRow) -> FileOut:
    return FileOut(
        id=row.id,
        name=row.name,
        mime=row.mime,
        size_bytes=row.size_bytes,
        project_id=row.project_id,
        folder_id=row.folder_id,
        status=row.status,
        error=row.error,
        doc_type=row.doc_type,
        week=row.week,
        chunk_count=row.chunk_count,
        created_at=row.created_at,
    )


@router.post("/uploads/presign", response_model=PresignResponse)
async def presign(
    body: PresignRequest,
    request: Request,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> PresignResponse:
    await _validate_scope(db, user, body.project_id, body.folder_id)
    # Two callers reach this endpoint:
    #   (a) the Worker, which has already minted an R2 presigned PUT URL
    #       (aws4fetch) and forwarded the original POST with the chosen
    #       ``blob_key`` in the ``X-Presigned-Blob-Key`` header. We must
    #       persist that key so ``/process`` HEADs the right object.
    #   (b) local dev / direct calls without the Worker — we mint our own
    #       URL via boto3 (or LocalBlobStore for filesystem dev).
    worker_blob_key = request.headers.get("X-Presigned-Blob-Key")
    blob = make_blob_store()
    if worker_blob_key:
        # Trust the Worker's key but never mint a usable URL here — the
        # Worker has already returned its own URL to the browser. Return a
        # placeholder so any caller relying on this URL fails loud rather
        # than silently uploading to the wrong place.
        url = "worker-presigned"
        key = worker_blob_key
    else:
        url, key = blob.presign_put(user_id=user.id, name=body.name, mime=body.mime)
    row = FileRow(
        user_id=user.id,
        name=body.name,
        mime=body.mime,
        blob_key=key,
        project_id=body.project_id,
        folder_id=body.folder_id,
        uploaded_by=user.id,
        status="uploaded",
        doc_type=body.doc_type,
        week=body.week,
    )
    db.add(row)
    await db.flush()
    job = IngestionJob(user_id=user.id, file_id=row.id, status="queued")
    db.add(job)
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
    import asyncio

    row = (
        await db.execute(select(FileRow).where(FileRow.id == file_id, FileRow.user_id == user.id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "file not found")
    if row.status == "ready":
        job = (
            await db.execute(
                select(IngestionJob).where(IngestionJob.file_id == row.id, IngestionJob.user_id == user.id)
            )
        ).scalar_one_or_none()
        return ProcessResponse(
            file_id=row.id, status=row.status, chunk_count=row.chunk_count, job_id=job.id if job else None
        )

    blob = make_blob_store()
    # R2 is strongly consistent for new-object reads, but we have observed
    # tail-latency cases where HEAD returns 404 within ~100 ms of a fresh
    # PUT (typically when the client polls /process immediately after the
    # browser sees a 200 from R2 but before R2's metadata replication
    # settles). Retry briefly before giving up — the cost of a few extra
    # HEADs is trivial vs. forcing the user to re-upload.
    blob_present = False
    for attempt in range(4):
        if blob.exists(row.blob_key):
            blob_present = True
            break
        if attempt < 3:
            await asyncio.sleep(0.25 * (attempt + 1))
    if not blob_present:
        row.status = "failed"
        row.error = "blob missing — upload not completed"
        await db.commit()
        # Log enough state to triage: if creds were missing make_blob_store()
        # above would have thrown, so reaching here with `not blob_present`
        # means R2 was contactable but either the key wasn't propagated yet
        # or auth was rejected. blob.exists() already logged the precise
        # HEAD response code; this adds the surrounding context.
        cfg = get_settings()
        log.warning(
            "blob missing after retries: file_id=%s blob_key=%s backend=%s "
            "bucket=%s account_id_set=%s access_key_set=%s endpoint=%s",
            file_id,
            row.blob_key,
            cfg.blob_backend,
            cfg.r2_bucket,
            bool(cfg.r2_account_id),
            bool(cfg.r2_access_key),
            cfg.r2_endpoint_url,
        )
        raise HTTPException(409, row.error)

    row.status = "processing"
    job = (
        await db.execute(
            select(IngestionJob).where(IngestionJob.file_id == row.id, IngestionJob.user_id == user.id)
        )
    ).scalar_one_or_none()
    if job is None:
        job = IngestionJob(user_id=user.id, file_id=row.id, status="processing")
        db.add(job)
    else:
        job.status = "processing"
        job.attempts += 1
        job.locked_at = _utcnow()
        job.error = None
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
            project_id=row.project_id,
            folder_id=row.folder_id,
            file_name=row.name,
        )
        row.status = "ready"
        row.chunk_count = chunk_count
        row.last_indexed_at = _utcnow()
        row.error = None
        job.status = "ready"
        job.error = None
    except ImportError:
        # Phase 2 not yet installed — leave row in 'processing' state.
        row.status = "queued"
        job.status = "queued"
        chunk_count = 0
    except Exception as e:
        log.exception("ingestion failed for file %s", file_id)
        row.status = "failed"
        row.error = str(e)[:1000]
        job.status = "failed"
        job.error = row.error
        chunk_count = 0

    await db.commit()
    return ProcessResponse(file_id=row.id, status=row.status, chunk_count=chunk_count, job_id=job.id)


@router.get("/files", response_model=list[FileOut])
async def list_files(
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
    project_id: Annotated[str | None, Query()] = None,
    folder_id: Annotated[str | None, Query()] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
) -> list[FileOut]:
    await _validate_scope(db, user, project_id, folder_id)
    query = select(FileRow).where(FileRow.user_id == user.id)
    if project_id:
        query = query.where(FileRow.project_id == project_id)
    if folder_id:
        query = query.where(FileRow.folder_id == folder_id)
    if status_filter:
        query = query.where(FileRow.status == status_filter)
    rows = (
        await db.execute(
            query.order_by(FileRow.created_at.desc())
        )
    ).scalars().all()
    return [_file_out(r) for r in rows]


@router.patch("/files/{file_id}", response_model=FileOut)
async def update_file(
    file_id: Annotated[str, PathParam()],
    body: FileUpdateRequest,
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> FileOut:
    row = (
        await db.execute(select(FileRow).where(FileRow.id == file_id, FileRow.user_id == user.id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "file not found")

    next_project_id = body.project_id if body.project_id is not None else row.project_id
    next_folder_id = body.folder_id if body.folder_id is not None else row.folder_id
    await _validate_scope(db, user, next_project_id, next_folder_id)

    if body.name is not None:
        row.name = body.name
    if body.project_id is not None:
        row.project_id = body.project_id
    if body.folder_id is not None:
        row.folder_id = body.folder_id
    if body.doc_type is not None:
        row.doc_type = body.doc_type
    if body.week is not None:
        row.week = body.week
    await db.execute(
        ChunkMeta.__table__.update()
        .where(ChunkMeta.file_id == row.id, ChunkMeta.user_id == user.id)
        .values(project_id=row.project_id, folder_id=row.folder_id, doc_type=row.doc_type, week=row.week)
    )
    chunk_ids = (
        await db.execute(select(ChunkMeta.id).where(ChunkMeta.file_id == row.id, ChunkMeta.user_id == user.id))
    ).scalars().all()
    if chunk_ids:
        try:
            from src.storage.vector_store import make_vector_store

            make_vector_store().update_metadata(
                chunk_ids,
                namespace=user.id,
                patch={
                    "project_id": row.project_id,
                    "folder_id": row.folder_id,
                    "type": row.doc_type,
                    "week": row.week,
                    "file_name": row.name,
                },
            )
        except NotImplementedError:
            log.warning("vector metadata update unavailable for file %s", file_id)
        except Exception as e:  # pragma: no cover
            log.warning("vector metadata update failed for file %s: %s", file_id, e)
    try:
        from src.agent import bm25

        bm25.invalidate(user.id)
    except Exception:  # pragma: no cover
        log.warning("bm25 invalidation failed for file move %s", file_id)
    await db.commit()
    await db.refresh(row)
    return _file_out(row)


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
        chunk_ids = (
            await db.execute(select(ChunkMeta.id).where(ChunkMeta.file_id == row.id, ChunkMeta.user_id == user.id))
        ).scalars().all()
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
