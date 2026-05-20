"""Per-file ingestion orchestrator.

``ingest_file`` is invoked by ``POST /uploads/{file_id}/process``. Steps:

  1. Resolve the blob to a local path (downloads from R2 if needed).
  2. Extract sections by MIME (PDF → per-page, DOCX → whole doc, image → VLM).
  3. Chunk text sections using the legal-tuned splitter.
  4. Embed chunks with the configured embedding model.
  5. Upsert vectors to the configured vector store under ``namespace=user_id``.
  6. Persist ``ChunkMeta`` rows in the relational DB.
  7. Invalidate the per-namespace BM25 cache so the next query rebuilds it.

Returns the number of chunks written.
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterable

from src.agent import bm25
from src.embeddings import ZeroEntropyEmbeddings
from src.storage.blob import make_blob_store
from src.storage.db import ChunkMeta, get_engine, get_session
from src.storage.vector_store import VectorItem, make_vector_store

from .chunker import chunk_text
from .extract import ExtractedSection, extract

log = logging.getLogger(__name__)


async def ingest_file(
    file_id: str,
    user_id: str,
    blob_key: str,
    mime: str,
    week: str | None = None,
    doc_type: str = "document",
) -> int:
    """Ingest one file end-to-end. Returns the chunk count."""
    blob = make_blob_store()
    local_path = blob.open_path(blob_key)

    sections = extract(local_path, mime=mime)
    if not sections:
        log.info("ingest_file %s: no extractable text", file_id)
        return 0

    # Build (chunk, metadata) pairs in one pass so embedding can batch.
    chunks: list[str] = []
    metas: list[dict] = []
    ids: list[str] = []
    is_image_mime = mime.startswith("image/")
    for section in sections:
        if is_image_mime:
            # VLM description ≈ a few hundred chars — keep as one chunk per image.
            section_chunks = [section.text]
        else:
            section_chunks = chunk_text(section.text)
        for i, c in enumerate(section_chunks):
            cid = uuid.uuid4().hex
            md = {
                "chunk_id": cid,
                "file_id": file_id,
                "namespace": user_id,
                "source": section.meta.get("source") or blob_key.rsplit("/", 1)[-1],
                "type": _doc_type_for_storage(doc_type, section, is_image_mime),
                "week": week,
                "page": section.page,
                "chunk_index": i,
            }
            if "image_path" in section.meta:
                md["image_path"] = section.meta["image_path"]
            chunks.append(c)
            metas.append(md)
            ids.append(cid)

    if not chunks:
        return 0

    embedder = ZeroEntropyEmbeddings()
    vectors = embedder.embed_documents(chunks)

    items: list[VectorItem] = [
        VectorItem(
            id=ids[i],
            vector=vectors[i],
            content=chunks[i],
            metadata=metas[i],
            namespace=user_id,
        )
        for i in range(len(chunks))
    ]

    vs = make_vector_store()
    vs.upsert(items)

    # Persist metadata rows so the UI can list / cite chunks per file.
    async with get_session(get_engine()) as session:
        for i in range(len(chunks)):
            session.add(
                ChunkMeta(
                    id=ids[i],
                    file_id=file_id,
                    user_id=user_id,
                    content=chunks[i],
                    page=metas[i].get("page"),
                    chunk_index=metas[i].get("chunk_index", 0),
                    source=metas[i]["source"],
                    doc_type=metas[i]["type"],
                    week=week,
                    image_path=metas[i].get("image_path"),
                )
            )
        await session.commit()

    bm25.invalidate(user_id)
    log.info("ingest_file %s: %d chunks ingested (ns=%s)", file_id, len(chunks), user_id)
    return len(chunks)


def _doc_type_for_storage(
    declared: str, section: ExtractedSection, is_image: bool
) -> str:
    """Resolve the metadata ``type`` to stamp on this section.

    Honour ingestion hints (``section.meta.doc_type_hint``) before falling
    back to the caller-declared ``doc_type``. Image modality is conveyed via
    ``metadata.image_path`` (set in ``extract_image``) so the slide-channel
    retrieval no longer relies on a special string here.
    """
    if section.meta.get("doc_type_hint"):
        return section.meta["doc_type_hint"]
    if is_image:
        return "image"
    return declared


# Synchronous wrapper for callers in non-async contexts (Streamlit, tests).
def ingest_file_sync(
    file_id: str,
    user_id: str,
    blob_key: str,
    mime: str,
    week: str | None = None,
    doc_type: str = "document",
) -> int:
    import asyncio

    return asyncio.run(
        ingest_file(
            file_id=file_id,
            user_id=user_id,
            blob_key=blob_key,
            mime=mime,
            week=week,
            doc_type=doc_type,
        )
    )
