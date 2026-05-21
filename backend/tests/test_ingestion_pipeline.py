"""End-to-end ingestion pipeline test.

Mocks ZeroEntropyEmbeddings + vector store + uses in-memory SQLite + a tmpdir
LocalBlobStore so it runs without network or persistent state.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from src.agent import bm25
from src.ingestion import pipeline as pipeline_mod
from src.storage import db as db_mod
from src.storage.blob import LocalBlobStore
from src.storage.db import ChunkMeta, File, User, init_db
from src.storage.vector_store import InMemoryVectorStore


def _fake_embed_documents(texts):
    """Deterministic mock embeddings — char-bag hashed into 8 dims."""
    out = []
    for t in texts:
        v = [0.0] * 8
        for c in t:
            v[ord(c) % 8] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


@pytest_asyncio.fixture
async def env(monkeypatch, tmp_path):
    """Wire blob root + sqlite engine + reusable vector store."""
    monkeypatch.setenv("BLOB_LOCAL_ROOT", str(tmp_path / "blobs"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    from src.config import reload_settings

    reload_settings()
    db_mod.reset_engine()
    engine = db_mod.get_engine()
    await init_db(engine)

    vs = InMemoryVectorStore()
    # Pre-create the user + file rows so the FK in ChunkMeta is satisfied.
    async with db_mod.get_session(engine) as session:
        u = User(clerk_id="usr_demo")
        session.add(u)
        await session.commit()
        await session.refresh(u)
        f = File(
            user_id=u.id, name="notes.txt", mime="text/plain", blob_key="usr_demo/a/notes.txt"
        )
        session.add(f)
        await session.commit()
        await session.refresh(f)
        user_id, file_id, blob_key = u.id, f.id, f.blob_key

    blob = LocalBlobStore(root=str(tmp_path / "blobs"))
    blob_path = tmp_path / "blobs" / blob_key
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_text(
        "Adverse possession requires factual possession with intent to possess. "
        "The court considered the elements: continuous, open, exclusive, hostile use.\n\n"
        "Estoppel in equity prevents asserting strict legal rights when reliance is shown."
    )

    yield {"user_id": user_id, "file_id": file_id, "blob_key": blob_key, "vs": vs}

    db_mod.reset_engine()


@pytest.mark.asyncio
async def test_ingest_text_file_writes_chunks_and_vectors(env):
    with patch.object(pipeline_mod, "make_vector_store", return_value=env["vs"]):
        with patch.object(
            pipeline_mod, "ZeroEntropyEmbeddings"
        ) as mock_emb:
            mock_emb.return_value.embed_documents.side_effect = _fake_embed_documents

            chunk_count = await pipeline_mod.ingest_file(
                file_id=env["file_id"],
                user_id=env["user_id"],
                blob_key=env["blob_key"],
                mime="text/plain",
                week="week_3",
                doc_type="note",
            )
    assert chunk_count >= 1

    # Vectors landed under the user's namespace.
    hits = env["vs"].search([0.1] * 8, namespace=env["user_id"], k=10)
    assert len(hits) == chunk_count
    assert all(h.metadata["namespace"] == env["user_id"] for h in hits)
    assert all(h.metadata["file_id"] == env["file_id"] for h in hits)

    # ChunkMeta rows persisted.
    from sqlalchemy import select

    async with db_mod.get_session() as session:
        rows = (await session.execute(select(ChunkMeta))).scalars().all()
        assert len(rows) == chunk_count
        assert {r.user_id for r in rows} == {env["user_id"]}
        assert {r.file_id for r in rows} == {env["file_id"]}
        assert {r.week for r in rows} == {"week_3"}


@pytest.mark.asyncio
async def test_ingest_invalidates_bm25_cache(env, monkeypatch):
    # Prime cache with a fake source.
    bm25.configure_bm25_source(lambda ns: [("old", "old content", {})])
    bm25.invalidate()
    bm25.get_bm25_index(env["user_id"])  # populate cache
    assert any(k.startswith(f"{env['user_id']}:") for k in bm25._index_cache)

    with patch.object(pipeline_mod, "make_vector_store", return_value=env["vs"]):
        with patch.object(pipeline_mod, "ZeroEntropyEmbeddings") as mock_emb:
            mock_emb.return_value.embed_documents.side_effect = _fake_embed_documents
            await pipeline_mod.ingest_file(
                file_id=env["file_id"],
                user_id=env["user_id"],
                blob_key=env["blob_key"],
                mime="text/plain",
            )

    # Cache entry for this namespace should be gone after ingest.
    assert not any(k.startswith(f"{env['user_id']}:") for k in bm25._index_cache)


@pytest.mark.asyncio
async def test_ingest_empty_text_returns_zero(env, tmp_path, monkeypatch):
    # Replace blob content with whitespace.
    blob_path = tmp_path / "blobs" / env["blob_key"]
    blob_path.write_text("   \n   ")
    with patch.object(pipeline_mod, "make_vector_store", return_value=env["vs"]):
        with patch.object(pipeline_mod, "ZeroEntropyEmbeddings") as mock_emb:
            mock_emb.return_value.embed_documents.side_effect = _fake_embed_documents
            n = await pipeline_mod.ingest_file(
                file_id=env["file_id"],
                user_id=env["user_id"],
                blob_key=env["blob_key"],
                mime="text/plain",
            )
    assert n == 0


@pytest.mark.asyncio
async def test_ingest_pdf_picks_per_page_sections(env, tmp_path):
    """PDF mime → per-page extraction → multiple chunks."""
    import fitz

    pdf_path = tmp_path / "blobs" / env["blob_key"].replace(".txt", ".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Page one: adverse possession requires factual possession.")
    page = doc.new_page()
    page.insert_text((72, 72), "Page two: estoppel in equity bars retraction.")
    doc.save(pdf_path)
    doc.close()

    blob_key = env["blob_key"].replace(".txt", ".pdf")
    with patch.object(pipeline_mod, "make_vector_store", return_value=env["vs"]):
        with patch.object(pipeline_mod, "ZeroEntropyEmbeddings") as mock_emb:
            mock_emb.return_value.embed_documents.side_effect = _fake_embed_documents
            n = await pipeline_mod.ingest_file(
                file_id=env["file_id"],
                user_id=env["user_id"],
                blob_key=blob_key,
                mime="application/pdf",
                doc_type="reading",
            )
    assert n == 2

    hits = env["vs"].search([0.1] * 8, namespace=env["user_id"], k=10)
    pages = sorted(int(h.metadata["page"]) for h in hits)
    assert pages == [1, 2]
    # Page-1 chunk's content should contain "possession" — sanity check on text routing.
    p1 = next(h for h in hits if h.metadata["page"] == 1)
    assert "possession" in p1.content.lower()
