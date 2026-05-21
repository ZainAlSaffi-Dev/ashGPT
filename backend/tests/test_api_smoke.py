"""End-to-end smoke tests against the FastAPI app via httpx ASGITransport.

Uses the dev-auth bypass (header ``X-Dev-User``) + in-memory SQLite to avoid
network and external services. The chat route is patched to skip running the
real LangGraph pipeline.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.config import reload_settings


@pytest_asyncio.fixture
async def client(monkeypatch, tmp_path):
    # Force dev auth + ephemeral file-backed SQLite (file-backed so multiple
    # async connections see the same database; ":memory:" is connection-local).
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DEV_AUTH_USER", "usr_demo")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("BLOB_BACKEND", "local")
    monkeypatch.setenv("BLOB_LOCAL_ROOT", str(tmp_path / "blobs"))
    monkeypatch.setenv("VECTOR_BACKEND", "memory")
    reload_settings()

    # Reset DB singletons so the new URL is picked up, then create tables
    # explicitly (httpx ASGITransport doesn't drive FastAPI lifespan).
    from src.storage import db as db_mod

    db_mod.reset_engine()
    await db_mod.init_db(db_mod.get_engine())

    from api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    db_mod.reset_engine()


def _auth_headers() -> dict[str, str]:
    return {"X-Dev-User": "usr_demo"}


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_sessions_lifecycle(client):
    r = await client.post("/sessions", json={"title": "My chat"}, headers=_auth_headers())
    assert r.status_code == 201, r.text
    session_id = r.json()["id"]

    r = await client.get("/sessions", headers=_auth_headers())
    assert r.status_code == 200
    titles = [s["title"] for s in r.json()]
    assert "My chat" in titles

    r = await client.get(f"/sessions/{session_id}/messages", headers=_auth_headers())
    assert r.status_code == 200
    assert r.json() == []

    r = await client.delete(f"/sessions/{session_id}", headers=_auth_headers())
    assert r.status_code == 204

    r = await client.get("/sessions", headers=_auth_headers())
    assert all(s["id"] != session_id for s in r.json())


@pytest.mark.asyncio
async def test_unauth_request_rejected(client):
    r = await client.get("/sessions")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_files_presign_and_list(client):
    r = await client.post(
        "/uploads/presign",
        json={"name": "notes.pdf", "mime": "application/pdf", "doc_type": "note"},
        headers=_auth_headers(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["upload_url"].startswith("/uploads/local/")
    assert body["method"] == "POST"
    assert body["blob_key"].endswith("notes.pdf")
    file_id = body["file_id"]

    r = await client.get("/files", headers=_auth_headers())
    assert r.status_code == 200
    files = r.json()
    assert any(f["id"] == file_id and f["status"] == "uploaded" for f in files)


@pytest.mark.asyncio
async def test_local_upload_path_authz(client):
    r = await client.post(
        "/uploads/presign",
        json={"name": "x.txt", "mime": "text/plain"},
        headers=_auth_headers(),
    )
    blob_key = r.json()["blob_key"]

    # Upload bytes to the local receive endpoint.
    upload_url = f"/uploads/local/{blob_key}"
    r = await client.post(upload_url, content=b"hello world", headers=_auth_headers())
    assert r.status_code == 204

    # Different user_id in path → 403.
    r = await client.post(
        f"/uploads/local/somebody_else/abc/x.txt",
        content=b"hi",
        headers=_auth_headers(),
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_scoped_local_upload_process_persists_project_folder(client, monkeypatch):
    project = (
        await client.post(
            "/projects",
            json={"name": "Evidence", "color": "#7a3b2e"},
            headers=_auth_headers(),
        )
    ).json()
    folder = (
        await client.post(
            f"/projects/{project['id']}/folders",
            json={"name": "Week 1"},
            headers=_auth_headers(),
        )
    ).json()
    r = await client.post(
        "/uploads/presign",
        json={
            "name": "notes.md",
            "mime": "application/octet-stream",
            "project_id": project["id"],
            "folder_id": folder["id"],
        },
        headers=_auth_headers(),
    )
    assert r.status_code == 200, r.text
    presign = r.json()
    await client.post(presign["upload_url"], content=b"# Notes\n\nEvidence Act", headers=_auth_headers())

    captured: dict[str, object] = {}

    async def fake_ingest_file(**kwargs):
        captured.update(kwargs)
        return 2

    from src.ingestion import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "ingest_file", fake_ingest_file)
    r = await client.post(
        f"/uploads/{presign['file_id']}/process",
        headers=_auth_headers(),
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ready"
    assert r.json()["chunk_count"] == 2
    assert captured["project_id"] == project["id"]
    assert captured["folder_id"] == folder["id"]
    assert captured["mime"] == "application/octet-stream"
    assert captured["file_name"] == "notes.md"

    r = await client.get(
        f"/files?project_id={project['id']}&folder_id={folder['id']}",
        headers=_auth_headers(),
    )
    assert r.status_code == 200
    scoped = r.json()
    assert len(scoped) == 1
    assert scoped[0]["id"] == presign["file_id"]
    assert scoped[0]["project_id"] == project["id"]
    assert scoped[0]["folder_id"] == folder["id"]
    assert scoped[0]["status"] == "ready"


@pytest.mark.asyncio
async def test_process_missing_blob_is_retryable_after_upload(client, monkeypatch):
    r = await client.post(
        "/uploads/presign",
        json={"name": "late.txt", "mime": "text/plain"},
        headers=_auth_headers(),
    )
    assert r.status_code == 200, r.text
    presign = r.json()

    r = await client.post(f"/uploads/{presign['file_id']}/process", headers=_auth_headers())
    assert r.status_code == 409

    await client.post(presign["upload_url"], content=b"late bytes", headers=_auth_headers())

    from src.ingestion import pipeline as pipeline_mod

    async def fake_ingest_file(**kwargs):
        return 1

    monkeypatch.setattr(pipeline_mod, "ingest_file", fake_ingest_file)
    r = await client.post(f"/uploads/{presign['file_id']}/process", headers=_auth_headers())
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ready"
    assert r.json()["chunk_count"] == 1


@pytest.mark.asyncio
async def test_chat_streams_events(client):
    """Smoke: chat endpoint streams SSE without touching real models."""
    fake_state = {
        "node_trace": ["router", "retrieval", "synthesis"],
        "intent": "general",
        "retrieved_texts": [{"source": "notes.pdf", "doc_type": "note", "week": None, "content": "ctx"}],
        "retrieved_slides": [],
        "final_answer": "It is the acquisition of title via long possession.",
        "verification_report": {"all_supported": True},
    }
    with patch("api.routes_chat.run_query", return_value=fake_state):
        async with client.stream(
            "POST",
            "/chat",
            json={"query": "What is adverse possession?"},
            headers=_auth_headers(),
        ) as resp:
            assert resp.status_code == 200
            body = await resp.aread()
    text = body.decode()
    assert "answer_chunk" in text
    assert "adverse possession" in text.lower() or "title via long" in text
    assert "\"intent\": \"general\"" in text or "intent" in text


@pytest.mark.asyncio
async def test_explicit_scope_on_legacy_session_is_rejected(client):
    session = (
        await client.post("/sessions", json={"title": "Legacy"}, headers=_auth_headers())
    ).json()
    project = (
        await client.post("/projects", json={"name": "Contracts"}, headers=_auth_headers())
    ).json()

    r = await client.post(
        "/chat",
        json={
            "query": "Only this project",
            "session_id": session["id"],
            "scope": {"type": "project", "project_id": project["id"]},
        },
        headers=_auth_headers(),
    )

    assert r.status_code == 409
