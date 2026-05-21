"""Tests for /users/me + /users/me/onboarded."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.config import reload_settings


@pytest_asyncio.fixture
async def client(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DEV_AUTH_USER", "usr_demo")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("BLOB_BACKEND", "local")
    monkeypatch.setenv("BLOB_LOCAL_ROOT", str(tmp_path / "blobs"))
    monkeypatch.setenv("VECTOR_BACKEND", "memory")
    reload_settings()

    from src.storage import db as db_mod

    db_mod.reset_engine()
    await db_mod.init_db(db_mod.get_engine())

    from api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    db_mod.reset_engine()


def _hdr() -> dict[str, str]:
    return {"X-Dev-User": "usr_demo"}


@pytest.mark.asyncio
async def test_me_returns_user_with_null_onboarded(client):
    r = await client.get("/users/me", headers=_hdr())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["clerk_id"] == "usr_demo"
    assert body["onboarded_at"] is None


@pytest.mark.asyncio
async def test_mark_onboarded_then_get(client):
    r = await client.post("/users/me/onboarded", headers=_hdr())
    assert r.status_code == 200, r.text
    stamped = r.json()["onboarded_at"]
    assert stamped is not None

    r = await client.get("/users/me", headers=_hdr())
    assert r.status_code == 200
    assert r.json()["onboarded_at"] == stamped


@pytest.mark.asyncio
async def test_mark_onboarded_is_idempotent(client):
    r1 = await client.post("/users/me/onboarded", headers=_hdr())
    first = r1.json()["onboarded_at"]
    r2 = await client.post("/users/me/onboarded", headers=_hdr())
    second = r2.json()["onboarded_at"]
    assert first == second, "second POST should not overwrite the timestamp"


@pytest.mark.asyncio
async def test_users_me_requires_auth(client):
    r = await client.get("/users/me")
    assert r.status_code == 401
