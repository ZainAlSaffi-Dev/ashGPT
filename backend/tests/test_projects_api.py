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

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    db_mod.reset_engine()


def auth(user: str = "usr_demo") -> dict[str, str]:
    if user != "usr_demo":
        return {"X-User-Id": user}
    return {"X-Dev-User": user}


@pytest.mark.asyncio
async def test_project_and_folder_crud_is_user_scoped(client):
    r = await client.post("/projects", json={"name": "LAWS1100 Torts"}, headers=auth())
    assert r.status_code == 201, r.text
    project = r.json()
    assert project["slug"] == "laws1100-torts"

    r = await client.post(
        f"/projects/{project['id']}/folders",
        json={"name": "Week 1"},
        headers=auth(),
    )
    assert r.status_code == 201, r.text
    folder = r.json()
    assert folder["project_id"] == project["id"]

    r = await client.get("/projects", headers=auth("usr_other"))
    assert r.status_code == 200
    assert r.json() == []

    r = await client.get(f"/projects/{project['id']}", headers=auth("usr_other"))
    assert r.status_code == 404

    r = await client.patch(
        f"/folders/{folder['id']}",
        json={"name": "Intro"},
        headers=auth(),
    )
    assert r.status_code == 200
    assert r.json()["name"] == "Intro"


@pytest.mark.asyncio
async def test_non_empty_folder_delete_requires_recursive(client):
    project = (
        await client.post("/projects", json={"name": "Equity"}, headers=auth())
    ).json()
    folder = (
        await client.post(
            f"/projects/{project['id']}/folders",
            json={"name": "Readings"},
            headers=auth(),
        )
    ).json()
    file_row = (
        await client.post(
            "/uploads/presign",
            json={
                "name": "reading.pdf",
                "mime": "application/pdf",
                "project_id": project["id"],
                "folder_id": folder["id"],
            },
            headers=auth(),
        )
    ).json()

    r = await client.delete(f"/folders/{folder['id']}", headers=auth())
    assert r.status_code == 409

    r = await client.delete(f"/folders/{folder['id']}?recursive=true", headers=auth())
    assert r.status_code == 204

    r = await client.get(f"/files?project_id={project['id']}", headers=auth())
    assert r.status_code == 200
    [moved] = [f for f in r.json() if f["id"] == file_row["file_id"]]
    assert moved["folder_id"] is None
