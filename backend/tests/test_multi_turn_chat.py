"""Multi-turn chat smoke tests.

Drives the FastAPI ``/chat`` endpoint with a mocked ``run_query`` so the
LangGraph pipeline is bypassed. Verifies:

  * ``session_id`` is stable across turns when passed back in the body.
  * D1 ``messages`` ordering survives multiple turns (user → assistant ×N).
  * ``run_query`` receives the rolling ``chat_history`` from D1, not just
    the current query, on every turn after the first.
  * When the graph fails mid-stream the orphan user message is deleted
    so the next turn doesn't see a dangling user row in history.
  * Verification blob carries ``synthesis_model`` + ``escalated`` for
    every assistant turn (escalation audit).
"""

from __future__ import annotations

import json
from unittest.mock import patch

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


def _auth() -> dict[str, str]:
    return {"X-Dev-User": "usr_demo"}


def _fake_state(answer: str) -> dict:
    return {
        "node_trace": ["router", "retrieval", "synthesis"],
        "intent": "general",
        "retrieved_texts": [],
        "retrieved_slides": [],
        "final_answer": answer,
        "verification_report": {"all_supported": True},
    }


def _fake_state_with_memory(answer: str) -> dict:
    state = _fake_state(answer)
    state["chat_history_overflow"] = {
        "dropped_turns": 12,
        "truncated_messages": 0,
        "memory_compressed": True,
        "compressed_turns": 12,
        "recent_messages": 24,
        "memory_fact_count": 3,
        "memory_summary_chars": 120,
    }
    state["memory_telemetry"] = {
        "memory_compressed": True,
        "compressed_turns": 12,
        "recent_messages": 24,
        "memory_fact_count": 3,
        "memory_summary_chars": 120,
        "truncated_messages": 0,
    }
    return state


async def _post_chat(client: AsyncClient, query: str, session_id: str | None) -> dict:
    body = {"query": query}
    if session_id:
        body["session_id"] = session_id
    async with client.stream("POST", "/chat", json=body, headers=_auth()) as resp:
        assert resp.status_code == 200
        raw = await resp.aread()
    text = raw.decode()
    # Pull the `done` event payload so the caller can read session_id back.
    done_payload: dict = {}
    for line in text.splitlines():
        if line.startswith("data: ") and "session_id" in line:
            try:
                payload = json.loads(line[len("data: "):])
                if "final_answer" in payload:
                    done_payload = payload
            except json.JSONDecodeError:
                continue
    return {"text": text, "done": done_payload}


@pytest.mark.asyncio
async def test_three_turns_share_session_and_grow_history(client):
    """run_query must see prior turns on turn 2 + 3."""
    seen_histories: list[list[dict]] = []

    def _fake(query: str, week_filter, chat_history, user_id):
        seen_histories.append(list(chat_history or []))
        return _fake_state(f"answer for: {query}")

    with patch("api.routes_chat.run_query", side_effect=_fake):
        r1 = await _post_chat(client, "What is bailment?", None)
        sid = r1["done"]["session_id"]
        assert sid

        r2 = await _post_chat(client, "Give me an example", sid)
        assert r2["done"]["session_id"] == sid

        r3 = await _post_chat(client, "Now apply that to consignment", sid)
        assert r3["done"]["session_id"] == sid

    assert len(seen_histories) == 3
    # Turn 1: empty history.
    assert seen_histories[0] == []
    # Turn 2: just turn-1 user + assistant pair.
    assert len(seen_histories[1]) == 2
    assert seen_histories[1][0]["role"] == "user"
    assert seen_histories[1][0]["content"] == "What is bailment?"
    assert seen_histories[1][1]["role"] == "assistant"
    # Turn 3: turn 1+2 pairs.
    assert len(seen_histories[2]) == 4
    assert seen_histories[2][2]["content"] == "Give me an example"


@pytest.mark.asyncio
async def test_failed_graph_rolls_back_orphan_user_message(client):
    """If run_query raises, the user row must be deleted so history stays clean."""
    seen_histories: list[list[dict]] = []
    call_count = {"n": 0}

    def _fake(query: str, week_filter, chat_history, user_id):
        seen_histories.append(list(chat_history or []))
        call_count["n"] += 1
        if call_count["n"] == 2:  # fail the second turn
            raise RuntimeError("synthetic graph failure")
        return _fake_state(f"answer for: {query}")

    with patch("api.routes_chat.run_query", side_effect=_fake):
        r1 = await _post_chat(client, "What is estoppel?", None)
        sid = r1["done"]["session_id"]

        # Turn 2 will fail mid-stream.
        async with client.stream(
            "POST",
            "/chat",
            json={"query": "Explain the elements", "session_id": sid},
            headers=_auth(),
        ) as resp:
            body = (await resp.aread()).decode()
        assert "error" in body

        # Turn 3 should see only the turn-1 pair — turn-2's user row was rolled back.
        r3 = await _post_chat(client, "Try again please", sid)

    # Turn 3 history must NOT include "Explain the elements" (the orphan).
    history_seen_by_turn_3 = seen_histories[-1]
    assert all(
        m["content"] != "Explain the elements" for m in history_seen_by_turn_3
    ), f"orphan user message leaked into history: {history_seen_by_turn_3}"


def test_public_error_detail_redacts_sql_and_embedding_params():
    from sqlalchemy.exc import OperationalError

    from api.routes_chat import _public_error_detail

    err = OperationalError(
        "SELECT id, content FROM vectors WHERE embedding <=> :emb",
        {"emb": "[very long embedding vector]"},
        Exception("SSL error: unexpected eof while reading"),
    )

    detail = _public_error_detail(err)

    assert "retry in a moment" in detail
    assert "SELECT" not in detail
    assert "embedding" not in detail


@pytest.mark.asyncio
async def test_verification_records_synthesis_model_and_escalated_flag(client):
    """The saved assistant message must record which synthesis model ran."""

    def _fake(query: str, week_filter, chat_history, user_id):
        # Simulate an escalated turn.
        state = _fake_state("escalated answer")
        state["escalated_from"] = "gpt-5.4-mini"
        state["escalated_to"] = "gpt-5.4"
        return state

    with patch("api.routes_chat.run_query", side_effect=_fake):
        r = await _post_chat(client, "Tricky question", None)
        sid = r["done"]["session_id"]

    msgs = (
        await client.get(f"/sessions/{sid}/messages", headers=_auth())
    ).json()
    assistant = next(m for m in msgs if m["role"] == "assistant")
    vr = assistant["verification"]
    assert vr["synthesis_model"] == "gpt-5.4"
    assert vr["escalated"] is True
    assert vr["escalated_from"] == "gpt-5.4-mini"


@pytest.mark.asyncio
async def test_chat_stream_emits_memory_compression_telemetry(client):
    def _fake(query: str, week_filter, chat_history, user_id):
        return _fake_state_with_memory("compressed answer")

    with patch("api.routes_chat.run_query", side_effect=_fake):
        r = await _post_chat(client, "Long chat follow-up", None)

    assert "event: history_overflow" in r["text"]
    assert "event: memory" in r["text"]
    assert '"memory_compressed": true' in r["text"]
