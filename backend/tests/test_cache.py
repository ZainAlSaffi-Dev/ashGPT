"""Semantic answer cache."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from src.agent.cache import (
    get,
    invalidate_user,
    make_cache_key,
    normalise_query,
    put,
)
from src.storage import db as db_mod


@pytest_asyncio.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    from src.config import reload_settings

    reload_settings()
    db_mod.reset_engine()
    await db_mod.init_db(db_mod.get_engine())
    yield
    db_mod.reset_engine()


def test_normalise_collapses_whitespace_and_lowercases():
    assert normalise_query("   What  is   Adverse  Possession?  ") == "what is adverse possession"


def test_normalise_strips_terminal_punctuation():
    assert normalise_query("Hello!") == "hello"
    assert normalise_query("Hello?") == "hello"
    assert normalise_query("Hello.") == "hello"


def test_normalise_empty():
    assert normalise_query("") == ""
    assert normalise_query("   ") == ""


def test_cache_key_stable_across_chunk_id_order():
    a = make_cache_key("usr", "q", ["c2", "c1", "c3"])
    b = make_cache_key("usr", "q", ["c1", "c2", "c3"])
    assert a == b


def test_cache_key_changes_with_user():
    a = make_cache_key("usr_a", "q", ["c1"])
    b = make_cache_key("usr_b", "q", ["c1"])
    assert a != b


def test_cache_key_changes_with_query():
    a = make_cache_key("usr", "q one", ["c1"])
    b = make_cache_key("usr", "q two", ["c1"])
    assert a != b


def test_cache_key_changes_with_chunks():
    a = make_cache_key("usr", "q", ["c1"])
    b = make_cache_key("usr", "q", ["c1", "c2"])
    assert a != b


def test_cache_key_is_sha256_hex():
    key = make_cache_key("usr", "q", ["c1"])
    assert isinstance(key, str)
    assert len(key) == 64
    int(key, 16)  # valid hex


@pytest.mark.asyncio
async def test_put_then_get_round_trip(env):
    key = make_cache_key("usr", "What is adverse possession?", ["c1", "c2"])
    await put(
        cache_key=key,
        user_id="usr",
        answer="Adverse possession is...",
        payload={"sources": [{"source": "smith.pdf"}]},
    )
    got = await get(key)
    assert got is not None
    assert got["answer"] == "Adverse possession is..."
    assert got["sources"][0]["source"] == "smith.pdf"


@pytest.mark.asyncio
async def test_get_miss_returns_none(env):
    assert (await get("nonexistent")) is None


@pytest.mark.asyncio
async def test_expired_entry_is_evicted(env):
    key = make_cache_key("usr", "q", [])
    await put(
        cache_key=key,
        user_id="usr",
        answer="stale",
        payload={},
        ttl=timedelta(seconds=-1),  # already expired
    )
    assert (await get(key)) is None


@pytest.mark.asyncio
async def test_overwrite_updates_payload(env):
    key = make_cache_key("usr", "q", [])
    await put(key, "usr", "v1", {"k": 1})
    await put(key, "usr", "v2", {"k": 2})
    got = await get(key)
    assert got is not None
    assert got["answer"] == "v2"
    assert got["k"] == 2


@pytest.mark.asyncio
async def test_invalidate_user_drops_only_user(env):
    await put(make_cache_key("alice", "q", []), "alice", "a", {})
    await put(make_cache_key("bob", "q", []), "bob", "b", {})
    n = await invalidate_user("alice")
    assert n == 1
    assert (await get(make_cache_key("alice", "q", []))) is None
    assert (await get(make_cache_key("bob", "q", []))) is not None


def test_cache_check_node_short_circuits_on_hit(env, monkeypatch):
    """A pre-seeded cache entry must populate state.final_answer and skip
    every downstream LLM call (no router/synthesis/verification fires)."""
    import asyncio

    from src.agent import nodes
    from src.agent.cache import make_cache_key, put

    user_id = "usr_test"
    query = "What is adverse possession?"
    retrieved_texts = [
        {"source": "readings_w3.md", "content": "x" * 100, "week": "week_3", "doc_type": "reading"}
    ]
    retrieved_slides = []
    chunk_ids = [d["source"] for d in retrieved_texts]
    key = make_cache_key(user_id, query, chunk_ids)
    asyncio.run(
        put(
            key,
            user_id,
            "CACHED ANSWER",
            {"intent": "ratio", "verification_report": {"confidence_score": 0.9}},
        )
    )

    # If any LLM dispatch fires below, the test fails loudly.
    def boom(*a, **kw):
        raise AssertionError("LLM must not be called on cache hit")

    monkeypatch.setattr(nodes, "llm_call", boom)

    state = {
        "query": query,
        "user_id": user_id,
        "retrieved_texts": retrieved_texts,
        "retrieved_slides": retrieved_slides,
        "intent": "ratio",  # cache_check runs after retrieval; router already set this
    }
    delta = nodes.cache_check_node(state)
    assert delta["cache_hit"] is True
    assert delta["final_answer"] == "CACHED ANSWER"
    assert delta["intent"] == "ratio"
    assert "cache_check" in delta["node_trace"]


def test_cache_check_node_misses_when_chat_history_present(env):
    """Multi-turn must bypass cache — follow-ups should always re-reason."""
    import asyncio

    from src.agent import nodes
    from src.agent.cache import make_cache_key, put

    user_id = "usr_test"
    query = "And what about that case?"
    chunk_ids = ["src.md"]
    key = make_cache_key(user_id, query, chunk_ids)
    asyncio.run(put(key, user_id, "STALE", {}))

    state = {
        "query": query,
        "user_id": user_id,
        "retrieved_texts": [{"source": "src.md", "content": "y", "week": "", "doc_type": ""}],
        "retrieved_slides": [],
        "chat_history": [{"role": "user", "content": "Earlier turn"}],
    }
    delta = nodes.cache_check_node(state)
    assert delta["cache_hit"] is False
    assert "final_answer" not in delta
