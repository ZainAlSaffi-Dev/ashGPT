"""Unit tests for multi-turn chat memory helpers (no API calls)."""

from __future__ import annotations

from src.agent.chat_memory import (
    build_retrieval_query,
    format_transcript_for_llm,
    get_chat_history,
    prepare_chat_history_for_run,
)


def test_prepare_drops_invalid_and_truncates_count() -> None:
    raw = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "bogus", "content": "x"},
        {"role": "user", "content": ""},
    ]
    out = prepare_chat_history_for_run(raw)
    assert len(out) == 2
    assert out[0]["role"] == "user"
    assert out[1]["role"] == "assistant"


def test_format_transcript_numbered_labels() -> None:
    msgs = prepare_chat_history_for_run(
        [
            {"role": "user", "content": "What is AP?"},
            {"role": "assistant", "content": "Adverse possession is …"},
        ]
    )
    text = format_transcript_for_llm(msgs)
    assert "[1] Student:" in text
    assert "[2] Tutor:" in text
    assert "Adverse possession" in text


def test_retrieval_query_standalone_is_unchanged() -> None:
    assert build_retrieval_query("fee simple", []) == "fee simple"


def test_retrieval_query_includes_prior_answer_excerpt() -> None:
    hist = prepare_chat_history_for_run(
        [
            {"role": "user", "content": "Explain bailment"},
            {"role": "assistant", "content": "Bailment is delivery of goods …" * 50},
        ]
    )
    q = build_retrieval_query("Give one exam tip", hist)
    assert "New student question:" in q
    assert "Give one exam tip" in q
    assert "Prior tutor answer" in q
    assert "Bailment" in q


def test_get_chat_history_safe() -> None:
    assert get_chat_history({"query": "x"}) == []
    assert get_chat_history({"query": "x", "chat_history": "bad"}) == []
