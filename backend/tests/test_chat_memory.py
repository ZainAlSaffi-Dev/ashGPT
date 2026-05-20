"""Unit tests for multi-turn chat memory helpers (no API calls)."""

from __future__ import annotations

from src.agent.chat_memory import (
    build_retrieval_query,
    format_transcript_for_llm,
    get_chat_history,
    prepare_chat_history_for_run,
)


def test_prepare_drops_invalid_and_returns_overflow_zero() -> None:
    raw = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "bogus", "content": "x"},
        {"role": "user", "content": ""},
    ]
    out, overflow = prepare_chat_history_for_run(raw)
    assert len(out) == 2
    assert out[0]["role"] == "user"
    assert out[1]["role"] == "assistant"
    assert overflow == {"dropped_turns": 0, "truncated_messages": 0}


def test_prepare_caps_turn_count_and_reports_overflow() -> None:
    from src.config import CHAT_HISTORY_MAX_MESSAGES

    raw = []
    for i in range(CHAT_HISTORY_MAX_MESSAGES + 5):
        raw.append({"role": "user", "content": f"q{i}"})
        raw.append({"role": "assistant", "content": f"a{i}"})
    out, overflow = prepare_chat_history_for_run(raw)
    assert len(out) == CHAT_HISTORY_MAX_MESSAGES
    assert overflow["dropped_turns"] == (
        len(raw) - CHAT_HISTORY_MAX_MESSAGES
    )


def test_prepare_truncates_long_messages_and_reports() -> None:
    from src.config import CHAT_HISTORY_MAX_CHARS_PER_MESSAGE

    big = "x" * (CHAT_HISTORY_MAX_CHARS_PER_MESSAGE + 500)
    out, overflow = prepare_chat_history_for_run(
        [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": big},
        ]
    )
    assert overflow["truncated_messages"] == 1
    assert "[truncated]" in out[1]["content"]


def test_format_transcript_numbered_labels() -> None:
    msgs, _ = prepare_chat_history_for_run(
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
    hist, _ = prepare_chat_history_for_run(
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
