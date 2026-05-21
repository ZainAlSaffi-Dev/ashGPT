"""Unit tests for multi-turn chat memory helpers (no API calls)."""

from __future__ import annotations

from src.agent.chat_memory import (
    build_retrieval_query,
    format_memory_for_llm,
    format_transcript_for_llm,
    get_chat_history,
    prepare_chat_memory_for_run,
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


def _long_history_with_early_context() -> list[dict[str, str]]:
    raw = [
        {
            "role": "user",
            "content": (
                "Assume Queensland law. AP means adverse possession. "
                "Only use uploaded sources. Explain Mabo v Queensland (No 2)."
            ),
        },
        {
            "role": "assistant",
            "content": "Mabo v Queensland (No 2) was discussed with source citations.",
        },
    ]
    for i in range(30):
        raw.append({"role": "user", "content": f"follow-up question {i}"})
        raw.append({"role": "assistant", "content": f"follow-up answer {i}"})
    return raw


def test_prepare_memory_keeps_recent_window_and_compresses_older_context() -> None:
    raw = _long_history_with_early_context()

    recent, telemetry, memory = prepare_chat_memory_for_run(raw)

    assert len(recent) == 24
    assert telemetry["memory_compressed"] is True
    assert telemetry["compressed_turns"] == len(raw) - 24
    assert memory is not None
    rendered = format_memory_for_llm(memory)
    assert "Queensland law" in rendered
    assert "AP = adverse possession" in rendered
    assert "Mabo v Queensland" in rendered
    assert "not source evidence" in rendered


def test_memory_corrections_replace_stale_jurisdiction() -> None:
    raw = [
        {"role": "user", "content": "Assume Queensland law. AP means adverse possession."},
        {"role": "assistant", "content": "Noted."},
        {
            "role": "user",
            "content": "Actually ignore that jurisdiction and use NSW law instead.",
        },
        {"role": "assistant", "content": "Noted."},
    ]
    raw.extend(
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"padding {i}"}
        for i in range(30)
    )

    _recent, _telemetry, memory = prepare_chat_memory_for_run(raw)

    assert memory is not None
    facts = memory["facts"]
    assert any("NSW law" in f["text"] for f in facts)
    assert not any(f["type"] == "jurisdiction" and "Queensland" in f["text"] for f in facts)
    assert any(f["type"] == "correction" for f in facts)


def test_memory_corrections_remove_ignored_shorthand() -> None:
    raw = [
        {"role": "user", "content": "AP means adverse possession."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Actually ignore that AP shorthand."},
        {"role": "assistant", "content": "Noted."},
    ]
    raw.extend(
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"padding {i}"}
        for i in range(30)
    )

    _recent, _telemetry, memory = prepare_chat_memory_for_run(raw)

    assert memory is not None
    assert not any("AP =" in f["text"] for f in memory["facts"])
    assert any(f["type"] == "correction" for f in memory["facts"])


def test_retrieval_query_uses_compressed_memory_for_coreference() -> None:
    raw = _long_history_with_early_context()
    recent, _telemetry, memory = prepare_chat_memory_for_run(raw)

    query = build_retrieval_query("What is the ratio in that case?", recent, memory)

    assert "Mabo v Queensland" in query
    assert "What is the ratio in that case?" in query


def test_memory_prompt_marks_source_grounding_boundary() -> None:
    _recent, _telemetry, memory = prepare_chat_memory_for_run(
        _long_history_with_early_context()
    )

    rendered = format_memory_for_llm(memory)

    assert "session context only" in rendered
    assert "not source evidence" in rendered
    assert "legal propositions" in rendered


def test_run_query_passes_recent_history_and_memory_to_graph(monkeypatch) -> None:
    captured: dict = {}

    class FakeGraph:
        def invoke(self, state):
            captured.update(state)
            return {
                **state,
                "node_trace": ["router"],
                "intent": "general",
                "final_answer": "ok",
            }

    monkeypatch.setattr("src.agent.graph.get_graph", lambda: FakeGraph())

    from src.agent.graph import run_query

    result = run_query(
        "What does AP require now?",
        chat_history=_long_history_with_early_context(),
        use_cache=False,
        use_escalation=False,
    )

    assert len(captured["chat_history"]) == 24
    assert captured["conversation_memory"]["compressed_turns"] > 24
    assert result["memory_telemetry"]["memory_compressed"] is True
    assert result["chat_history_overflow"]["dropped_turns"] > 24
