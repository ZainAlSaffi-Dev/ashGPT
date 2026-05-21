"""Coreference rewriter tests — Stage 4 of the polish plan."""

from __future__ import annotations

from src.agent.chat_memory import rewrite_followup_query


def test_rewrite_returns_input_unchanged_when_history_empty() -> None:
    assert rewrite_followup_query("What is bailment?", []) == "What is bailment?"


def test_rewrite_uses_llm_to_resolve_pronouns() -> None:
    history = [
        {"role": "user", "content": "Explain Mabo v Queensland (No 2)."},
        {
            "role": "assistant",
            "content": "Mabo v Queensland (No 2) recognised native title at common law.",
        },
    ]

    captured: dict = {}

    def fake_llm(prompt: str, model: str, system_instruction: str) -> str:
        captured["prompt"] = prompt
        captured["model"] = model
        captured["system"] = system_instruction
        return "Explain the ratio decidendi in Mabo v Queensland (No 2)."

    out = rewrite_followup_query(
        "What's the ratio decidendi?",
        history,
        llm_call=fake_llm,
        model="test-model",
    )
    assert out == "Explain the ratio decidendi in Mabo v Queensland (No 2)."
    assert "Mabo" in captured["prompt"]
    assert captured["model"] == "test-model"
    assert "self-contained legal search query" in captured["system"]


def test_rewrite_falls_back_on_llm_failure() -> None:
    history = [{"role": "user", "content": "What about adverse possession?"}]

    def explode(*_args, **_kwargs):
        raise RuntimeError("model timeout")

    out = rewrite_followup_query(
        "explain it further", history, llm_call=explode, model="x"
    )
    assert out == "explain it further"


def test_rewrite_strips_quotes_and_extra_lines() -> None:
    history = [{"role": "user", "content": "Tell me about negligence."}]

    def fake_llm(prompt: str, model: str, system_instruction: str) -> str:
        return '"Define duty of care in negligence."\nfoo bar'

    out = rewrite_followup_query(
        "what's duty of care?", history, llm_call=fake_llm, model="x"
    )
    assert out == "Define duty of care in negligence."


def test_rewrite_returns_original_when_llm_returns_empty() -> None:
    history = [{"role": "user", "content": "ok"}]

    def empty(*_args, **_kwargs) -> str:
        return "   \n  "

    out = rewrite_followup_query("again?", history, llm_call=empty, model="x")
    assert out == "again?"


def test_rewrite_can_use_compressed_memory_without_recent_history() -> None:
    memory = {
        "summary": "",
        "facts": [
            {
                "type": "authority_discussed",
                "text": "Earlier authority discussed: Mabo v Queensland (No 2).",
            }
        ],
        "source_grounding": "Compressed conversation memory is not source evidence.",
        "compressed_turns": 30,
    }
    captured: dict = {}

    def fake_llm(prompt: str, model: str, system_instruction: str) -> str:
        captured["prompt"] = prompt
        return "Explain the ratio decidendi in Mabo v Queensland (No 2)."

    out = rewrite_followup_query(
        "what was the ratio?",
        [],
        memory=memory,
        llm_call=fake_llm,
        model="x",
    )

    assert out == "Explain the ratio decidendi in Mabo v Queensland (No 2)."
    assert "COMPRESSED CONVERSATION MEMORY" in captured["prompt"]
    assert "Mabo v Queensland" in captured["prompt"]
