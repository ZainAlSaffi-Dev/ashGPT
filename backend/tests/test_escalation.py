"""Confidence-gated synthesis escalation."""

from __future__ import annotations

from unittest.mock import patch

from src.agent.graph import _maybe_escalate


def _state(confidence: float) -> dict:
    return {
        "query": "What is adverse possession?",
        "final_answer": "Draft answer with citations.",
        "retrieved_texts": [
            {"content": "ctx", "source": "smith.pdf", "week": "", "doc_type": "", "image_path": None}
        ],
        "retrieved_slides": [],
        "verification_report": {"confidence_score": confidence, "all_supported": confidence >= 1.0},
    }


def test_no_escalation_when_confidence_above_threshold():
    state = _state(confidence=0.85)
    out = _maybe_escalate(state)
    assert "escalated_from" not in out
    assert out["final_answer"] == state["final_answer"]


def test_escalates_when_below_threshold():
    state = _state(confidence=0.3)
    with patch(
        "src.agent.nodes.synthesis_node",
        return_value={"final_answer": "Stronger answer"},
    ) as mock_node:
        out = _maybe_escalate(state)
    assert out["final_answer"] == "Stronger answer"
    assert out["escalated_from"]
    assert out["escalated_to"]
    # Confirm the override model was passed through state.
    called_state = mock_node.call_args[0][0]
    assert called_state["_override_synthesis_model"]


def test_escalation_failure_keeps_original_answer():
    state = _state(confidence=0.2)
    with patch(
        "src.agent.nodes.synthesis_node",
        side_effect=RuntimeError("LLM down"),
    ):
        out = _maybe_escalate(state)
    assert out["final_answer"] == state["final_answer"]
    assert "escalated_from" not in out


def test_no_escalation_when_no_verification_report():
    state = {
        "query": "q",
        "final_answer": "a",
        "retrieved_texts": [],
        "retrieved_slides": [],
    }
    out = _maybe_escalate(state)
    assert out == state


def test_empty_new_answer_keeps_original():
    state = _state(confidence=0.1)
    with patch("src.agent.nodes.synthesis_node", return_value={"final_answer": ""}):
        out = _maybe_escalate(state)
    assert out["final_answer"] == state["final_answer"]
    assert "escalated_from" not in out
