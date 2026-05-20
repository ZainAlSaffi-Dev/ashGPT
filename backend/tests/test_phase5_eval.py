"""Slim eval-suite tests — pure-function helpers + summary + plots."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.eval.run_evals import (
    EVAL_CASES,
    RETRIEVAL_MODES,
    TEST_QUERIES,
    _compute_summary,
    _eval_case_last_turn,
    _generate_plots,
    _judge_question_for_case,
    _ranking_metrics_from_binary_verdicts,
    check_irac_compliance,
    check_mermaid_validity,
    compare_retrieval_modes,
    judge_context_precision,
)


# ── Eval-case shape ──────────────────────────────────────────────────────────


def test_eval_cases_present_and_aligned_with_test_queries():
    assert len(EVAL_CASES) >= 8
    assert len(TEST_QUERIES) == len(EVAL_CASES)


def test_eval_cases_cover_all_query_families():
    fams = {c["query_family"] for c in EVAL_CASES}
    assert fams == {
        "factual_retrieval",
        "cross_modal_retrieval",
        "analytical_synthesis",
        "conversational_followup",
    }


def test_judge_question_singles_returns_only_turn():
    case = next(c for c in EVAL_CASES if len(c["user_turns"]) == 1)
    assert _judge_question_for_case(case) == case["user_turns"][0]


def test_judge_question_multi_turn_includes_history():
    conv = next(c for c in EVAL_CASES if c["case_id"] == "conv_ap_followup")
    assert _eval_case_last_turn(conv) == conv["user_turns"][-1]
    jq = _judge_question_for_case(conv)
    assert "Multi-turn" in jq
    assert conv["user_turns"][0] in jq
    assert conv["user_turns"][-1] in jq


# ── Ranking metrics ───────────────────────────────────────────────────────────


def test_mrr_first_relevant_at_two():
    m = _ranking_metrics_from_binary_verdicts([False, True, False])
    assert m["mrr"] == 0.5
    assert m["hit_at_k"] == 1.0
    assert m["k"] == 3


def test_all_irrelevant():
    m = _ranking_metrics_from_binary_verdicts([False, False])
    assert m["mrr"] == 0.0
    assert m["hit_at_k"] == 0.0


def test_all_relevant_perfect_ndcg():
    m = _ranking_metrics_from_binary_verdicts([True, True, True])
    assert m["mrr"] == 1.0
    assert m["ndcg_at_k"] == 1.0


def test_empty_verdicts():
    m = _ranking_metrics_from_binary_verdicts([])
    assert m["k"] == 0
    assert m["mrr"] == 0.0


# ── Structural checks ─────────────────────────────────────────────────────────


def test_mermaid_validity_full_score():
    code = "flowchart TD\nA --> B\nB --> C"
    out = check_mermaid_validity(code)
    assert out["score"] == 1.0


def test_mermaid_validity_partial_no_edges():
    out = check_mermaid_validity("flowchart TD")
    assert out["score"] == 0.5


def test_mermaid_validity_empty():
    assert check_mermaid_validity("")["score"] == 0.0
    assert check_mermaid_validity(None)["score"] == 0.0  # type: ignore[arg-type]


def test_irac_compliance_full():
    text = "Issue: ...\nRule: ...\nApplication: ...\nConclusion: ..."
    assert check_irac_compliance(text, intent="ratio")["score"] == 1.0


def test_irac_compliance_partial():
    text = "Issue: ...\nRule: ..."
    out = check_irac_compliance(text, intent="ratio")
    assert out["score"] == 0.5


def test_irac_compliance_not_applicable_for_chronology():
    out = check_irac_compliance("anything", intent="chronology")
    assert out["score"] is None


# ── Context precision judge — mocked LLM ─────────────────────────────────────


def test_context_precision_with_no_chunks_returns_zero():
    out = judge_context_precision("q", [])
    assert out["precision_at_k"] == 0.0
    assert out["total_count"] == 0


def test_context_precision_mocks_judge_per_chunk():
    chunks = [
        {"source": "a.pdf", "content": "adverse possession content"},
        {"source": "b.pdf", "content": "irrelevant filler"},
        {"source": "c.pdf", "content": "another adverse possession passage"},
    ]
    side = [
        {"relevant": True},
        {"relevant": False},
        {"relevant": True},
    ]
    with patch("src.eval.run_evals._judge_call", side_effect=side):
        out = judge_context_precision("adverse possession", chunks)
    assert out["precision_at_k"] == round(2 / 3, 3)
    assert out["relevant_count"] == 2
    assert out["verdicts"] == [True, False, True]
    assert out["mrr"] == 1.0


# ── Retrieval-mode comparison (mocked retrieval + judge) ─────────────────────


def test_retrieval_modes_definition():
    names = {m["name"] for m in RETRIEVAL_MODES}
    assert names == {
        "dense_only_no_rerank",
        "dense_only_rerank",
        "hybrid_no_rerank",
        "hybrid_rerank",
    }


def test_compare_retrieval_modes_runs_per_mode_and_aggregates():
    case = EVAL_CASES[0]
    fake_chunks_by_mode = {
        "dense_only_no_rerank": [
            {"source": "a", "content": "x"},
            {"source": "b", "content": "y"},
        ],
        "dense_only_rerank": [
            {"source": "a", "content": "x"},
            {"source": "b", "content": "y"},
        ],
        "hybrid_no_rerank": [
            {"source": "a", "content": "x"},
            {"source": "b", "content": "y"},
        ],
        "hybrid_rerank": [
            {"source": "a", "content": "x"},
            {"source": "b", "content": "y"},
        ],
    }
    # Verdict pattern per mode → hybrid_rerank gets the best precision.
    verdicts_by_mode = {
        "dense_only_no_rerank": [False, False],
        "dense_only_rerank":    [True,  False],
        "hybrid_no_rerank":     [True,  False],
        "hybrid_rerank":        [True,  True],
    }
    call_order = {"i": 0}
    seq_modes = ["dense_only_no_rerank", "dense_only_rerank", "hybrid_no_rerank", "hybrid_rerank"]

    def fake_retrieve(query, use_hybrid, use_reranker):
        # Mirror the call order in RETRIEVAL_MODES (verified by names below).
        name = seq_modes[call_order["i"] % 4]
        call_order["i"] += 1
        return fake_chunks_by_mode[name]

    judge_calls = {"i": 0}

    def fake_judge(prompt, system, model=None):
        # The compare loop runs 4 modes × N chunks judge calls. Cycle verdicts.
        mode_idx = (judge_calls["i"] // 2) % 4
        within = judge_calls["i"] % 2
        judge_calls["i"] += 1
        mode = seq_modes[mode_idx]
        return {"relevant": verdicts_by_mode[mode][within]}

    with patch("src.eval.run_evals._retrieve_with_mode", side_effect=fake_retrieve):
        with patch("src.eval.run_evals._judge_call", side_effect=fake_judge):
            out = compare_retrieval_modes([case])

    per = out["per_case"][0]["modes"]
    assert per["dense_only_no_rerank"]["precision_at_k"] == 0.0
    assert per["dense_only_rerank"]["precision_at_k"] == 0.5
    assert per["hybrid_no_rerank"]["precision_at_k"] == 0.5
    assert per["hybrid_rerank"]["precision_at_k"] == 1.0
    # Hybrid+rerank tops the aggregate.
    assert out["avg"]["hybrid_rerank"]["precision_at_k"] == 1.0
    assert out["avg"]["dense_only_no_rerank"]["precision_at_k"] == 0.0


# ── Summary + plots ───────────────────────────────────────────────────────────


def _mock_case_result(case_id: str, **overrides) -> dict:
    base = {
        "case_id": case_id,
        "query_family": "factual_retrieval",
        "intent": "general",
        "node_trace": ["router", "retrieval", "synthesis"],
        "latency_s": 8.0,
        "source_diversity": 3,
        "answer": "...",
        "groundedness": {"score": 4},
        "answer_relevancy": {"score": 5},
        "context_precision": {"precision_at_k": 0.75, "mrr": 1.0, "ndcg_at_k": 0.9, "k": 12},
        "mermaid_validity": {"score": 0.0},
        "irac_compliance": {"score": 1.0},
        "verification": {"all_supported": True},
        "token_usage": {"total_input_tokens": 100, "total_output_tokens": 50, "total_tokens": 150},
        "escalated_to": None,
    }
    base.update(overrides)
    return base


def test_summary_computes_means_and_groups_by_family():
    results = [
        _mock_case_result("a", query_family="factual_retrieval", groundedness={"score": 5}),
        _mock_case_result("b", query_family="factual_retrieval", groundedness={"score": 3}),
        _mock_case_result(
            "c", query_family="conversational_followup",
            groundedness={"score": 4}, latency_s=12.0,
        ),
    ]
    retrieval = {
        "avg": {
            "dense_only_no_rerank": {"precision_at_k": 0.4, "mrr": 0.5, "ndcg_at_k": 0.6, "hit_at_k": 1.0},
            "hybrid_rerank":        {"precision_at_k": 0.8, "mrr": 0.9, "ndcg_at_k": 0.95, "hit_at_k": 1.0},
        }
    }
    summary = _compute_summary(results, retrieval)
    assert summary["n_cases"] == 3
    assert summary["agent"]["avg_groundedness"] == 4.0
    assert summary["agent"]["avg_latency_s"] == round((8.0 + 8.0 + 12.0) / 3, 3)
    assert summary["by_query_family"]["factual_retrieval"]["avg_groundedness"] == 4.0
    assert summary["by_query_family"]["conversational_followup"]["avg_groundedness"] == 4.0
    # Headline lift visible.
    assert summary["retrieval_modes"]["hybrid_rerank"]["precision_at_k"] == 0.8


def test_summary_skips_errored_cases():
    results = [
        _mock_case_result("a"),
        {"case_id": "b", "query_family": "factual_retrieval", "error": "boom"},
    ]
    summary = _compute_summary(results, {})
    assert summary["n_cases"] == 2
    assert summary["n_successful"] == 1


def test_plots_generate_without_error(tmp_path: Path):
    results = [
        _mock_case_result("a"),
        _mock_case_result("b", query_family="analytical_synthesis"),
    ]
    retrieval = {
        "avg": {
            "dense_only_no_rerank": {"precision_at_k": 0.3, "mrr": 0.4, "ndcg_at_k": 0.5, "hit_at_k": 0.8},
            "dense_only_rerank":    {"precision_at_k": 0.55, "mrr": 0.6, "ndcg_at_k": 0.7, "hit_at_k": 0.9},
            "hybrid_no_rerank":     {"precision_at_k": 0.6, "mrr": 0.65, "ndcg_at_k": 0.75, "hit_at_k": 1.0},
            "hybrid_rerank":        {"precision_at_k": 0.85, "mrr": 0.9, "ndcg_at_k": 0.93, "hit_at_k": 1.0},
        }
    }
    _generate_plots(results, retrieval, tmp_path)
    assert (tmp_path / "groundedness.png").exists()
    assert (tmp_path / "context_precision.png").exists()
    assert (tmp_path / "latency.png").exists()
    assert (tmp_path / "retrieval_modes.png").exists()


def test_plots_handle_only_retrieval(tmp_path: Path):
    """No agent results — plots still produced for retrieval modes."""
    retrieval = {
        "avg": {
            "dense_only_no_rerank": {"precision_at_k": 0.3, "mrr": 0.4, "ndcg_at_k": 0.5, "hit_at_k": 0.8},
            "hybrid_rerank":        {"precision_at_k": 0.85, "mrr": 0.9, "ndcg_at_k": 0.93, "hit_at_k": 1.0},
        }
    }
    _generate_plots([], retrieval, tmp_path)
    assert (tmp_path / "retrieval_modes.png").exists()
    # No agent plots without agent rows.
    assert not (tmp_path / "groundedness.png").exists()


# ── Live integration (requires API keys) ─────────────────────────────────────


class TestLive:
    @pytest.mark.integration
    @pytest.mark.slow
    def test_groundedness_judge_returns_valid_score(self):
        from src.eval.run_evals import judge_groundedness

        result = judge_groundedness(
            query="What is adverse possession?",
            context="Adverse possession lets someone gain title to land after 12 years of open use.",
            answer="It is a legal doctrine where occupation for a statutory period gives title.",
        )
        assert 1 <= result["score"] <= 5
        assert "reasoning" in result

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_evaluation_smoke(self, tmp_path: Path):
        from src.eval.run_evals import run_evaluation

        # One-case sample, retrieval-only — keeps the live cost minimal.
        summary = run_evaluation(
            output_dir=tmp_path,
            skip_agent=True,
            sample_size=1,
        )
        assert "retrieval_modes" in summary
