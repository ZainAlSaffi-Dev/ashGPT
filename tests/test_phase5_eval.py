"""Phase 5 tests: evaluation framework, baselines, and plotting."""

from __future__ import annotations

from pathlib import Path

import pytest


def _mock_agent(**kwargs) -> dict:
    defaults = {
        "groundedness": {"score": 5},
        "answer_relevancy": {"score": 5},
        "latency_s": 10.0,
        "source_diversity": 3,
        "node_latencies": {"router": 2.0, "retrieval": 1.0, "synthesis": 7.0},
        "mermaid_validity": {"score": 0.0},
        "irac_compliance": {"score": 0.0},
        "context_precision": {
            "precision_at_k": 0.5,
            "mrr": 0.5,
            "hit_at_k": 1.0,
            "ndcg_at_k": 0.8,
            "k": 12,
        },
        "token_usage": {"total_input_tokens": 100, "total_output_tokens": 50, "total_tokens": 150},
    }
    defaults.update(kwargs)
    return defaults


def _mock_baseline(**kwargs) -> dict:
    defaults = {
        "groundedness": {"score": 3},
        "answer_relevancy": {"score": 4},
        "latency_s": 5.0,
        "source_diversity": 0,
        "mermaid_validity": {"score": 0.0},
        "irac_compliance": {"score": 0.0},
        "token_usage": {"total_input_tokens": 10, "total_output_tokens": 20, "total_tokens": 30},
    }
    defaults.update(kwargs)
    return defaults


def _mock_ablation(**kwargs) -> dict:
    defaults = {
        "groundedness": {"score": 4},
        "answer_relevancy": {"score": 5},
        "latency_s": 7.0,
        "source_diversity": 3,
        "mermaid_validity": {"score": 0.0},
        "irac_compliance": {"score": 0.0},
        "context_precision": {
            "precision_at_k": 0.5,
            "mrr": 0.5,
            "hit_at_k": 1.0,
            "ndcg_at_k": 0.8,
            "k": 12,
        },
        "token_usage": {"total_input_tokens": 80, "total_output_tokens": 40, "total_tokens": 120},
    }
    defaults.update(kwargs)
    return defaults


class TestRetrievalRankingMetrics:
    def test_mrr_first_relevant_at_two(self) -> None:
        from src.eval.run_evals import _ranking_metrics_from_binary_verdicts

        m = _ranking_metrics_from_binary_verdicts([False, True, False])
        assert m["mrr"] == 0.5
        assert m["hit_at_k"] == 1.0
        assert m["k"] == 3

    def test_all_irrelevant(self) -> None:
        from src.eval.run_evals import _ranking_metrics_from_binary_verdicts

        m = _ranking_metrics_from_binary_verdicts([False, False])
        assert m["mrr"] == 0.0
        assert m["hit_at_k"] == 0.0


class TestEvalCaseHelpers:
    def test_judge_question_multi_turn_includes_prior_turns(self) -> None:
        from src.eval.run_evals import EVAL_CASES, _eval_case_last_turn, _judge_question_for_case

        conv = next(c for c in EVAL_CASES if c["case_id"] == "conv_ap_followup")
        assert _eval_case_last_turn(conv) == conv["user_turns"][-1]
        jq = _judge_question_for_case(conv)
        assert "Multi-turn" in jq
        assert conv["user_turns"][0] in jq
        assert conv["user_turns"][-1] in jq


class TestEvalImports:
    def test_eval_module_imports(self) -> None:
        from src.eval.run_evals import (
            EVAL_CASES,
            TEST_QUERIES,
            judge_groundedness,
            run_agent_with_metrics,
            run_baseline,
            run_evaluation,
        )

        assert len(EVAL_CASES) >= 8
        assert len(TEST_QUERIES) == len(EVAL_CASES)

    def test_eval_cases_cover_all_query_families(self) -> None:
        from src.eval.run_evals import EVAL_CASES

        fams = {c["query_family"] for c in EVAL_CASES}
        assert fams == {
            "factual_retrieval",
            "cross_modal_retrieval",
            "analytical_synthesis",
            "conversational_followup",
        }

    def test_plot_generation_imports(self) -> None:
        from src.eval.run_evals import _compute_summary, _generate_plots

        assert callable(_compute_summary)
        assert callable(_generate_plots)


class TestComputeSummary:
    def test_summary_from_mock_results(self) -> None:
        from src.eval.run_evals import _compute_summary

        mock_results = [
            {
                "query": "test",
                "query_family": "factual_retrieval",
                "case_id": "mock_a",
                "agent": _mock_agent(),
                "baseline": _mock_baseline(),
                "ablation_no_ratio": _mock_ablation(),
            }
        ]

        summary = _compute_summary(mock_results)
        assert summary["agent"]["avg_groundedness"] == 5.0
        assert summary["baseline"]["avg_groundedness"] == 3.0
        assert summary["agent"]["avg_latency_s"] == 10.0
        assert summary["baseline"]["avg_source_diversity"] == 0.0
        assert "avg_node_latencies" in summary["agent"]
        assert summary["agent"]["avg_node_latencies"]["router"] == 2.0
        assert "by_query_family" in summary
        assert "factual_retrieval" in summary["by_query_family"]

    def test_summary_handles_multiple_results(self) -> None:
        from src.eval.run_evals import _compute_summary

        mock_results = [
            {
                "query": "q1",
                "query_family": "factual_retrieval",
                "case_id": "m1",
                "agent": _mock_agent(groundedness={"score": 4}, node_latencies={"router": 2.0}),
                "baseline": _mock_baseline(groundedness={"score": 2}),
                "ablation_no_ratio": _mock_ablation(groundedness={"score": 3}),
            },
            {
                "query": "q2",
                "query_family": "analytical_synthesis",
                "case_id": "m2",
                "agent": _mock_agent(groundedness={"score": 5}, node_latencies={"router": 4.0}),
                "baseline": _mock_baseline(groundedness={"score": 4}),
                "ablation_no_ratio": _mock_ablation(groundedness={"score": 5}),
            },
        ]
        summary = _compute_summary(mock_results)
        assert summary["agent"]["avg_groundedness"] == 4.5
        assert summary["agent"]["avg_node_latencies"]["router"] == 3.0


class TestPlotGeneration:
    def test_plots_generate_without_error(self, tmp_path: Path) -> None:
        from src.eval.run_evals import _compute_summary, _generate_plots

        mock_results = [
            {
                "query": "What is adverse possession?",
                "query_family": "factual_retrieval",
                "case_id": "p1",
                "agent": _mock_agent(
                    groundedness={"score": 5},
                    latency_s=30.0,
                    source_diversity=3,
                    node_latencies={"router": 3.0, "retrieval": 2.0, "ratio_extractor": 15.0, "synthesis": 10.0},
                ),
                "baseline": _mock_baseline(groundedness={"score": 3}, latency_s=8.0),
                "ablation_no_ratio": _mock_ablation(groundedness={"score": 4}, latency_s=15.0),
            },
            {
                "query": "Explain the Torrens system",
                "query_family": "analytical_synthesis",
                "case_id": "p2",
                "agent": _mock_agent(
                    groundedness={"score": 4},
                    latency_s=25.0,
                    source_diversity=2,
                    node_latencies={"router": 2.5, "retrieval": 1.5, "synthesis": 12.0},
                ),
                "baseline": _mock_baseline(groundedness={"score": 2}, latency_s=6.0),
                "ablation_no_ratio": _mock_ablation(groundedness={"score": 3}, latency_s=12.0),
            },
        ]

        summary = _compute_summary(mock_results)
        _generate_plots(mock_results, summary, tmp_path)

        assert (tmp_path / "groundedness_comparison.png").exists()
        assert (tmp_path / "latency_comparison.png").exists()
        assert (tmp_path / "per_query_groundedness.png").exists()
        assert (tmp_path / "source_diversity.png").exists()
        assert (tmp_path / "node_latency_breakdown.png").exists()
        assert (tmp_path / "groundedness_by_query_family.png").exists()
        assert (tmp_path / "retrieval_ranking_metrics.png").exists()


class TestBaseline:
    @pytest.mark.integration
    def test_baseline_returns_answer(self) -> None:
        from src.eval.run_evals import run_baseline

        result = run_baseline("What is a fee simple estate?")
        assert result["answer"], "Baseline should produce an answer"
        assert result["latency_s"] > 0
        assert result["source_diversity"] == 0
        assert result["node_trace"] == ["baseline_llm"]


class TestAgentWithMetrics:
    @pytest.mark.integration
    @pytest.mark.slow
    def test_agent_returns_node_latencies(self) -> None:
        from src.eval.run_evals import run_agent_with_metrics

        result = run_agent_with_metrics("What is adverse possession?", week="week_3")
        assert "node_latencies" in result
        assert "router" in result["node_latencies"]
        assert "retrieval" in result["node_latencies"]
        assert "synthesis" in result["node_latencies"]
        assert result["context"], "Should capture context for judge"


class TestGroundednessJudge:
    @pytest.mark.integration
    def test_judge_returns_valid_score(self) -> None:
        from src.eval.run_evals import judge_groundedness

        result = judge_groundedness(
            query="What is adverse possession?",
            context="Adverse possession is a legal doctrine where a squatter gains title after 12 years.",
            answer="Adverse possession allows someone to gain title to land after occupying it for 12 years.",
        )
        assert 1 <= result["score"] <= 5
        assert "reasoning" in result
