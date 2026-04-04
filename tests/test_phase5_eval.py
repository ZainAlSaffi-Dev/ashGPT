"""Phase 5 tests: evaluation framework, baselines, and plotting."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestEvalImports:
    def test_eval_module_imports(self) -> None:
        from src.eval.run_evals import (
            TEST_QUERIES,
            judge_groundedness,
            run_agent_with_metrics,
            run_baseline,
            run_evaluation,
        )
        assert len(TEST_QUERIES) >= 8

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
                "week": None,
                "agent": {
                    "groundedness": {"score": 5},
                    "latency_s": 10.0,
                    "source_diversity": 3,
                    "node_latencies": {"router": 2.0, "retrieval": 1.0, "synthesis": 7.0},
                },
                "baseline": {
                    "groundedness": {"score": 3},
                    "latency_s": 5.0,
                    "source_diversity": 0,
                },
                "ablation_no_ratio": {
                    "groundedness": {"score": 4},
                    "latency_s": 7.0,
                    "source_diversity": 3,
                },
            }
        ]

        summary = _compute_summary(mock_results)
        assert summary["agent"]["avg_groundedness"] == 5.0
        assert summary["baseline"]["avg_groundedness"] == 3.0
        assert summary["agent"]["avg_latency_s"] == 10.0
        assert summary["baseline"]["avg_source_diversity"] == 0.0
        assert "avg_node_latencies" in summary["agent"]
        assert summary["agent"]["avg_node_latencies"]["router"] == 2.0

    def test_summary_handles_multiple_results(self) -> None:
        from src.eval.run_evals import _compute_summary

        mock_results = [
            {
                "query": "q1", "week": None,
                "agent": {"groundedness": {"score": 4}, "latency_s": 10.0, "source_diversity": 3, "node_latencies": {"router": 2.0}},
                "baseline": {"groundedness": {"score": 2}, "latency_s": 5.0, "source_diversity": 0},
                "ablation_no_ratio": {"groundedness": {"score": 3}, "latency_s": 7.0, "source_diversity": 3},
            },
            {
                "query": "q2", "week": None,
                "agent": {"groundedness": {"score": 5}, "latency_s": 20.0, "source_diversity": 5, "node_latencies": {"router": 4.0}},
                "baseline": {"groundedness": {"score": 4}, "latency_s": 8.0, "source_diversity": 0},
                "ablation_no_ratio": {"groundedness": {"score": 5}, "latency_s": 12.0, "source_diversity": 5},
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
                "week": "week_3",
                "agent": {
                    "groundedness": {"score": 5}, "latency_s": 30.0,
                    "source_diversity": 3,
                    "node_latencies": {"router": 3.0, "retrieval": 2.0, "ratio_extractor": 15.0, "synthesis": 10.0},
                },
                "baseline": {"groundedness": {"score": 3}, "latency_s": 8.0, "source_diversity": 0},
                "ablation_no_ratio": {"groundedness": {"score": 4}, "latency_s": 15.0, "source_diversity": 3},
            },
            {
                "query": "Explain the Torrens system",
                "week": None,
                "agent": {
                    "groundedness": {"score": 4}, "latency_s": 25.0,
                    "source_diversity": 2,
                    "node_latencies": {"router": 2.5, "retrieval": 1.5, "synthesis": 12.0},
                },
                "baseline": {"groundedness": {"score": 2}, "latency_s": 6.0, "source_diversity": 0},
                "ablation_no_ratio": {"groundedness": {"score": 3}, "latency_s": 12.0, "source_diversity": 2},
            },
        ]

        summary = _compute_summary(mock_results)
        _generate_plots(mock_results, summary, tmp_path)

        assert (tmp_path / "groundedness_comparison.png").exists()
        assert (tmp_path / "latency_comparison.png").exists()
        assert (tmp_path / "per_query_groundedness.png").exists()
        assert (tmp_path / "source_diversity.png").exists()
        assert (tmp_path / "node_latency_breakdown.png").exists()


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
