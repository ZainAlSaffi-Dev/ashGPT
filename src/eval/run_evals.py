"""Evaluation and ablation framework for the Property Law Exam Assistant.

Compares the full agent pipeline against:
  1. A plain LLM baseline (no retrieval, no graph nodes)
  2. Ablated variant (retrieval + synthesis only, no specialised reasoning nodes)

Metrics collected per run:
  - Groundedness score (LLM-as-a-judge: 1-5 scale)
  - Source diversity (unique sources retrieved)
  - Total latency (seconds)
  - Per-node latency breakdown (for full agent)
  - Node trace (which nodes fired)

Results are exported as JSON and plots as PNG for the report.

Usage:
    python -m src.eval.run_evals
    python -m src.eval.run_evals --output-dir eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.agent.nodes import (
    _format_context,
    chronology_node,
    ratio_extractor_node,
    retrieval_node,
    router_node,
    synthesis_node,
)
from src.agent.state import AgentState
from src.config import REASONING_MODEL

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Test Queries ───────────────────────────────────────────────────────────────

TEST_QUERIES = [
    # Ratio intent
    {"query": "What is the ratio decidendi for adverse possession?", "week": "week_3"},
    {"query": "What is the legal test for establishing possession of chattels from the week 4 readings?", "week": "week_4"},
    # Chronology intent
    {"query": "Show me the timeline of events in Perry v Clissold", "week": "week_3"},
    {"query": "Map out the chain of events from the week 2 readings", "week": "week_2"},
    # Summary intent
    {"query": "Summarise the key concepts from week 1", "week": "week_1"},
    {"query": "Explain the law relating to sub-bailment from week 6", "week": "week_6"},
    # General / cross-week
    {"query": "What is a fee simple estate?", "week": None},
    {"query": "How does the concept of possession differ between land and chattels?", "week": None},
    {"query": "Explain the Torrens system of land registration", "week": None},
    {"query": "What are the requirements for a valid bailment?", "week": "week_5"},
]


# ── Groundedness Judge ─────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an impartial evaluator assessing the groundedness of an AI answer. "
    "You will be given:\n"
    "1. The original question.\n"
    "2. The source material that was available to the AI.\n"
    "3. The AI-generated answer.\n\n"
    "Score the answer on a scale of 1 to 5:\n"
    "  5 = Fully grounded: every factual claim (cases, dates, statutes, legal "
    "tests) is supported by the source material.\n"
    "  4 = Mostly grounded: core claims are supported, minor unsupported details.\n"
    "  3 = Partially grounded: some claims supported, some not in the sources.\n"
    "  2 = Mostly ungrounded: significant claims not traceable to the sources.\n"
    "  1 = Ungrounded: answer contradicts or largely ignores the source material.\n\n"
    "Note: The AI is allowed to PARAPHRASE and EXPLAIN concepts in plain English. "
    "Only penalise for introducing factual claims (case names, dates, statutes, "
    "legal holdings) that do not appear in the sources.\n\n"
    "Respond with ONLY a JSON object: {\"score\": <int>, \"reasoning\": \"<brief explanation>\"}"
)


def judge_groundedness(query: str, context: str, answer: str) -> dict:
    """Use the LLM as a judge to score groundedness of an answer."""
    prompt = (
        f"QUESTION: {query}\n\n"
        f"SOURCE MATERIAL:\n{context}\n\n"
        f"AI ANSWER:\n{answer}\n\n"
        "Evaluate the groundedness of this answer."
    )
    client = genai.Client()
    config = types.GenerateContentConfig(
        temperature=0.0,
        system_instruction=JUDGE_SYSTEM,
    )
    response = client.models.generate_content(
        model=REASONING_MODEL,
        contents=prompt,
        config=config,
    )
    try:
        cleaned = re.sub(r"```json\s*|\s*```", "", response.text).strip()
        parsed = json.loads(cleaned)
        return {"score": int(parsed["score"]), "reasoning": parsed.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        log.warning("Judge parse failed, defaulting to score=3")
        return {"score": 3, "reasoning": "Parse error"}


# ── Timed node runner ─────────────────────────────────────────────────────────


def _run_node(node_fn, state: AgentState) -> tuple[dict, float]:
    """Run a node function and return (result_dict, elapsed_seconds)."""
    start = time.time()
    result = node_fn(state)
    elapsed = time.time() - start
    return result, round(elapsed, 2)


# ── Full Agent Run with Per-Node Metrics ──────────────────────────────────────


def run_agent_with_metrics(query: str, week: str | None = None) -> dict:
    """Run the full agent pipeline with per-node latency tracking."""
    node_latencies: dict[str, float] = {}
    total_start = time.time()

    state: AgentState = {"query": query}
    if week:
        state["week_filter"] = week

    # Router
    result, lat = _run_node(router_node, state)
    state.update(result)
    node_latencies["router"] = lat

    # Retrieval
    result, lat = _run_node(retrieval_node, state)
    state.update(result)
    node_latencies["retrieval"] = lat

    # Conditional reasoning based on intent
    intent = state.get("intent", "general")

    if intent in ("ratio", "summary"):
        result, lat = _run_node(ratio_extractor_node, state)
        state.update(result)
        node_latencies["ratio_extractor"] = lat

    if intent in ("chronology", "summary"):
        result, lat = _run_node(chronology_node, state)
        state.update(result)
        node_latencies["chronology"] = lat

    # Synthesis
    result, lat = _run_node(synthesis_node, state)
    state.update(result)
    node_latencies["synthesis"] = lat

    total_elapsed = round(time.time() - total_start, 2)

    texts = state.get("retrieved_texts", [])
    slides = state.get("retrieved_slides", [])
    all_sources = set(d["source"] for d in texts + slides)

    # Build context string from what the agent actually retrieved
    context = _format_context(state)

    return {
        "answer": state.get("final_answer", ""),
        "context": context,
        "latency_s": total_elapsed,
        "node_latencies": node_latencies,
        "node_trace": state.get("node_trace", []),
        "source_diversity": len(all_sources),
        "intent": intent,
        "ratio_decidendi": state.get("ratio_decidendi", ""),
        "has_mermaid": bool(state.get("mermaid_diagram", "")),
        "retrieved_text_count": len(texts),
        "retrieved_slide_count": len(slides),
    }


# ── Baseline: Plain LLM ───────────────────────────────────────────────────────


def run_baseline(query: str) -> dict:
    """Send query directly to the LLM with no retrieval or graph."""
    start = time.time()
    client = genai.Client()
    config = types.GenerateContentConfig(
        temperature=0.2,
        system_instruction=(
            "You are an Australian Property Law tutor. Answer the student's "
            "question as accurately as possible."
        ),
    )
    response = client.models.generate_content(
        model=REASONING_MODEL,
        contents=query,
        config=config,
    )
    elapsed = time.time() - start
    return {
        "answer": response.text,
        "latency_s": round(elapsed, 2),
        "node_trace": ["baseline_llm"],
        "source_diversity": 0,
    }


# ── Ablation: Retrieval + Synthesis Only ──────────────────────────────────────


def run_ablation_no_ratio(query: str, week: str | None = None) -> dict:
    """Run retrieval + synthesis only, skipping all specialised reasoning nodes."""
    total_start = time.time()

    state: AgentState = {"query": query, "intent": "general"}
    if week:
        state["week_filter"] = week

    state.update(retrieval_node(state))
    state.update(synthesis_node(state))

    total_elapsed = round(time.time() - total_start, 2)

    texts = state.get("retrieved_texts", [])
    slides = state.get("retrieved_slides", [])
    all_sources = set(d["source"] for d in texts + slides)
    context = _format_context(state)

    return {
        "answer": state.get("final_answer", ""),
        "context": context,
        "latency_s": total_elapsed,
        "node_trace": state.get("node_trace", []),
        "source_diversity": len(all_sources),
    }


# ── Evaluation Runner ─────────────────────────────────────────────────────────


def run_evaluation(output_dir: Path) -> dict:
    """Run the full evaluation suite and return results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for i, tq in enumerate(TEST_QUERIES):
        query = tq["query"]
        week = tq["week"]
        log.info("═══ Query %d/%d: %s ═══", i + 1, len(TEST_QUERIES), query[:60])

        # 1. Full agent (captures its own context)
        log.info("Running: full agent")
        agent_result = run_agent_with_metrics(query, week)

        # 2. Baseline (plain LLM)
        log.info("Running: baseline LLM")
        baseline_result = run_baseline(query)

        # 3. Ablation: retrieval + synthesis only (captures its own context)
        log.info("Running: ablation (no reasoning nodes)")
        ablation_result = run_ablation_no_ratio(query, week)

        # 4. Judge groundedness using EACH config's own context
        log.info("Judging groundedness...")
        agent_context = agent_result.pop("context")
        ablation_context = ablation_result.pop("context")

        agent_ground = judge_groundedness(query, agent_context, agent_result["answer"])
        baseline_ground = judge_groundedness(query, agent_context, baseline_result["answer"])
        ablation_ground = judge_groundedness(query, ablation_context, ablation_result["answer"])

        result_entry = {
            "query": query,
            "week": week,
            "agent": {**agent_result, "groundedness": agent_ground},
            "baseline": {**baseline_result, "groundedness": baseline_ground},
            "ablation_no_ratio": {**ablation_result, "groundedness": ablation_ground},
        }
        all_results.append(result_entry)
        log.info(
            "Scores — Agent: %d, Baseline: %d, Ablated: %d",
            agent_ground["score"],
            baseline_ground["score"],
            ablation_ground["score"],
        )

    # Save raw results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s", results_path)

    # Generate summary
    summary = _compute_summary(all_results)
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", summary_path)

    # Generate plots
    _generate_plots(all_results, summary, output_dir)
    log.info("Plots saved to %s", output_dir)

    return summary


def _compute_summary(results: list[dict]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    configs = ["agent", "baseline", "ablation_no_ratio"]
    summary = {}

    for config in configs:
        scores = [r[config]["groundedness"]["score"] for r in results]
        latencies = [r[config]["latency_s"] for r in results]
        diversities = [r[config].get("source_diversity", 0) for r in results]

        summary[config] = {
            "avg_groundedness": round(sum(scores) / len(scores), 2),
            "avg_latency_s": round(sum(latencies) / len(latencies), 2),
            "avg_source_diversity": round(sum(diversities) / len(diversities), 2),
            "groundedness_scores": scores,
            "latencies": latencies,
        }

    # Per-node latency averages (agent only)
    node_names = set()
    for r in results:
        node_names.update(r["agent"].get("node_latencies", {}).keys())

    node_avg: dict[str, float] = {}
    for node in sorted(node_names):
        vals = [
            r["agent"]["node_latencies"].get(node, 0)
            for r in results
            if "node_latencies" in r["agent"]
        ]
        if vals:
            node_avg[node] = round(sum(vals) / len(vals), 2)
    summary["agent"]["avg_node_latencies"] = node_avg

    return summary


# ── Plot Generation ────────────────────────────────────────────────────────────


def _generate_plots(results: list[dict], summary: dict, output_dir: Path) -> None:
    """Generate comparison plots and save as PNG."""
    configs = ["agent", "baseline", "ablation_no_ratio"]
    labels = ["Full Agent", "Plain LLM\n(Baseline)", "No Ratio\n(Ablation)"]

    # 1. Groundedness comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    scores = [summary[c]["avg_groundedness"] for c in configs]
    bars = ax.bar(labels, scores, color=["#2563eb", "#9ca3af", "#f59e0b"], edgecolor="black")
    ax.set_ylabel("Average Groundedness Score (1-5)")
    ax.set_title("Groundedness: Full Agent vs Baseline vs Ablation")
    ax.set_ylim(0, 5.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{score:.1f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "groundedness_comparison.png", dpi=150)
    plt.close()

    # 2. Latency comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    latencies = [summary[c]["avg_latency_s"] for c in configs]
    bars = ax.bar(labels, latencies, color=["#2563eb", "#9ca3af", "#f59e0b"], edgecolor="black")
    ax.set_ylabel("Average Latency (seconds)")
    ax.set_title("Latency: Full Agent vs Baseline vs Ablation")
    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{lat:.1f}s", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.png", dpi=150)
    plt.close()

    # 3. Per-query groundedness breakdown
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(results))
    query_labels = [f"Q{i+1}" for i in x]

    for i, (config, label) in enumerate(zip(configs, ["Agent", "Baseline", "Ablation"])):
        scores = [r[config]["groundedness"]["score"] for r in results]
        offset = (i - 1) * 0.25
        ax.bar([xi + offset for xi in x], scores, width=0.25, label=label)

    ax.set_ylabel("Groundedness Score")
    ax.set_title("Per-Query Groundedness Breakdown")
    ax.set_xticks(list(x))
    ax.set_xticklabels(query_labels)
    ax.set_ylim(0, 5.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_query_groundedness.png", dpi=150)
    plt.close()

    # 4. Source diversity comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    diversities = [summary[c]["avg_source_diversity"] for c in configs]
    bars = ax.bar(labels, diversities, color=["#2563eb", "#9ca3af", "#f59e0b"], edgecolor="black")
    ax.set_ylabel("Average Unique Sources Retrieved")
    ax.set_title("Source Diversity: Full Agent vs Baseline vs Ablation")
    for bar, div in zip(bars, diversities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{div:.1f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "source_diversity.png", dpi=150)
    plt.close()

    # 5. Per-node latency breakdown (stacked bar for agent)
    node_lats = summary["agent"].get("avg_node_latencies", {})
    if node_lats:
        fig, ax = plt.subplots(figsize=(8, 5))
        nodes = list(node_lats.keys())
        values = list(node_lats.values())
        bars = ax.barh(nodes, values, color="#2563eb", edgecolor="black")
        ax.set_xlabel("Average Latency (seconds)")
        ax.set_title("Per-Node Latency Breakdown (Full Agent)")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}s", va="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "node_latency_breakdown.png", dpi=150)
        plt.close()

    log.info("Generated %d evaluation plots", 5 if node_lats else 4)


# ── CLI Entry Point ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation and ablation suite")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save results and plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log.info("Starting evaluation suite — results will be saved to %s/", output_dir)

    summary = run_evaluation(output_dir)

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for config, metrics in summary.items():
        print(f"\n  {config}:")
        print(f"    Avg Groundedness:     {metrics['avg_groundedness']}/5.0")
        print(f"    Avg Latency:          {metrics['avg_latency_s']}s")
        print(f"    Avg Source Diversity:  {metrics['avg_source_diversity']}")
        if "avg_node_latencies" in metrics:
            print(f"    Node Latencies:       {metrics['avg_node_latencies']}")


if __name__ == "__main__":
    main()
