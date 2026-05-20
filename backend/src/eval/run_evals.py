"""Slim evaluation suite for the current LawGPT implementation.

Focus: validate the production retrieval + reranker + agent pipeline. The
prior assignment-era suite ran four configs (full agent, baseline, mega-
prompt ablation, no-reranker ablation) over 22 cases with ~20 plots. This
revision drops the ablations and reduces the case set, keeping the metrics
that still matter:

  * **Per-case agent quality** — groundedness (two-stage judge), answer
    relevancy, context precision + MRR + Hit@K + NDCG@K, structural checks
    (Mermaid validity for chronology, IRAC compliance for ratio), per-node
    latency, source diversity.
  * **Retrieval mode comparison (the headline lift)** — for each query,
    run retrieval-only in four modes:
        dense_only_no_rerank, dense_only_rerank,
        hybrid_no_rerank, hybrid_rerank
    score each with the context-precision judge and compare. This is the
    direct measurement of the hybrid-BM25 + Cohere-rerank investment.

Run:

    python -m src.eval.run_evals --output-dir eval_results
    python -m src.eval.run_evals --skip-retrieval-modes      # agent only
    python -m src.eval.run_evals --skip-agent                # retrieval only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from src.agent.chat_memory import prepare_chat_history_for_run
from src.agent.graph import run_query
from src.agent.tools import retrieve_all
from src.config import JUDGE_CRITIQUE_MODEL, JUDGE_DRAFT_MODEL
from src.llm import (
    get_token_usage,
    llm_call,
    reset_token_usage,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


# ── Eval cases (slim, 10 across four query families) ─────────────────────────


class EvalCase(TypedDict):
    case_id: str
    query_family: str
    rationale: str
    user_turns: list[str]


EVAL_CASES: list[EvalCase] = [
    # Factual retrieval (2)
    {
        "case_id": "fact_fee_simple",
        "query_family": "factual_retrieval",
        "rationale": "Core estate definition — single-concept factual answer.",
        "user_turns": ["What is a fee simple estate?"],
    },
    {
        "case_id": "fact_moran_test",
        "query_family": "factual_retrieval",
        "rationale": "Named-case lookup — should retrieve specific ratio/test.",
        "user_turns": ["What is the legal test from Buckinghamshire County Council v Moran?"],
    },
    # Cross-modal (2)
    {
        "case_id": "xmodal_chattels_slides",
        "query_family": "cross_modal_retrieval",
        "rationale": "Forces reliance on VLM-described lecture slides for chattels.",
        "user_turns": [
            "According to the indexed lecture slide materials, what two elements are "
            "required to establish possession of chattels?",
        ],
    },
    {
        "case_id": "xmodal_torrens_indefeasibility",
        "query_family": "cross_modal_retrieval",
        "rationale": "Targets lecture slides on Torrens indefeasibility.",
        "user_turns": [
            "Looking at the indexed lecture slides on the Torrens system, how do those "
            "materials present the doctrine of indefeasibility of title?",
        ],
    },
    # Analytical synthesis (3 — covers ratio, chronology, summary intents)
    {
        "case_id": "anal_ap_elements",
        "query_family": "analytical_synthesis",
        "rationale": "Ratio path: factual possession + animus.",
        "user_turns": [
            "What is the ratio decidendi for adverse possession across the readings?",
        ],
    },
    {
        "case_id": "anal_chronology_ap",
        "query_family": "analytical_synthesis",
        "rationale": "Chronology path: should trigger Mermaid output.",
        "user_turns": [
            "Show the chronological sequence of how adverse possession is established.",
        ],
    },
    {
        "case_id": "anal_torrens_framework",
        "query_family": "analytical_synthesis",
        "rationale": "Summary path: framework across readings + slides.",
        "user_turns": [
            "Summarise the legal framework for land registration under the Torrens system.",
        ],
    },
    # Conversational follow-up (3)
    {
        "case_id": "conv_ap_followup",
        "query_family": "conversational_followup",
        "rationale": "Coreference: 'that ratio' resolves to turn-1.",
        "user_turns": [
            "What is the ratio decidendi for adverse possession?",
            "Give one short exam tip tailored to that ratio.",
        ],
    },
    {
        "case_id": "conv_torrens_followup",
        "query_family": "conversational_followup",
        "rationale": "Contrast question after Torrens explanation.",
        "user_turns": [
            "Explain the Torrens system of land registration.",
            "In one paragraph, how does that differ from a deeds registry?",
        ],
    },
    {
        "case_id": "conv_chattels_extension",
        "query_family": "conversational_followup",
        "rationale": "Hypothetical extension referencing prior topic.",
        "user_turns": [
            "What is the legal test for establishing possession of chattels?",
            "Does our indexed material suggest the answer changes if the chattel was found "
            "on the ground rather than handed over?",
        ],
    },
]


TEST_QUERIES: list[str] = [c["user_turns"][-1] for c in EVAL_CASES]


def _eval_case_last_turn(case: EvalCase) -> str:
    return case["user_turns"][-1]


def _judge_question_for_case(case: EvalCase) -> str:
    """For multi-turn cases the judge sees the full turn context."""
    turns = case["user_turns"]
    if len(turns) == 1:
        return turns[0]
    history = "\n".join(f"  Student: {t}" for t in turns[:-1])
    return (
        f"Multi-turn case.\n"
        f"Prior turns:\n{history}\n"
        f"Current question: {turns[-1]}"
    )


# ── LLM judge helpers ────────────────────────────────────────────────────────


def _judge_call(prompt: str, system: str, model: str | None = None) -> dict:
    try:
        response = llm_call(
            prompt, model=model or JUDGE_DRAFT_MODEL, system_instruction=system, temperature=0.0
        )
    except Exception as e:
        log.warning("Judge call failed: %s", e)
        return {}
    cleaned = re.sub(r"```json\s*|\s*```", "", response or "").strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        log.warning("Judge parse failed for prompt: %s...", prompt[:80])
        return {}


JUDGE_GROUNDEDNESS_SYSTEM = (
    "You are an impartial evaluator assessing groundedness of an AI answer. "
    "Score 1–5: 5 = fully grounded (every factual claim — cases, dates, statutes, "
    "legal tests — supported by sources); 4 = mostly grounded with minor unsupported "
    "details; 3 = partially grounded; 2 = mostly ungrounded; 1 = contradicts/ignores "
    "sources. Paraphrasing and plain-English explanations are ALLOWED. Only penalise "
    "invented factual claims.\n\n"
    'Respond with JSON only: {"score": <int>, "reasoning": "<brief>"}'
)


JUDGE_CRITIQUE_SYSTEM = (
    "You are a senior evaluator reviewing a junior judge's assessment. Agree or "
    "override with a corrected score (1–5). Paraphrasing is allowed; penalise only "
    "invented factual claims.\n\n"
    'Respond with JSON only: '
    '{"score": <int 1-5>, "reasoning": "<your assessment>", "agreed_with_draft": <bool>}'
)


def judge_groundedness(query: str, context: str, answer: str) -> dict:
    """Two-stage judge (draft + critique) for cross-provider robustness."""
    base = f"QUESTION: {query}\n\nSOURCE MATERIAL:\n{context}\n\nAI ANSWER:\n{answer}\n\n"
    draft = _judge_call(
        base + "Evaluate the groundedness of this answer.",
        JUDGE_GROUNDEDNESS_SYSTEM,
        model=JUDGE_DRAFT_MODEL,
    )
    draft_score = int(draft.get("score", 3))
    draft_reasoning = draft.get("reasoning", "No reasoning")
    critique = _judge_call(
        base
        + f"JUNIOR JUDGE ASSESSMENT:\n  Score: {draft_score}/5\n  Reasoning: {draft_reasoning}\n\n"
        "Review and agree or override.",
        JUDGE_CRITIQUE_SYSTEM,
        model=JUDGE_CRITIQUE_MODEL,
    )
    return {
        "score": int(critique.get("score", draft_score)),
        "reasoning": critique.get("reasoning", draft_reasoning),
        "draft_score": draft_score,
        "agreed_with_draft": critique.get("agreed_with_draft", True),
    }


JUDGE_RELEVANCY_SYSTEM = (
    "Assess whether an AI answer is relevant to the student's question. Score 1–5: "
    "5 = directly and completely addresses the question; 1 = does not address it.\n\n"
    'Respond with JSON only: {"score": <int>, "reasoning": "<brief>"}'
)


def judge_answer_relevancy(query: str, answer: str) -> dict:
    result = _judge_call(
        f"QUESTION: {query}\n\nAI ANSWER:\n{answer}\n\nEvaluate relevancy.",
        JUDGE_RELEVANCY_SYSTEM,
    )
    return {
        "score": int(result.get("score", 3)),
        "reasoning": result.get("reasoning", "Parse error"),
    }


JUDGE_CONTEXT_PRECISION_SYSTEM = (
    "Is this retrieved chunk relevant to the question? A chunk is relevant if it "
    "contains information that helps answer the question.\n\n"
    'Respond with JSON only: {"relevant": <true|false>, "reasoning": "<brief>"}'
)


def _ranking_metrics_from_binary_verdicts(verdicts: list[bool]) -> dict:
    """MRR + Hit@K + NDCG@K from ranked-order binary relevance verdicts."""
    k = len(verdicts)
    if k == 0:
        return {"mrr": 0.0, "hit_at_k": 0.0, "ndcg_at_k": 0.0, "k": 0}
    mrr = 0.0
    for i, v in enumerate(verdicts):
        if v:
            mrr = 1.0 / (i + 1)
            break
    hit = 1.0 if any(verdicts) else 0.0
    dcg = sum((1.0 if v else 0.0) / math.log2(i + 2) for i, v in enumerate(verdicts))
    rel_count = sum(verdicts)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(rel_count, k)))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return {"mrr": round(mrr, 3), "hit_at_k": hit, "ndcg_at_k": round(ndcg, 3), "k": k}


def judge_context_precision(query: str, chunks: list[dict]) -> dict:
    """Judge each chunk for relevance → precision@k + ranking metrics."""
    if not chunks:
        return {
            "precision_at_k": 0.0,
            "relevant_count": 0,
            "total_count": 0,
            "verdicts": [],
            **_ranking_metrics_from_binary_verdicts([]),
        }
    verdicts: list[bool] = []
    for chunk in chunks:
        prompt = (
            f"QUESTION: {query}\n\n"
            f"RETRIEVED CHUNK (from {chunk.get('source', 'unknown')}):\n"
            f"{chunk.get('content', '')[:1500]}\n\n"
            "Is this chunk relevant?"
        )
        out = _judge_call(prompt, JUDGE_CONTEXT_PRECISION_SYSTEM)
        verdicts.append(bool(out.get("relevant", False)))
    rel = sum(verdicts)
    return {
        "precision_at_k": round(rel / len(chunks), 3),
        "relevant_count": rel,
        "total_count": len(chunks),
        "verdicts": verdicts,
        **_ranking_metrics_from_binary_verdicts(verdicts),
    }


# ── Structural checks (Mermaid + IRAC) ───────────────────────────────────────


def _extract_mermaid(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```mermaid\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def check_mermaid_validity(mermaid_code: str) -> dict:
    """Lightweight syntactic check — confirms diagram header + at least one edge."""
    code = (mermaid_code or "").strip()
    if not code:
        return {"score": 0.0, "reason": "empty"}
    header_ok = bool(re.match(r"(flowchart|graph|sequenceDiagram|stateDiagram)", code))
    edge_ok = bool(re.search(r"-->|---|==>|->>|-->>", code))
    score = (0.5 if header_ok else 0.0) + (0.5 if edge_ok else 0.0)
    return {
        "score": round(score, 2),
        "reason": "ok" if score == 1.0 else f"header={header_ok} edge={edge_ok}",
    }


def check_irac_compliance(irac_text: str, intent: str | None = None) -> dict:
    """Score 0–1 by IRAC heading presence. Only meaningful for ratio/summary intents."""
    if intent not in (None, "ratio", "summary"):
        return {"score": None, "reason": f"n/a for intent={intent}"}
    if not irac_text:
        return {"score": 0.0, "reason": "empty"}
    text = irac_text.lower()
    parts = {
        "issue": "issue" in text,
        "rule": "rule" in text,
        "application": "application" in text or "analysis" in text,
        "conclusion": "conclusion" in text,
    }
    score = sum(parts.values()) / 4
    return {"score": round(score, 2), "parts": parts}


# ── Agent runner with per-node latencies ─────────────────────────────────────


def _run_one_turn_with_timing(query: str, week: str | None, history: list[dict]) -> dict:
    """Single-turn run capturing per-stage wall time. Uses run_query so the
    production cache + escalation paths are exercised."""
    reset_token_usage()
    start = time.time()
    result = run_query(query=query, week_filter=week, chat_history=history)
    latency = time.time() - start
    usage = get_token_usage().summary()

    texts = result.get("retrieved_texts") or []
    slides = result.get("retrieved_slides") or []
    chunks = [*texts, *slides]
    context = "\n\n".join((d.get("content", "") or "") for d in chunks)

    return {
        "query": query,
        "answer": result.get("final_answer", ""),
        "intent": result.get("intent"),
        "node_trace": result.get("node_trace", []),
        "chunks": [
            {
                "source": d.get("source"),
                "doc_type": d.get("doc_type"),
                "week": d.get("week"),
                "content": d.get("content"),
            }
            for d in chunks
        ],
        "context": context,
        "latency_s": round(latency, 2),
        "source_diversity": len({d.get("source") for d in chunks if d.get("source")}),
        "mermaid_diagram": result.get("mermaid_diagram") or _extract_mermaid(result.get("final_answer", "")),
        "irac_analysis": result.get("irac_analysis", ""),
        "verification": result.get("verification_report"),
        "token_usage": usage,
        "escalated_to": result.get("escalated_to"),
    }


def run_agent_with_metrics(case: EvalCase) -> dict:
    """Replay all user turns; the metrics report on the FINAL turn."""
    prior: list[dict] = []
    last: dict = {}
    for turn in case["user_turns"]:
        last = _run_one_turn_with_timing(turn, week=None, history=prior)
        prior.extend(
            [
                {"role": "user", "content": turn},
                {"role": "assistant", "content": last["answer"]},
            ]
        )
    return last


# ── Retrieval-mode comparison (the headline plot) ────────────────────────────


RETRIEVAL_MODES: list[dict] = [
    {"name": "dense_only_no_rerank", "use_hybrid": False, "use_reranker": False},
    {"name": "dense_only_rerank",    "use_hybrid": False, "use_reranker": True},
    {"name": "hybrid_no_rerank",     "use_hybrid": True,  "use_reranker": False},
    {"name": "hybrid_rerank",        "use_hybrid": True,  "use_reranker": True},
]


def _retrieve_with_mode(query: str, use_hybrid: bool, use_reranker: bool) -> list[dict]:
    """Toggle USE_HYBRID_RETRIEVAL via monkeypatch-style attribute set + pass
    use_reranker through retrieve_all. Restores original at the end."""
    from src.agent import tools as tools_mod

    original = tools_mod.USE_HYBRID_RETRIEVAL
    tools_mod.USE_HYBRID_RETRIEVAL = bool(use_hybrid)
    try:
        texts, slides = retrieve_all(
            query, week=None, k_text=8, k_slides=4, use_reranker=use_reranker
        )
    finally:
        tools_mod.USE_HYBRID_RETRIEVAL = original
    return [*texts, *slides]


def compare_retrieval_modes(cases: list[EvalCase]) -> dict:
    """For each case + mode, judge retrieved chunks and compute precision/MRR/NDCG."""
    per_case: list[dict] = []
    for case in cases:
        q = _eval_case_last_turn(case)
        modes: dict[str, dict] = {}
        for mode in RETRIEVAL_MODES:
            chunks = _retrieve_with_mode(q, mode["use_hybrid"], mode["use_reranker"])
            metrics = judge_context_precision(q, chunks)
            modes[mode["name"]] = {
                "precision_at_k": metrics["precision_at_k"],
                "mrr": metrics["mrr"],
                "ndcg_at_k": metrics["ndcg_at_k"],
                "hit_at_k": metrics["hit_at_k"],
                "k": metrics["k"],
                "verdicts": metrics["verdicts"],
                "n_chunks": len(chunks),
            }
        per_case.append({"case_id": case["case_id"], "query": q, "modes": modes})

    avg: dict[str, dict[str, float]] = {}
    for mode in RETRIEVAL_MODES:
        name = mode["name"]
        rows = [c["modes"][name] for c in per_case]
        avg[name] = {
            "precision_at_k": round(sum(r["precision_at_k"] for r in rows) / len(rows), 3),
            "mrr": round(sum(r["mrr"] for r in rows) / len(rows), 3),
            "ndcg_at_k": round(sum(r["ndcg_at_k"] for r in rows) / len(rows), 3),
            "hit_at_k": round(sum(r["hit_at_k"] for r in rows) / len(rows), 3),
        }
    return {"per_case": per_case, "avg": avg}


# ── Full per-case agent evaluation ───────────────────────────────────────────


def _evaluate_case(case: EvalCase) -> dict:
    """Run the agent + score all metrics for one case."""
    agent = run_agent_with_metrics(case)
    judge_q = _judge_question_for_case(case)

    grounded = judge_groundedness(judge_q, agent["context"], agent["answer"])
    relevancy = judge_answer_relevancy(judge_q, agent["answer"])
    precision = judge_context_precision(judge_q, agent["chunks"])

    mermaid = check_mermaid_validity(agent["mermaid_diagram"])
    irac = check_irac_compliance(agent["irac_analysis"], intent=agent["intent"])

    return {
        "case_id": case["case_id"],
        "query_family": case["query_family"],
        "query": _eval_case_last_turn(case),
        "intent": agent["intent"],
        "node_trace": agent["node_trace"],
        "latency_s": agent["latency_s"],
        "source_diversity": agent["source_diversity"],
        "answer": agent["answer"],
        "groundedness": grounded,
        "answer_relevancy": relevancy,
        "context_precision": precision,
        "mermaid_validity": mermaid,
        "irac_compliance": irac,
        "verification": agent["verification"],
        "token_usage": agent["token_usage"],
        "escalated_to": agent["escalated_to"],
    }


def run_evaluation(
    output_dir: Path,
    *,
    skip_agent: bool = False,
    skip_retrieval_modes: bool = False,
    sample_size: int | None = None,
    seed: int = 4205,
) -> dict:
    """Run the full eval. Writes JSON + plots to ``output_dir``. Returns summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = list(EVAL_CASES)
    if sample_size is not None and sample_size < len(cases):
        random.seed(seed)
        cases = random.sample(cases, sample_size)

    agent_results: list[dict] = []
    if not skip_agent:
        log.info("Running agent evaluation (%d cases)", len(cases))
        for i, case in enumerate(cases, 1):
            log.info("[%d/%d] %s", i, len(cases), case["case_id"])
            try:
                agent_results.append(_evaluate_case(case))
            except Exception as e:  # one case shouldn't kill the suite
                log.exception("case %s failed: %s", case["case_id"], e)
                agent_results.append(
                    {"case_id": case["case_id"], "query_family": case["query_family"], "error": str(e)}
                )

    retrieval_results: dict = {}
    if not skip_retrieval_modes:
        log.info("Running retrieval-mode comparison (%d cases × 4 modes)", len(cases))
        retrieval_results = compare_retrieval_modes(cases)

    summary = _compute_summary(agent_results, retrieval_results)
    (output_dir / "eval_results.json").write_text(
        json.dumps(
            {"agent": agent_results, "retrieval_modes": retrieval_results, "summary": summary},
            indent=2,
            default=str,
        )
    )
    _generate_plots(agent_results, retrieval_results, output_dir)
    log.info("Wrote %s", output_dir / "eval_results.json")
    return summary


# ── Summary + plots ──────────────────────────────────────────────────────────


def _avg(rows: list[dict], path: tuple[str, ...]) -> float | None:
    """Mean of nested key path across rows; None when no row carries it."""
    vals = []
    for r in rows:
        cur = r
        for k in path:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        if isinstance(cur, (int, float)):
            vals.append(float(cur))
    return round(sum(vals) / len(vals), 3) if vals else None


def _compute_summary(agent_results: list[dict], retrieval_results: dict) -> dict:
    ok = [r for r in agent_results if "error" not in r]
    summary: dict = {
        "n_cases": len(agent_results),
        "n_successful": len(ok),
        "agent": {
            "avg_groundedness": _avg(ok, ("groundedness", "score")),
            "avg_answer_relevancy": _avg(ok, ("answer_relevancy", "score")),
            "avg_precision_at_k": _avg(ok, ("context_precision", "precision_at_k")),
            "avg_mrr": _avg(ok, ("context_precision", "mrr")),
            "avg_ndcg_at_k": _avg(ok, ("context_precision", "ndcg_at_k")),
            "avg_latency_s": _avg(ok, ("latency_s",)),
            "avg_source_diversity": _avg(ok, ("source_diversity",)),
            "avg_total_tokens": _avg(ok, ("token_usage", "total_tokens")),
        },
    }
    by_family: dict[str, dict] = {}
    for fam in sorted({r.get("query_family", "") for r in ok}):
        fam_rows = [r for r in ok if r.get("query_family") == fam]
        by_family[fam] = {
            "n": len(fam_rows),
            "avg_groundedness": _avg(fam_rows, ("groundedness", "score")),
            "avg_precision_at_k": _avg(fam_rows, ("context_precision", "precision_at_k")),
            "avg_mrr": _avg(fam_rows, ("context_precision", "mrr")),
        }
    summary["by_query_family"] = by_family

    if retrieval_results:
        summary["retrieval_modes"] = retrieval_results.get("avg", {})

    return summary


def _generate_plots(agent_results: list[dict], retrieval_results: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in agent_results if "error" not in r]

    # 1) Groundedness per case
    if ok:
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [r["case_id"] for r in ok]
        scores = [(r.get("groundedness") or {}).get("score", 0) for r in ok]
        ax.bar(labels, scores, color="#7a3b2e")
        ax.set_ylim(0, 5)
        ax.set_ylabel("Groundedness (1–5)")
        ax.set_title("Per-case groundedness")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / "groundedness.png", dpi=120)
        plt.close(fig)

    # 2) Context precision metrics (precision@k, MRR, NDCG)
    if ok:
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [r["case_id"] for r in ok]
        cp = [r.get("context_precision") or {} for r in ok]
        p_at_k = [c.get("precision_at_k", 0) for c in cp]
        mrr = [c.get("mrr", 0) for c in cp]
        ndcg = [c.get("ndcg_at_k", 0) for c in cp]
        x = list(range(len(labels)))
        w = 0.27
        ax.bar([i - w for i in x], p_at_k, w, label="P@k", color="#7a3b2e")
        ax.bar(x, mrr, w, label="MRR", color="#c08070")
        ax.bar([i + w for i in x], ndcg, w, label="NDCG@k", color="#e3b8a4")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Context precision metrics per case")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "context_precision.png", dpi=120)
        plt.close(fig)

    # 3) Latency per case
    if ok:
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [r["case_id"] for r in ok]
        lat = [r.get("latency_s", 0.0) for r in ok]
        ax.bar(labels, lat, color="#4a5060")
        ax.set_ylabel("Seconds")
        ax.set_title("Per-case end-to-end latency")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / "latency.png", dpi=120)
        plt.close(fig)

    # 4) Retrieval-mode comparison (the headline lift)
    avg = (retrieval_results or {}).get("avg") or {}
    if avg:
        mode_names = [m["name"] for m in RETRIEVAL_MODES]
        metrics = ["precision_at_k", "mrr", "ndcg_at_k"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = list(range(len(mode_names)))
        w = 0.27
        colors = ["#7a3b2e", "#c08070", "#e3b8a4"]
        for i, m in enumerate(metrics):
            vals = [avg.get(name, {}).get(m, 0.0) for name in mode_names]
            ax.bar([xi + (i - 1) * w for xi in x], vals, w, label=m, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(mode_names, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Retrieval mode comparison (mean over cases)")
        ax.set_ylabel("Score (higher is better)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "retrieval_modes.png", dpi=120)
        plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="LawGPT slim eval")
    parser.add_argument("--output-dir", default="eval_results", type=Path)
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--skip-retrieval-modes", action="store_true")
    parser.add_argument("--sample", type=int, default=None, help="Run a random subset of N cases")
    args = parser.parse_args()

    summary = run_evaluation(
        output_dir=args.output_dir,
        skip_agent=args.skip_agent,
        skip_retrieval_modes=args.skip_retrieval_modes,
        sample_size=args.sample,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
