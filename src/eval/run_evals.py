"""Evaluation and ablation framework for the Property Law Exam Assistant.

Compares the full agent pipeline against:
  1. A plain LLM baseline (no retrieval, no graph nodes)
  2. Mega-prompt ablation (retrieval + single consolidated LLM call replacing all nodes)

Eval cases are defined in ``EVAL_CASES``: each has a **query_family** tag
(`factual_retrieval`, `cross_modal_retrieval`, `analytical_synthesis`,
`conversational_followup`) so reports satisfy explicit per-family testing.
Multi-turn cases run the agent (and ablation) sequentially with memory; the
baseline always sees only the **final** user turn.

Metrics collected per run:
  - Groundedness score (LLM-as-a-judge: 1-5 scale)
  - Answer relevancy (LLM-as-a-judge: 1-5 scale)
  - Context precision@K, MRR, Hit@K, NDCG@K (LLM-judged, binary relevance)
  - Optional recall-vs-pool (``--retrieval-pool-eval``): expanded retrieve + judge
  - Source diversity (unique sources retrieved)
  - Total latency (seconds)
  - Per-node latency breakdown (for full agent)
  - Node trace (which nodes fired)
  - Cross-modal retrieval diagnostics on slide-targeted cases

Results are exported as JSON and plots as PNG for the report.

Usage:
    python -m src.eval.run_evals
    python -m src.eval.run_evals --output-dir eval_results
    python -m src.eval.run_evals --retrieval-pool-eval   # optional recall-vs-pool (slow)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.agent.chat_memory import prepare_chat_history_for_run
from src.agent.nodes import (
    _format_context,
    chronology_node,
    mega_prompt_node,
    ratio_extractor_node,
    retrieval_node,
    router_node,
    synthesis_node,
)
from src.agent.state import AgentState
from src.agent.tools import retrieve_all
from src.config import (
    BASELINE_MODEL,
    EVAL_RETRIEVAL_POOL_K_SLIDES,
    EVAL_RETRIEVAL_POOL_K_TEXT,
    JUDGE_CRITIQUE_MODEL,
    JUDGE_DRAFT_MODEL,
)
from src.llm import get_token_usage, llm_call, reset_token_usage

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Query families (assignment-style explicit coverage) ─────────────────────────
#
# Each eval case is tagged with exactly one family so reports can show per-family
# success/failure. Families mirror the coursework rubric: factual retrieval,
# cross-modal retrieval (text + indexed slide descriptions), analytical synthesis,
# and conversational follow-up (multi-turn with session memory).

QueryFamily = Literal[
    "factual_retrieval",
    "cross_modal_retrieval",
    "analytical_synthesis",
    "conversational_followup",
]


QUERY_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "factual_retrieval": (
        "Direct retrieval of stored knowledge: doctrine, tests, and definitions "
        "that should appear verbatim or paraphrased from a small number of chunks."
    ),
    "cross_modal_retrieval": (
        "Requires the lecture-slide channel: questions are phrased to target "
        "VLM-indexed slide descriptions alongside PDF readings (multimodal KB)."
    ),
    "analytical_synthesis": (
        "Multi-evidence answers: compare, relate, summarise across concepts, or "
        "reconstruct sequences using several retrieved sources."
    ),
    "conversational_followup": (
        "Multi-turn: first user message establishes topic; second message is a "
        "follow-up resolved using chat_history (baseline sees only the last turn)."
    ),
}


class EvalCase(TypedDict):
    """One evaluation scenario with explicit family tagging."""

    case_id: str
    query_family: QueryFamily
    rationale: str
    user_turns: list[str]


def _eval_case_last_turn(case: EvalCase) -> str:
    return case["user_turns"][-1]


EVAL_CASES: list[EvalCase] = [
    # ── Factual retrieval (direct KB lookup) ───────────────────────────────
    {
        "case_id": "fact_moran_test",
        "query_family": "factual_retrieval",
        "rationale": "Named case test — should retrieve specific ratio/test from readings.",
        "user_turns": [
            "What is the legal test from Buckinghamshire County Council v Moran?",
        ],
    },
    {
        "case_id": "fact_fee_simple",
        "query_family": "factual_retrieval",
        "rationale": "Core estate definition — single-concept factual answer.",
        "user_turns": ["What is a fee simple estate?"],
    },
    {
        "case_id": "fact_bailment_ratio",
        "query_family": "factual_retrieval",
        "rationale": "Ratio-style question tied to course materials on bailment.",
        "user_turns": ["What is the ratio decidendi regarding bailment at will?"],
    },
    {
        "case_id": "fact_perry_possessory",
        "query_family": "factual_retrieval",
        "rationale": "Named case holding from indexed readings.",
        "user_turns": [
            "What is the ratio in Perry v Clissold regarding possessory title?",
        ],
    },
    # ── Cross-modal retrieval (slide / lecture description channel) ─────────
    {
        "case_id": "xmodal_chattels_slides",
        "query_family": "cross_modal_retrieval",
        "rationale": "Forces reliance on VLM-described lecture slides for chattels possession.",
        "user_turns": [
            "According to the indexed lecture slide materials in the knowledge base, "
            "what two elements are required to establish possession of chattels?",
        ],
    },
    {
        "case_id": "xmodal_moukataff_slide",
        "query_family": "cross_modal_retrieval",
        "rationale": "Slide-anchored case (Moukataff) should appear in slide descriptions.",
        "user_turns": [
            "What does the indexed lecture content describe about Moukataff v BOAC "
            "and baggage or chattel possession?",
        ],
    },
    {
        "case_id": "xmodal_ap_slides",
        "query_family": "cross_modal_retrieval",
        "rationale": "Asks for emphasis from slide-indexed adverse possession teaching.",
        "user_turns": [
            "Based on the VLM-described lecture slides on adverse possession in the "
            "knowledge base, what sequence or steps do those materials emphasise?",
        ],
    },
    # ── Analytical / multi-hop synthesis ────────────────────────────────────
    {
        "case_id": "anal_ap_elements",
        "query_family": "analytical_synthesis",
        "rationale": "Must relate factual possession and animus across sources.",
        "user_turns": [
            "Explain the relationship between factual possession and animus possidendi "
            "for adverse possession, using evidence from multiple parts of the course "
            "readings and notes.",
        ],
    },
    {
        "case_id": "anal_lease_licence",
        "query_family": "analytical_synthesis",
        "rationale": "Compare two institutions from materials (distinction test).",
        "user_turns": [
            "What legal principle governs the distinction between a lease and a licence, "
            "and how does each concept apply in outline?",
        ],
    },
    {
        "case_id": "anal_chronology_ap",
        "query_family": "analytical_synthesis",
        "rationale": "Chronological reconstruction across multiple factual beats.",
        "user_turns": [
            "Show the chronological sequence of how adverse possession is established "
            "under the materials we indexed.",
        ],
    },
    {
        "case_id": "anal_torrens_framework",
        "query_family": "analytical_synthesis",
        "rationale": "Framework summary spanning readings/slides on Torrens.",
        "user_turns": [
            "Summarise the legal framework for land registration under the Torrens "
            "system as presented in the indexed course materials.",
        ],
    },
    {
        "case_id": "anal_taxonomy",
        "query_family": "analytical_synthesis",
        "rationale": "Taxonomy spans multiple categories (real vs personal property).",
        "user_turns": [
            "Summarise the key concepts of property law taxonomy including real and "
            "personal property as covered in our sources.",
        ],
    },
    # ── Conversational follow-up / personalised context (multi-turn) ───────
    {
        "case_id": "conv_ap_followup",
        "query_family": "conversational_followup",
        "rationale": "Second turn references 'that ratio' — needs session memory.",
        "user_turns": [
            "What is the ratio decidendi for adverse possession?",
            "Give one short exam tip tailored to that ratio for an open-book exam.",
        ],
    },
    {
        "case_id": "conv_torrens_followup",
        "query_family": "conversational_followup",
        "rationale": "Contrast question after Torrens explanation — coreference to prior answer.",
        "user_turns": [
            "Explain the Torrens system of land registration.",
            "In one paragraph, how does that differ from a deeds registry? Stay within "
            "our indexed course sources only.",
        ],
    },
    {
        "case_id": "conv_chattels_found",
        "query_family": "conversational_followup",
        "rationale": "Hypothetical extension after test — requires prior topic + retrieval.",
        "user_turns": [
            "What is the legal test for establishing possession of chattels?",
            "Does anything in our indexed materials suggest the answer changes if the "
            "chattel was found on the ground rather than handed over? Explain briefly.",
        ],
    },
]


# Backwards compatibility for imports / smoke tests
TEST_QUERIES: list[str] = [_eval_case_last_turn(c) for c in EVAL_CASES]


def _judge_question_for_case(case: EvalCase) -> str:
    """Question text passed to relevancy/groundedness judges (context for follow-ups)."""
    turns = case["user_turns"]
    if len(turns) == 1:
        return turns[0]
    prior = "\n".join(f"User (earlier): {t}" for t in turns[:-1])
    return (
        "Multi-turn evaluation. Score the final assistant answer in light of the "
        "whole thread.\n\n"
        f"{prior}\n\n"
        f"Latest user message (primary focus): {turns[-1]}"
    )


def _cross_modal_signals(result: dict) -> dict:
    """Diagnostics for cross-modal cases: did both text and slide chunks appear?"""
    texts = result.get("retrieved_text_count", 0)
    slides = result.get("retrieved_slide_count", 0)
    modalities: list[str] = []
    if texts > 0:
        modalities.append("text_pdf")
    if slides > 0:
        modalities.append("slide_vlm")
    return {
        "retrieved_text_chunks": texts,
        "retrieved_slide_chunks": slides,
        "modalities_present": modalities,
        "both_modalities_retrieved": texts > 0 and slides > 0,
    }


def _run_agent_turns_for_eval(case: EvalCase, week: str | None = None) -> dict:
    """Run the full agent for each user turn; return metrics dict for the final turn."""
    history: list[dict[str, str]] = []
    last: dict = {}
    for turn in case["user_turns"]:
        prior = prepare_chat_history_for_run(history) if history else None
        last = run_agent_with_metrics(turn, week=week, chat_history=prior)
        history.append({"role": "user", "content": turn})
        history.append({"role": "assistant", "content": last.get("answer", "")})
    last["conversation_turns"] = list(case["user_turns"])
    return last


def _run_ablation_turns_for_eval(case: EvalCase, week: str | None = None) -> dict:
    history: list[dict[str, str]] = []
    last: dict = {}
    for turn in case["user_turns"]:
        prior = prepare_chat_history_for_run(history) if history else None
        last = run_ablation_mega_prompt(turn, week=week, chat_history=prior)
        history.append({"role": "user", "content": turn})
        history.append({"role": "assistant", "content": last.get("answer", "")})
    last["conversation_turns"] = list(case["user_turns"])
    return last


# ── LLM Judge Helpers ─────────────────────────────────────────────────────────


def _judge_call(prompt: str, system: str, model: str | None = None) -> dict:
    """Make a judge LLM call and parse the JSON response."""
    response = llm_call(
        prompt,
        model=model or JUDGE_DRAFT_MODEL,
        system_instruction=system,
        temperature=0.0,
    )
    try:
        cleaned = re.sub(r"```json\s*|\s*```", "", response).strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, KeyError, ValueError):
        log.warning("Judge parse failed for prompt: %s...", prompt[:80])
        return {}


# ── Groundedness Judge ─────────────────────────────────────────────────────────

JUDGE_GROUNDEDNESS_SYSTEM = (
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


JUDGE_CRITIQUE_SYSTEM = (
    "You are a senior evaluator reviewing a junior judge's assessment of an AI answer. "
    "You will be given:\n"
    "1. The original question.\n"
    "2. The source material.\n"
    "3. The AI answer being evaluated.\n"
    "4. The junior judge's initial score and reasoning.\n\n"
    "Your job: critically review the junior judge's assessment. Either:\n"
    "- AGREE with the score if the reasoning is sound, or\n"
    "- OVERRIDE with a corrected score if the junior judge was too lenient or too harsh.\n\n"
    "Remember: paraphrasing and plain-English explanations are ALLOWED. "
    "Only penalise for invented factual claims (case names, dates, statutes, holdings).\n\n"
    "Respond with ONLY a JSON object: "
    "{\"score\": <int 1-5>, \"reasoning\": \"<your assessment>\", \"agreed_with_draft\": <bool>}"
)


def judge_groundedness(query: str, context: str, answer: str) -> dict:
    """Two-stage judge: draft model scores, critique model reviews and finalises."""
    base_prompt = (
        f"QUESTION: {query}\n\n"
        f"SOURCE MATERIAL:\n{context}\n\n"
        f"AI ANSWER:\n{answer}\n\n"
    )

    # Stage 1: Draft judgment
    draft = _judge_call(
        base_prompt + "Evaluate the groundedness of this answer.",
        JUDGE_GROUNDEDNESS_SYSTEM,
        model=JUDGE_DRAFT_MODEL,
    )
    draft_score = int(draft.get("score", 3))
    draft_reasoning = draft.get("reasoning", "No reasoning provided")

    # Stage 2: Critique judgment
    critique_prompt = (
        base_prompt
        + f"JUNIOR JUDGE ASSESSMENT:\n"
        f"  Score: {draft_score}/5\n"
        f"  Reasoning: {draft_reasoning}\n\n"
        "Review this assessment. Agree or override with a corrected score."
    )
    critique = _judge_call(
        critique_prompt,
        JUDGE_CRITIQUE_SYSTEM,
        model=JUDGE_CRITIQUE_MODEL,
    )

    final_score = int(critique.get("score", draft_score))
    agreed = critique.get("agreed_with_draft", True)

    return {
        "score": final_score,
        "reasoning": critique.get("reasoning", draft_reasoning),
        "draft_score": draft_score,
        "draft_reasoning": draft_reasoning,
        "agreed_with_draft": agreed,
    }


# ── Answer Relevancy Judge ────────────────────────────────────────────────────

JUDGE_RELEVANCY_SYSTEM = (
    "You are an impartial evaluator assessing whether an AI answer is relevant "
    "to the student's question.\n\n"
    "Score the answer on a scale of 1 to 5:\n"
    "  5 = Perfectly relevant: directly and completely addresses the question.\n"
    "  4 = Mostly relevant: addresses the core question with minor tangents.\n"
    "  3 = Partially relevant: some parts address the question, others drift.\n"
    "  2 = Mostly irrelevant: answer is largely off-topic.\n"
    "  1 = Irrelevant: answer does not address the question at all.\n\n"
    "Respond with ONLY a JSON object: {\"score\": <int>, \"reasoning\": \"<brief explanation>\"}"
)


def judge_answer_relevancy(query: str, answer: str) -> dict:
    """Use the LLM as a judge to score answer relevancy."""
    prompt = (
        f"QUESTION: {query}\n\n"
        f"AI ANSWER:\n{answer}\n\n"
        "Evaluate how relevant this answer is to the question."
    )
    result = _judge_call(prompt, JUDGE_RELEVANCY_SYSTEM)
    return {
        "score": int(result.get("score", 3)),
        "reasoning": result.get("reasoning", "Parse error"),
    }


# ── Context Precision Judge ───────────────────────────────────────────────────


def _ranking_metrics_from_binary_verdicts(verdicts: list[bool]) -> dict[str, float | int]:
    """MRR, Hit@K, and NDCG@K from LLM relevance verdicts in ranked order.

    Interprets ``verdicts[i]`` as relevance of the chunk at rank ``i+1`` (production
    retrieval order: texts then slides, as returned by the agent).
    """
    k = len(verdicts)
    if k == 0:
        return {"mrr": 0.0, "hit_at_k": 0.0, "ndcg_at_k": 0.0, "k": 0}

    mrr = 0.0
    for i, v in enumerate(verdicts):
        if v:
            mrr = 1.0 / (i + 1)
            break

    hit_at_k = 1.0 if any(verdicts) else 0.0

    dcg = sum((1.0 if v else 0.0) / math.log2(i + 2) for i, v in enumerate(verdicts))
    rel_count = int(sum(1 for v in verdicts if v))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(rel_count, k)))
    ndcg_at_k = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "mrr": round(mrr, 3),
        "hit_at_k": hit_at_k,
        "ndcg_at_k": round(ndcg_at_k, 3),
        "k": k,
    }


JUDGE_CONTEXT_PRECISION_SYSTEM = (
    "You are an impartial evaluator assessing retrieval quality. "
    "You will be given a question and a single retrieved text chunk.\n\n"
    "Determine whether this chunk is RELEVANT to answering the question.\n"
    "A chunk is relevant if it contains information that would help answer "
    "the question — even partially.\n\n"
    "Respond with ONLY a JSON object: {\"relevant\": true} or {\"relevant\": false}"
)


def judge_context_precision(query: str, chunks: list[dict]) -> dict:
    """Judge each retrieved chunk for relevance.

    Returns precision@K, MRR, Hit@K, NDCG@K (binary labels), and per-chunk verdicts.
    """
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
            "Is this chunk relevant to the question?"
        )
        result = _judge_call(prompt, JUDGE_CONTEXT_PRECISION_SYSTEM)
        verdicts.append(bool(result.get("relevant", False)))

    relevant_count = sum(verdicts)
    ranking = _ranking_metrics_from_binary_verdicts(verdicts)
    return {
        "precision_at_k": round(relevant_count / len(chunks), 2),
        "relevant_count": relevant_count,
        "total_count": len(chunks),
        "verdicts": verdicts,
        **ranking,
    }


def judge_context_precision_with_pool(
    query: str,
    *,
    week: str | None = None,
    k_text_prod: int = 8,
    k_slide_prod: int = 4,
    k_text_pool: int = EVAL_RETRIEVAL_POOL_K_TEXT,
    k_slide_pool: int = EVAL_RETRIEVAL_POOL_K_SLIDES,
) -> dict:
    """LLM-judge a **deep** retrieval pool, then score production slice + recall estimate.

    **Recall vs pool:** ``relevant_in_production / max(relevant_in_pool, 1)`` — treats
    all judge-positive chunks in the pool as a proxy for the relevant set. This is
    **not** corpus-wide recall (no human qrels); document that limitation in the report.

    Ranking metrics (MRR, NDCG, Hit) use **production** ordering (first *k_text_prod*
    text chunks + first *k_slide_prod* slides from the same ranked lists).
    """
    texts, slides = retrieve_all(
        query,
        week=week,
        k_text=k_text_pool,
        k_slides=k_slide_pool,
    )
    pool_chunks = list(texts) + list(slides)
    base = judge_context_precision(query, pool_chunks)
    verdicts: list[bool] = base["verdicts"]
    n_t = len(texts)
    n_s = len(slides)
    prod_indices = list(range(min(k_text_prod, n_t))) + list(
        range(n_t, n_t + min(k_slide_prod, n_s))
    )
    prod_verdicts = [verdicts[i] for i in prod_indices if i < len(verdicts)]
    rel_prod = sum(prod_verdicts)
    total_rel_pool = sum(verdicts)
    recall_vs_pool: float | None = None
    if total_rel_pool > 0:
        recall_vs_pool = round(rel_prod / total_rel_pool, 3)

    rank_prod = _ranking_metrics_from_binary_verdicts(prod_verdicts)

    return {
        "precision_at_k": round(rel_prod / max(len(prod_verdicts), 1), 2),
        "relevant_count": int(rel_prod),
        "total_count": len(prod_verdicts),
        "verdicts": prod_verdicts,
        **rank_prod,
        "recall_vs_pool": recall_vs_pool,
        "relevant_count_pool": int(total_rel_pool),
        "pool_total_chunks": len(pool_chunks),
        "pool_k_text": k_text_pool,
        "pool_k_slides": k_slide_pool,
        "eval_mode": "pool",
    }


# ── Mermaid.js Validity ───────────────────────────────────────────────────────


def _extract_mermaid_from_text(text: str) -> str:
    """Return the first fenced ```mermaid block body, or empty string."""
    if not text:
        return ""
    match = re.search(r"```mermaid\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _mermaid_for_structural_metric(dedicated: str, final_answer: str) -> str:
    """Prefer the chronology node's diagram; else any Mermaid embedded in the final answer."""
    d = (dedicated or "").strip()
    if d:
        return d
    return _extract_mermaid_from_text(final_answer or "")


def _irac_for_structural_metric(dedicated: str, final_answer: str) -> str:
    """Prefer the ratio extractor artefact; else the student-facing answer (ablation / baseline)."""
    d = (dedicated or "").strip()
    if d:
        return d
    return final_answer or ""


def check_mermaid_validity(mermaid_code: str) -> dict:
    """Check if the Mermaid diagram is structurally valid.

    Validates: non-empty, has graph declaration, has nodes, has edges,
    no reserved-word conflicts. Returns pass/fail with detail.
    """
    if not mermaid_code or not mermaid_code.strip():
        return {"valid": False, "score": 0.0, "reason": "No Mermaid diagram produced"}

    checks = {
        "has_graph_declaration": bool(re.search(r"graph\s+(TD|LR|TB|BT|RL)", mermaid_code)),
        "has_nodes": bool(re.search(r'\w+\[".+?"\]', mermaid_code) or re.search(r"\w+\[.+?\]", mermaid_code)),
        "has_edges": bool(re.search(r"-->", mermaid_code)),
        "min_3_nodes": len(re.findall(r'\w+\[', mermaid_code)) >= 3,
        "no_empty_labels": not bool(re.search(r'\[\s*\]', mermaid_code)),
    }

    passed = sum(checks.values())
    total = len(checks)
    score = round(passed / total, 2)
    failed = [k for k, v in checks.items() if not v]

    return {
        "valid": passed == total,
        "score": score,
        "checks": checks,
        "reason": f"Passed {passed}/{total}" + (f" — failed: {failed}" if failed else ""),
    }


# ── IRAC Structural Compliance ───────────────────────────────────────────────


def check_irac_compliance(irac_text: str) -> dict:
    """Check if the IRAC analysis contains all four required components.

    Looks for Issue, Rule, Application, and Conclusion sections.
    """
    if not irac_text or not irac_text.strip():
        return {"compliant": False, "score": 0.0, "reason": "No IRAC analysis produced", "components": {}}

    components = {
        "issue": bool(re.search(r"\b(issue|legal question)\b", irac_text, re.IGNORECASE)),
        "rule": bool(re.search(r"\b(rule|ratio decidendi|legal (rule|principle|test))\b", irac_text, re.IGNORECASE)),
        "application": bool(re.search(r"\bapplicat", irac_text, re.IGNORECASE)),
        "conclusion": bool(re.search(r"\bconclus", irac_text, re.IGNORECASE)),
    }

    passed = sum(components.values())
    total = len(components)
    score = round(passed / total, 2)

    return {
        "compliant": passed == total,
        "score": score,
        "components": components,
        "reason": f"IRAC {passed}/{total} components found",
    }


# ── Timed node runner ─────────────────────────────────────────────────────────


def _run_node(node_fn, state: AgentState) -> tuple[dict, float]:
    """Run a node function and return (result_dict, elapsed_seconds)."""
    start = time.time()
    result = node_fn(state)
    elapsed = time.time() - start
    return result, round(elapsed, 2)


# ── Full Agent Run with Per-Node Metrics ──────────────────────────────────────


def run_agent_with_metrics(
    query: str,
    week: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> dict:
    """Run the full agent pipeline with per-node latency tracking."""
    node_latencies: dict[str, float] = {}
    total_start = time.time()

    state: AgentState = {"query": query}
    if week:
        state["week_filter"] = week
    prepared = prepare_chat_history_for_run(chat_history)
    if prepared:
        state["chat_history"] = prepared

    result, lat = _run_node(router_node, state)
    state.update(result)
    node_latencies["router"] = lat

    result, lat = _run_node(retrieval_node, state)
    state.update(result)
    node_latencies["retrieval"] = lat

    intent = state.get("intent", "general")

    if intent in ("ratio", "summary"):
        result, lat = _run_node(ratio_extractor_node, state)
        state.update(result)
        node_latencies["ratio_extractor"] = lat

    if intent in ("chronology", "summary"):
        result, lat = _run_node(chronology_node, state)
        state.update(result)
        node_latencies["chronology"] = lat

    result, lat = _run_node(synthesis_node, state)
    state.update(result)
    node_latencies["synthesis"] = lat

    total_elapsed = round(time.time() - total_start, 2)

    texts = state.get("retrieved_texts", [])
    slides = state.get("retrieved_slides", [])
    all_sources = set(d["source"] for d in texts + slides)
    context = _format_context(state)

    return {
        "answer": state.get("final_answer", ""),
        "context": context,
        "retrieved_chunks": texts + slides,
        "latency_s": total_elapsed,
        "node_latencies": node_latencies,
        "node_trace": state.get("node_trace", []),
        "source_diversity": len(all_sources),
        "intent": intent,
        "ratio_decidendi": state.get("ratio_decidendi", ""),
        "irac_analysis": state.get("irac_analysis", ""),
        "mermaid_diagram": state.get("mermaid_diagram", ""),
        "retrieved_text_count": len(texts),
        "retrieved_slide_count": len(slides),
    }


# ── Baseline: Plain LLM ───────────────────────────────────────────────────────


def run_baseline(query: str) -> dict:
    """Send query directly to the LLM with no retrieval or graph."""
    start = time.time()
    answer = llm_call(
        query,
        model=BASELINE_MODEL,
        system_instruction=(
            "You are an Australian Property Law tutor. Answer the student's "
            "question as accurately as possible."
        ),
    )
    elapsed = time.time() - start
    return {
        "answer": answer,
        "latency_s": round(elapsed, 2),
        "node_trace": ["baseline_llm"],
        "source_diversity": 0,
    }


# ── Ablation: Mega-Prompt (single consolidated LLM call) ──────────────────────


def run_ablation_mega_prompt(
    query: str,
    week: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> dict:
    """Run retrieval then a single mega-prompt LLM call, bypassing the LangGraph router.

    Tests whether one consolidated prompt can match the quality of the full
    multi-node pipeline for IRAC extraction, Mermaid generation, and synthesis.
    """
    total_start = time.time()

    state: AgentState = {"query": query, "intent": "summary"}
    if week:
        state["week_filter"] = week
    prepared = prepare_chat_history_for_run(chat_history)
    if prepared:
        state["chat_history"] = prepared

    state.update(retrieval_node(state))
    state.update(mega_prompt_node(state))

    total_elapsed = round(time.time() - total_start, 2)

    texts = state.get("retrieved_texts", [])
    slides = state.get("retrieved_slides", [])
    all_sources = set(d["source"] for d in texts + slides)
    context = _format_context(state)

    return {
        "answer": state.get("final_answer", ""),
        "context": context,
        "retrieved_chunks": texts + slides,
        "latency_s": total_elapsed,
        "node_trace": state.get("node_trace", []),
        "source_diversity": len(all_sources),
        "intent": state.get("intent", "summary"),
        "ratio_decidendi": state.get("ratio_decidendi", ""),
        "irac_analysis": state.get("irac_analysis", ""),
        "mermaid_diagram": state.get("mermaid_diagram", ""),
    }


# ── Evaluation Runner ─────────────────────────────────────────────────────────


def run_evaluation(output_dir: Path, *, retrieval_pool_eval: bool = False) -> dict:
    """Run the full evaluation suite and return results.

    If ``retrieval_pool_eval`` is True, retrieval metrics use a larger retrieved pool
    (see ``judge_context_precision_with_pool``) — many more judge API calls.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for i, case in enumerate(EVAL_CASES):
        last_q = _eval_case_last_turn(case)
        judge_q = _judge_question_for_case(case)
        log.info(
            "═══ Case %d/%d [%s / %s] ═══",
            i + 1,
            len(EVAL_CASES),
            case["query_family"],
            case["case_id"],
        )

        # Token usage is tracked per eval case (multi-turn agent runs accumulate per case)
        reset_token_usage()
        log.info("Running: full agent")
        if len(case["user_turns"]) == 1:
            agent_result = run_agent_with_metrics(case["user_turns"][0])
        else:
            agent_result = _run_agent_turns_for_eval(case)
        agent_result["token_usage"] = get_token_usage().summary()

        reset_token_usage()
        log.info("Running: baseline LLM (final turn only)")
        baseline_result = run_baseline(last_q)
        baseline_result["token_usage"] = get_token_usage().summary()

        reset_token_usage()
        log.info("Running: ablation (mega-prompt, single LLM call)")
        if len(case["user_turns"]) == 1:
            ablation_result = run_ablation_mega_prompt(case["user_turns"][0])
        else:
            ablation_result = _run_ablation_turns_for_eval(case)
        ablation_result["token_usage"] = get_token_usage().summary()

        # 4. Extract context and chunks before popping
        agent_context = agent_result.pop("context")
        agent_chunks = agent_result.pop("retrieved_chunks")
        ablation_context = ablation_result.pop("context")
        ablation_chunks = ablation_result.pop("retrieved_chunks")

        # 5. Judge groundedness (judge_q includes multi-turn framing where needed)
        log.info("Judging groundedness...")
        agent_ground = judge_groundedness(judge_q, agent_context, agent_result["answer"])
        baseline_ground = judge_groundedness(judge_q, agent_context, baseline_result["answer"])
        ablation_ground = judge_groundedness(judge_q, ablation_context, ablation_result["answer"])

        # 6. Judge answer relevancy
        log.info("Judging answer relevancy...")
        agent_relevancy = judge_answer_relevancy(judge_q, agent_result["answer"])
        baseline_relevancy = judge_answer_relevancy(judge_q, baseline_result["answer"])
        ablation_relevancy = judge_answer_relevancy(judge_q, ablation_result["answer"])

        # 7. Retrieval metrics (relevance of chunks to the **final** user utterance)
        if retrieval_pool_eval:
            log.info("Judging retrieval (expanded pool — slow)...")
            agent_precision = judge_context_precision_with_pool(last_q, week=None)
            ablation_precision = judge_context_precision_with_pool(last_q, week=None)
        else:
            log.info("Judging context precision + ranking (production chunks)...")
            agent_precision = judge_context_precision(last_q, agent_chunks)
            ablation_precision = judge_context_precision(last_q, ablation_chunks)

        # 8. Mermaid validity and IRAC compliance (hypothesis-specific metrics)
        # Use dedicated node artefacts when present; otherwise score embedded content in
        # final_answer so ablation/baseline are comparable to the full agent.
        agent_mermaid = check_mermaid_validity(
            _mermaid_for_structural_metric(
                agent_result.get("mermaid_diagram", ""),
                agent_result.get("answer", ""),
            )
        )
        ablation_mermaid = check_mermaid_validity(
            _mermaid_for_structural_metric(
                ablation_result.get("mermaid_diagram", ""),
                ablation_result.get("answer", ""),
            )
        )
        baseline_mermaid = check_mermaid_validity(
            _mermaid_for_structural_metric("", baseline_result.get("answer", ""))
        )

        agent_irac = check_irac_compliance(
            _irac_for_structural_metric(
                agent_result.get("irac_analysis", ""),
                agent_result.get("answer", ""),
            )
        )
        ablation_irac = check_irac_compliance(
            _irac_for_structural_metric(
                ablation_result.get("irac_analysis", ""),
                ablation_result.get("answer", ""),
            )
        )
        baseline_irac = check_irac_compliance(
            _irac_for_structural_metric("", baseline_result.get("answer", ""))
        )

        log.info(
            "Structural — Mermaid: Agent=%.0f%% | Ablation=%.0f%% | Baseline=%.0f%% | "
            "IRAC: Agent=%.0f%% | Ablation=%.0f%% | Baseline=%.0f%%",
            agent_mermaid["score"] * 100,
            ablation_mermaid["score"] * 100,
            baseline_mermaid["score"] * 100,
            agent_irac["score"] * 100,
            ablation_irac["score"] * 100,
            baseline_irac["score"] * 100,
        )

        cross_modal_block: dict | None = None
        if case["query_family"] == "cross_modal_retrieval":
            cross_modal_block = _cross_modal_signals(agent_result)

        result_entry = {
            "query_family": case["query_family"],
            "case_id": case["case_id"],
            "case_rationale": case["rationale"],
            "query": last_q,
            "judge_question": judge_q,
            "user_turns": list(case["user_turns"]),
            "agent": {
                **agent_result,
                "groundedness": agent_ground,
                "answer_relevancy": agent_relevancy,
                "context_precision": agent_precision,
                "mermaid_validity": agent_mermaid,
                "irac_compliance": agent_irac,
            },
            "baseline": {
                **baseline_result,
                "groundedness": baseline_ground,
                "answer_relevancy": baseline_relevancy,
                "mermaid_validity": baseline_mermaid,
                "irac_compliance": baseline_irac,
            },
            "ablation_mega_prompt": {
                **ablation_result,
                "groundedness": ablation_ground,
                "answer_relevancy": ablation_relevancy,
                "context_precision": ablation_precision,
                "mermaid_validity": ablation_mermaid,
                "irac_compliance": ablation_irac,
            },
        }
        if cross_modal_block is not None:
            result_entry["cross_modal_signals"] = cross_modal_block

        all_results.append(result_entry)
        log.info(
            "Groundedness — Agent: %d, Baseline: %d, Ablated: %d | "
            "Relevancy — Agent: %d, Baseline: %d, Ablated: %d | "
            "P@K %.2f / MRR %.3f / NDCG %.3f (agent) | "
            "P@K %.2f / MRR %.3f / NDCG %.3f (ablation)",
            agent_ground["score"], baseline_ground["score"], ablation_ground["score"],
            agent_relevancy["score"], baseline_relevancy["score"], ablation_relevancy["score"],
            agent_precision["precision_at_k"],
            agent_precision.get("mrr", 0.0),
            agent_precision.get("ndcg_at_k", 0.0),
            ablation_precision["precision_at_k"],
            ablation_precision.get("mrr", 0.0),
            ablation_precision.get("ndcg_at_k", 0.0),
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

    # Generate failure analysis
    _generate_failure_analysis(all_results, output_dir)

    return summary


def _compute_summary(results: list[dict]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    configs = ["agent", "baseline", "ablation_mega_prompt"]
    summary = {}

    for config in configs:
        scores = [r[config]["groundedness"]["score"] for r in results]
        relevancies = [r[config]["answer_relevancy"]["score"] for r in results]
        latencies = [r[config]["latency_s"] for r in results]
        diversities = [r[config].get("source_diversity", 0) for r in results]

        mermaid_scores = [r[config]["mermaid_validity"]["score"] for r in results]
        irac_scores = [r[config]["irac_compliance"]["score"] for r in results]

        entry = {
            "avg_groundedness": round(sum(scores) / len(scores), 2),
            "avg_answer_relevancy": round(sum(relevancies) / len(relevancies), 2),
            "avg_mermaid_validity": round(sum(mermaid_scores) / len(mermaid_scores), 2),
            "avg_irac_compliance": round(sum(irac_scores) / len(irac_scores), 2),
            "avg_latency_s": round(sum(latencies) / len(latencies), 2),
            "avg_source_diversity": round(sum(diversities) / len(diversities), 2),
            "groundedness_scores": scores,
            "answer_relevancy_scores": relevancies,
            "mermaid_scores": mermaid_scores,
            "irac_scores": irac_scores,
            "latencies": latencies,
        }

        # Context precision (agent and ablation only)
        if "context_precision" in results[0].get(config, {}):
            precisions = [r[config]["context_precision"]["precision_at_k"] for r in results]
            entry["avg_context_precision"] = round(sum(precisions) / len(precisions), 2)
            entry["context_precisions"] = precisions
            cp0 = results[0][config]["context_precision"]
            if "mrr" in cp0:
                entry["avg_mrr"] = round(
                    sum(r[config]["context_precision"]["mrr"] for r in results) / len(results),
                    3,
                )
                entry["avg_hit_at_k"] = round(
                    sum(r[config]["context_precision"]["hit_at_k"] for r in results) / len(results),
                    3,
                )
                entry["avg_ndcg_at_k"] = round(
                    sum(r[config]["context_precision"]["ndcg_at_k"] for r in results) / len(results),
                    3,
                )
                entry["mrr_per_query"] = [r[config]["context_precision"]["mrr"] for r in results]
                entry["hit_at_k_per_query"] = [
                    r[config]["context_precision"]["hit_at_k"] for r in results
                ]
                entry["ndcg_at_k_per_query"] = [
                    r[config]["context_precision"]["ndcg_at_k"] for r in results
                ]
            recalls = [
                r[config]["context_precision"].get("recall_vs_pool")
                for r in results
                if r[config]["context_precision"].get("recall_vs_pool") is not None
            ]
            if recalls:
                entry["avg_recall_vs_pool"] = round(sum(recalls) / len(recalls), 3)

        # Token usage
        if "token_usage" in results[0].get(config, {}):
            total_input = sum(r[config]["token_usage"]["total_input_tokens"] for r in results)
            total_output = sum(r[config]["token_usage"]["total_output_tokens"] for r in results)
            entry["total_input_tokens"] = total_input
            entry["total_output_tokens"] = total_output
            entry["total_tokens"] = total_input + total_output
            entry["avg_tokens_per_query"] = round((total_input + total_output) / len(results))

        summary[config] = entry

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

    summary["by_query_family"] = _summary_by_query_family(results)

    return summary


def _summary_by_query_family(results: list[dict]) -> dict[str, dict]:
    """Aggregate agent metrics per ``query_family`` for the report and plots."""
    grouped: dict[str, list[dict]] = {}
    for r in results:
        fam = r.get("query_family", "unknown")
        grouped.setdefault(fam, []).append(r)

    out: dict[str, dict] = {}
    for fam, rows in grouped.items():
        n = len(rows)
        agent_g = sum(x["agent"]["groundedness"]["score"] for x in rows) / n
        agent_r = sum(x["agent"]["answer_relevancy"]["score"] for x in rows) / n
        precisions = [
            x["agent"]["context_precision"]["precision_at_k"]
            for x in rows
            if x["agent"].get("context_precision")
        ]
        entry: dict = {
            "description": QUERY_FAMILY_DESCRIPTIONS.get(fam, ""),
            "case_count": n,
            "case_ids": [x.get("case_id") for x in rows],
            "agent_avg_groundedness": round(agent_g, 2),
            "agent_avg_answer_relevancy": round(agent_r, 2),
        }
        if precisions:
            entry["agent_avg_context_precision"] = round(sum(precisions) / len(precisions), 2)
        if fam == "cross_modal_retrieval":
            hits = sum(
                1
                for x in rows
                if x.get("cross_modal_signals", {}).get("both_modalities_retrieved")
            )
            entry["fraction_cases_with_text_and_slides_retrieved"] = round(hits / n, 2)
        out[fam] = entry
    return out


# ── Failure Analysis ──────────────────────────────────────────────────────────


def _generate_failure_analysis(results: list[dict], output_dir: Path) -> None:
    """Analyse evaluation results and generate a failure analysis report."""
    lines: list[str] = [
        "# Failure Analysis Report",
        "",
        "Auto-generated from evaluation results. This report identifies queries where "
        "the system underperformed and categorises the root causes.",
        "",
    ]

    # ── 0. Explicit query family coverage (coursework requirement) ───────────
    lines.append("## 0. Query family coverage")
    lines.append("")
    lines.append(
        "Every eval case is assigned **exactly one** of four families. "
        "Use this table for success/failure discussion per family."
    )
    lines.append("")
    lines.append("| Family | Cases | Case IDs |")
    lines.append("|--------|-------|----------|")
    fam_order = [
        "factual_retrieval",
        "cross_modal_retrieval",
        "analytical_synthesis",
        "conversational_followup",
    ]
    by_fam: dict[str, list[dict]] = {k: [] for k in fam_order}
    for r in results:
        fam = r.get("query_family", "unknown")
        by_fam.setdefault(fam, []).append(r)
    for fam in fam_order + [k for k in sorted(by_fam.keys()) if k not in fam_order]:
        rows = by_fam.get(fam, [])
        if not rows:
            continue
        ids = ", ".join(str(x.get("case_id", "?")) for x in rows)
        lines.append(f"| **{fam}** | {len(rows)} | {ids} |")
    lines.append("")
    for fam in fam_order:
        desc = QUERY_FAMILY_DESCRIPTIONS.get(fam)
        if desc:
            lines.append(f"- **{fam}:** {desc}")
    lines.append("")

    for fam in fam_order:
        rows = by_fam.get(fam, [])
        if not rows:
            continue
        lines.append(f"### Family: `{fam}`")
        lines.append("")
        for r in rows:
            cid = r.get("case_id", "?")
            ag = r["agent"]["groundedness"]["score"]
            ar = r["agent"]["answer_relevancy"]["score"]
            turns = r.get("user_turns", [r.get("query", "")])
            turn_note = f"{len(turns)} turn(s)" if len(turns) > 1 else "single turn"
            lines.append(f"- **{cid}** ({turn_note}): groundedness **{ag}/5**, relevancy **{ar}/5**")
            rat = r.get("case_rationale")
            if rat:
                lines.append(f"  - *Rationale:* {rat}")
            xm = r.get("cross_modal_signals")
            if xm:
                lines.append(
                    f"  - *Retrieval:* text chunks={xm.get('retrieved_text_chunks', 0)}, "
                    f"slide chunks={xm.get('retrieved_slide_chunks', 0)}, "
                    f"both_modalities={xm.get('both_modalities_retrieved')}"
                )
        lines.append("")

    # ── 1. Low groundedness (agent scored < 4) ─────────────────────────────
    low_ground = [
        r for r in results
        if r["agent"]["groundedness"]["score"] < 4
    ]
    lines.append("## 1. Low Groundedness (Agent < 4/5)")
    lines.append("")
    if low_ground:
        for r in low_ground:
            score = r["agent"]["groundedness"]["score"]
            reasoning = r["agent"]["groundedness"]["reasoning"]
            fam = r.get("query_family", "N/A")
            cid = r.get("case_id", "N/A")
            lines.append(f"### [{fam} / {cid}] \"{r['query']}\"")
            lines.append(f"- **Score:** {score}/5")
            lines.append(f"- **Judge reasoning:** {reasoning}")
            lines.append(f"- **Intent:** {r['agent'].get('intent', 'N/A')}")
            lines.append(f"- **Sources retrieved:** {r['agent'].get('source_diversity', 0)} unique")
            lines.append("")
    else:
        lines.append("No queries scored below 4. All agent answers were well-grounded.")
        lines.append("")

    # ── 2. Agent scored lower than ablation ────────────────────────────────
    ablation_wins = [
        r for r in results
        if r["ablation_mega_prompt"]["groundedness"]["score"] > r["agent"]["groundedness"]["score"]
    ]
    lines.append("## 2. Ablation Outperformed Agent")
    lines.append("")
    lines.append(
        "Cases where the single mega-prompt outperformed the full agent on groundedness, "
        "suggesting the multi-node pipeline introduced unverifiable inferences on those queries."
    )
    lines.append("")
    if ablation_wins:
        for r in ablation_wins:
            agent_s = r["agent"]["groundedness"]["score"]
            ablation_s = r["ablation_mega_prompt"]["groundedness"]["score"]
            agent_reason = r["agent"]["groundedness"]["reasoning"]
            lines.append(f"### Q: \"{r['query']}\"")
            lines.append(f"- **Agent:** {agent_s}/5 | **Ablation:** {ablation_s}/5")
            lines.append(f"- **Agent judge reasoning:** {agent_reason}")
            lines.append("")
    else:
        lines.append("The agent matched or outperformed the ablation on all queries.")
        lines.append("")

    # ── 3. Low context precision (retrieval misses) ────────────────────────
    low_precision = [
        r for r in results
        if r["agent"].get("context_precision", {}).get("precision_at_k", 1.0) < 0.7
    ]
    lines.append("## 3. Low Context Precision (Retrieval < 70%)")
    lines.append("")
    lines.append(
        "Queries where fewer than 70% of retrieved chunks were judged relevant, "
        "indicating the retrieval missed the target topic."
    )
    lines.append("")
    if low_precision:
        for r in low_precision:
            prec = r["agent"]["context_precision"]
            lines.append(f"### Q: \"{r['query']}\"")
            lines.append(
                f"- **Precision@K:** {prec['precision_at_k']:.0%} "
                f"({prec['relevant_count']}/{prec['total_count']} relevant)"
            )
            lines.append(f"- **Verdicts:** {prec['verdicts']}")
            lines.append("")
    else:
        lines.append("All queries achieved >= 70% context precision.")
        lines.append("")

    # ── 4. Low answer relevancy ────────────────────────────────────────────
    low_relevancy = [
        r for r in results
        if r["agent"]["answer_relevancy"]["score"] < 4
    ]
    lines.append("## 4. Low Answer Relevancy (Agent < 4/5)")
    lines.append("")
    if low_relevancy:
        for r in low_relevancy:
            score = r["agent"]["answer_relevancy"]["score"]
            reasoning = r["agent"]["answer_relevancy"]["reasoning"]
            lines.append(f"### Q: \"{r['query']}\"")
            lines.append(f"- **Score:** {score}/5")
            lines.append(f"- **Judge reasoning:** {reasoning}")
            lines.append("")
    else:
        lines.append("All agent answers scored 4+ on relevancy.")
        lines.append("")

    # ── 5. Baseline outperformed agent ─────────────────────────────────────
    baseline_wins = [
        r for r in results
        if r["baseline"]["groundedness"]["score"] > r["agent"]["groundedness"]["score"]
    ]
    lines.append("## 5. Baseline Outperformed Agent")
    lines.append("")
    if baseline_wins:
        lines.append(
            "**Critical finding:** These queries show the plain LLM scored higher "
            "than the full agent, suggesting retrieval may have introduced noise."
        )
        lines.append("")
        for r in baseline_wins:
            agent_s = r["agent"]["groundedness"]["score"]
            baseline_s = r["baseline"]["groundedness"]["score"]
            lines.append(f"### Q: \"{r['query']}\"")
            lines.append(f"- **Agent:** {agent_s}/5 | **Baseline:** {baseline_s}/5")
            lines.append(f"- **Agent reasoning:** {r['agent']['groundedness']['reasoning']}")
            lines.append(f"- **Baseline reasoning:** {r['baseline']['groundedness']['reasoning']}")
            lines.append("")
    else:
        lines.append("The agent matched or outperformed the baseline on all queries.")
        lines.append("")

    # ── 6. Summary statistics ──────────────────────────────────────────────
    total = len(results)
    lines.append("## 6. Summary")
    lines.append("")
    lines.append(f"| Category | Count | Percentage |")
    lines.append(f"|----------|-------|------------|")
    lines.append(f"| Total queries | {total} | 100% |")
    lines.append(f"| Low groundedness (<4) | {len(low_ground)} | {len(low_ground)/total*100:.0f}% |")
    lines.append(f"| Ablation outperformed agent | {len(ablation_wins)} | {len(ablation_wins)/total*100:.0f}% |")
    lines.append(f"| Low context precision (<70%) | {len(low_precision)} | {len(low_precision)/total*100:.0f}% |")
    lines.append(f"| Low answer relevancy (<4) | {len(low_relevancy)} | {len(low_relevancy)/total*100:.0f}% |")
    lines.append(f"| Baseline outperformed agent | {len(baseline_wins)} | {len(baseline_wins)/total*100:.0f}% |")
    lines.append("")

    report_path = output_dir / "failure_analysis.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    log.info("Failure analysis saved to %s", report_path)


# ── Plot Generation ────────────────────────────────────────────────────────────


def _generate_plots(results: list[dict], summary: dict, output_dir: Path) -> None:
    """Generate comparison plots and save as PNG."""
    configs = ["agent", "baseline", "ablation_mega_prompt"]
    labels = ["Full Agent", "Plain LLM\n(Baseline)", "Mega-Prompt\n(Ablation)"]

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

    # 2. Answer relevancy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    relevancies = [summary[c]["avg_answer_relevancy"] for c in configs]
    bars = ax.bar(labels, relevancies, color=["#2563eb", "#9ca3af", "#f59e0b"], edgecolor="black")
    ax.set_ylabel("Average Answer Relevancy (1-5)")
    ax.set_title("Answer Relevancy: Full Agent vs Baseline vs Ablation")
    ax.set_ylim(0, 5.5)
    for bar, rel in zip(bars, relevancies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{rel:.1f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "answer_relevancy_comparison.png", dpi=150)
    plt.close()

    # 3. Context precision comparison (agent and ablation only)
    retrieval_configs = ["agent", "ablation_mega_prompt"]
    retrieval_labels = ["Full Agent\n(MMR)", "Mega-Prompt\n(Ablation)"]
    precisions = [summary[c].get("avg_context_precision", 0) for c in retrieval_configs]
    if any(p > 0 for p in precisions):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(retrieval_labels, precisions, color=["#2563eb", "#f59e0b"], edgecolor="black")
        ax.set_ylabel("Average Context Precision@K")
        ax.set_title("Retrieval Quality: Context Precision@K")
        ax.set_ylim(0, 1.1)
        for bar, prec in zip(bars, precisions):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{prec:.2f}", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "context_precision.png", dpi=150)
        plt.close()

        # 3b. Ranking metrics (agent vs ablation)
        if all("avg_mrr" in summary[c] for c in retrieval_configs):
            fig, ax = plt.subplots(figsize=(9, 5))
            metrics_rank = ["MRR", "Hit@K", "NDCG@K"]
            xw = range(len(metrics_rank))
            width = 0.35
            agent_vals = [
                summary["agent"]["avg_mrr"],
                summary["agent"]["avg_hit_at_k"],
                summary["agent"]["avg_ndcg_at_k"],
            ]
            ab_vals = [
                summary["ablation_mega_prompt"]["avg_mrr"],
                summary["ablation_mega_prompt"]["avg_hit_at_k"],
                summary["ablation_mega_prompt"]["avg_ndcg_at_k"],
            ]
            ax.bar([i - width / 2 for i in xw], agent_vals, width, label="Full Agent", color="#2563eb", edgecolor="black")
            ax.bar([i + width / 2 for i in xw], ab_vals, width, label="Ablation", color="#f59e0b", edgecolor="black")
            ax.set_xticks(list(xw))
            ax.set_xticklabels(metrics_rank)
            ax.set_ylabel("Score (0–1)")
            ax.set_title("Retrieval ranking metrics (LLM-judged binary relevance)")
            ax.set_ylim(0, 1.1)
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "retrieval_ranking_metrics.png", dpi=150)
            plt.close()

    # 4. Latency comparison
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

    # 5. Per-query groundedness breakdown
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

    # 6. Source diversity comparison
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

    # 7. Per-node latency breakdown
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

    # 8. Radar / multi-metric summary
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_names = ["Groundedness", "Answer\nRelevancy", "Source\nDiversity\n(norm)"]
    for config, label, color in zip(configs, ["Full Agent", "Baseline", "Ablation"],
                                     ["#2563eb", "#9ca3af", "#f59e0b"]):
        vals = [
            summary[config]["avg_groundedness"],
            summary[config]["avg_answer_relevancy"],
            min(summary[config]["avg_source_diversity"] / 8.0 * 5, 5.0),
        ]
        if "avg_context_precision" in summary[config]:
            metric_names_ext = metric_names + ["Context\nPrecision"]
            vals.append(summary[config]["avg_context_precision"] * 5)
        else:
            metric_names_ext = metric_names
        x_pos = range(len(vals))
        ax.plot(list(x_pos), vals, "o-", label=label, color=color, linewidth=2, markersize=8)

    ax.set_xticks(range(len(metric_names_ext)))
    ax.set_xticklabels(metric_names_ext)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Score (normalised to 0-5)")
    ax.set_title("Multi-Metric Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "multi_metric_comparison.png", dpi=150)
    plt.close()

    # 9. Token usage comparison (stacked: input vs output)
    if all("total_tokens" in summary[c] for c in configs):
        fig, ax = plt.subplots(figsize=(8, 5))
        input_tokens = [summary[c]["total_input_tokens"] for c in configs]
        output_tokens = [summary[c]["total_output_tokens"] for c in configs]
        bars1 = ax.bar(labels, input_tokens, label="Input Tokens", color="#2563eb", edgecolor="black")
        bars2 = ax.bar(labels, output_tokens, bottom=input_tokens, label="Output Tokens", color="#60a5fa", edgecolor="black")
        ax.set_ylabel("Total Tokens (across all queries)")
        ax.set_title("Token Usage: Full Agent vs Baseline vs Ablation")
        ax.legend()
        for bar, inp, out in zip(bars1, input_tokens, output_tokens):
            total = inp + out
            ax.text(bar.get_x() + bar.get_width() / 2, total + max(input_tokens) * 0.02,
                    f"{total:,}", ha="center", fontweight="bold", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / "token_usage_comparison.png", dpi=150)
        plt.close()

    # 10. Structural metrics: Mermaid validity + IRAC compliance
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(configs))
    width = 0.35
    mermaid_vals = [summary[c]["avg_mermaid_validity"] * 100 for c in configs]
    irac_vals = [summary[c]["avg_irac_compliance"] * 100 for c in configs]

    bars1 = ax.bar([xi - width / 2 for xi in x], mermaid_vals, width, label="Mermaid Validity", color="#2563eb", edgecolor="black")
    bars2 = ax.bar([xi + width / 2 for xi in x], irac_vals, width, label="IRAC Compliance", color="#f59e0b", edgecolor="black")
    ax.set_ylabel("Score (%)")
    ax.set_title("Structural Output Quality: Mermaid Validity & IRAC Compliance")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend()
    for bar, val in zip(list(bars1) + list(bars2), mermaid_vals + irac_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f}%", ha="center", fontweight="bold", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 2,
                    "0%", ha="center", fontweight="bold", fontsize=9, color="red")
    plt.tight_layout()
    plt.savefig(output_dir / "structural_metrics.png", dpi=150)
    plt.close()

    # 11. Agent groundedness by query family
    bf = summary.get("by_query_family")
    if bf:
        fam_keys = list(bf.keys())
        fig, ax = plt.subplots(figsize=(9, 5))
        scores = [bf[k]["agent_avg_groundedness"] for k in fam_keys]
        colours = ["#2563eb", "#7c3aed", "#059669", "#ea580c"]
        bar_cols = [colours[i % len(colours)] for i in range(len(fam_keys))]
        bars = ax.bar(fam_keys, scores, color=bar_cols, edgecolor="black")
        ax.set_ylabel("Average groundedness (1–5)")
        ax.set_title("Full agent: groundedness by query family")
        ax.set_ylim(0, 5.5)
        ax.tick_params(axis="x", rotation=15)
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                f"{score:.2f}",
                ha="center",
                fontweight="bold",
                fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(output_dir / "groundedness_by_query_family.png", dpi=150)
        plt.close()

    log.info("Generated evaluation plots")


# ── CLI Entry Point ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation and ablation suite")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save results and plots",
    )
    parser.add_argument(
        "--retrieval-pool-eval",
        action="store_true",
        help=(
            "Re-run retrieval with a larger k (text+slides), judge all pool chunks, "
            "and add recall-vs-pool. Much slower and more judge API calls."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log.info("Starting evaluation suite — results will be saved to %s/", output_dir)

    summary = run_evaluation(output_dir, retrieval_pool_eval=args.retrieval_pool_eval)

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    by_fam = summary.pop("by_query_family", {})
    for config in ("agent", "baseline", "ablation_mega_prompt"):
        metrics = summary[config]
        print(f"\n  {config}:")
        print(f"    Avg Groundedness:       {metrics['avg_groundedness']}/5.0")
        print(f"    Avg Answer Relevancy:   {metrics['avg_answer_relevancy']}/5.0")
        print(f"    Avg Mermaid Validity:   {metrics['avg_mermaid_validity']*100:.0f}%")
        print(f"    Avg IRAC Compliance:    {metrics['avg_irac_compliance']*100:.0f}%")
        if "avg_context_precision" in metrics:
            print(f"    Avg Context Precision:  {metrics['avg_context_precision']}")
        if config in ("agent", "ablation_mega_prompt") and "avg_mrr" in metrics:
            print(f"    Avg MRR / Hit@K / NDCG: {metrics['avg_mrr']} / {metrics['avg_hit_at_k']} / {metrics['avg_ndcg_at_k']}")
        if config in ("agent", "ablation_mega_prompt") and metrics.get("avg_recall_vs_pool") is not None:
            print(f"    Avg recall vs pool:     {metrics['avg_recall_vs_pool']}")
        print(f"    Avg Latency:            {metrics['avg_latency_s']}s")
        print(f"    Avg Source Diversity:    {metrics['avg_source_diversity']}")
        if "total_tokens" in metrics:
            print(f"    Total Tokens:           {metrics['total_tokens']:,}")
            print(f"    Avg Tokens/Query:       {metrics['avg_tokens_per_query']:,}")
        if "avg_node_latencies" in metrics:
            print(f"    Node Latencies:         {metrics['avg_node_latencies']}")

    if by_fam:
        print("\n  By query family (full agent):")
        for fam, block in by_fam.items():
            print(f"    {fam}: groundedness {block['agent_avg_groundedness']}/5, "
                  f"relevancy {block['agent_avg_answer_relevancy']}/5, "
                  f"n={block['case_count']}")
            if block.get("fraction_cases_with_text_and_slides_retrieved") is not None:
                print(
                    "      cross-modal both text+slides retrieved: "
                    f"{block['fraction_cases_with_text_and_slides_retrieved']:.0%} of cases"
                )


if __name__ == "__main__":
    main()
