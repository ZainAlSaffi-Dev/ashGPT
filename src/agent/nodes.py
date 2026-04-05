"""Cognitive nodes for the Property Law LangGraph pipeline.

Each node is a pure function: AgentState → partial AgentState update.
Nodes are composed into a graph in Phase 4 (graph.py).

Node pipeline:
    router_node → retrieval_node → [ratio_extractor_node | chronology_node] → synthesis_node
"""

from __future__ import annotations

import json
import logging
import re

from dotenv import load_dotenv

from src.agent.state import AgentState
from src.agent.tools import retrieve_all
from src.config import (
    CHRONOLOGY_MODEL,
    RATIO_EXTRACTOR_MODEL,
    ROUTER_MODEL,
    SYNTHESIS_MODEL,
)
from src.llm import llm_call

load_dotenv()

log = logging.getLogger(__name__)


def _append_trace(state: AgentState, node_name: str) -> list[str]:
    """Return a new node_trace list with the current node appended."""
    trace = list(state.get("node_trace", []))
    trace.append(node_name)
    return trace


# ── Helper: format retrieved docs for prompts ─────────────────────────────────


def _format_context(state: AgentState) -> str:
    """Compile retrieved texts and slides into a single context block."""
    sections: list[str] = []

    texts = state.get("retrieved_texts", [])
    if texts:
        sections.append("=== RETRIEVED TEXT SOURCES ===")
        for i, doc in enumerate(texts, 1):
            sections.append(
                f"\n--- Source {i}: {doc['source']} "
                f"[{doc['week']}, {doc['doc_type']}] ---\n"
                f"{doc['content']}"
            )

    slides = state.get("retrieved_slides", [])
    if slides:
        sections.append("\n=== RETRIEVED LECTURE SLIDES ===")
        for i, doc in enumerate(slides, 1):
            sections.append(
                f"\n--- Slide {i}: {doc['source']} "
                f"[{doc['week']}] ---\n"
                f"{doc['content']}"
            )

    return "\n".join(sections) if sections else "(No context retrieved.)"


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: ROUTER
# ══════════════════════════════════════════════════════════════════════════════


ROUTER_SYSTEM = (
    "You are a query classifier for a Property Law study assistant. "
    "Analyse the user's question and return ONLY a JSON object with two keys:\n"
    '  "intent": one of "ratio", "chronology", "summary", "general"\n'
    '  "week_filter": a string like "week_3" if the user mentions a specific '
    "week, or null if not.\n\n"
    "Intent definitions:\n"
    '- "ratio": user wants the ratio decidendi, binding legal rule, or IRAC analysis of a case.\n'
    '- "chronology": user wants a timeline, flowchart, chain of title, or sequence of events.\n'
    '- "summary": user wants a general case summary or explanation of a topic.\n'
    '- "general": anything else (greetings, meta-questions, off-topic).\n\n'
    "Respond with ONLY the JSON object, no markdown fences."
)


def router_node(state: AgentState) -> dict:
    """Classify the user's intent and extract any week filter.

    Reads:  query
    Writes: intent, week_filter, node_trace
    """
    query = state["query"]
    log.info("RouterNode: classifying query")

    raw = llm_call(query, model=ROUTER_MODEL, system_instruction=ROUTER_SYSTEM)

    try:
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
        parsed = json.loads(cleaned)
        intent = parsed.get("intent", "general")
        week_filter = parsed.get("week_filter")
    except (json.JSONDecodeError, AttributeError):
        log.warning("Router failed to parse LLM output, defaulting to 'summary'")
        intent = "summary"
        week_filter = None

    valid_intents = {"ratio", "chronology", "summary", "general"}
    if intent not in valid_intents:
        intent = "summary"

    log.info("RouterNode: intent=%s, week_filter=%s", intent, week_filter)
    return {
        "intent": intent,
        "week_filter": week_filter,
        "node_trace": _append_trace(state, "router"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════


def retrieval_node(state: AgentState) -> dict:
    """Query ChromaDB for relevant text chunks and slide descriptions.

    Reads:  query, week_filter
    Writes: retrieved_texts, retrieved_slides, node_trace
    """
    query = state["query"]
    week = state.get("week_filter")
    log.info("RetrievalNode: searching KB (week=%s)", week)

    texts, slides = retrieve_all(query, week=week, k_text=8, k_slides=4)

    log.info(
        "RetrievalNode: retrieved %d text chunks, %d slides",
        len(texts),
        len(slides),
    )
    return {
        "retrieved_texts": texts,
        "retrieved_slides": slides,
        "node_trace": _append_trace(state, "retrieval"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3: RATIO EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════


RATIO_SYSTEM = (
    "You are an expert Australian Property Law tutor. Your task is to analyse "
    "the provided source material and extract the ratio decidendi — the binding "
    "legal principle that the court applied to reach its decision.\n\n"
    "You MUST structure your response as an IRAC analysis:\n"
    "1. **Issue**: State the legal question the court had to decide.\n"
    "2. **Rule**: State the legal rule or principle (this IS the ratio decidendi).\n"
    "3. **Application**: Explain how the court applied the rule to the facts.\n"
    "4. **Conclusion**: State the court's decision.\n\n"
    "GROUNDING RULES:\n"
    "- FACTS must come from the sources: case names, dates, statutory references, "
    "holdings, and specific legal tests MUST appear in the source material. "
    "Do NOT invent cases, statutes, or factual claims.\n"
    "- EXPLANATIONS are encouraged: you may paraphrase, simplify, and explain "
    "legal concepts in plain English to help the student understand. This is "
    "your role as a tutor.\n"
    "- If you cite a case or statute by name, it MUST appear in the sources.\n"
    "- If the source material is insufficient, say so explicitly rather than "
    "filling gaps from your own knowledge."
)


def ratio_extractor_node(state: AgentState) -> dict:
    """Extract the ratio decidendi and produce an IRAC analysis.

    Reads:  query, retrieved_texts, retrieved_slides
    Writes: ratio_decidendi, irac_analysis, node_trace
    """
    query = state["query"]
    context = _format_context(state)

    prompt = (
        f"USER QUESTION: {query}\n\n"
        f"SOURCE MATERIAL:\n{context}\n\n"
        "Based on the source material above, provide:\n"
        "1. The RATIO DECIDENDI (the exact binding legal rule) in one clear sentence.\n"
        "2. A full IRAC analysis."
    )

    log.info("RatioExtractorNode: generating IRAC analysis")
    response = llm_call(prompt, model=RATIO_EXTRACTOR_MODEL, system_instruction=RATIO_SYSTEM)

    ratio_line = ""
    for line in response.split("\n"):
        if "ratio" in line.lower() and "decidendi" in line.lower():
            ratio_line = line.strip().lstrip("#*- ")
            break
    if not ratio_line:
        ratio_line = response.split("\n")[0].strip()

    return {
        "ratio_decidendi": ratio_line,
        "irac_analysis": response,
        "node_trace": _append_trace(state, "ratio_extractor"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4: CHRONOLOGY GENERATOR
# ══════════════════════════════════════════════════════════════════════════════


CHRONOLOGY_SYSTEM = (
    "You are an expert Australian Property Law tutor specialising in creating "
    "precise chronological flowcharts of property transactions and legal events.\n\n"
    "Your task:\n"
    "1. Extract the chronological sequence of events from the source material.\n"
    "2. Output a valid Mermaid.js flowchart using `graph TD` syntax.\n"
    "3. After the Mermaid block, provide a brief plain-English summary of the timeline.\n\n"
    "STRICT Mermaid formatting rules for readability:\n"
    "- Use `graph TD` (top-down) direction.\n"
    "- Node IDs must be camelCase (no spaces): e.g. `ownerDies`, `titlePasses`.\n"
    "- KEEP NODE LABELS SHORT — maximum 8-10 words per node. Put dates at the start.\n"
    "  GOOD: `e1[\"1881: Clissold enters possession\"]`\n"
    "  BAD:  `e1[\"1881: Frederick Clissold enters into possession of the land and begins to enclose it\"]`\n"
    "- Put extra detail on the EDGE labels, not the nodes:\n"
    "  GOOD: `e1 -->|\"encloses land, pays rates\"| e2`\n"
    "- Use round brackets for decision nodes: `d1{\"Owner consents?\"}`\n"
    "- Use subgraphs to group related phases: `subgraph phase1 [\"Phase 1: Possession\"]`\n"
    "- Wrap ALL labels in double quotes.\n"
    "- Do NOT use the keyword `end` as a node ID.\n"
    "- Aim for 5-12 nodes total. Fewer is better — each node is one key event.\n\n"
    "Example of a GOOD diagram:\n"
    "```\n"
    "graph TD\n"
    "    subgraph possession [\"Establishing Possession\"]\n"
    "        e1[\"1881: Clissold takes possession\"] -->|\"encloses land, pays rates\"| e2[\"1881-1891: Continuous occupation\"]\n"
    "    end\n"
    "    subgraph limitation [\"Limitation Period\"]\n"
    "        e2 -->|\"12 years pass\"| e3[\"1893: Limitation period expires\"]\n"
    "    end\n"
    "    e3 -->|\"owner's title extinguished\"| e4[\"Title vests in possessor\"]\n"
    "```\n\n"
    "Format your full response as:\n"
    "```mermaid\n<your flowchart>\n```\n\n"
    "**Timeline Summary:**\n<plain-English summary>"
)


def chronology_node(state: AgentState) -> dict:
    """Generate a Mermaid.js chronological flowchart from retrieved facts.

    Reads:  query, retrieved_texts, retrieved_slides
    Writes: mermaid_diagram, chronology_summary, node_trace
    """
    query = state["query"]
    context = _format_context(state)

    prompt = (
        f"USER QUESTION: {query}\n\n"
        f"SOURCE MATERIAL:\n{context}\n\n"
        "Extract the chronological sequence of events and generate a Mermaid.js "
        "flowchart showing the chain of title or timeline of legal events.\n\n"
        "IMPORTANT: Only include events, dates, and parties that appear in the "
        "source material above. Do not invent events. You may label diagram "
        "edges with brief plain-English explanations of what happened."
    )

    log.info("ChronologyNode: generating Mermaid.js diagram")
    response = llm_call(prompt, model=CHRONOLOGY_MODEL, system_instruction=CHRONOLOGY_SYSTEM)

    mermaid_match = re.search(r"```mermaid\s*\n(.*?)```", response, re.DOTALL)
    mermaid_diagram = mermaid_match.group(1).strip() if mermaid_match else ""

    summary_match = re.search(
        r"\*\*Timeline Summary[:\s]*\*\*(.*)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    chronology_summary = summary_match.group(1).strip() if summary_match else ""

    if not mermaid_diagram:
        log.warning("ChronologyNode: no Mermaid block found in LLM response")

    return {
        "mermaid_diagram": mermaid_diagram,
        "chronology_summary": chronology_summary,
        "node_trace": _append_trace(state, "chronology"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5: SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════════


SYNTHESIS_SYSTEM = (
    "You are an expert Australian Property Law tutor providing a final answer "
    "to a student's question.\n\n"
    "You will be given:\n"
    "- The student's original question.\n"
    "- Retrieved source material from readings and lecture slides.\n"
    "- Optionally, an IRAC analysis with the ratio decidendi.\n"
    "- Optionally, a chronological timeline summary.\n\n"
    "GROUNDING RULES:\n"
    "- FACTS must come from the sources: all case names, dates, statutory "
    "references, legal tests, and holdings you mention MUST appear in the "
    "provided source material or upstream IRAC/chronology analysis. Do NOT "
    "invent or import cases, statutes, or facts from your own knowledge.\n"
    "- EXPLANATIONS are encouraged: paraphrase, simplify, and connect ideas "
    "in plain English to help the student understand. You are a tutor, not "
    "a copy machine.\n"
    "- Use inline citations where you rely on specific sources, e.g. "
    "(Source: Readings Week 3) or (Source: Lecture 2 Slide 5).\n"
    "- If the sources are insufficient, say: \"The provided sources do not "
    "cover [topic] in detail. Please consult your additional readings.\"\n"
    "- If a Mermaid diagram was generated, reference it in your answer.\n\n"
    "Structure your answer clearly with headings and paragraphs."
)


def synthesis_node(state: AgentState) -> dict:
    """Compile all upstream outputs into a final answer for the student.

    Reads:  query, retrieved_texts, retrieved_slides, intent,
            ratio_decidendi, irac_analysis, mermaid_diagram, chronology_summary
    Writes: final_answer, node_trace
    """
    query = state["query"]
    intent = state.get("intent", "general")
    context = _format_context(state)

    sections: list[str] = [
        f"STUDENT QUESTION: {query}",
        f"DETECTED INTENT: {intent}",
        f"\n{'='*60}",
        f"PRIMARY EVIDENCE (ground your answer in this):\n{context}",
        f"{'='*60}",
    ]

    irac = state.get("irac_analysis", "")
    ratio = state.get("ratio_decidendi", "")
    mermaid = state.get("mermaid_diagram", "")
    chrono = state.get("chronology_summary", "")

    if irac or ratio or mermaid or chrono:
        sections.append(
            "\nDERIVED ANALYSIS (use for structure and framing, but verify "
            "all facts against the PRIMARY EVIDENCE above):"
        )
        if ratio:
            sections.append(f"\nRatio Decidendi: {ratio}")
        if irac:
            sections.append(f"\nIRAC Analysis:\n{irac}")
        if mermaid:
            sections.append(f"\nChronological Flowchart:\n```mermaid\n{mermaid}\n```")
        if chrono:
            sections.append(f"\nTimeline Summary:\n{chrono}")

    prompt = "\n".join(sections) + (
        "\n\nSynthesise a final answer for the student. Use the DERIVED "
        "ANALYSIS for structure and framing, but ensure every factual claim "
        "(cases, dates, statutes, legal tests) is supported by the PRIMARY "
        "EVIDENCE. If the derived analysis mentions something not in the "
        "primary evidence, omit it."
    )

    log.info("SynthesisNode: compiling final answer")
    final_answer = llm_call(prompt, model=SYNTHESIS_MODEL, system_instruction=SYNTHESIS_SYSTEM)

    return {
        "final_answer": final_answer,
        "node_trace": _append_trace(state, "synthesis"),
    }
