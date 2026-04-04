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
from google import genai
from google.genai import types

from src.agent.state import AgentState
from src.agent.tools import retrieve_all
from src.config import REASONING_MODEL

load_dotenv()

log = logging.getLogger(__name__)

# ── Shared LLM client ─────────────────────────────────────────────────────────

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazily initialise and cache the Gemini client."""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _llm_call(prompt: str, system_instruction: str | None = None) -> str:
    """Make a single LLM call and return the text response."""
    client = _get_client()
    config = types.GenerateContentConfig(
        temperature=0.2,
        system_instruction=system_instruction,
    )
    response = client.models.generate_content(
        model=REASONING_MODEL,
        contents=prompt,
        config=config,
    )
    return response.text


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

    raw = _llm_call(query, system_instruction=ROUTER_SYSTEM)

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

    texts, slides = retrieve_all(query, week=week, k_text=6, k_slides=4)

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
    "Be precise. Cite case names exactly as they appear in the sources. "
    "Do not invent facts or rules not supported by the provided context. "
    "If the context is insufficient, say so explicitly."
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
    response = _llm_call(prompt, system_instruction=RATIO_SYSTEM)

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
    "Mermaid rules:\n"
    "- Use `graph TD` (top-down) direction.\n"
    "- Node IDs must be single words or camelCase (no spaces).\n"
    "- Use square brackets for labels: `nodeId[\"Label text here\"]`.\n"
    "- Use arrows with labels for transitions: `A -->|\"action\"| B`.\n"
    "- Wrap labels containing special characters in double quotes.\n"
    "- Do NOT use the keyword `end` as a node ID.\n\n"
    "Format your response as:\n"
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
        "flowchart showing the chain of title or timeline of legal events."
    )

    log.info("ChronologyNode: generating Mermaid.js diagram")
    response = _llm_call(prompt, system_instruction=CHRONOLOGY_SYSTEM)

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
    "You are an expert Australian Property Law tutor providing a final, "
    "comprehensive answer to a student's question.\n\n"
    "You will be given:\n"
    "- The student's original question.\n"
    "- Retrieved source material from readings and lecture slides.\n"
    "- Optionally, an IRAC analysis with the ratio decidendi.\n"
    "- Optionally, a chronological timeline summary.\n\n"
    "Synthesise all available information into a clear, well-structured answer. "
    "Ensure 100% fidelity to the source material — do not invent facts. "
    "Cite case names and statutory references as they appear in the sources. "
    "If a Mermaid diagram was generated, reference it in your answer."
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
        f"\nSOURCE MATERIAL:\n{context}",
    ]

    irac = state.get("irac_analysis", "")
    if irac:
        sections.append(f"\nIRAC ANALYSIS (from Ratio Extractor):\n{irac}")

    ratio = state.get("ratio_decidendi", "")
    if ratio:
        sections.append(f"\nRATIO DECIDENDI: {ratio}")

    mermaid = state.get("mermaid_diagram", "")
    if mermaid:
        sections.append(
            f"\nCHRONOLOGICAL FLOWCHART (Mermaid.js):\n```mermaid\n{mermaid}\n```"
        )

    chrono = state.get("chronology_summary", "")
    if chrono:
        sections.append(f"\nTIMELINE SUMMARY:\n{chrono}")

    prompt = "\n".join(sections) + (
        "\n\nUsing all the information above, provide a comprehensive, "
        "well-structured final answer to the student's question."
    )

    log.info("SynthesisNode: compiling final answer")
    final_answer = _llm_call(prompt, system_instruction=SYNTHESIS_SYSTEM)

    return {
        "final_answer": final_answer,
        "node_trace": _append_trace(state, "synthesis"),
    }
