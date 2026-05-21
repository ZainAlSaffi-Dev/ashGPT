"""Cognitive nodes for the LangGraph legal-study pipeline.

Each node is a pure function: AgentState → partial AgentState update.
Nodes are composed into a graph in graph.py.

Pipeline:
    router_node → retrieval_node → [ratio_extractor_node | chronology_node] → synthesis_node
"""

from __future__ import annotations

import functools
import json
import logging
import re
import time

from dotenv import load_dotenv

from src.agent.chat_memory import (
    build_retrieval_query,
    format_memory_for_llm,
    format_transcript_for_llm,
    get_chat_history,
    get_conversation_memory,
    rewrite_followup_query,
)
from src.agent.state import AgentState
from src.agent.tools import retrieve_all
from src.agent.verification import find_unsupported_cases
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


def _timed(name: str):
    """Decorate a node function to record its wall-clock duration into ``state.timings``.

    The wrapped node may opt into sub-hop timings by including ``_timing_sub``
    in its returned delta (e.g. retrieval reporting embed/chroma/bm25/rerank
    milliseconds). The decorator pops that key off the delta and stores it
    under the per-node entry so the public state schema stays clean.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state: AgentState) -> dict:
            t0 = time.perf_counter()
            delta = fn(state) or {}
            ms = round((time.perf_counter() - t0) * 1000, 1)
            timings = list(state.get("timings", []))
            entry: dict = {"node": name, "ms": ms}
            sub = delta.pop("_timing_sub", None) if isinstance(delta, dict) else None
            if sub:
                entry["sub"] = sub
            timings.append(entry)
            if isinstance(delta, dict):
                delta["timings"] = timings
            return delta

        return wrapper

    return deco


# ── Helper: format retrieved docs for prompts ─────────────────────────────────


def _format_context(state: AgentState) -> str:
    """Compile retrieved chunks into a context block with stable citation labels.

    Each source gets a sequential ``[S#]`` label that the synthesis prompt
    instructs the model to cite inline. The UI parses the same tokens out of
    the final answer and renders them as clickable badges linking to the
    matching ``sources`` entry. Text and image-bearing chunks share one
    numbering sequence so a citation always resolves to exactly one source
    regardless of modality.
    """
    sections: list[str] = []

    def _tags(doc: dict) -> str:
        bits = []
        if doc.get("doc_type"):
            bits.append(doc["doc_type"])
        if doc.get("week"):
            bits.append(doc["week"])
        return f" [{', '.join(bits)}]" if bits else ""

    label = 0
    texts = state.get("retrieved_texts", [])
    if texts:
        sections.append("=== RETRIEVED TEXT SOURCES ===")
        for doc in texts:
            label += 1
            sections.append(
                f"\n[S{label}] {doc['source']}{_tags(doc)}\n{doc['content']}"
            )

    slides = state.get("retrieved_slides", [])
    if slides:
        sections.append("\n=== RETRIEVED IMAGES / DIAGRAMS ===")
        for doc in slides:
            label += 1
            sections.append(
                f"\n[S{label}] {doc['source']}{_tags(doc)}\n{doc['content']}"
            )

    return "\n".join(sections) if sections else "(No context retrieved.)"


def _retrieved_chunk_ids(state: AgentState) -> list[str]:
    """Stable per-chunk identifiers for cache keying. Matches graph._chunk_ids_from."""
    ids: list[str] = []
    for d in (state.get("retrieved_texts") or []) + (state.get("retrieved_slides") or []):
        ids.append(d.get("chunk_id") or d.get("source") or (d.get("content") or "")[:80])
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: ROUTER
# ══════════════════════════════════════════════════════════════════════════════


ROUTER_SYSTEM = (
    "You are a query classifier for a legal study assistant. The student is "
    "studying any area of law (their own uploaded corpus — cases, statutes, "
    "notes, lecture material). You may receive a prior conversation plus a "
    "LATEST student message. Classify **only** the latest message, but use "
    "the transcript to resolve vague references (e.g. \"that case\", \"the "
    "diagram\", \"explain further\").\n\n"
    "Return ONLY a JSON object with two keys:\n"
    '  "intent": one of "ratio", "chronology", "summary", "general"\n'
    '  "week_filter": a string like "week_3" if the student explicitly '
    'references a labelled week, topic block, or module in their own corpus; '
    'null otherwise. Do not invent a week.\n\n'
    "Intent definitions:\n"
    '- "ratio": user wants the ratio decidendi, binding legal rule, or IRAC analysis of a case.\n'
    '- "chronology": user ONLY wants a visual timeline, flowchart, diagram, chain of title, '
    'sequence of events, chain of events, "who did what", "what happened and when", or '
    '"walk me through the events" — with NO request for a broader explanation or summary.\n'
    '- "summary": use this when the user wants (a) a general case or topic explanation '
    'OR (b) asks for BOTH a sequence/timeline AND a broader explanation or summary. '
    'Examples that must map to "summary": "explain X and give me the sequence of events", '
    '"who was doing what and provide a summary", "walk me through the case". '
    'This intent runs both the chronology diagram AND the IRAC analysis, so it is the '
    'correct choice whenever the query combines a timeline request with any explanatory goal.\n'
    '- "general": anything else (greetings, meta-questions, off-topic, definitional, '
    'doctrinal explanations not tied to a single case).\n\n'
    "DECISION RULE: If the query contains timeline/sequence language (e.g. \"sequence of "
    "events\", \"who did what\", \"chain of events\", \"what happened\", \"walk me through\") "
    "alongside any request for explanation, summary, or overview — classify as \"summary\", "
    "NOT \"chronology\". Only use \"chronology\" when the user is asking solely for a "
    "diagram or timeline with no broader explanation requested.\n\n"
    "Respond with ONLY the JSON object, no markdown fences."
)


@_timed("router")
def router_node(state: AgentState) -> dict:
    """Classify the user's intent and extract any week filter.

    Reads:  query, chat_history, week_filter (UI may pre-set week; it wins over the model)
    Writes: intent, week_filter, node_trace
    """
    query = state["query"]
    history = get_chat_history(state)
    memory = get_conversation_memory(state)
    log.info("RouterNode: classifying query (history_turns=%d)", len(history))

    if history or memory:
        memory_block = format_memory_for_llm(memory)
        memory_prefix = f"{memory_block}\n\n" if memory_block else ""
        transcript = format_transcript_for_llm(history)
        router_input = (
            f"{memory_prefix}"
            f"RECENT CONVERSATION SO FAR:\n{transcript}\n\n"
            f"LATEST STUDENT MESSAGE (classify this):\n{query}"
        )
    else:
        router_input = query

    raw = llm_call(router_input, model=ROUTER_MODEL, system_instruction=ROUTER_SYSTEM)

    try:
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
        parsed = json.loads(cleaned)
        intent = parsed.get("intent", "general")
        parsed_week = parsed.get("week_filter")
    except (json.JSONDecodeError, AttributeError):
        log.warning("Router failed to parse LLM output, defaulting to 'summary'")
        intent = "summary"
        parsed_week = None

    valid_intents = {"ratio", "chronology", "summary", "general"}
    if intent not in valid_intents:
        intent = "summary"

    existing_week = state.get("week_filter")
    week_filter = existing_week if existing_week else parsed_week

    log.info("RouterNode: intent=%s, week_filter=%s", intent, week_filter)
    return {
        "intent": intent,
        "week_filter": week_filter,
        "node_trace": _append_trace(state, "router"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════


@_timed("retrieval")
def retrieval_node(state: AgentState) -> dict:
    """Retrieve text + image chunks from the user's vector store.

    Reads:  query, week_filter, chat_history, user_id (tenant namespace),
            use_reranker (optional override)
    Writes: retrieved_texts, retrieved_slides, rewritten_query, node_trace
    """
    from src.config import USE_QUERY_REWRITER

    history = get_chat_history(state)
    memory = get_conversation_memory(state)
    raw_query = state["query"]

    # On follow-up turns optionally run a coreference rewrite so the retriever
    # sees a self-contained query ("explain the elements" → "Explain the
    # elements of adverse possession in Mabo v Queensland (No 2)"). Skip on
    # first turns — no transcript to resolve and the LLM hop is dead weight.
    rewritten_query: str | None = None
    if (history or memory) and USE_QUERY_REWRITER:
        rewritten_query = rewrite_followup_query(raw_query, history, memory=memory)
        if rewritten_query == raw_query:
            rewritten_query = None  # rewriter returned the input unchanged

    embed_query = rewritten_query or raw_query
    # Always run the deterministic packer on top — it appends the prior
    # assistant excerpt so the dense embedding has even more anchor signal.
    search_q = build_retrieval_query(embed_query, history, memory)

    week = state.get("week_filter")
    namespace = state.get("user_id")
    from src.agent.scope import RetrievalScope

    scope = RetrievalScope.from_input(state.get("retrieval_scope"), week_filter=week)
    use_reranker_override = state.get("use_reranker")
    log.info(
        "RetrievalNode: searching KB (week=%s, ns=%s, follow_up=%s, rewrote=%s, rerank_override=%s)",
        week,
        namespace,
        bool(history),
        bool(rewritten_query),
        use_reranker_override,
    )

    sub_timings: dict[str, float] = {}
    texts, slides = retrieve_all(
        search_q,
        week=week,
        k_text=8,
        k_slides=4,
        use_reranker=use_reranker_override,
        namespace=namespace,
        timings=sub_timings,
        scope=scope,
    )

    log.info(
        "RetrievalNode: retrieved %d text chunks, %d slides (timings_ms=%s)",
        len(texts),
        len(slides),
        sub_timings,
    )
    out: dict = {
        "retrieved_texts": texts,
        "retrieved_slides": slides,
        "retrieval_scope_hash": scope.scope_hash(),
        "no_material_reason": "no material in the selected scope"
        if scope.explicit and not texts and not slides
        else "",
        "node_trace": _append_trace(state, "retrieval"),
        "_timing_sub": sub_timings or None,
    }
    if rewritten_query:
        out["rewritten_query"] = rewritten_query
    return out


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2.5: CACHE LOOKUP (post-retrieval short-circuit)
# ══════════════════════════════════════════════════════════════════════════════


@_timed("cache_check")
def cache_check_node(state: AgentState) -> dict:
    """Look up a cached final answer keyed by (user_id, normalised_query, chunk_ids).

    Runs after retrieval so the chunk-id-aware key matches exactly what
    ``run_query`` writes on completion. On hit, returns the cached
    ``final_answer`` and signals ``cache_hit=True``; the conditional edge in
    :mod:`src.agent.graph` then routes straight to END, skipping every
    reasoning + verification LLM call. Disabled when chat_history is set
    (multi-turn reasoning bypass) or when ``user_id`` is unset (legacy/shared
    collection — no tenant scope to key against).
    """
    from src.config import USE_ANSWER_CACHE

    user_id = state.get("user_id")
    history = get_chat_history(state)
    if not USE_ANSWER_CACHE or not user_id or history:
        return {"cache_hit": False, "node_trace": _append_trace(state, "cache_check")}

    from src.agent.cache import get as cache_get
    from src.agent.cache import make_cache_key

    chunk_ids = _retrieved_chunk_ids(state)
    if not chunk_ids:
        return {"cache_hit": False, "node_trace": _append_trace(state, "cache_check")}

    cache_key = make_cache_key(
        user_id, state["query"], chunk_ids, scope_hash=state.get("retrieval_scope_hash")
    )

    import asyncio

    try:
        hit = asyncio.run(cache_get(cache_key))
    except RuntimeError:
        # Already inside an event loop (e.g. FastAPI thread executor calls
        # `loop.run_in_executor` → run_query is sync but the parent loop may
        # leak in via newer Python asyncio). Fall through as a miss rather
        # than raise.
        hit = None
    except Exception as e:  # pragma: no cover — cache failure should never block
        log.warning("cache lookup failed (%s); treating as miss", e)
        hit = None

    if not hit:
        return {"cache_hit": False, "node_trace": _append_trace(state, "cache_check")}

    log.info("cache HIT user=%s key=%s", user_id, cache_key[:12])
    return {
        "cache_hit": True,
        "final_answer": hit.get("answer", ""),
        "intent": hit.get("intent") or state.get("intent", "general"),
        "verification_report": hit.get("verification_report"),
        "node_trace": _append_trace(state, "cache_check"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3: RATIO EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════


RATIO_SYSTEM = (
    "You are an expert legal tutor working across any area of law (the "
    "student's own uploaded corpus). Analyse the provided source material and "
    "extract the ratio decidendi — the binding legal principle the court "
    "applied to reach its decision. Stay neutral on jurisdiction: follow "
    "whatever the sources indicate (Australian, English, US, EU, comparative, "
    "etc.).\n\n"
    "Sources arrive labelled ``[S1]``, ``[S2]``, …. Use those exact tokens "
    "for inline citations.\n\n"
    "You MUST structure your response as an IRAC analysis:\n"
    "1. **Issue**: State the legal question the court had to decide.\n"
    "2. **Rule**: State the legal rule or principle (this IS the ratio decidendi).\n"
    "3. **Application**: Explain how the court applied the rule to the facts.\n"
    "4. **Conclusion**: State the court's decision.\n\n"
    "GROUNDING RULES — non-negotiable:\n"
    "- Every factual claim (case name, date, statute, holding, legal test) "
    "MUST end with one or more ``[S#]`` tokens lifted verbatim from the "
    "source labels.\n"
    "- If a claim is not in any source, either drop it or mark the sentence "
    "with a literal ``[external]`` prefix and explain why you needed outside "
    "knowledge. Never silently smuggle facts in from your own training.\n"
    "- Explanations and paraphrasing that don't introduce new facts don't "
    "need a citation.\n"
    "- If the source material is insufficient, say so explicitly with an "
    "``[external]`` marker rather than filling gaps."
)


@_timed("ratio_extractor")
def ratio_extractor_node(state: AgentState) -> dict:
    """Extract the ratio decidendi and produce an IRAC analysis.

    Reads:  query, retrieved_texts, retrieved_slides
    Writes: ratio_decidendi, irac_analysis, node_trace
    """
    query = state["query"]
    context = _format_context(state)
    history = get_chat_history(state)
    memory = get_conversation_memory(state)
    memory_block = (
        f"{format_memory_for_llm(memory)}\n\n"
        if memory
        else ""
    )
    hist_block = (
        f"CONVERSATION HISTORY:\n{format_transcript_for_llm(history)}\n\n"
        if history
        else ""
    )

    prompt = (
        f"{memory_block}"
        f"{hist_block}"
        f"CURRENT USER QUESTION: {query}\n\n"
        f"SOURCE MATERIAL:\n{context}\n\n"
        "Based on the source material above, provide:\n"
        "1. The RATIO DECIDENDI (the exact binding legal rule) in one clear sentence.\n"
        "2. A full IRAC analysis.\n\n"
        "If the question refers to earlier turns, use the conversation history only "
        "to interpret what the student means — all legal facts must still come from "
        "the source material."
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
    "You are an expert legal tutor specialising in turning case facts, "
    "statutory histories, or procedural records into precise chronological "
    "flowcharts. The student's corpus may cover any area of law.\n\n"
    "Your task:\n"
    "1. Extract the chronological sequence of events from the source material.\n"
    "2. Output a valid Mermaid.js flowchart using `graph TD` syntax.\n"
    "3. After the Mermaid block, provide a brief plain-English summary of the timeline.\n\n"
    "STRICT Mermaid formatting rules for readability:\n"
    "- Use `graph TD` (top-down) direction.\n"
    "- Node IDs must be camelCase (no spaces): e.g. `claimFiled`, `appealHeard`.\n"
    "- KEEP NODE LABELS SHORT — maximum 8-10 words per node. Put dates at the start.\n"
    "  GOOD: `e1[\"2019-03: claim filed\"]`\n"
    "  BAD:  `e1[\"On 14 March 2019 the plaintiff filed a statement of claim against the defendant…\"]`\n"
    "- Put extra detail on the EDGE labels, not the nodes:\n"
    "  GOOD: `e1 -->|\"plaintiff serves statement of claim\"| e2`\n"
    "- Use curly brackets for decision nodes: `d1{\"Defence served?\"}`\n"
    "- Use subgraphs to group related phases: `subgraph phase1 [\"Pleadings\"]`\n"
    "- Wrap ALL labels in double quotes.\n"
    "- Do NOT use the keyword `end` as a node ID.\n"
    "- Aim for 5-12 nodes total. Fewer is better — each node is one key event.\n\n"
    "Example of a GOOD diagram:\n"
    "```\n"
    "graph TD\n"
    "    subgraph pleadings [\"Pleadings\"]\n"
    "        e1[\"2019-03: claim filed\"] -->|\"plaintiff serves statement of claim\"| e2[\"2019-05: defence filed\"]\n"
    "    end\n"
    "    subgraph trial [\"Trial\"]\n"
    "        e2 -->|\"matter heard at first instance\"| e3[\"2020-08: trial judgment\"]\n"
    "    end\n"
    "    e3 -->|\"appeal lodged within time\"| e4[\"2021-02: appeal decided\"]\n"
    "```\n\n"
    "Format your full response as:\n"
    "```mermaid\n<your flowchart>\n```\n\n"
    "**Timeline Summary:**\n<plain-English summary>"
)


@_timed("chronology")
def chronology_node(state: AgentState) -> dict:
    """Generate a Mermaid.js chronological flowchart from retrieved facts.

    Reads:  query, retrieved_texts, retrieved_slides
    Writes: mermaid_diagram, chronology_summary, node_trace
    """
    query = state["query"]
    context = _format_context(state)
    history = get_chat_history(state)
    memory = get_conversation_memory(state)
    memory_block = (
        f"{format_memory_for_llm(memory)}\n\n"
        if memory
        else ""
    )
    hist_block = (
        f"CONVERSATION HISTORY:\n{format_transcript_for_llm(history)}\n\n"
        if history
        else ""
    )

    prompt = (
        f"{memory_block}"
        f"{hist_block}"
        f"CURRENT USER QUESTION: {query}\n\n"
        f"SOURCE MATERIAL:\n{context}\n\n"
        "Extract the chronological sequence of events and generate a Mermaid.js "
        "flowchart showing the chain of title or timeline of legal events.\n\n"
        "IMPORTANT: Only include events, dates, and parties that appear in the "
        "source material above. Do not invent events. You may label diagram "
        "edges with brief plain-English explanations of what happened.\n\n"
        "Use conversation history only to interpret follow-up phrasing; facts "
        "must come from the sources."
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
    "You are an expert legal tutor providing a final answer to a student's "
    "question. The student may be studying any area of law (their own "
    "uploaded corpus — cases, statutes, notes, slides, transcripts).\n\n"
    "You will be given:\n"
    "- The student's original question.\n"
    "- Retrieved source material from their corpus, each chunk labelled with "
    "a stable identifier of the form ``[S1]``, ``[S2]``, …. These labels are "
    "the only citation tokens the UI understands.\n"
    "- Optionally, an IRAC analysis with the ratio decidendi.\n"
    "- Optionally, a Mermaid.js chronological flowchart and timeline summary.\n\n"
    "GROUNDING RULES — non-negotiable:\n"
    "1. Every factual claim (case name, date, statute, holding, legal test, "
    "definition, doctrine attribution) MUST be followed by one or more "
    "citation tokens, e.g. ``…the doctrine of estoppel [S1]`` or "
    "``…s 31 of the Property Law Act [S2][S4]``. Use the ``[S#]`` tokens "
    "EXACTLY as they appear in the source labels — do not invent S-numbers, "
    "do not rephrase them as ``(Source: …)`` or footnotes.\n"
    "2. If a claim is not covered by any provided source you MUST do ONE of "
    "the following — never silently rely on outside knowledge:\n"
    "    (a) Drop the claim entirely; or\n"
    "    (b) Mark it explicitly: prefix the sentence with ``[external]`` "
    "(literal brackets, lowercase) and add a short reason, e.g. ``[external] "
    "Background context not in the supplied sources: …``. The UI surfaces "
    "this to the student as a warning.\n"
    "3. EXPLANATIONS, paraphrasing, plain-English restatements, and "
    "definitional setup that do NOT introduce new facts do not need a "
    "citation — only factual claims do.\n"
    "4. If the sources are insufficient to answer the question, say so "
    "explicitly with an ``[external]`` marker and suggest the student upload "
    "more material — do not paper over the gap with general knowledge.\n"
    "5. If CONVERSATION HISTORY is provided, resolve pronouns and follow-up "
    "references using it, but every **new** factual claim must still be "
    "either backed by a ``[S#]`` token or marked ``[external]``.\n\n"
    "CRITICAL — MERMAID DIAGRAM PRESERVATION RULE:\n"
    "If the DERIVED ANALYSIS section contains a ```mermaid code block, you MUST "
    "copy that exact ```mermaid ... ``` block verbatim into your final answer. "
    "The UI renders it as an interactive diagram — if you omit the code block or "
    "convert it to bullet points or prose, the diagram will be lost entirely. "
    "Do NOT paraphrase, summarise, or rewrite the Mermaid block in any form. "
    "Place the ```mermaid block where it fits naturally in your answer "
    "(e.g. after introducing the timeline), then continue with your explanation.\n\n"
    "Structure your answer clearly with headings and paragraphs."
)


@_timed("synthesis")
def synthesis_node(state: AgentState) -> dict:
    """Compile all upstream outputs into a final answer for the student.

    Reads:  query, retrieved_texts, retrieved_slides, intent,
            ratio_decidendi, irac_analysis, mermaid_diagram, chronology_summary
    Writes: final_answer, node_trace
    """
    query = state["query"]
    intent = state.get("intent", "general")
    context = _format_context(state)
    history = get_chat_history(state)
    memory = get_conversation_memory(state)

    sections: list[str] = []
    if memory:
        sections.append(format_memory_for_llm(memory))
        sections.append("")
    if history:
        sections.append(
            "CONVERSATION HISTORY (session memory — use to interpret follow-ups):\n"
            f"{format_transcript_for_llm(history)}"
        )
        sections.append("")

    sections.extend(
        [
            f"CURRENT STUDENT MESSAGE: {query}",
            f"DETECTED INTENT: {intent}",
            f"\n{'='*60}",
            f"PRIMARY EVIDENCE (ground your answer in this):\n{context}",
            f"{'='*60}",
        ]
    )

    irac = state.get("irac_analysis", "")
    ratio = state.get("ratio_decidendi", "")
    mermaid = state.get("mermaid_diagram", "")
    chrono = state.get("chronology_summary", "")

    if irac or ratio or mermaid or chrono:
        sections.append(
            "\nDERIVED ANALYSIS (incorporate this directly into your answer — "
            "any ```mermaid block MUST be reproduced verbatim; verify all "
            "factual claims against the PRIMARY EVIDENCE above):"
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
        "\n\nSynthesise a final answer for the student. "
        "IMPORTANT: If a ```mermaid code block appears in the DERIVED ANALYSIS, "
        "you MUST include that exact ```mermaid ... ``` block verbatim in your "
        "response — do not convert it to bullet points or prose. "
        "Ensure every factual claim (cases, dates, statutes, legal tests) is "
        "supported by the PRIMARY EVIDENCE. If the derived analysis mentions "
        "something not in the primary evidence, omit it."
    )

    log.info("SynthesisNode: compiling final answer")
    # Phase 5: state may carry ``_override_synthesis_model`` when the
    # confidence-gated escalation path re-invokes synthesis with a stronger
    # model. Default to SYNTHESIS_MODEL otherwise.
    model = state.get("_override_synthesis_model") or SYNTHESIS_MODEL
    final_answer = llm_call(prompt, model=model, system_instruction=SYNTHESIS_SYSTEM)

    return {
        "final_answer": final_answer,
        "node_trace": _append_trace(state, "synthesis"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6: VERIFICATION (post-synthesis fact check)
# ══════════════════════════════════════════════════════════════════════════════


VERIFICATION_REWRITE_SYSTEM = (
    "You are a careful legal editor. You will be given:\n"
    "1. The student's question.\n"
    "2. The retrieved source material from the student's corpus.\n"
    "3. A draft answer.\n"
    "4. A list of CASE CITATIONS in the draft that DO NOT appear in the sources.\n\n"
    "Your task: rewrite the draft answer so that EACH unsupported citation is "
    "either (a) removed entirely, or (b) hedged with a phrase that signals it is "
    "not in the provided materials (e.g. \"is not covered in the uploaded "
    "sources\"). Preserve every supported claim, the structure of the answer, "
    "every Markdown heading, and any ```mermaid code block VERBATIM. Do not "
    "introduce new citations. Do not mention this rewrite process to the "
    "student. Return ONLY the rewritten answer."
)


def _collect_sources_text(state: AgentState) -> str:
    """Concatenate raw content from retrieved texts and slides for citation lookup."""
    parts: list[str] = []
    for doc in state.get("retrieved_texts", []) or []:
        parts.append(doc.get("content", ""))
        parts.append(doc.get("source", ""))
    for doc in state.get("retrieved_slides", []) or []:
        parts.append(doc.get("content", ""))
        parts.append(doc.get("source", ""))
    return "\n".join(p for p in parts if p)


@_timed("verification")
def verification_node(state: AgentState) -> dict:
    """Fact-check cited cases in the final answer against retrieved sources.

    For each italicised or plain ``X v Y`` citation in ``final_answer``, check
    whether the case name appears anywhere in the retrieved chunk content.
    If unsupported citations exist, ask the synthesis model to rewrite the
    answer with those claims removed or hedged. Supported claims are
    preserved verbatim by prompt design.

    Reads:  final_answer, retrieved_texts, retrieved_slides, query
    Writes: final_answer (possibly rewritten), verification_report, node_trace
    """
    answer = state.get("final_answer", "") or ""
    sources_text = _collect_sources_text(state)
    unsupported = find_unsupported_cases(answer, sources_text)

    # Cheap confidence proxy: 1.0 when no unsupported claims, else fraction
    # of total cited cases that were supported. Used by Phase 5 escalation.
    from src.agent.verification import extract_case_citations

    cited = extract_case_citations(answer)
    total = len(cited) or 1
    supported = max(0, total - len(unsupported))
    confidence = supported / total

    # Count ``[S#]`` citation tokens + ``[external]`` declarations so the UI
    # can show the user how grounded the answer was and whether the model
    # had to reach outside the indexed sources for anything.
    citation_tokens = re.findall(r"\[S\d+\]", answer)
    external_markers = re.findall(r"\[external\]", answer, flags=re.IGNORECASE)

    report = {
        "unsupported_claims": unsupported,
        "rewrites_applied": False,
        "claims_total": len(cited),
        "claims_supported": supported,
        "confidence_score": round(confidence, 3),
        "all_supported": len(unsupported) == 0,
        "citation_count": len(citation_tokens),
        "distinct_citations": sorted(set(citation_tokens)),
        "used_external_knowledge": bool(external_markers),
        "external_marker_count": len(external_markers),
    }

    if not unsupported:
        log.info("VerificationNode: all citations supported — no rewrite needed")
        return {
            "verification_report": report,
            "node_trace": _append_trace(state, "verification"),
        }

    log.info(
        "VerificationNode: %d unsupported citation(s): %s",
        len(unsupported),
        unsupported,
    )

    rewrite_prompt = (
        f"STUDENT QUESTION: {state.get('query', '')}\n\n"
        f"RETRIEVED SOURCE MATERIAL:\n{_format_context(state)}\n\n"
        f"DRAFT ANSWER:\n{answer}\n\n"
        f"UNSUPPORTED CITATIONS (must be removed or hedged):\n"
        + "\n".join(f"- {c}" for c in unsupported)
        + "\n\nRewrite the answer per the instructions. Return ONLY the rewritten answer."
    )

    try:
        rewritten = llm_call(
            rewrite_prompt,
            model=SYNTHESIS_MODEL,
            system_instruction=VERIFICATION_REWRITE_SYSTEM,
        )
        rewritten = (rewritten or "").strip()
        if rewritten:
            report["rewrites_applied"] = True
            return {
                "final_answer": rewritten,
                "verification_report": report,
                "node_trace": _append_trace(state, "verification"),
            }
    except Exception as e:  # graceful: leave answer unchanged on rewrite failure
        log.warning("VerificationNode: rewrite failed (%s); keeping draft", e)

    return {
        "verification_report": report,
        "node_trace": _append_trace(state, "verification"),
    }
