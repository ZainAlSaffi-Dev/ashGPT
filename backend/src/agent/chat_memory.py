"""Multi-turn chat helpers: trim, validate, format history, and pack retrieval queries.

``chat_history`` in :class:`AgentState` holds **prior** turns only; the current user
utterance always lives in ``query``. This keeps retrieval embeddings focused while
still giving the router and synthesis models conversational context.
"""

from __future__ import annotations

import logging
import re

from src.agent.state import AgentState, ChatMessage, ConversationMemory
from src.config import (
    CHAT_HISTORY_MAX_ASSISTANT_TAIL_CHARS,
    CHAT_HISTORY_MAX_CHARS_PER_MESSAGE,
    CHAT_HISTORY_MAX_MESSAGES,
    CHAT_MEMORY_MAX_FACT_CHARS,
    CHAT_MEMORY_MAX_FACTS,
    CHAT_MEMORY_MAX_SUMMARY_CHARS,
)

log = logging.getLogger(__name__)


def _normalise_messages(raw: list[dict[str, str]] | None) -> tuple[list[ChatMessage], dict]:
    overflow = {"dropped_turns": 0, "truncated_messages": 0}
    if not raw:
        return [], overflow
    out: list[ChatMessage] = []
    for m in raw:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        cap = CHAT_HISTORY_MAX_CHARS_PER_MESSAGE
        if len(content) > cap:
            content = content[:cap] + "\n… [truncated]"
            overflow["truncated_messages"] += 1
        out.append({"role": role, "content": content})  # type: ignore[typeddict-item]
    return out, overflow


def prepare_chat_history_for_run(
    raw: list[dict[str, str]] | None,
) -> tuple[list[ChatMessage], dict]:
    """Normalise roles, strip empties, truncate long messages, cap turn count.

    Returns the prepared list **and** an overflow report so the caller can
    surface it to the user / logs ("3 older turns trimmed to fit context").
    The report shape is ``{"dropped_turns": int, "truncated_messages": int}``.
    """
    out, overflow = _normalise_messages(raw)
    if len(out) > CHAT_HISTORY_MAX_MESSAGES:
        overflow["dropped_turns"] = len(out) - CHAT_HISTORY_MAX_MESSAGES
        out = out[-CHAT_HISTORY_MAX_MESSAGES:]
    return out, overflow


def prepare_chat_memory_for_run(
    raw: list[dict[str, str]] | None,
) -> tuple[list[ChatMessage], dict, ConversationMemory | None]:
    """Prepare recent verbatim history plus compressed memory for older turns.

    The recent window remains the only verbatim transcript passed through the
    graph. Older session rows are deterministically compressed into a short
    memory block whose contract is explicit: it may resolve references,
    preferences, shorthand, and study goals, but it is never source evidence
    for legal claims.
    """
    messages, overflow = _normalise_messages(raw)
    telemetry = {
        "dropped_turns": 0,
        "truncated_messages": overflow["truncated_messages"],
        "memory_compressed": False,
        "compressed_turns": 0,
        "recent_messages": len(messages),
        "memory_fact_count": 0,
        "memory_summary_chars": 0,
    }
    if len(messages) <= CHAT_HISTORY_MAX_MESSAGES:
        return messages, telemetry, None

    older = messages[: -CHAT_HISTORY_MAX_MESSAGES]
    recent = messages[-CHAT_HISTORY_MAX_MESSAGES:]
    memory = build_conversation_memory(older)
    fact_count = len(memory.get("facts", []))
    summary_chars = len(memory.get("summary", ""))
    telemetry.update(
        {
            "dropped_turns": len(older),
            "memory_compressed": True,
            "compressed_turns": len(older),
            "recent_messages": len(recent),
            "memory_fact_count": fact_count,
            "memory_summary_chars": summary_chars,
        }
    )
    return recent, telemetry, memory


def build_conversation_memory(messages: list[ChatMessage]) -> ConversationMemory:
    facts = _extract_memory_facts(messages)
    summary = _summarise_older_turns(messages)
    corrections = [
        _trim_fact(m["content"])
        for m in messages
        if m["role"] == "user" and _is_correction(m["content"])
    ][-6:]
    return {
        "summary": summary,
        "facts": facts[:CHAT_MEMORY_MAX_FACTS],
        "corrections": corrections,
        "compressed_turns": len(messages),
        "source_grounding": (
            "Compressed conversation memory is session context only. Use it to "
            "resolve references, preferences, shorthand, and corrections; do "
            "not treat it as source evidence for legal propositions."
        ),
    }


def _summarise_older_turns(messages: list[ChatMessage]) -> str:
    user_topics: list[str] = []
    assistant_topics: list[str] = []
    for m in messages:
        text = _trim_fact(m["content"], 180)
        if not text:
            continue
        if m["role"] == "user":
            if _is_correction(text):
                continue
            user_topics.append(text)
        elif _looks_like_substantive_answer(text):
            assistant_topics.append(text)

    parts: list[str] = []
    if user_topics:
        parts.append("Earlier student requests: " + "; ".join(user_topics[-8:]))
    if assistant_topics:
        parts.append("Earlier tutor coverage: " + "; ".join(assistant_topics[-5:]))
    summary = "\n".join(parts)
    if len(summary) > CHAT_MEMORY_MAX_SUMMARY_CHARS:
        summary = summary[:CHAT_MEMORY_MAX_SUMMARY_CHARS].rstrip() + "\n… [compressed]"
    return summary


def _extract_memory_facts(messages: list[ChatMessage]) -> list[dict[str, str]]:
    facts: list[dict[str, str]] = []
    jurisdiction: str | None = None
    corrections: list[str] = []
    ignored_terms: set[str] = set()

    def add(kind: str, text: str) -> None:
        text = _trim_fact(text, CHAT_MEMORY_MAX_FACT_CHARS)
        if not text:
            return
        low = text.lower()
        if any(term and term in low for term in ignored_terms):
            return
        if any(f["type"] == kind and f["text"].lower() == low for f in facts):
            return
        facts.append({"type": kind, "text": text})

    for m in messages:
        text = m["content"]
        low = text.lower()
        if m["role"] == "user" and _is_correction(text):
            correction = _trim_fact(text)
            corrections.append(correction)
            for term in re.findall(r"\b[A-Z]{2,}\b", text):
                ignored_terms.add(term.lower())
            for term in re.findall(r"['\"]([^'\"]+)['\"]", text):
                ignored_terms.add(term.lower())
            if "ignore" in low or "forget" in low or "disregard" in low:
                ignored_terms.update(_keywords_from_correction(text))
            new_jurisdiction = _extract_jurisdiction(text)
            if new_jurisdiction:
                jurisdiction = new_jurisdiction
            continue

        if m["role"] == "user":
            new_jurisdiction = _extract_jurisdiction(text)
            if new_jurisdiction:
                jurisdiction = new_jurisdiction
            for shorthand in _extract_shorthand(text):
                add("shorthand", shorthand)
            for goal in _extract_study_goal(text):
                add("study_goal", goal)
            for constraint in _extract_source_constraints(text):
                add("source_constraint", constraint)

        for authority in _extract_authorities(text):
            add("authority_discussed", f"Earlier authority discussed: {authority}.")

    if ignored_terms:
        facts = [
            f
            for f in facts
            if not any(term and term in f["text"].lower() for term in ignored_terms)
        ]

    if jurisdiction:
        facts = [f for f in facts if f["type"] != "jurisdiction"]
        facts.insert(0, {"type": "jurisdiction", "text": jurisdiction})
    for correction in corrections[-4:]:
        facts.append({"type": "correction", "text": f"Correction to honour: {correction}"})
    return facts


def _extract_jurisdiction(text: str) -> str | None:
    low = text.lower()
    known = [
        ("queensland", "Current jurisdiction/course context: Queensland law."),
        ("nsw", "Current jurisdiction/course context: NSW law."),
        ("new south wales", "Current jurisdiction/course context: NSW law."),
        ("victoria", "Current jurisdiction/course context: Victorian law."),
        ("australia", "Current jurisdiction/course context: Australian law."),
        ("australian", "Current jurisdiction/course context: Australian law."),
        ("england", "Current jurisdiction/course context: English law."),
        ("uk", "Current jurisdiction/course context: UK law."),
        ("united states", "Current jurisdiction/course context: US law."),
        (" us ", "Current jurisdiction/course context: US law."),
    ]
    if (
        "jurisdiction" in low
        or "assume" in low
        or "context" in low
        or "switch" in low
        or "use" in low
        or "instead" in low
    ):
        for needle, label in known:
            if needle in f" {low} ":
                return label
    return None


def _extract_shorthand(text: str) -> list[str]:
    out: list[str] = []
    patterns = [
        r"\b([A-Z][A-Z0-9]{1,12})\s+(?:means|=|stands for)\s+([^.;\n]+)",
        r"\b(?:call|refer to)\s+([^.;\n]+?)\s+as\s+['\"]?([A-Z][A-Z0-9]{1,12})['\"]?",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if len(match.groups()) == 2:
                a, b = match.group(1).strip(), match.group(2).strip()
                if re.fullmatch(r"[A-Z][A-Z0-9]{1,12}", a):
                    out.append(f"Student shorthand: {a} = {b}.")
                else:
                    out.append(f"Student shorthand: {b} = {a}.")
    return out


def _extract_study_goal(text: str) -> list[str]:
    patterns = [
        r"\b(?:my goal is|i want to|help me|focus on)\s+([^.;\n]+)",
        r"\b(?:i am|i'm) studying\s+([^.;\n]+)",
    ]
    return [f"Student study goal: {m.group(1).strip()}." for p in patterns for m in re.finditer(p, text, flags=re.IGNORECASE)]


def _extract_source_constraints(text: str) -> list[str]:
    low = text.lower()
    if "only use uploaded" in low or "stick to the sources" in low or "no external" in low:
        return ["Student asked to stick to uploaded sources and avoid uncited external law."]
    return []


def _extract_authorities(text: str) -> list[str]:
    pattern = r"\b([A-Z][A-Za-z'’.-]+(?:\s+\([^)]+\))?(?:\s+[A-Z][A-Za-z'’.-]+)*\s+v\s+[A-Z][A-Za-z'’.-]+(?:\s+\([^)]+\))?(?:\s+[A-Z][A-Za-z'’.-]+)*)"
    found = []
    for match in re.finditer(pattern, text):
        authority = _clean_authority(match.group(1).strip(" .,:;"))
        if 5 <= len(authority) <= 120:
            found.append(authority)
    return found


def _clean_authority(authority: str) -> str:
    return re.sub(
        r"^(?:explain|discuss|summarise|summarize|apply|compare)\s+",
        "",
        authority,
        flags=re.IGNORECASE,
    )


def _is_correction(text: str) -> bool:
    low = text.lower()
    return any(
        phrase in low
        for phrase in (
            "actually ignore",
            "ignore that",
            "forget that",
            "disregard",
            "correction:",
            "instead use",
            "switch to",
            "actually use",
        )
    )


def _keywords_from_correction(text: str) -> set[str]:
    low = text.lower()
    after = re.split(r"ignore|forget|disregard", low, maxsplit=1)
    if len(after) < 2:
        return set()
    words = re.findall(r"[a-z][a-z0-9_-]{1,}", after[1])
    stop = {"that", "this", "about", "please", "instead", "use", "the"}
    return {w for w in words if w not in stop}


def _looks_like_substantive_answer(text: str) -> bool:
    return len(text) > 40 and not text.lower().startswith(("sure", "yes", "ok"))


def _trim_fact(text: str, limit: int = CHAT_MEMORY_MAX_FACT_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "…"
    return text


def format_transcript_for_llm(messages: list[ChatMessage]) -> str:
    """Render history as numbered Student/Tutor lines for node prompts."""
    if not messages:
        return ""
    parts: list[str] = []
    for i, m in enumerate(messages, start=1):
        label = "Student" if m["role"] == "user" else "Tutor"
        parts.append(f"[{i}] {label}: {m['content']}")
    return "\n\n".join(parts)


def format_memory_for_llm(memory: ConversationMemory | None) -> str:
    """Render compressed memory for prompts with its grounding boundary."""
    if not memory:
        return ""
    parts = [
        "COMPRESSED CONVERSATION MEMORY (session context only; not source evidence):",
        memory.get("source_grounding", ""),
    ]
    facts = memory.get("facts") or []
    if facts:
        parts.append("Structured memory facts:")
        parts.extend(f"- {f.get('type', 'fact')}: {f.get('text', '')}" for f in facts)
    corrections = memory.get("corrections") or []
    if corrections:
        parts.append("Corrections to honour:")
        parts.extend(f"- {c}" for c in corrections)
    summary = memory.get("summary")
    if summary:
        parts.append("Compressed older-turn summary:")
        parts.append(summary)
    return "\n".join(p for p in parts if p)


def get_conversation_memory(state: AgentState) -> ConversationMemory | None:
    """Safe read of compressed ``conversation_memory`` from graph state."""
    memory = state.get("conversation_memory")
    return memory if isinstance(memory, dict) else None


def build_retrieval_query(
    current_query: str,
    messages: list[ChatMessage],
    memory: ConversationMemory | None = None,
) -> str:
    """Prefer the latest user question for embedding; add tutor excerpt on follow-ups.

    Concatenating a **short** prior answer helps disambiguate "Explain that further"
    without drowning the vector query in an entire prior diagram or IRAC block.
    """
    memory_block = _format_memory_for_retrieval(memory)
    if not messages:
        if memory_block:
            return (
                "[Conversation follow-up — retrieve material relevant to the "
                "compressed session memory and the new question.]\n\n"
                f"{memory_block}\n\nNew student question:\n{current_query}"
            )
        return current_query
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] != "assistant":
            continue
        excerpt = messages[i]["content"][:CHAT_HISTORY_MAX_ASSISTANT_TAIL_CHARS].strip()
        if not excerpt:
            break
        memory_prefix = f"{memory_block}\n\n" if memory_block else ""
        return (
            "[Conversation follow-up — retrieve material relevant to both the prior "
            "answer and the new question.]\n\n"
            f"{memory_prefix}"
            f"Prior tutor answer (excerpt):\n{excerpt}\n\n"
            f"New student question:\n{current_query}"
        )
    if memory_block:
        return (
            "[Conversation follow-up — retrieve material relevant to the "
            "compressed session memory and the new question.]\n\n"
            f"{memory_block}\n\nNew student question:\n{current_query}"
        )
    return current_query


def _format_memory_for_retrieval(memory: ConversationMemory | None) -> str:
    if not memory:
        return ""
    facts = [
        f.get("text", "")
        for f in memory.get("facts", [])
        if f.get("type") in {"jurisdiction", "shorthand", "authority_discussed", "study_goal"}
    ][:10]
    facts = [fact for fact in facts if fact]
    if not facts:
        return ""
    return "Compressed session memory anchors:\n" + "\n".join(f"- {f}" for f in facts)


def get_chat_history(state: AgentState) -> list[ChatMessage]:
    """Safe read of ``chat_history`` from graph state."""
    h = state.get("chat_history")
    return h if isinstance(h, list) else []


_REWRITER_SYSTEM = (
    "You rewrite a student's follow-up message into a self-contained legal "
    "search query. The query is fed into a vector + BM25 retriever — it must "
    "carry every referent the search needs (case names, doctrines, statutory "
    "section numbers) so that none of them have to be inferred from a "
    "transcript.\n\n"
    "Rules:\n"
    "1. Resolve pronouns and demonstratives (\"it\", \"that case\", \"the "
    "second one\") using the transcript.\n"
    "2. Expand abbreviated case names to their full form when the transcript "
    "shows the full name earlier.\n"
    "3. Keep the user's *intent verb* (\"explain\", \"summarise\", \"give me "
    "the ratio\") intact.\n"
    "4. Output ONE sentence (no preamble, no markdown).\n"
    "5. If the latest message is already self-contained, return it unchanged."
)


def rewrite_followup_query(
    current_query: str,
    messages: list[ChatMessage],
    *,
    memory: ConversationMemory | None = None,
    llm_call=None,
    model: str | None = None,
) -> str:
    """LLM-based coreference rewriter for follow-up retrieval.

    Resolves pronouns / expands abbreviated case names so the retriever sees
    a self-contained query. ``llm_call`` and ``model`` are injectable for
    tests; production callers pass the real ``llm.llm_call`` and
    ``ROUTER_MODEL``. Falls back to ``current_query`` on any failure.
    """
    if not (messages or memory) or not current_query.strip():
        return current_query
    # Lazy import to keep this module's surface small and break a circular
    # import (src.llm imports from src.config which is fine, but tests may
    # want to avoid network).
    if llm_call is None:
        from src.llm import llm_call as _llm_call

        llm_call = _llm_call
    if model is None:
        from src.config import ROUTER_MODEL

        model = ROUTER_MODEL

    transcript = format_transcript_for_llm(messages)
    memory_block = format_memory_for_llm(memory)
    memory_prefix = f"{memory_block}\n\n" if memory_block else ""
    prompt = (
        f"{memory_prefix}"
        f"RECENT CONVERSATION SO FAR:\n{transcript}\n\n"
        f"LATEST STUDENT MESSAGE:\n{current_query}\n\n"
        "Rewrite the latest message as a standalone search query."
    )
    try:
        rewritten = llm_call(prompt, model=model, system_instruction=_REWRITER_SYSTEM)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("query rewriter failed (%s); using raw query", exc)
        return current_query
    # Cap to a single line — the rewriter is told to return one sentence but
    # we don't fully trust it. Strip everything past the first newline before
    # cleaning quotes so wrapping quotes still strip when the model adds
    # trailing junk.
    rewritten = (rewritten or "").strip().split("\n", 1)[0].strip()
    rewritten = rewritten.strip('"').strip("'").strip()
    if not rewritten:
        return current_query
    return rewritten
