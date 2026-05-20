"""Multi-turn chat helpers: trim, validate, format history, and pack retrieval queries.

``chat_history`` in :class:`AgentState` holds **prior** turns only; the current user
utterance always lives in ``query``. This keeps retrieval embeddings focused while
still giving the router and synthesis models conversational context.
"""

from __future__ import annotations

import logging

from src.agent.state import AgentState, ChatMessage
from src.config import (
    CHAT_HISTORY_MAX_ASSISTANT_TAIL_CHARS,
    CHAT_HISTORY_MAX_CHARS_PER_MESSAGE,
    CHAT_HISTORY_MAX_MESSAGES,
)

log = logging.getLogger(__name__)


def prepare_chat_history_for_run(
    raw: list[dict[str, str]] | None,
) -> tuple[list[ChatMessage], dict]:
    """Normalise roles, strip empties, truncate long messages, cap turn count.

    Returns the prepared list **and** an overflow report so the caller can
    surface it to the user / logs ("3 older turns trimmed to fit context").
    The report shape is ``{"dropped_turns": int, "truncated_messages": int}``.
    """
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
        out.append({"role": role, "content": content})
    if len(out) > CHAT_HISTORY_MAX_MESSAGES:
        overflow["dropped_turns"] = len(out) - CHAT_HISTORY_MAX_MESSAGES
        out = out[-CHAT_HISTORY_MAX_MESSAGES:]
    return out, overflow


def format_transcript_for_llm(messages: list[ChatMessage]) -> str:
    """Render history as numbered Student/Tutor lines for node prompts."""
    if not messages:
        return ""
    parts: list[str] = []
    for i, m in enumerate(messages, start=1):
        label = "Student" if m["role"] == "user" else "Tutor"
        parts.append(f"[{i}] {label}: {m['content']}")
    return "\n\n".join(parts)


def build_retrieval_query(current_query: str, messages: list[ChatMessage]) -> str:
    """Prefer the latest user question for embedding; add tutor excerpt on follow-ups.

    Concatenating a **short** prior answer helps disambiguate "Explain that further"
    without drowning the vector query in an entire prior diagram or IRAC block.
    """
    if not messages:
        return current_query
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] != "assistant":
            continue
        excerpt = messages[i]["content"][:CHAT_HISTORY_MAX_ASSISTANT_TAIL_CHARS].strip()
        if not excerpt:
            break
        return (
            "[Conversation follow-up — retrieve material relevant to both the prior "
            "answer and the new question.]\n\n"
            f"Prior tutor answer (excerpt):\n{excerpt}\n\n"
            f"New student question:\n{current_query}"
        )
    return current_query


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
    llm_call=None,
    model: str | None = None,
) -> str:
    """LLM-based coreference rewriter for follow-up retrieval.

    Resolves pronouns / expands abbreviated case names so the retriever sees
    a self-contained query. ``llm_call`` and ``model`` are injectable for
    tests; production callers pass the real ``llm.llm_call`` and
    ``ROUTER_MODEL``. Falls back to ``current_query`` on any failure.
    """
    if not messages or not current_query.strip():
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
    prompt = (
        f"CONVERSATION SO FAR:\n{transcript}\n\n"
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
