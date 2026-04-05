"""Multi-turn chat helpers: trim, validate, format history, and pack retrieval queries.

``chat_history`` in :class:`AgentState` holds **prior** turns only; the current user
utterance always lives in ``query``. This keeps retrieval embeddings focused while
still giving the router and synthesis models conversational context.
"""

from __future__ import annotations

from src.agent.state import AgentState, ChatMessage
from src.config import (
    CHAT_HISTORY_MAX_ASSISTANT_TAIL_CHARS,
    CHAT_HISTORY_MAX_CHARS_PER_MESSAGE,
    CHAT_HISTORY_MAX_MESSAGES,
)


def prepare_chat_history_for_run(raw: list[dict[str, str]] | None) -> list[ChatMessage]:
    """Normalise roles, strip empties, truncate long messages, cap turn count."""
    if not raw:
        return []
    out: list[ChatMessage] = []
    for m in raw:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        cap = CHAT_HISTORY_MAX_CHARS_PER_MESSAGE
        if len(content) > cap:
            content = content[:cap] + "\n… [truncated]"
        out.append({"role": role, "content": content})
    if len(out) > CHAT_HISTORY_MAX_MESSAGES:
        out = out[-CHAT_HISTORY_MAX_MESSAGES:]
    return out


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
