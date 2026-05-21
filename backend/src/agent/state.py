"""LangGraph state definition for the LawGPT study assistant.

The state flows through the graph as follows:

    User Query → Router → Retrieval → [Ratio Extractor | Chronology Generator] → Synthesis

Each node reads from and writes to specific keys, keeping the data pipeline
explicit and inspectable for evaluation and ablation.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class RetrievedDocument(TypedDict):
    """A single document retrieved from the knowledge base."""

    content: str
    source: str
    week: str
    doc_type: str
    image_path: str | None


class ChatMessage(TypedDict):
    """One turn in the session transcript (Streamlit roles map directly)."""

    role: Literal["user", "assistant"]
    content: str


class ConversationMemory(TypedDict, total=False):
    """Compressed session memory built from turns outside the recent window."""

    summary: str
    facts: list[dict[str, str]]
    corrections: list[str]
    compressed_turns: int
    source_grounding: str


class AgentState(TypedDict, total=False):
    """Full state flowing through the LangGraph pipeline.

    Using total=False so nodes only need to set the keys they produce.
    Every key is optional at graph entry; the router populates intent,
    retrieval populates documents, and downstream nodes populate their
    respective output fields.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    query: str
    week_filter: str | None
    chat_history: list[ChatMessage]  # prior turns only; current message is ``query``
    conversation_memory: ConversationMemory  # compressed older turns; never source evidence
    user_id: str | None  # tenant namespace; None ⇒ shared/legacy collection

    # ── Router output ──────────────────────────────────────────────────────
    intent: Literal["ratio", "chronology", "summary", "general"]

    # ── Retrieval output ───────────────────────────────────────────────────
    retrieved_texts: list[RetrievedDocument]
    retrieved_slides: list[RetrievedDocument]

    # ── Ratio Extractor output ─────────────────────────────────────────────
    ratio_decidendi: str
    irac_analysis: str

    # ── Chronology Generator output ────────────────────────────────────────
    mermaid_diagram: str
    chronology_summary: str

    # ── Synthesis output ───────────────────────────────────────────────────
    final_answer: str

    # ── Verification node output ───────────────────────────────────────────
    verification_report: dict

    # ── Ablation overrides (set by the eval harness; agent reads if present) ─
    use_reranker: bool  # if absent, retrieval_node falls back to USE_RERANKER

    # ── Diagnostics (for evaluation / ablation) ────────────────────────────
    node_trace: list[str]
    timings: list[dict]  # [{"node": str, "ms": float, "sub": dict | None}]
    cache_hit: bool      # True when run_query short-circuited from semantic cache

    # ── Chat-history audit ─────────────────────────────────────────────────
    # Populated by prepare_chat_history_for_run when the incoming transcript
    # was trimmed (turn-cap or per-message char-cap). UI can warn the user.
    chat_history_overflow: dict  # {"dropped_turns": int, "truncated_messages": int}
    memory_telemetry: dict  # compression stats safe to expose over SSE

    # ── Follow-up rewriter (Stage 4) ───────────────────────────────────────
    # Set by query_rewriter_node when a coreference-resolved search query is
    # produced for retrieval. Logged so eval can compare with raw query.
    rewritten_query: str

    # ── Escalation audit ───────────────────────────────────────────────────
    # When confidence-gated escalation kicks in, the synthesis node sees the
    # override here and the route layer records both fields into the saved
    # message's verification blob.
    _override_synthesis_model: str
    escalated: bool
