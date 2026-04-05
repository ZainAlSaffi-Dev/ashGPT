"""LangGraph state definition for the Property Law Exam Assistant.

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

    # ── Diagnostics (for evaluation / ablation) ────────────────────────────
    node_trace: list[str]
