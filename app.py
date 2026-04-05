"""Streamlit frontend for the ashGPT Property Law Exam Assistant.

Provides an interactive chat interface with:
  - Intent classification display
  - IRAC analysis rendering
  - Mermaid.js diagram rendering
  - Source attribution panel
  - Node trace visualisation

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import re
import time

import streamlit as st
from dotenv import load_dotenv
from streamlit_mermaid import st_mermaid

load_dotenv()

from src.agent.graph import run_query


def render_message(content: str) -> None:
    """Render a message, splitting out any ```mermaid blocks for st_mermaid.

    Plain markdown segments are passed to st.markdown; Mermaid blocks are
    rendered with st_mermaid so they display as interactive diagrams both
    on first render and when replayed from session history.
    """
    parts = re.split(r"(```mermaid\n.*?```)", content, flags=re.DOTALL)
    for part in parts:
        m = re.fullmatch(r"```mermaid\n(.*?)```", part, flags=re.DOTALL)
        if m:
            st_mermaid(m.group(1).strip(), height="auto")
        elif part.strip():
            st.markdown(part)

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ashGPT — Property Law Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚖️ ashGPT")
    st.caption("Property Law Exam Assistant")
    st.divider()

    st.subheader("Filters")
    week_options = ["All Weeks"] + [f"week_{i}" for i in range(1, 7)]
    selected_week = st.selectbox("Restrict to week:", week_options)
    week_filter = None if selected_week == "All Weeks" else selected_week

    st.divider()
    st.subheader("About")
    st.markdown(
        "**Design Hypothesis:** Does separating *ratio decidendi* extraction "
        "from chronological fact-extraction (Mermaid.js) produce more accurate "
        "causal analyses than a single-prompt baseline?"
    )
    st.caption("INFS4205 Systems Engineering Project")

    if st.button("Clear conversation", help="Reset session memory for this browser tab"):
        st.session_state.messages = []
        st.rerun()

# ── Chat History ───────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Property Law Exam Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_message(msg["content"])

# ── Chat Input ─────────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask a property law question...")

if prompt:
    chat_prior = [dict(m) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            result = run_query(
                prompt,
                week_filter=week_filter,
                chat_history=chat_prior or None,
            )
            elapsed = time.time() - start

        # ── Intent & Trace Banner ──────────────────────────────────────────
        intent = result.get("intent", "general")
        trace = result.get("node_trace", [])

        col1, col2, col3 = st.columns(3)
        col1.metric("Intent", intent)
        col2.metric("Latency", f"{elapsed:.1f}s")
        col3.metric("Nodes", " → ".join(trace))

        st.divider()

        # ── Final Answer (with inline Mermaid rendering) ───────────────────
        final_answer = result.get("final_answer", "")
        mermaid = result.get("mermaid_diagram", "")

        # Build the stored content (diagram embedded so history replays it)
        response_content = final_answer
        if mermaid and f"```mermaid" not in final_answer:
            # Synthesis didn't include the block inline — append it so the
            # stored message and the expander both have the code available
            response_content += f"\n\n```mermaid\n{mermaid}\n```"

        render_message(response_content)

        # ── Raw code expander (only when a diagram exists) ─────────────────
        if mermaid:
            with st.expander("Raw Mermaid code (copy for mermaid.live)"):
                st.code(mermaid, language="text")

        # ── IRAC Analysis ─────────────────────────────────────────────────
        irac = result.get("irac_analysis", "")
        if irac:
            with st.expander("Full IRAC Analysis", expanded=False):
                st.markdown(irac)

        # ── Sources Panel ──────────────────────────────────────────────────
        texts = result.get("retrieved_texts", [])
        slides = result.get("retrieved_slides", [])
        if texts or slides:
            with st.expander(
                f"Sources ({len(texts)} texts, {len(slides)} slides)", expanded=False
            ):
                if texts:
                    st.markdown("**Text Sources:**")
                    for t in texts:
                        st.markdown(
                            f"- `[{t['week']}]` **{t['source']}** "
                            f"({t['doc_type']}, {len(t['content'])} chars)"
                        )
                if slides:
                    st.markdown("**Slide Sources:**")
                    for s in slides:
                        st.markdown(f"- `[{s['week']}]` **{s['source']}**")

    st.session_state.messages.append({"role": "assistant", "content": response_content})
