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

import time

import streamlit as st
from dotenv import load_dotenv
from streamlit_mermaid import st_mermaid

load_dotenv()

from src.agent.graph import run_query

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

# ── Chat History ───────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Property Law Exam Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Chat Input ─────────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask a property law question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            result = run_query(prompt, week_filter=week_filter)
            elapsed = time.time() - start

        # ── Intent & Trace Banner ──────────────────────────────────────────
        intent = result.get("intent", "general")
        trace = result.get("node_trace", [])

        col1, col2, col3 = st.columns(3)
        col1.metric("Intent", intent)
        col2.metric("Latency", f"{elapsed:.1f}s")
        col3.metric("Nodes", " → ".join(trace))

        st.divider()

        # ── Final Answer ───────────────────────────────────────────────────
        final_answer = result.get("final_answer", "")
        st.markdown(final_answer)

        # ── Mermaid Diagram ────────────────────────────────────────────────
        mermaid = result.get("mermaid_diagram", "")
        if mermaid:
            st.divider()
            st.subheader("Chronological Flowchart")
            st_mermaid(mermaid, height="auto")

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

    response_content = final_answer
    if mermaid:
        response_content += f"\n\n```mermaid\n{mermaid}\n```"
    st.session_state.messages.append({"role": "assistant", "content": response_content})
