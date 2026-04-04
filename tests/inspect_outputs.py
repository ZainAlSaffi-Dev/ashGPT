"""Interactive output inspection script.

Run this to vet the quality of each node's output before moving to Phase 5.
It runs three representative queries (one per intent type) and prints
everything the pipeline produces in a human-readable format.

Usage:
    python -m tests.inspect_outputs
    python -m tests.inspect_outputs --query "your own question here"
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

from src.agent.graph import run_query


def _divider(title: str) -> str:
    return f"\n{'=' * 70}\n  {title}\n{'=' * 70}"


def _section(label: str, content: str, wrap: bool = True) -> None:
    print(f"\n--- {label} ---")
    if not content:
        print("  (empty)")
        return
    if wrap:
        print(textwrap.indent(textwrap.fill(content, width=80), "  "))
    else:
        print(textwrap.indent(content, "  "))


def inspect_query(query: str, week: str | None = None) -> None:
    """Run a query and display all pipeline outputs."""
    print(_divider(f"QUERY: {query}"))
    if week:
        print(f"  Week filter: {week}")

    result = run_query(query, week_filter=week)

    # Routing
    print(f"\n  Intent:     {result.get('intent', '?')}")
    print(f"  Week:       {result.get('week_filter', 'None')}")
    print(f"  Node trace: {result.get('node_trace', [])}")

    # Retrieval
    texts = result.get("retrieved_texts", [])
    slides = result.get("retrieved_slides", [])
    print(f"\n  Retrieved:  {len(texts)} text chunks, {len(slides)} slides")
    if texts:
        print("  Text sources:")
        for t in texts:
            print(f"    - [{t['week']}] {t['doc_type']}: {t['source']} ({len(t['content'])} chars)")
    if slides:
        print("  Slide sources:")
        for s in slides:
            print(f"    - [{s['week']}] {s['source']}")

    # Ratio Extractor
    ratio = result.get("ratio_decidendi", "")
    irac = result.get("irac_analysis", "")
    if ratio or irac:
        _section("RATIO DECIDENDI", ratio)
        _section("FULL IRAC ANALYSIS", irac, wrap=False)

    # Chronology
    mermaid = result.get("mermaid_diagram", "")
    chrono = result.get("chronology_summary", "")
    if mermaid or chrono:
        _section("MERMAID DIAGRAM", f"```mermaid\n{mermaid}\n```" if mermaid else "", wrap=False)
        _section("TIMELINE SUMMARY", chrono)

    # Final answer
    _section("FINAL ANSWER", result.get("final_answer", ""), wrap=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect pipeline outputs")
    parser.add_argument("--query", type=str, help="Run a single custom query")
    parser.add_argument("--week", type=str, default=None, help="Week filter (e.g. week_3)")
    parser.add_argument("--all", action="store_true", help="Run all three sample queries")
    args = parser.parse_args()

    if args.query:
        inspect_query(args.query, week=args.week)
        return

    if args.all:
        samples = [
            ("What is the ratio decidendi for adverse possession?", "week_3"),
            ("Show me the timeline of events from the week 3 readings", None),
            ("Summarise the key concepts from week 1", None),
        ]
        for query, week in samples:
            inspect_query(query, week=week)
        return

    # Default: interactive mode
    print("ashGPT Output Inspector")
    print("Type a question (or 'quit' to exit).\n")
    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        week = input("Week filter (leave blank for none): ").strip() or None
        inspect_query(query, week=week)


if __name__ == "__main__":
    main()
