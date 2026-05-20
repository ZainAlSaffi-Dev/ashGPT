"""Unit tests for namespace handling in src.agent.tools._build_filter.

These don't load Chroma — they exercise the filter-builder pure function.
"""

from __future__ import annotations

from src.agent.tools import _build_filter


def test_empty_filter_returns_none():
    assert _build_filter() is None


def test_week_only():
    assert _build_filter(week="week_1") == {"week": {"$eq": "week_1"}}


def test_namespace_only():
    assert _build_filter(namespace="usr_demo") == {"namespace": {"$eq": "usr_demo"}}


def test_week_and_namespace_combine_with_and():
    f = _build_filter(week="week_2", namespace="usr_demo")
    assert f == {
        "$and": [
            {"week": {"$eq": "week_2"}},
            {"namespace": {"$eq": "usr_demo"}},
        ]
    }


def test_namespace_with_multiple_doc_types():
    f = _build_filter(doc_types=["reading", "tutorial"], namespace="alice")
    assert f == {
        "$and": [
            {"type": {"$in": ["reading", "tutorial"]}},
            {"namespace": {"$eq": "alice"}},
        ]
    }


def test_namespace_falsy_skipped():
    # Empty string namespace must NOT add a filter clause.
    assert _build_filter(namespace="") is None
    assert _build_filter(week="week_1", namespace="") == {"week": {"$eq": "week_1"}}
