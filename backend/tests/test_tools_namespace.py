"""Unit tests for namespace handling in src.agent.tools._build_filter.

These don't load Chroma — they exercise the filter-builder pure function.
"""

from __future__ import annotations

from src.agent.tools import _build_filter
from src.agent.scope import RetrievalScope


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


def test_project_folder_file_scope_filters_combine_with_namespace():
    scope = RetrievalScope(
        type="folder",
        project_id="proj_1",
        folder_id="fold_1",
        file_ids=("file_a", "file_b"),
        explicit=True,
    )
    f = _build_filter(namespace="alice", scope=scope)
    assert f == {
        "$and": [
            {"namespace": {"$eq": "alice"}},
            {"project_id": {"$eq": "proj_1"}},
            {"folder_id": {"$eq": "fold_1"}},
            {"file_id": {"$in": ["file_a", "file_b"]}},
        ]
    }


def test_explicit_empty_files_scope_is_not_all_library():
    scope = RetrievalScope(type="files", file_ids=(), explicit=True)
    assert scope.is_explicit_empty()


def test_missing_required_explicit_scope_is_not_all_library():
    assert RetrievalScope.from_input({"type": "project"}).is_explicit_empty()
    assert RetrievalScope.from_input({"type": "folder"}).is_explicit_empty()
    assert RetrievalScope.from_input({"type": "week"}).is_explicit_empty()
    assert RetrievalScope.from_input({"type": "doc_type", "doc_types": []}).is_explicit_empty()
    assert RetrievalScope.from_input({"type": "bogus"}).is_explicit_empty()
