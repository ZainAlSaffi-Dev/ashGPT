"""LocalBlobStore tests."""

from __future__ import annotations

import pytest

from src.storage.blob import LocalBlobStore, _safe_filename


def test_safe_filename_strips_unsafe_chars():
    assert _safe_filename("hello.pdf") == "hello.pdf"
    # Slashes are replaced; dots in filenames are kept (path traversal is prevented
    # by joining under a controlled prefix, not by stripping dots).
    assert _safe_filename("../../etc/passwd") == ".._.._etc_passwd"
    assert _safe_filename("week 1 — adverse possession.docx") == "week_1___adverse_possession.docx"
    assert _safe_filename("") == "upload"


def test_local_put_then_read_then_delete(tmp_path):
    store = LocalBlobStore(root=str(tmp_path))
    key = store.put("usr_demo", "notes.pdf", b"%PDF-1.4 hello")
    assert key.startswith("usr_demo/")
    assert key.endswith("notes.pdf")
    assert store.exists(key)
    assert store.read(key) == b"%PDF-1.4 hello"
    p = store.open_path(key)
    assert str(tmp_path) in p
    store.delete(key)
    assert not store.exists(key)


def test_presign_put_is_a_url_pointing_at_local_endpoint(tmp_path):
    store = LocalBlobStore(root=str(tmp_path))
    url, key = store.presign_put("usr_demo", "notes.pdf", "application/pdf")
    assert url.startswith("/uploads/local/")
    assert key.startswith("usr_demo/")
    assert url.endswith(key)


def test_delete_missing_is_noop(tmp_path):
    store = LocalBlobStore(root=str(tmp_path))
    store.delete("usr_demo/zzz/missing.pdf")  # should not raise


@pytest.mark.parametrize(
    "name,expected_suffix",
    [
        ("contract.docx", ".docx"),
        ("slide.png", ".png"),
        ("notes (final).md", ".md"),
    ],
)
def test_put_preserves_extension(tmp_path, name, expected_suffix):
    store = LocalBlobStore(root=str(tmp_path))
    key = store.put("usr", name, b"x")
    assert key.endswith(expected_suffix)
