"""Blob storage adapter — raw user uploads.

Two impls: ``LocalBlobStore`` (filesystem under ``settings.blob_local_root``)
and ``R2BlobStore`` (Cloudflare R2 over the S3-compatible API via boto3). Both
expose the same surface so callers do not care which is active.

Example:
    >>> store = make_blob_store()
    >>> key = store.put("usr_demo", "notes.pdf", b"%PDF-1.4 ...")  # doctest: +SKIP
    >>> store.exists(key)                                          # doctest: +SKIP
    True
    >>> store.read(key)[:5]                                        # doctest: +SKIP
    b'%PDF-'
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol


class BlobStore(Protocol):
    def put(self, user_id: str, name: str, data: bytes) -> str: ...

    def read(self, key: str) -> bytes: ...

    def open_path(self, key: str) -> str:
        """Return a local filesystem path for ``key`` (downloads if remote)."""
        ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None: ...

    def presign_put(self, user_id: str, name: str, mime: str, expires_seconds: int = 900) -> tuple[str, str]:
        """Return ``(upload_url, key)`` for direct client → blob uploads."""
        ...


# ── Local filesystem ──────────────────────────────────────────────────────────


class LocalBlobStore:
    """Stores blobs under ``<root>/<user_id>/<random>/<name>``."""

    def __init__(self, root: str):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _abs(self, key: str) -> Path:
        return self._root / key

    def put(self, user_id: str, name: str, data: bytes) -> str:
        import uuid

        safe_name = _safe_filename(name)
        key = f"{user_id}/{uuid.uuid4().hex}/{safe_name}"
        dst = self._abs(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return key

    def read(self, key: str) -> bytes:
        return self._abs(key).read_bytes()

    def open_path(self, key: str) -> str:
        return str(self._abs(key))

    def exists(self, key: str) -> bool:
        return self._abs(key).exists()

    def delete(self, key: str) -> None:
        p = self._abs(key)
        if p.exists():
            p.unlink()
            # remove empty parent dirs up to the root
            for parent in [p.parent, p.parent.parent]:
                try:
                    if parent.is_dir() and not any(parent.iterdir()):
                        parent.rmdir()
                except OSError:
                    break

    def presign_put(
        self, user_id: str, name: str, mime: str, expires_seconds: int = 900
    ) -> tuple[str, str]:
        """For local, return a relative URL that the FastAPI app will accept via
        a streaming POST endpoint (``/uploads/local/{key}``). Direct uploads
        without going through the backend are not supported locally."""
        import uuid

        safe_name = _safe_filename(name)
        key = f"{user_id}/{uuid.uuid4().hex}/{safe_name}"
        # The actual ingestion endpoint will be /uploads/local/{key}.
        return (f"/uploads/local/{key}", key)


# ── Cloudflare R2 (S3-compatible) ─────────────────────────────────────────────


class R2BlobStore:
    """R2 via boto3 + AWS Signature V4 (S3 compatible)."""

    def __init__(
        self,
        bucket: str,
        account_id: str,
        access_key: str,
        secret_key: str,
        endpoint_url: str = "",
    ):
        import boto3
        from botocore.config import Config

        endpoint = endpoint_url or f"https://{account_id}.r2.cloudflarestorage.com"
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4", region_name="auto"),
        )

    def put(self, user_id: str, name: str, data: bytes) -> str:
        import uuid

        key = f"{user_id}/{uuid.uuid4().hex}/{_safe_filename(name)}"
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
        return key

    def read(self, key: str) -> bytes:
        obj = self._client.get_object(Bucket=self._bucket, Key=key)
        return obj["Body"].read()

    def open_path(self, key: str) -> str:
        """Download to a tempfile and return the path."""
        import tempfile

        suffix = Path(key).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        self._client.download_file(self._bucket, key, tmp.name)
        return tmp.name

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception as e:
            # Distinguish "object not yet uploaded" (404 — normal during the
            # poll window after presigned PUT) from auth / misconfig errors
            # (401/403 — needs operator action). The former is silent; the
            # latter is shouted so the next /process attempt's logs reveal
            # exactly why the HEAD failed.
            import logging

            log = logging.getLogger(__name__)
            err = getattr(e, "response", {}).get("Error", {}) if hasattr(e, "response") else {}
            code = err.get("Code") or err.get("HTTPStatusCode") or ""
            status_code = getattr(getattr(e, "response", None), "get", lambda *_: None)(
                "ResponseMetadata", {}
            ).get("HTTPStatusCode", "")
            msg = err.get("Message") or str(e)
            if str(code) in ("404", "NoSuchKey", "NotFound") or status_code == 404:
                log.info("R2 HEAD miss bucket=%s key=%s", self._bucket, key)
            else:
                log.error(
                    "R2 HEAD failed bucket=%s key=%s code=%s status=%s msg=%s",
                    self._bucket, key, code, status_code, msg,
                )
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def presign_put(
        self, user_id: str, name: str, mime: str, expires_seconds: int = 900
    ) -> tuple[str, str]:
        import uuid

        key = f"{user_id}/{uuid.uuid4().hex}/{_safe_filename(name)}"
        url = self._client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self._bucket, "Key": key, "ContentType": mime},
            ExpiresIn=expires_seconds,
        )
        return url, key


def _safe_filename(name: str) -> str:
    keep = "-_.()[] "
    cleaned = "".join(c if c.isalnum() or c in keep else "_" for c in name)
    return cleaned.strip().replace(" ", "_") or "upload"


def make_blob_store() -> BlobStore:
    from src.config import get_settings

    s = get_settings()
    if s.blob_backend == "local":
        return LocalBlobStore(s.blob_local_root)
    if s.blob_backend == "r2":
        # Fail loud at construction time if R2 creds are missing — otherwise
        # boto3 silently builds a client that 401s on every call, which
        # surfaces downstream as "blob missing" 409s and hours of debugging.
        missing = [
            n for n, v in [
                ("r2_bucket", s.r2_bucket),
                ("r2_account_id", s.r2_account_id),
                ("r2_access_key", s.r2_access_key),
                ("r2_secret_key", s.r2_secret_key),
            ]
            if not v
        ]
        if missing:
            raise RuntimeError(
                f"R2 backend selected but missing env vars: {missing}. "
                "Set R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY (or R2_ACCESS_KEY / "
                "R2_SECRET_KEY) and R2_ACCOUNT_ID / R2_BUCKET."
            )
        return R2BlobStore(
            bucket=s.r2_bucket,
            account_id=s.r2_account_id,
            access_key=s.r2_access_key,
            secret_key=s.r2_secret_key,
            endpoint_url=s.r2_endpoint_url,
        )
    raise ValueError(f"unknown blob backend: {s.blob_backend}")
