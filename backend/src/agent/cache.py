"""Semantic answer cache.

Avoids re-running the full pipeline when the user re-asks an identical or
near-identical question against the same retrieved context. Key is a hash of:

  (user_id, normalised_query, sorted retrieved-chunk-ids)

so two queries that retrieve the same chunks share a cache entry. TTL 7 days.

Example:
    >>> k = make_cache_key("usr_demo", " What is adverse possession? ", ["c2","c1"])
    >>> isinstance(k, str) and len(k) == 64
    True
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from sqlalchemy import select

from src.storage.db import AnswerCache, get_engine, get_session

log = logging.getLogger(__name__)

DEFAULT_TTL = timedelta(days=7)


_WS_RE = re.compile(r"\s+")


def normalise_query(q: str) -> str:
    """Collapse whitespace, lowercase, strip terminal punctuation. Stable across
    incidental phrasing differences without losing semantic meaning."""
    if not q:
        return ""
    cleaned = _WS_RE.sub(" ", q).strip().lower()
    cleaned = cleaned.rstrip(".?!,;:")
    return cleaned


def make_cache_key(
    user_id: str, query: str, chunk_ids: Iterable[str], scope_hash: str | None = None
) -> str:
    """SHA-256 of namespace + normalised query + canonicalised chunk-id set."""
    ids = sorted({str(c) for c in chunk_ids if c})
    raw = f"{user_id}\x1f{normalise_query(query)}\x1f{scope_hash or 'all'}\x1f{'|'.join(ids)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def get(cache_key: str) -> dict | None:
    """Return the cached payload dict (with ``answer`` merged in) or None."""
    async with get_session(get_engine()) as session:
        row = (
            await session.execute(
                select(AnswerCache).where(AnswerCache.cache_key == cache_key)
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        # SQLite drops tzinfo on roundtrip. Coerce both sides to UTC for the
        # comparison; Postgres in prod preserves it so this is harmless there.
        expires = row.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires < datetime.now(timezone.utc):
            await session.delete(row)
            await session.commit()
            return None
        return {"answer": row.answer, **(row.payload or {})}


async def put(
    cache_key: str,
    user_id: str,
    answer: str,
    payload: dict[str, Any],
    ttl: timedelta = DEFAULT_TTL,
    scope_hash: str | None = None,
) -> None:
    """Upsert a cache entry."""
    expires = datetime.now(timezone.utc) + ttl
    async with get_session(get_engine()) as session:
        existing = (
            await session.execute(
                select(AnswerCache).where(AnswerCache.cache_key == cache_key)
            )
        ).scalar_one_or_none()
        if existing:
            existing.answer = answer
            existing.payload = payload
            existing.scope_hash = scope_hash
            existing.expires_at = expires
        else:
            session.add(
                AnswerCache(
                    cache_key=cache_key,
                    user_id=user_id,
                    scope_hash=scope_hash,
                    answer=answer,
                    payload=payload,
                    expires_at=expires,
                )
            )
        await session.commit()


async def invalidate_user(user_id: str) -> int:
    """Drop all cache entries for ``user_id`` (e.g. after file delete)."""
    from sqlalchemy import delete

    async with get_session(get_engine()) as session:
        result = await session.execute(
            delete(AnswerCache).where(AnswerCache.user_id == user_id)
        )
        await session.commit()
        return result.rowcount or 0
