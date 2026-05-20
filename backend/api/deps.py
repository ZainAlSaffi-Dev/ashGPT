"""FastAPI dependency wrappers around auth + storage singletons."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.clerk import AuthError, ClerkClaims, require_user
from src.storage.db import User, get_engine, get_or_create_user, get_session


async def current_claims(
    authorization: Annotated[str | None, Header()] = None,
    x_dev_user: Annotated[str | None, Header(alias="X-Dev-User")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> ClerkClaims:
    # When the edge Worker has already verified the Clerk JWT it forwards the
    # resolved user via X-User-Id. The container is only reachable through the
    # Worker (Durable Object binding) so we trust that header in prod.
    if x_user_id:
        return ClerkClaims(user_id=x_user_id, email=None, raw={"sub": x_user_id, "via": "worker"})
    try:
        return await require_user(authorization=authorization, x_dev_user=x_dev_user)
    except AuthError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


async def db_session() -> AsyncSession:
    """Yield an async session bound to the configured engine."""
    engine = get_engine()
    async with get_session(engine) as s:
        yield s


async def current_user(
    claims: Annotated[ClerkClaims, Depends(current_claims)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> User:
    return await get_or_create_user(db, clerk_id=claims.user_id, email=claims.email)
