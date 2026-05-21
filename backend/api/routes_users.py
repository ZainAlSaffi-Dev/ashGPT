"""User profile + onboarding state."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.db import User

from .deps import current_user, db_session
from .schemas import UserMe

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserMe)
async def get_me(
    user: Annotated[User, Depends(current_user)],
) -> UserMe:
    return UserMe(
        id=user.id,
        clerk_id=user.clerk_id,
        email=user.email,
        onboarded_at=user.onboarded_at,
        created_at=user.created_at,
    )


@router.post("/me/onboarded", response_model=UserMe, status_code=status.HTTP_200_OK)
async def mark_onboarded(
    user: Annotated[User, Depends(current_user)],
    db: Annotated[AsyncSession, Depends(db_session)],
) -> UserMe:
    """Idempotent: first call stamps ``onboarded_at``; further calls return the existing value."""
    if user.onboarded_at is None:
        user.onboarded_at = datetime.now(timezone.utc)
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return UserMe(
        id=user.id,
        clerk_id=user.clerk_id,
        email=user.email,
        onboarded_at=user.onboarded_at,
        created_at=user.created_at,
    )
