"""Clerk JWT verification.

Verifies session tokens issued by Clerk against the project's JWKS endpoint
(``<CLERK_ISSUER>/.well-known/jwks.json``). Returns the Clerk user id (``sub``)
on success.

Dev bypass: when ``settings.dev_auth_user`` is set, requests carrying header
``X-Dev-User: <id>`` are treated as that user. This lets local docker compose
and tests skip the Clerk round-trip without bypassing auth in production
(``dev_auth_user`` is empty in prod env files).

Example (verification):
    >>> verifier = ClerkVerifier(issuer="https://abc.clerk.accounts.dev")  # doctest: +SKIP
    >>> claims = verifier.verify("eyJhbGc...")                              # doctest: +SKIP
    >>> claims["sub"]                                                       # doctest: +SKIP
    'user_2abcdef...'
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient


class AuthError(Exception):
    """Raised when a token is missing, malformed, or invalid."""


@dataclass
class ClerkClaims:
    user_id: str
    email: str | None
    raw: dict[str, Any]


class ClerkVerifier:
    """JWKS-backed verifier. JWKS is cached in-process for the TTL of the
    ``PyJWKClient`` (default ~5 minutes for the underlying urllib cache)."""

    def __init__(
        self,
        issuer: str,
        jwks_url: str | None = None,
        audience: str | None = None,
        leeway_seconds: int = 30,
    ):
        if not issuer:
            raise ValueError("Clerk issuer required")
        self._issuer = issuer.rstrip("/")
        self._jwks_url = jwks_url or f"{self._issuer}/.well-known/jwks.json"
        self._jwks = PyJWKClient(self._jwks_url, cache_jwk_set=True, lifespan=300)
        self._audience = audience
        self._leeway = leeway_seconds

    def verify(self, token: str) -> ClerkClaims:
        if not token:
            raise AuthError("missing token")
        try:
            signing_key = self._jwks.get_signing_key_from_jwt(token).key
        except Exception as e:
            raise AuthError(f"jwks lookup failed: {e}") from e
        try:
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                issuer=self._issuer,
                audience=self._audience,
                leeway=self._leeway,
                options={
                    "require": ["exp", "iat", "sub"],
                    "verify_aud": self._audience is not None,
                },
            )
        except jwt.PyJWTError as e:
            raise AuthError(f"jwt invalid: {e}") from e

        sub = claims.get("sub")
        if not sub:
            raise AuthError("token missing sub claim")
        return ClerkClaims(user_id=sub, email=claims.get("email"), raw=claims)


# ── Dev bypass + FastAPI dependency ───────────────────────────────────────────


_verifier: ClerkVerifier | None = None


def _get_verifier() -> ClerkVerifier:
    global _verifier
    from src.config import get_settings

    if _verifier is None:
        s = get_settings()
        if not s.clerk_issuer:
            raise AuthError("Clerk issuer not configured")
        _verifier = ClerkVerifier(issuer=s.clerk_issuer, jwks_url=s.clerk_jwks_url)
    return _verifier


def reset_verifier() -> None:
    """Drop the cached verifier (used by tests after env changes)."""
    global _verifier
    _verifier = None


def extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


async def require_user(
    authorization: str | None = None,
    x_dev_user: str | None = None,
) -> ClerkClaims:
    """FastAPI-style dependency. Pure-function for easy unit testing.

    Resolution order:
      1. If ``dev_auth_user`` is set in settings and ``X-Dev-User`` matches, bypass.
      2. Otherwise verify the Bearer token against Clerk JWKS.
    """
    from src.config import get_settings

    s = get_settings()
    if s.dev_auth_user and x_dev_user and x_dev_user == s.dev_auth_user:
        return ClerkClaims(user_id=x_dev_user, email=None, raw={"sub": x_dev_user, "dev": True})

    token = extract_bearer(authorization)
    if not token:
        raise AuthError("missing Authorization: Bearer <token>")
    return _get_verifier().verify(token)
