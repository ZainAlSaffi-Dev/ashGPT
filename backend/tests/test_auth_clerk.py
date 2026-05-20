"""Clerk auth: dev bypass + bearer parsing. JWKS verification is mocked
because production JWKS depends on Clerk-issued keys."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.auth.clerk import AuthError, ClerkClaims, extract_bearer, require_user
from src.config import reload_settings


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    # Clear any prior settings cache.
    monkeypatch.delenv("DEV_AUTH_USER", raising=False)
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    reload_settings()
    yield
    reload_settings()


def test_extract_bearer():
    assert extract_bearer("Bearer abc") == "abc"
    assert extract_bearer("bearer  spaced  ") == "spaced"
    assert extract_bearer("Token abc") is None
    assert extract_bearer(None) is None
    assert extract_bearer("") is None


@pytest.mark.asyncio
async def test_dev_bypass_when_header_matches(monkeypatch):
    monkeypatch.setenv("DEV_AUTH_USER", "usr_demo")
    reload_settings()
    claims = await require_user(authorization=None, x_dev_user="usr_demo")
    assert isinstance(claims, ClerkClaims)
    assert claims.user_id == "usr_demo"
    assert claims.raw.get("dev") is True


@pytest.mark.asyncio
async def test_dev_bypass_wrong_id_falls_through_to_jwt(monkeypatch):
    monkeypatch.setenv("DEV_AUTH_USER", "usr_real")
    reload_settings()
    with pytest.raises(AuthError):
        # Wrong dev user header + no bearer → must fail.
        await require_user(authorization=None, x_dev_user="usr_wrong")


@pytest.mark.asyncio
async def test_no_auth_no_bypass_raises():
    with pytest.raises(AuthError):
        await require_user(authorization=None, x_dev_user=None)


@pytest.mark.asyncio
async def test_bearer_route_invokes_verifier(monkeypatch):
    monkeypatch.setenv("CLERK_ISSUER", "https://test.clerk.accounts.dev")
    reload_settings()

    fake_claims = ClerkClaims(user_id="usr_xyz", email="x@y.z", raw={"sub": "usr_xyz"})

    with patch("src.auth.clerk._get_verifier") as mv:
        mv.return_value.verify.return_value = fake_claims
        out = await require_user(authorization="Bearer faketoken", x_dev_user=None)
        assert out.user_id == "usr_xyz"
        mv.return_value.verify.assert_called_once_with("faketoken")
