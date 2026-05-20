"""Shared pytest fixtures and configuration for the ashGPT test suite.

Markers:
    integration: Tests that require live API keys (GOOGLE_API_KEY, ZEMBED_API_KEY).
                 Skipped automatically if keys are missing.
    slow:        Tests that take >5 seconds (VLM calls, large embeddings).
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: requires live API keys")
    config.addinivalue_line("markers", "slow: takes >5 seconds")


def _has_api_key(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


@pytest.fixture(autouse=True)
def _skip_integration_without_keys(request: pytest.FixtureRequest) -> None:
    """Auto-skip integration tests when the required API keys are absent."""
    marker = request.node.get_closest_marker("integration")
    if marker is None:
        return
    missing = [
        k for k in ("ZEMBED_API_KEY", "GOOGLE_API_KEY")
        if not _has_api_key(k)
    ]
    if missing:
        pytest.skip(f"Missing API key(s): {', '.join(missing)}")


@pytest.fixture(scope="session")
def zembed_api_key() -> str:
    key = os.getenv("ZEMBED_API_KEY", "")
    if not key:
        pytest.skip("ZEMBED_API_KEY not set")
    return key
