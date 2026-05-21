"""Shared pytest fixtures and configuration for the ashGPT test suite.

Markers:
    integration: Tests that talk to live services (Cohere embed/rerank, Anthropic,
                 a populated Postgres+pgvector). Auto-skipped unless the relevant
                 API keys are set AND ``RUN_INTEGRATION_TESTS=1`` is exported.
    slow:        Tests that take >5 seconds (VLM calls, large embeddings).
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: requires live services")
    config.addinivalue_line("markers", "slow: takes >5 seconds")


def _has_api_key(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


# Integration tests need a populated DB + the modern provider keys (Cohere for
# embed/rerank, Anthropic for synthesis). Even with keys set, contributors
# without a seeded Postgres instance shouldn't see noisy failures, so we
# additionally require an explicit ``RUN_INTEGRATION_TESTS=1`` opt-in.
_INTEGRATION_KEYS = ("COHERE_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture(autouse=True)
def _skip_integration_without_keys(request: pytest.FixtureRequest) -> None:
    """Auto-skip integration tests unless explicitly opted in."""
    marker = request.node.get_closest_marker("integration")
    if marker is None:
        return
    if os.getenv("RUN_INTEGRATION_TESTS", "").strip() != "1":
        pytest.skip("Set RUN_INTEGRATION_TESTS=1 to run integration tests")
    missing = [k for k in _INTEGRATION_KEYS if not _has_api_key(k)]
    if missing:
        pytest.skip(f"Missing API key(s): {', '.join(missing)}")
