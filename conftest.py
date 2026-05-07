"""Shared pytest configuration. Package is installed via pyproject.toml so no sys.path manipulation needed."""

import pytest


@pytest.fixture(autouse=True, scope="session")
def init_tracing_once():
    """Configure Logfire once for the entire test session.

    Must run before any test so the _logfire_core guard in tracing.py is won
    by this call (with the token present) rather than by whichever test file
    happens to import first. autouse=True + scope="session" guarantees exactly
    one call, before any test body executes.
    """
    from voice_agent.tracing import init_tracing
    init_tracing(service_name="voice-agent-evals")
    yield
    try:
        import logfire
        logfire.force_flush()
    except Exception:
        pass
