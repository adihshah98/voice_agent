"""Shared pytest configuration. Package is installed via pyproject.toml so no sys.path manipulation needed."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--cases",
        default=None,
        help=(
            "Comma-separated persona names to run in test_tier3_trajectories. "
            "Example: --cases context_drift,context_compression"
        ),
    )


@pytest.fixture(scope="session")
def cases_filter(request: pytest.FixtureRequest) -> set[str] | None:
    """Returns the set of persona names from --cases, or None to run all."""
    raw = request.config.getoption("--cases")
    if not raw:
        return None
    return {name.strip() for name in raw.split(",")}


@pytest.fixture(autouse=True, scope="session")
def init_tracing_once():
    """Configure Logfire once for the entire test session.

    Must run before any test so the _logfire_core guard in tracing.py is won
    by this call (with the token present) rather than by whichever test file
    happens to import first. autouse=True + scope="session" guarantees exactly
    one call, before any test body executes.
    """
    from voice_agent.tracing import init_tracing
    init_tracing(service_name="voice-agent-evals", send_to_logfire=True)
    yield
    try:
        import logfire
        logfire.force_flush()
    except Exception:
        pass
