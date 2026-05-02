"""Regression tests for multi-provider prompt caching helpers."""

from voice_agent.config import Settings
from voice_agent.interviewer_llm_caching import (
    ANTHROPIC_CACHE_TTL,
    anthropic_interviewer_settings,
    openai_interviewer_settings,
    user_message_cache_breakpoint,
)


def test_user_cache_breakpoint_matches_anthropic_ttl() -> None:
    cp = user_message_cache_breakpoint()
    assert cp.ttl == ANTHROPIC_CACHE_TTL


def test_anthropic_settings_ttl_aligned() -> None:
    s = anthropic_interviewer_settings()
    assert s["anthropic_cache_instructions"] == ANTHROPIC_CACHE_TTL


def test_openai_settings_none_when_key_blank() -> None:
    s = Settings(openai_prompt_cache_key="")
    assert openai_interviewer_settings(s) is None


def test_openai_settings_appends_version_suffix() -> None:
    s = Settings(openai_prompt_cache_key="myproj", interviewer_prompt_cache_version="3")
    out = openai_interviewer_settings(s)
    assert out is not None
    assert out["openai_prompt_cache_key"] == "myproj:3"
    assert out["openai_prompt_cache_retention"] == "24h"
