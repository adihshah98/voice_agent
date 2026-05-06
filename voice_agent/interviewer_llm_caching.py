"""Interviewer prompt caching across the FallbackModel chain.

One user prompt is built in ``prepare_interviewer_turn`` and reused for every fallback tier.
pydantic-ai adapts it per provider:

- **Anthropic**: ``CachePoint`` after COVERED_SUBTOPICS plus ``anthropic_cache_instructions`` on
  the system block. TTL is single-sourced in :data:`ANTHROPIC_CACHE_TTL`.
- **OpenAI** and **Cerebras** (OpenAI-compatible): ``CachePoint`` is stripped; prefix caching uses
  ``prompt_cache_key`` and retention on the model (see :func:`openai_interviewer_settings`).
- **Gemini**: inline ``CachePoint`` is ignored; explicit caches would use
  ``GoogleModelSettings.google_cached_content`` (not configured here).
- **Groq**: ``CachePoint`` is stripped.

Bump ``Settings.interviewer_prompt_cache_version`` when ``INTERVIEWER_PROMPT`` or the user-prefix
layout changes so OpenAI cache grouping stays meaningful across releases.
"""

from __future__ import annotations

from typing import Literal

from pydantic_ai.messages import CachePoint
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIChatModelSettings

from voice_agent.config import Settings

# Must stay aligned: Anthropic counts this toward its cache breakpoint budget next to system cache.
ANTHROPIC_CACHE_TTL: Literal["5m", "1h"] = "1h"


def user_message_cache_breakpoint() -> CachePoint:
    """Breakpoint after the semi-static COVERED_SUBTOPICS block (Anthropic prompt cache)."""
    return CachePoint(ttl=ANTHROPIC_CACHE_TTL)


def anthropic_interviewer_settings(temperature: float | None = None) -> AnthropicModelSettings:
    """Cache system instructions; pairs with :func:`user_message_cache_breakpoint`.

    `temperature` is forwarded when set (e.g. 0.0 in eval mode for determinism).
    """
    kwargs: dict = {"anthropic_cache_instructions": ANTHROPIC_CACHE_TTL}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return AnthropicModelSettings(**kwargs)


def openai_interviewer_settings(settings: Settings) -> OpenAIChatModelSettings | None:
    """OpenAI Chat Completions prefix cache; ``None`` disables explicit cache-key params.

    The effective key is ``{openai_prompt_cache_key}:{interviewer_prompt_cache_version}``
    when both are non-empty so deploys can invalidate grouping without renaming the base key.
    Temperature is forwarded when set (e.g. 0.0 in eval mode for determinism).
    """
    base = settings.openai_prompt_cache_key.strip()
    temp = settings.interviewer_temperature
    if not base and temp is None:
        return None
    kwargs: dict = {}
    if base:
        ver = settings.interviewer_prompt_cache_version.strip()
        kwargs["openai_prompt_cache_key"] = f"{base}:{ver}" if ver else base
        kwargs["openai_prompt_cache_retention"] = settings.openai_prompt_cache_retention
    if temp is not None:
        kwargs["temperature"] = temp
    return OpenAIChatModelSettings(**kwargs)
