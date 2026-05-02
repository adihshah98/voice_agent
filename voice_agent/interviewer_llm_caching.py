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


def anthropic_interviewer_settings() -> AnthropicModelSettings:
    """Cache system instructions; pairs with :func:`user_message_cache_breakpoint`."""
    return AnthropicModelSettings(anthropic_cache_instructions=ANTHROPIC_CACHE_TTL)


def openai_interviewer_settings(settings: Settings) -> OpenAIChatModelSettings | None:
    """OpenAI Chat Completions prefix cache; ``None`` disables explicit cache-key params.

    The effective key is ``{openai_prompt_cache_key}:{interviewer_prompt_cache_version}``
    when both are non-empty so deploys can invalidate grouping without renaming the base key.
    """
    base = settings.openai_prompt_cache_key.strip()
    if not base:
        return None
    ver = settings.interviewer_prompt_cache_version.strip()
    key = f"{base}:{ver}" if ver else base
    return OpenAIChatModelSettings(
        openai_prompt_cache_key=key,
        openai_prompt_cache_retention=settings.openai_prompt_cache_retention,
    )
