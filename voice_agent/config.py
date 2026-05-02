"""Central model and runtime configuration.

Single source of truth for model IDs, latency budgets, scheduling knobs, and env settings.
Swap a model here — nothing else changes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Core
    anthropic_api_key: str | None = None  # validated by Anthropic SDK at call time
    google_api_key: str | None = None     # validated by Google GenAI SDK at call time (GOOGLE_API_KEY)
    openai_api_key: str | None = None     # optional first interviewer tier (OPENAI_API_KEY)
    groq_api_key: str | None = None       # validated by Groq SDK at call time (GROQ_API_KEY)
    cerebras_api_key: str | None = None   # optional; Cerebras skipped from chain when absent
    database_url: str = "sqlite:///voice_agent.db"

    # Logfire (no token = console output only)
    logfire_token: str | None = None
    logfire_project: str = "research-agent"
    logfire_project_path: str | None = None  # falls back to logfire_project

    # Vapi (all optional — empty means local simulation only)
    vapi_api_key: str = ""
    vapi_phone_number_id: str = ""
    vapi_webhook_secret: str = ""             # HMAC secret key from Vapi dashboard credential; empty = skip check (dev)
    vapi_server_credential_id: str = ""       # HMAC credential ID — attached to server (webhook) URL on dial
    vapi_signature_header: str = "x-signature"   # must match "Signature Header" in Vapi HMAC credential
    vapi_timestamp_header: str = "x-timestamp"   # must match "Timestamp Header"; empty = omit timestamp from payload
    webhook_url: str = ""
    llm_secret_token: str = ""   # static secret sent by Vapi in X-Vapi-Secret header; empty = skip check (dev)

    # Voice
    vapi_voice_provider: str = "11labs"
    vapi_voice_id: str | None = None  # default depends on provider
    vapi_voice_model: str | None = None  # ElevenLabs model, e.g. "eleven_flash_v2_5" (~75ms latency)
    vapi_voice_stability: float | None = None
    vapi_voice_similarity_boost: float | None = None
    vapi_voice_style: float | None = None
    vapi_voice_speed: float | None = None
    # ElevenLabs via Vapi: flush streamed text to TTS after this many chars. High values delay
    # first audio on short fillers; 1 starts TTS as soon as text arrives.
    vapi_voice_chunk_min_characters: int = 1

    # Voice pipeline timing & interruption
    vapi_wait_seconds: float = 0.0          # delay before assistant speaks after turn ends (Vapi default: 0.4)
    vapi_stop_num_words: int = 3            # words user must say before assistant stops (blocks backchannel interrupts)
    vapi_stop_backoff_seconds: float = 0.5  # wait after real interruption before speaking again

    # After assistant TTS ends, if the user does not speak within this window, end the call (Vapi DELETE).
    # 0 = disabled. Avoids unbounded "Still there?" loops until maxDurationSeconds.
    vapi_extended_silence_seconds: float = 0.0

    # Spoken line when the model response cannot be parsed into a usable utterance + metadata.
    interviewer_recovery_utterance: str = (
        "Hey, sorry I could not hear you — could you come again?"
    )

    # Interviewer model chain (env-overridable): OpenAI (if key + model) → Haiku → Gemini → Groq → Cerebras (optional).
    openai_model: str = "openai:gpt-4.1-mini"  # "" skips OpenAI tier even when OPENAI_API_KEY is set
    # OpenAI Chat Completions prompt caching — shared prefix (system + COVERED_SUBTOPICS block). Empty = omit API params.
    openai_prompt_cache_key: str = "voice_agent_interviewer"
    openai_prompt_cache_retention: Literal["in_memory", "24h"] = "24h"
    # Bump when INTERVIEWER_PROMPT or user-prefix layout changes (suffixes OpenAI prompt_cache_key).
    interviewer_prompt_cache_version: str = "1"
    haiku_model: str = "anthropic:claude-haiku-4-5-20251001"
    gemini_model: str = "google-gla:gemini-2.0-flash"
    groq_model: str = "groq:llama-3.3-70b-versatile"
    cerebras_model: str = "cerebras:llama3.1-8b"  # same provider:model form; "" skips Cerebras in the chain

    # Dev
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("vapi_voice_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        stripped = str(v).strip().lower()
        return stripped if stripped else "11labs"

    @property
    def effective_logfire_project_path(self) -> str:
        return self.logfire_project_path or self.logfire_project


settings = Settings()

# Models — interviewer chain: OpenAI (optional) → Haiku → Gemini → Groq → Cerebras (optional)
# Env: OPENAI_MODEL, HAIKU_MODEL, GEMINI_MODEL, GROQ_MODEL, CEREBRAS_MODEL; OpenAI/Cerebras omitted when key/model unset.
INTERVIEWER_OPENAI_MODEL = settings.openai_model
INTERVIEWER_HAIKU_MODEL = settings.haiku_model
INTERVIEWER_GEMINI_MODEL = settings.gemini_model
INTERVIEWER_GROQ_MODEL = settings.groq_model
INTERVIEWER_CEREBRAS_MODEL = settings.cerebras_model
ANALYST_MODEL = "anthropic:claude-sonnet-4-6"              # async, quality-sensitive
SYNTHESIS_MODEL = "anthropic:claude-sonnet-4-6"            # post-call, no latency constraint

# Post-call synthesis report (LLM). Set False to skip generation + polling during testing.
ENABLE_SYNTHESIS_REPORT: bool = False

# Interviewer hard deadline (seconds). Haiku + structured output can exceed ~2.5 s under variance;
# 5 s reduces premature scripted fallbacks while still bounding hangs.
INTERVIEWER_BUDGET_S: float = 5.0

INTERVIEWER_RECOVERY_UTTERANCE: str = settings.interviewer_recovery_utterance

# Streaming filler: when > 0, TurnPipeline yields a brief acknowledgment as soon as the
# LLM stream starts so TTS begins immediately; real tokens follow. 0.0 = disabled.
FILLER_THRESHOLD_S: float = 0.4

# Vapi HMAC replay-attack window — reject requests whose timestamp is older than this.
VAPI_TIMESTAMP_TOLERANCE_S: int = 300  # 5 minutes
