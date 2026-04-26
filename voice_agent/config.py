"""Central model and runtime configuration.

Single source of truth for model IDs, latency budgets, scheduling knobs, and env settings.
Swap a model here — nothing else changes.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Core
    anthropic_api_key: str | None = None  # validated by Anthropic SDK at call time
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
    vapi_voice_stability: float | None = None
    vapi_voice_similarity_boost: float | None = None
    vapi_voice_style: float | None = None
    vapi_voice_speed: float | None = None

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

# Models
INTERVIEWER_MODEL = "anthropic:claude-haiku-4-5-20251001"  # real-time, latency-critical
ANALYST_MODEL = "anthropic:claude-sonnet-4-6"              # async, quality-sensitive
SYNTHESIS_MODEL = "anthropic:claude-sonnet-4-6"            # post-call, no latency constraint

# Post-call synthesis report (LLM). Set False to skip generation + polling during testing.
ENABLE_SYNTHESIS_REPORT: bool = False

# Interviewer hard deadline (seconds). Haiku + structured output can exceed ~2.5 s under variance;
# 5 s reduces premature scripted fallbacks while still bounding hangs.
INTERVIEWER_BUDGET_S: float = 5.0

# Vapi HMAC replay-attack window — reject requests whose timestamp is older than this.
VAPI_TIMESTAMP_TOLERANCE_S: int = 300  # 5 minutes
