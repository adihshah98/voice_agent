"""Central model and runtime configuration.

Single source of truth for model IDs, latency budgets, and scheduling knobs.
Swap a model here — nothing else changes.
"""

# Models
INTERVIEWER_MODEL = "anthropic:claude-haiku-4-5-20251001"  # real-time, latency-critical
ANALYST_MODEL = "anthropic:claude-sonnet-4-6"              # async, quality-sensitive
SYNTHESIS_MODEL = "anthropic:claude-sonnet-4-6"            # post-call, no latency constraint

# Post-call synthesis report (LLM). Set False to skip generation + polling during testing.
ENABLE_SYNTHESIS_REPORT: bool = False

# Interviewer hard deadline (seconds). Haiku + structured output can exceed ~2.5 s under variance;
# 5 s reduces premature scripted fallbacks while still bounding hangs.
INTERVIEWER_BUDGET_S: float = 5.0
