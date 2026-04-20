"""Central model and runtime configuration.

Single source of truth for model IDs, latency budgets, and scheduling knobs.
Swap a model here — nothing else changes.
"""

# Models
INTERVIEWER_MODEL = "anthropic:claude-haiku-4-5-20251001"  # real-time, latency-critical
ANALYST_MODEL = "anthropic:claude-sonnet-4-6"              # async, quality-sensitive
SYNTHESIS_MODEL = "anthropic:claude-sonnet-4-6"            # post-call, no latency constraint

# Interviewer hard deadline (seconds). Haiku with structured output runs ~1.7–1.8 s in practice;
# 2.5 s gives headroom while still catching genuine hangs.
INTERVIEWER_BUDGET_S: float = 2.5

# Fire the analyst every N respondent turns (1 = every turn, 2 = every other, …).
# Reduces redundant passes on short back-to-back utterances.
ANALYST_EVERY_N_TURNS: int = 2
