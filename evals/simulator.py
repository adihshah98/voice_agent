"""Respondent simulator for Tier 3 trajectory evals.

A PydanticAI agent that plays a research respondent. Each persona has a
distinct system prompt loaded from datasets/personas.yaml. The agent receives
the full conversation history and returns the next respondent utterance.

Usage:
    personas = load_personas()
    history = [{"speaker": "interviewer", "text": "How do you use the product?"}]
    reply = await simulate_turn(personas[0], history)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

load_dotenv()

from voice_agent.models import Persona, SimulatedReply

SIMULATOR_MODEL = "anthropic:claude-haiku-4-5-20251001"


@dataclass
class SimulatorDeps:
    persona: Persona


respondent_agent: Agent[SimulatorDeps, SimulatedReply] = Agent(
    SIMULATOR_MODEL,
    deps_type=SimulatorDeps,
    output_type=SimulatedReply,
    instrument=True,
)


@respondent_agent.system_prompt
def _persona_system_prompt(ctx: RunContext[SimulatorDeps]) -> str:
    return (
        ctx.deps.persona.system.strip()
        + "\n\n"
        "You are on a live phone market-research call. Respond naturally to the "
        "interviewer as the character described above. Keep replies conversational "
        "and realistic — this is spoken dialogue, not a written essay. Do NOT "
        "include speaker labels, stage directions, or quotes in your response. "
        "Just say what your character would say."
    )


def load_personas(path: str | Path | None = None) -> list[Persona]:
    """Load all personas from the YAML file."""
    if path is None:
        path = Path(__file__).parent / "datasets" / "personas.yaml"
    raw = yaml.safe_load(Path(path).read_text())
    return [Persona.model_validate(p) for p in raw["personas"]]


def get_persona(name: str, personas: list[Persona]) -> Persona:
    """Return a persona by name, raise if not found."""
    for p in personas:
        if p.name == name:
            return p
    raise ValueError(f"Persona {name!r} not found. Available: {[p.name for p in personas]}")


async def simulate_turn(
    persona: Persona,
    history: list[dict[str, str]],
) -> str:
    """Simulate the respondent's next utterance given the full conversation history.

    Args:
        persona: The persona to simulate.
        history: Full conversation so far as a list of
                 {"speaker": "interviewer"|"respondent", "text": "..."} dicts.
                 The last entry should be the latest interviewer utterance.

    Returns:
        The respondent's reply as a plain string.
    """
    lines: list[str] = []
    for turn in history:
        prefix = "Interviewer" if turn["speaker"] == "interviewer" else "You"
        lines.append(f"{prefix}: {turn['text']}")

    prompt = "Conversation so far:\n" + "\n".join(lines) + "\n\nYour response:"

    deps = SimulatorDeps(persona=persona)
    result = await respondent_agent.run(prompt, deps=deps)
    return result.output.text
