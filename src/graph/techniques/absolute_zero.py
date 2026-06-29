"""Absolute-Zero prompting technique (experimental; full algorithm deferred).

Absolute Zero Reasoner / Self-Play — propose your OWN tasks, verify them against
a code oracle, then learn from the dual of proposing and solving, maximizing
learnability rather than solving a fixed problem set. Strongest on formal /
code-verifiable goals.

This module ships the *prompting body* the :class:`TechniqueSelector` injects
when its opt-in flag is on. The full self-play loop — alternating proposer and
solver rolls against a verifier with reinforcement from dual feedback — is
deferred to a later phase; see :func:`apply`.
"""

from __future__ import annotations

from src.graph.enums import TaskComplexity
from src.graph.prompts.technique_selector import NODE_EXECUTE, Technique

from ._errors import TechniqueDeferredError

TECHNIQUE: Technique = Technique(
    name="absolute_zero",
    body=(
        "Self-play the problem: first pose a sharper, verifiable version of the "
        "task that would teach the original, solve it, and check it against an "
        "objective oracle; then carry the verified insight back to the original "
        "goal. Prefer problems you can actually verify over ones you can only "
        "assert."
    ),
    applies_to_complexities=frozenset({TaskComplexity.CRITICAL}),
    nodes=frozenset({NODE_EXECUTE}),
    goal_patterns=frozenset({"code", "reasoning"}),
    token_cost_estimate=110,
    layer="reasoning",
    priority=76,
)


async def apply() -> None:
    """Deferred: the full Absolute-Zero proposer↔solver↔verifier self-play loop.

    The flag injects only the prompting ``body`` today; wiring the self-play
    controller with verifier feedback is deferred to a later phase.
    """
    raise TechniqueDeferredError(
        "The full Absolute-Zero proposer↔solver↔verifier self-play loop is "
        "deferred. The opt-in flag injects only the prompting body into node "
        "prompts via TechniqueSelector; the self-play controller is not wired. "
        "Enable the body via EXPERIMENTAL_TECHNIQUES_ENABLED + "
        "EXPERIMENTAL_TECHNIQUES_ABSOLUTE_ZERO_ENABLED."
    )
