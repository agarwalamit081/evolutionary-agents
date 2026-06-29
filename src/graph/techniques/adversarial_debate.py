"""Adversarial-Debate prompting technique (experimental; full algorithm deferred).

Adversarial / multi-agent debate — split the reasoning into a proponent and a
critic: the proponent argues the answer, the critic attacks its weakest
assumptions, and the pressure surfaces hidden flaws before a verdict is
committed. Strongest at the verify/reflect step of high-stakes goals.

This module ships the *prompting body* the :class:`TechniqueSelector` injects
when its opt-in flag is on. The full multi-turn proponent↔critic debate —
distinct advocate rolls with a judge reconciliation — is deferred to a later
phase; see :func:`apply`.
"""

from __future__ import annotations

from src.graph.enums import TaskComplexity
from src.graph.prompts.technique_selector import NODE_REFLECT, NODE_VERIFY, Technique

from ._errors import TechniqueDeferredError

TECHNIQUE: Technique = Technique(
    name="adversarial_debate",
    body=(
        "Debate the answer before committing it: argue the strongest case FOR "
        "the conclusion, then immediately attack its weakest assumptions as a "
        "hostile critic would, and reconcile only what survives both sides. "
        "Treat your first verdict as a hypothesis to be broken, not a result."
    ),
    applies_to_complexities=frozenset({TaskComplexity.CRITICAL}),
    nodes=frozenset({NODE_VERIFY, NODE_REFLECT}),
    goal_patterns=frozenset({"reasoning", "analysis"}),
    token_cost_estimate=100,
    layer="verification",
    priority=84,
)


async def apply() -> None:
    """Deferred: the full multi-turn proponent↔critic↔judge debate.

    The flag injects only the prompting ``body`` today; wiring the distinct
    advocate rolls with judge reconciliation is deferred to a later phase.
    """
    raise TechniqueDeferredError(
        "The full multi-turn proponent↔critic↔judge debate is deferred. The "
        "opt-in flag injects only the prompting body into node prompts via "
        "TechniqueSelector; the distinct advocate rolls are not wired. Enable "
        "the body via EXPERIMENTAL_TECHNIQUES_ENABLED + "
        "EXPERIMENTAL_TECHNIQUES_ADVERSARIAL_DEBATE_ENABLED."
    )
