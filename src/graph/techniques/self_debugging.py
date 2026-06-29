"""Self-Debugging prompting technique (experimental; full algorithm deferred).

Self-Debugging (Chen et al., 2023) — produce a draft, run or trace it against
the target, read the *exact* failure, then revise only what that failure
pinpoints, iterating draft→run→diagnose→fix until it passes.

This module ships the *prompting body* the :class:`TechniqueSelector` injects
when its opt-in flag is on. The full generate→execute→debug→fix controller — a
multi-turn execute↔debug loop over the gateway — is deferred to a later phase;
see :func:`apply`.
"""

from __future__ import annotations

from src.graph.enums import TaskComplexity
from src.graph.prompts.technique_selector import NODE_EXECUTE, NODE_REFLECT, Technique

from ._errors import TechniqueDeferredError

TECHNIQUE: Technique = Technique(
    name="self_debugging",
    body=(
        "Self-debug in a tight loop: produce a first draft, run or trace it "
        "against the target, read the EXACT failure (the error message, the wrong "
        "row, the failing assertion), then revise only what that failure "
        "pinpoints. Repeat draft→run→diagnose→fix until it passes — never "
        "re-guess blindly after a known failure."
    ),
    applies_to_complexities=frozenset({TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}),
    nodes=frozenset({NODE_EXECUTE, NODE_REFLECT}),
    goal_patterns=frozenset({"code"}),
    token_cost_estimate=110,
    layer="reasoning",
    priority=82,
)


async def apply() -> None:
    """Deferred: the full Self-Debugging generate→execute→debug→fix controller.

    The flag injects only the prompting ``body`` today; wiring the multi-turn
    execute↔debug loop is deferred to a later phase.
    """
    raise TechniqueDeferredError(
        "The full Self-Debugging generate→execute→debug→fix controller is "
        "deferred. The opt-in flag injects only the prompting body into node "
        "prompts via TechniqueSelector; the multi-turn controller is not wired. "
        "Enable the body via EXPERIMENTAL_TECHNIQUES_ENABLED + "
        "EXPERIMENTAL_TECHNIQUES_SELF_DEBUGGING_ENABLED."
    )
