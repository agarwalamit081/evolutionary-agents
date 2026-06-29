"""Gödel-Agent prompting technique (experimental; full algorithm deferred).

Gödel Agent (a self-referential, self-improving reasoner) — at each step,
reflect on which reasoning *strategy* has been working and adaptively switch to
a better one, meta-reasoning over your own process rather than committing to a
single fixed technique for the whole task.

This module ships the *prompting body* the :class:`TechniqueSelector` injects
when its opt-in flag is on. The full self-referential strategy-rewrite loop —
where the agent rewrites its own reasoning rule from trajectory feedback — is
deferred to a later phase; see :func:`apply`.
"""

from __future__ import annotations

from src.graph.enums import TaskComplexity
from src.graph.prompts.technique_selector import NODE_EXECUTE, NODE_PLAN, Technique

from ._errors import TechniqueDeferredError

TECHNIQUE: Technique = Technique(
    name="godel_agent",
    body=(
        "Meta-reason over your own process: after each step, judge which "
        "reasoning strategy is actually paying off for THIS task and switch to a "
        "better one rather than grinding one fixed approach. Treat your "
        "reasoning rule itself as something you can improve mid-task."
    ),
    applies_to_complexities=frozenset({TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}),
    nodes=frozenset({NODE_PLAN, NODE_EXECUTE}),
    goal_patterns=frozenset({"reasoning"}),
    token_cost_estimate=100,
    layer="reasoning",
    priority=79,
)


async def apply() -> None:
    """Deferred: the full Gödel-Agent self-referential strategy-rewrite loop.

    The flag injects only the prompting ``body`` today; wiring the rule-rewrite
    controller is deferred to a later phase.
    """
    raise TechniqueDeferredError(
        "The full Gödel-Agent self-referential strategy-rewrite loop is "
        "deferred. The opt-in flag injects only the prompting body into node "
        "prompts via TechniqueSelector; the rule-rewrite controller is not "
        "wired. Enable the body via EXPERIMENTAL_TECHNIQUES_ENABLED + "
        "EXPERIMENTAL_TECHNIQUES_GODEL_AGENT_ENABLED."
    )
