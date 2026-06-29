"""WebDreamer prompting technique (experimental; full algorithm deferred).

WebDreamer (model-based planning over web/tool actions) — before taking a real
and possibly irreversible action, *dream* a few candidate action sequences
forward using the LLM as a world model, predict each outcome, and commit only
the most promising first action — model-predictive control over the tool
interface.

This module ships the *prompting body* the :class:`TechniqueSelector` injects
when its opt-in flag is on. The full lookahead world-model rollout — where
candidate action chains are simulated and scored before any real tool call — is
deferred to a later phase; see :func:`apply`.
"""

from __future__ import annotations

from src.graph.enums import TaskComplexity
from src.graph.prompts.technique_selector import NODE_PLAN, Technique

from ._errors import TechniqueDeferredError

TECHNIQUE: Technique = Technique(
    name="web_dreamer",
    body=(
        "Before committing a consequential action, dream 2-3 candidate next "
        "actions forward: predict what each would produce, score the predicted "
        "outcomes, and take only the first step of the most promising chain. "
        "Treat irreversible actions as needing a lookahead, not a guess."
    ),
    applies_to_complexities=frozenset({TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}),
    nodes=frozenset({NODE_PLAN}),
    goal_patterns=frozenset({"research", "code"}),
    token_cost_estimate=100,
    layer="reasoning",
    priority=78,
)


async def apply() -> None:
    """Deferred: the full WebDreamer world-model lookahead rollout.

    The flag injects only the prompting ``body`` today; wiring the simulated
    candidate-chain rollout is deferred to a later phase.
    """
    raise TechniqueDeferredError(
        "The full WebDreamer world-model lookahead rollout is deferred. The "
        "opt-in flag injects only the prompting body into node prompts via "
        "TechniqueSelector; the simulated candidate-chain rollout is not wired. "
        "Enable the body via EXPERIMENTAL_TECHNIQUES_ENABLED + "
        "EXPERIMENTAL_TECHNIQUES_WEB_DREAMER_ENABLED."
    )
