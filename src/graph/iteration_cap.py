"""Complexity-aware runtime iteration cap (B1).

The runtime iteration cap — when the routers and error handler decide to stop /
reflect / accept-partial / abort — is complexity-aware: a TRIVIAL goal stops
loop-hunting at a small cap, a COMPLEX goal keeps full headroom. An explicit
per-run cap pinned by the caller (``state["max_iterations"]`` from the CLI
``--max-iterations``, an eval spec, or the worker schema) ALWAYS wins — callers
that pin a cap know their budget. When no cap is pinned, the cap falls back to
the complexity tier instead of a flat ``agent.max_iterations``.

LangGraph's ``recursion_limit`` (``runner.py``) is a SEPARATE, build-time fan-
out ceiling computed before complexity is classified, so it is NOT complexity-
aware — this helper only governs the runtime termination decisions.

Why a single helper: ``complexity`` lives on the ``Goal`` object at
``state["current_goal"]``, not as a top-level state key. Every max-iterations
reader must extract it identically; a single site defaulting to a different
complexity lets a low-cap run overshoot or force-terminate inconsistently.
"""

from __future__ import annotations

from src.config import get_settings
from src.graph.enums import TaskComplexity
from src.graph.state import AgentState


def _goal_complexity(state: AgentState) -> TaskComplexity:
    """The active goal's complexity, defaulting to SIMPLE when unset.

    A run that hasn't classified yet (or the heuristic-fallback path) yields
    SIMPLE — the historical default cap basis. ``current_goal`` may briefly be a
    plain string or None; ``getattr`` returns None there → SIMPLE.
    """
    goal = state.get("current_goal")
    complexity = getattr(goal, "complexity", None) if goal is not None else None
    return complexity or TaskComplexity.SIMPLE


def effective_max_iterations(state: AgentState) -> int:
    """The runtime iteration cap for ``state``.

    An explicit per-run cap (``state["max_iterations"]``) wins. Otherwise the cap
    is the complexity-tier default from ``AgentSettings.max_iterations_<tier>``.
    """
    explicit = state.get("max_iterations")
    if explicit:
        return int(explicit)

    settings = get_settings().agent
    complexity = _goal_complexity(state)
    return {
        TaskComplexity.TRIVIAL: settings.max_iterations_trivial,
        TaskComplexity.SIMPLE: settings.max_iterations_simple,
        TaskComplexity.COMPLEX: settings.max_iterations_complex,
        TaskComplexity.CRITICAL: settings.max_iterations_critical,
    }[complexity]
