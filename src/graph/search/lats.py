"""LATS / MCTS tree-search execution primitive (Phase 5 G3a).

A real search primitive — not a prompt fragment — that explores alternative
next-steps in *reasoning space* before the irreversible real execution node
runs a single chosen step. Faithful LATS (LLM value function + UCB1
lookahead over candidate actions), side-effect-safe and checkpoint-safe.

Two design pivots from a textbook rollout (see the Phase-5 plan):
- **Reasoning-only rollouts.** ``execute`` has irreversible side effects
  (deliverable writes, ``terminal_command``, ``http_request``). Branching into
  N real rollouts would fan those out (conflicting writes, duplicate charges)
  and multiply cost N-fold. So expansion + rollout + value are *gateway-only*
  calls that imagine and score hypothetical trajectories; real tool execution
  stays single-trajectory. LATS only *selects* the best next step.
- **Stateless per-call tree.** Each ``lats_search`` invocation builds a fresh
  in-memory MCTS tree for that one decision, commits the UCB-best root action,
  then discards it. No cross-call UCB state, no new ``AgentState`` field —
  checkpoint/resume-safe (grounding accumulates naturally via the real
  ``reflect`` outcomes already in ``messages``).

Default-off (``LATS_ENABLED``). When off, ``route_after_reflect`` never returns
``"lats_search"`` and this node is unreachable → topology is byte-identical.
Fail-safe: any error (gateway, budget-exhausted, parse) ⇒ returns ``{}`` so the
plan's original next step runs unchanged. LATS can never break a run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import json_repair
from loguru import logger

from src.config.settings import get_settings
from src.graph.enums import Confidence, TaskComplexity
from src.graph.models import PlanStep
from src.graph.state import AgentState, objective_goal_text

# Neutral value for an unevaluated / unparseable leaf. Deliberately mid-range:
# an unevaluated child never wins outright, but a real alternative that beats
# it can displace the incumbent, and a failed parse never aborts the search.
_NEUTRAL_SCORE: float = 0.5

# The subset of PlanStep fields the expansion LLM may propose. Everything else
# (id/status/result/tokens_used/duration_ms) keeps its PlanStep default.
_EXPAND_FIELDS: tuple[str, ...] = (
    "description",
    "tool_name",
    "tool_input",
    "expected_output",
    "depends_on",
)

# ── Prompts (module-level constants — E3 structure_analysis idiom; no f-strings) ──

_LATS_EXPAND_SYSTEM = (
    "You are a planning search engine for an autonomous agent that is stuck on a "
    "CRITICAL task. The current next step may be suboptimal. Propose {n} DISTINCT "
    "alternative next-steps that could unblock progress toward the objective. Each "
    "alternative must use a real available tool and be concretely different from the "
    "others and from the incumbent step. Output STRICT JSON only: "
    '{{"candidates": [{{"description": str, "tool_name": str, '
    '"tool_input": dict, "expected_output": str}}]}}. '
    "tool_input values must be JSON primitives or nested objects — never code."
)

_LATS_EXPAND_USER = (
    "OBJECTIVE:\n{objective}\n\n"
    "COMPLETED SO FAR:\n{completed}\n\n"
    "AVAILABLE TOOLS:\n{tools}\n\n"
    "INCUMBENT NEXT STEP (the current plan):\n{current_step}\n\n"
    "Propose {n} alternative next-steps that better advance the objective. "
    "Return ONLY the JSON object."
)

_LATS_ROLLOUT_SYSTEM = (
    "You imagine — without executing any tool — what would plausibly happen if an "
    "agent took a given step on a CRITICAL task. Describe the immediate outcome and "
    "the 1-2 likely follow-on steps. Be concrete but concise. Output STRICT JSON: "
    '{{"trajectory": str}}.'
)

_LATS_ROLLOUT_USER = (
    "OBJECTIVE:\n{objective}\n\n"
    "AVAILABLE TOOLS:\n{tools}\n\n"
    "PROPOSED STEP:\n{step}\n\n"
    "Imagine this step running. Output ONLY the JSON object."
)

_LATS_VALUE_SYSTEM = (
    "You are a value function for a search tree. Score how likely the imagined "
    "trajectory of a proposed step is to advance the stated objective, on [0.0, 1.0]. "
    "Reward concrete, tool-grounded progress toward the objective; penalize vagueness, "
    "irrelevance, or steps that ignore the available tools. Output STRICT JSON: "
    '{{"score": float, "rationale": str}}. score must be a number in [0.0, 1.0].'
)

_LATS_VALUE_USER = (
    "OBJECTIVE:\n{objective}\n\n"
    "COMPLETED SO FAR:\n{completed}\n\n"
    "PROPOSED STEP:\n{step}\n\n"
    "IMAGINED TRAJECTORY:\n{trajectory}\n\n"
    "Score this branch. Output ONLY the JSON object."
)


# ── Tree node ──────────────────────────────────────────────────────────────────


@dataclass
class _Child:
    """One frontier node of the per-call MCTS tree."""

    step: PlanStep
    is_original: bool = False
    visits: int = 0
    value_sum: float = 0.0
    trajectory: str = ""


# ── Engage guard ───────────────────────────────────────────────────────────────


def _lats_should_engage(state: AgentState) -> bool:
    """Return True iff LATS should engage for this decision.

    CRITICAL complexity is always required (LATS is for hard tasks). The
    confidence gate applies only in the default ``"stall"`` scope — only
    LOW/VERY_LOW (i.e. single-trajectory has stalled). ``scope == "always"``
    engages on any confidence. Always requires a remaining plan step to replace.
    """
    try:
        settings = get_settings().lats
    except Exception:  # noqa: BLE001 — settings must never gate a run out
        return False
    if not settings.enabled:
        return False

    goal = state.get("current_goal")
    if getattr(goal, "complexity", None) != TaskComplexity.CRITICAL:
        return False

    scope_always = str(getattr(settings, "scope", "stall")).lower() == "always"
    if not scope_always:
        if state.get("confidence") not in (Confidence.LOW, Confidence.VERY_LOW):
            return False

    plan_steps = state.get("plan_steps") or []
    idx = int(state.get("current_step_index") or 0)
    return idx < len(plan_steps)


# ── Context ────────────────────────────────────────────────────────────────────


def _safe_tool_names(tools: Any) -> list[str]:
    """Best-effort list of registered tool names; ``[]`` on any failure.

    Prefers the direct ``list_names()`` accessor; falls back to parsing
    ``list_tools()`` descriptors (whose ``name`` may be top-level OR nested under
    a ``function`` key, the LangChain bind_tools shape). Advisory context only —
    a failure just yields ``(unknown)`` in the expand prompt, never an error.
    """
    # Preferred: direct name list.
    try:
        direct = tools.list_names()
        if isinstance(direct, list):
            names = [n for n in direct if isinstance(n, str) and n]
            if names:
                return names
    except Exception:  # noqa: BLE001 — advisory only
        pass

    # Fallback: parse bind_tools-style descriptors.
    try:
        descriptors = tools.list_tools()
    except Exception:  # noqa: BLE001 — advisory only
        return []
    names: list[str] = []
    for desc in descriptors:
        if not isinstance(desc, dict):
            continue
        name = desc.get("name")
        if not isinstance(name, str):
            fn = desc.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _describe_step(step: PlanStep) -> str:
    """Compact human-readable rendering of a step for prompts."""
    tool = step.tool_name or "(none)"
    return f"- {step.description} [tool={tool}] expects: {step.expected_output or '(unspecified)'}"


def _build_context(state: AgentState, tools: Any) -> dict[str, str]:
    """Assemble the shared prompt context from live state (objective-anchored)."""
    plan_steps: list[PlanStep] = list(state.get("plan_steps") or [])
    idx = int(state.get("current_step_index") or 0)

    completed: list[str] = []
    for step in plan_steps[: max(idx, 0)]:
        status = str(getattr(step, "status", "")).lower()
        if status.endswith("completed"):
            completed.append(_describe_step(step))

    current = plan_steps[idx] if 0 <= idx < len(plan_steps) else None
    return {
        "objective": objective_goal_text(state) or "(unspecified)",
        "completed": "\n".join(completed[-6:]) or "(none yet)",
        "tools": ", ".join(_safe_tool_names(tools)) or "(unknown)",
        "current_step": _describe_step(current) if current else "(none)",
    }


# ── Gateway helpers (default routing — structure_analysis idiom) ───────────────


async def _llm_json(gateway: Any, system: str, user: str) -> Any:
    """One gateway call returning a json_repair-parsed object (or ``{}``)."""
    response = await gateway.acompletion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    content = (getattr(response, "content", "") or "").strip()
    if not content:
        return {}
    return json_repair.loads(content)


async def _expand_candidates(
    gateway: Any, context: dict[str, str], max_expansions: int
) -> list[PlanStep]:
    """Ask the gateway for up to ``max_expansions`` distinct alternative steps."""
    system = _LATS_EXPAND_SYSTEM.format(n=max_expansions)
    user = _LATS_EXPAND_USER.format(n=max_expansions, **context)
    data = await _llm_json(gateway, system, user)
    raw = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []

    candidates: list[PlanStep] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        filtered = {k: entry[k] for k in _EXPAND_FIELDS if isinstance(entry.get(k), (str, dict, list))}
        desc = filtered.get("description")
        if not isinstance(desc, str) or not desc.strip():
            continue
        key = desc.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            candidates.append(PlanStep(**filtered))
        except Exception:  # noqa: BLE001 — skip a malformed candidate, keep the rest
            continue
        if len(candidates) >= max_expansions:
            break
    return candidates


async def _rollout(
    gateway: Any, step: PlanStep, context: dict[str, str], depth: int
) -> str:
    """Gateway-only imagined trajectory for ``step`` (depth-bounded, no tool calls)."""
    trajectory = ""
    current = step
    for _ in range(max(depth, 0)):
        system = _LATS_ROLLOUT_SYSTEM
        user = _LATS_ROLLOUT_USER.format(step=_describe_step(current), **context)
        data = await _llm_json(gateway, system, user)
        piece = ""
        if isinstance(data, dict) and isinstance(data.get("trajectory"), str):
            piece = data["trajectory"].strip()
        if not piece:
            break
        trajectory = f"{trajectory}\n{piece}".strip() if trajectory else piece
        # The imagined next step is qualitative — stop the chain here rather than
        # recursively expanding hypotheticals (keeps rollouts bounded + cheap).
        break
    return trajectory


async def _value_score(
    gateway: Any, step: PlanStep, trajectory: str, context: dict[str, str]
) -> float:
    """LLM value function → score in [0, 1]; neutral on any failure."""
    system = _LATS_VALUE_SYSTEM
    user = _LATS_VALUE_USER.format(
        step=_describe_step(step),
        trajectory=trajectory or "(no imagined trajectory)",
        **context,
    )
    data = await _llm_json(gateway, system, user)
    score = data.get("score") if isinstance(data, dict) else None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return _NEUTRAL_SCORE
    return max(0.0, min(1.0, float(score)))


# ── Selection (UCB1) ───────────────────────────────────────────────────────────


def _ucb(child: _Child, total_visits: int, exploration: float) -> float:
    """UCB1: exploit + explore. Unvisited children score +inf (always tried first)."""
    if child.visits == 0:
        return math.inf
    exploit = child.value_sum / child.visits
    explore = exploration * math.sqrt(math.log(total_visits) / child.visits)
    return exploit + explore


def _mean(child: _Child) -> float:
    """Mean value of a child; unevaluated children are neutral."""
    return child.value_sum / child.visits if child.visits > 0 else _NEUTRAL_SCORE


async def _evaluate_frontier(
    gateway: Any,
    children: list[_Child],
    context: dict[str, str],
    *,
    max_evaluations: int,
    rollout_depth: int,
    exploration: float,
) -> None:
    """Roll out + value-score the frontier under a bounded evaluation budget.

    Phase 1 (rollout): each child gets an imagined trajectory when
    ``rollout_depth > 0`` (gateway-only; not counted against value budget so the
    value-selection unit test stays deterministic at ``rollout_depth=0``).
    Phase 2 (value): visit each child once in order until the budget is spent,
    then spend any remaining budget on UCB1-selected re-evaluations.
    """
    # Phase 1 — rollouts (optional).
    if rollout_depth > 0:
        for child in children:
            try:
                child.trajectory = await _rollout(gateway, child.step, context, rollout_depth)
            except Exception as exc:  # noqa: BLE001 — a bad rollout never aborts
                logger.debug("LATS rollout failed: {}", exc)
                child.trajectory = ""

    # Phase 2 — value evaluations.
    evaluations = 0
    for child in children:  # first pass: one visit each, in order, under budget
        if evaluations >= max_evaluations:
            break
        score = await _value_score(gateway, child.step, child.trajectory, context)
        child.visits = 1
        child.value_sum = score
        evaluations += 1

    while evaluations < max_evaluations and len(children) > 1:
        total = sum(c.visits for c in children)
        if total <= 0:
            break
        # Re-evaluate the most promising visited child (UCB1).
        candidates = [c for c in children if c.visits > 0]
        target = max(candidates, key=lambda c: _ucb(c, total, exploration))
        score = await _value_score(gateway, target.step, target.trajectory, context)
        target.visits += 1
        target.value_sum += score
        evaluations += 1


# ── Commit ─────────────────────────────────────────────────────────────────────


def _commit_step(plan_steps: list[PlanStep], idx: int, chosen: PlanStep) -> list[PlanStep]:
    """Return a new plan with ``plan_steps[idx]`` replaced by ``chosen``.

    The chosen step keeps the incumbent's ``id`` (via ``model_copy``) so any
    later step that ``depends_on`` it still resolves. Length is unchanged →
    execute's completion check (``step_index >= len(plan_steps)``) is unaffected.
    """
    revised = list(plan_steps)
    revised[idx] = chosen.model_copy(update={"id": plan_steps[idx].id})
    return revised


# ── Node ───────────────────────────────────────────────────────────────────────


async def lats_search_node(
    state: AgentState, *, gateway: Any, tools: Any
) -> dict[str, Any]:
    """Tree-search the next step in reasoning space; commit the UCB-best branch.

    Returns ``{}`` (transparent pass-through → execute runs the incumbent step)
    when LATS is off, declines to engage, or anything fails. On a non-incumbent
    winner, returns ``{"plan_steps": revised}`` with the chosen step swapped in.
    """
    if not _lats_should_engage(state):
        return {}

    try:
        settings = get_settings().lats
        plan_steps: list[PlanStep] = list(state.get("plan_steps") or [])
        idx = int(state.get("current_step_index") or 0)
        if idx >= len(plan_steps):
            return {}

        context = _build_context(state, tools)
        incumbent = plan_steps[idx]

        # Expand → alternatives; incumbent is always child[0] (stay-the-course).
        candidates = await _expand_candidates(gateway, context, settings.max_expansions)
        seen = {incumbent.description.strip().lower()}
        children: list[_Child] = [_Child(step=incumbent, is_original=True)]
        for cand in candidates:
            key = cand.description.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            children.append(_Child(step=cand, is_original=False))

        # Search: bounded rollouts + UCB1 value evaluations.
        await _evaluate_frontier(
            gateway,
            children,
            context,
            max_evaluations=max(int(settings.max_evaluations), 1),
            rollout_depth=max(int(settings.rollout_depth), 0),
            exploration=float(settings.exploration),
        )

        # Select argmax-mean (ties → first child = incumbent → stay course).
        chosen = max(children, key=_mean)
        if getattr(chosen, "is_original", False) or chosen.step is incumbent:
            logger.debug(
                "LATS kept incumbent step (best mean={:.3f} over {} children)",
                _mean(chosen),
                len(children),
            )
            return {}

        revised = _commit_step(plan_steps, idx, chosen.step)
        logger.info(
            "LATS replaced step {} with alternative branch (mean={:.3f} over {} children)",
            idx,
            _mean(chosen),
            len(children),
        )
        return {"plan_steps": revised}
    except Exception as exc:  # noqa: BLE001 — fail-safe: LATS never breaks a run
        logger.warning("LATS search failed; running incumbent step unchanged: {}", exc)
        return {}
