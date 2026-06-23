"""Plan node — generates an execution plan from the classified task."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

from src.graph.enums import GoalStatus, Phase, Strategy, TaskComplexity
from src.graph.models import Goal, PlanStep
from src.graph.state import AgentState, objective_goal_text

if TYPE_CHECKING:
    from src.graph.schemas import PlanQuality
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry


async def plan_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Generate an execution plan based on the classified goal and strategy.

    Creates a list of PlanSteps from the goal and strategy.
    When a gateway is provided, LLM-based planning can generate richer plans.

    Args:
        state: Current agent state with classified goal.
        gateway: Optional LLM gateway for LLM-enhanced planning.
        tools: Optional ToolRegistry for dynamic tool list in prompts.

    Returns:
        Partial state update with plan_steps and phase transition.
    """
    goal = state.get("current_goal")
    strategy = state.get("strategy", Strategy.REACT)
    iteration_count = state.get("iteration_count", 0)

    if not goal or not goal.text:
        return {
            "phase": Phase.ERROR_HANDLER,
            "errors": ["plan: No goal text available"],
        }

    logger.info(f"Planning for goal (strategy={strategy.value}): {goal.text[:80]}...")

    # Try LLM planning first, fall back to heuristics
    plan_steps: list[PlanStep] | None = None
    if gateway is not None:
        plan_steps = await _llm_plan(gateway, goal, strategy, state, tools)

    if plan_steps is None:
        plan_steps = _generate_plan(goal.text, strategy)

    # Budget-aware sizing: cap the plan to fit the remaining iteration budget
    # so large multi-unit goals decompose within max_iterations instead of
    # stretching the run to exhaustion. Applies to both LLM and heuristic plans.
    from src.config import get_settings
    max_iter = state.get("max_iterations") or get_settings().agent.max_iterations
    remaining = max(0, max_iter - iteration_count)
    max_steps = min(get_settings().agent.planning_max_steps, max(1, remaining))
    if len(plan_steps) > max_steps:
        logger.info(
            f"Capping plan from {len(plan_steps)} to {max_steps} steps "
            f"(remaining iterations: {remaining})"
        )
        plan_steps = plan_steps[:max_steps]

    logger.info(f"Generated {len(plan_steps)} plan steps")

    # Feature C: per-step atomicity. Always compute + attach ``plan_quality``
    # as advisory telemetry (a model_dump dict — checkpoint-safe). When
    # ``plan_atomicity_enforce`` is on and a too_coarse step is found, run ONE
    # bounded heuristic split (guarded by ``atomicity_replan_done`` so a
    # reflect→plan loop can't re-split the same step forever). Pure heuristic —
    # zero LLM cost, fully deterministic.
    atomicity_replan_done = bool(state.get("atomicity_replan_done"))
    quality = _validate_step_atomicity(plan_steps)
    if (
        get_settings().agent.plan_atomicity_enforce
        and not atomicity_replan_done
        and quality.too_coarse_count > 0
    ):
        # Mark attempted regardless of outcome — a coarse step that won't split
        # must not be retried every re-plan (the loop guard).
        atomicity_replan_done = True
        refined_steps: list[PlanStep] = []
        for step in plan_steps:
            verdict = next(
                (v for v in quality.per_step if v.step_id == step.id), None
            )
            if verdict is not None and verdict.flag == "too_coarse":
                refined_steps.extend(_split_coarse_step(step))
            else:
                refined_steps.append(step)
        if len(refined_steps) != len(plan_steps):
            plan_steps = refined_steps[:max_steps]
            quality = _validate_step_atomicity(plan_steps)
            logger.info(
                f"Atomicity enforce: decomposed coarse step(s) -> "
                f"{len(plan_steps)} steps"
            )
    if not quality.atomic:
        logger.warning(
            f"Plan not atomic: {quality.too_coarse_count} too_coarse, "
            f"{quality.too_fine_count} too_fine"
        )

    result: dict[str, Any] = {
        "phase": Phase.RETRIEVE_MEMORY,
        "plan_steps": plan_steps,
        "current_step_index": 0,
        "iteration_count": iteration_count + 1,
        "plan_quality": quality.model_dump(),
    }
    if atomicity_replan_done:
        result["atomicity_replan_done"] = True
    return result


# Keywords marking a failure as a content-hash / handoff-integrity problem. The
# upstream artifact is already correct (a passing check asserts it) but a
# downstream manifest carries a stale hash / aggregate. The fix is to RE-READ
# the upstream and RECOMPUTE the derived value — regenerating the upstream
# re-breaks every artifact that depends on it (the exact battery-04 q08
# plateau: each re-plan rewrote raw_findings.jsonl, so the manifest's
# input_sha256 was stale forever).
_INTEGRITY_MARKERS: tuple[str, ...] = (
    "sha256", "sha", "hash", "mismatch", "integrity", "handoff",
    "did not read", "stale",
)


def _eval_failure_reason(check: dict[str, Any]) -> str:
    """Concise reason an eval check (JSON-dict form) failed.

    Mirrors ``verify._failure_reason`` but operates on the ``model_dump`` dict
    stored in ``state["eval_checks"]`` — plan runs as a peer of verify, so the
    check is a plain dict here, not a ``CheckResult`` object. Prefer the
    execution probe's stdout verdict (its last line states the specific
    failure), then a stored ``reason``, then ``error``.
    """
    evidence = check.get("evidence")
    if isinstance(evidence, dict):
        stdout = str(evidence.get("stdout") or "").strip()
        if stdout:
            last = stdout.splitlines()[-1].strip()
            if last:
                return last[:200]
        reason = str(evidence.get("reason") or "").strip()
        if reason:
            return reason[:200]
    err = check.get("error")
    if err:
        return str(err).strip()[:200]
    return "failed"


def _correction_context(state: AgentState) -> str:
    """Targeted re-plan directive when a prior attempt's eval checks failed.

    Without this the planner regenerates the WHOLE pipeline every re-plan,
    which (for content-hash handoff goals like battery-04 q08) perpetually
    invalidates the manifest's input_sha256 for an upstream artifact the agent
    just regenerated. Telling the planner what already passes (reuse, do not
    overwrite) and what to fix yields a plan that re-reads existing files
    instead of recreating them.

    Returns ``""`` for a fresh plan (no failed checks in state) so the
    first-attempt prompt — and the heuristic-fallback path — is unchanged.
    """
    raw_checks = state.get("eval_checks") or []
    checks = [c for c in raw_checks if isinstance(c, dict)]
    failed = [c for c in checks if not c.get("passed") and not c.get("skipped")]
    if not failed:
        return ""

    passing = [c for c in checks if c.get("passed") and not c.get("skipped")]

    lines: list[str] = [
        "CORRECTION RE-PLAN — a previous attempt already produced deliverables "
        "on disk, but correctness checks failed. Do NOT regenerate the whole "
        "pipeline. Read the existing deliverable files and fix ONLY the failing "
        "checks.",
    ]
    if passing:
        names = ", ".join(str(c.get("check_name", "?")) for c in passing)
        lines.append(
            f"PASSED (these deliverables are already correct — read and reuse "
            f"the existing files; DO NOT overwrite or regenerate them): {names}"
        )
    fail_block = ["FAILED (your plan must fix ONLY these checks):"]
    for c in failed:
        fail_block.append(
            f"  - {c.get('check_name', '?')}: {_eval_failure_reason(c)}"
        )
    lines.append("\n".join(fail_block))

    # A hash/integrity failure means the upstream is already correct (a passing
    # check confirms it). Drive the re-plan toward re-read + recompute, NOT
    # regenerating upstream — that only re-breaks the dependents.
    all_reasons = " ".join(_eval_failure_reason(c).lower() for c in failed)
    if any(kw in all_reasons for kw in _INTEGRITY_MARKERS):
        lines.append(
            "CRITICAL: a hash/mismatch/integrity failure means the UPSTREAM "
            "artifact is already correct. Re-read the existing upstream file and "
            "recompute the derived value (sha256 / count / aggregate) INTO the "
            "FAILING deliverable. Regenerating the upstream artifact re-breaks "
            "every artifact that depends on it."
        )
    else:
        lines.append(
            "Fix the failing deliverables in place; reuse the passing ones unchanged."
        )
    return "\n".join(lines)


async def _llm_plan(
    gateway: LLMGateway,
    goal: Goal,
    strategy: Strategy,
    state: AgentState,
    tools: ToolRegistry | None = None,
) -> list[PlanStep] | None:
    """Attempt LLM-based plan generation. Returns None on failure."""
    try:
        from src.graph.prompts import (
            NODE_PLAN,
            PLAN_SYSTEM,
            PLAN_USER,
            build_messages,
            select_techniques_for_node,
        )
        from src.graph.schemas import GeneratedPlan
        from src.llm.structured_output import StructuredOutputManager
        from src.config import get_settings
        from src.tools.selection import select_tools_for_query

        # Build memory context if available. Rendered as a bare bulleted list —
        # the plan_user.j2 template wraps it in an explicit ADVISORY ONLY frame
        # (objective-drift guard #254: recalled memory must read as advisory
        # technique hints, never as the objective, so a goal whose recall
        # surfaces a different run's skill cannot contaminate the plan).
        memories = state.get("retrieved_memories", [])
        memory_ctx = ""
        if memories:
            memory_ctx = "\n".join(f"- {m}" for m in memories[:5])

        max_iterations = state.get("max_iterations") or get_settings().agent.max_iterations
        iteration_count = state.get("iteration_count", 0)
        remaining_iterations = max(0, max_iterations - iteration_count)
        # Deliverable-aware re-plan: when verify already scored this run's
        # deliverables (state["eval_checks"]), tell the planner what passes
        # (reuse) vs what failed (fix in place). Empty for a fresh plan.
        correction_ctx = _correction_context(state)
        # Feature B advisory: the disambiguate cascade's proposed resolution
        # + assumptions + evidence. Rendered into plan_user.j2's ADVISORY
        # block — it explains the goal, never rewrites it (the OBJECTIVE slot
        # still holds the literal goal_text above). Empty when the cascade did
        # not run (default-off) so the template drops the block entirely.
        disambig_ctx = str(state.get("disambiguation_context", "") or "")
        user_prompt = PLAN_USER.format(
            goal_text=objective_goal_text(state),
            strategy=strategy.value,
            complexity=goal.complexity.value if goal.complexity else "simple",
            estimated_steps="auto",
            remaining_iterations=remaining_iterations,
            max_iterations=max_iterations,
            memory_context=memory_ctx,
            disambiguation_context=disambig_ctx,
            correction_context=correction_ctx,
        )
        # Build dynamic tool list for the plan prompt. When tool retrieval is
        # enabled (findings-05), select_tools_for_query returns the built-ins
        # plus the top-k dynamic tools nearest the goal instead of every active
        # tool; otherwise the full set (the default, unchanged behavior).
        if tools is not None:
            selected = await select_tools_for_query(goal.text, tools, get_settings())
            tool_names = [t["function"]["name"] for t in selected]
        else:
            tool_names = [
                "code_executor", "web_search", "file_reader",
                "file_writer", "code_validator", "self_inspect", "memory_search",
            ]
        system_prompt = PLAN_SYSTEM.format(available_tools=", ".join(tool_names))

        # §5: select prompting techniques for this call and splice their bodies
        # into the system prompt above the JSON-schema footer.
        plan_complexity = goal.complexity or TaskComplexity.SIMPLE
        techniques = select_techniques_for_node(
            complexity=plan_complexity, node=NODE_PLAN, goal_text=goal.text,
            # Feature D: thread Feature A's refined_intent so audience/
            # uncertainty signals shape the plan's technique mix. ``or None``
            # collapses the empty heuristic-path default so generic goals
            # (and tests with no refined_intent) infer nothing → unchanged.
            refined_intent=(state.get("refined_intent") or None),
        )
        messages = build_messages(system_prompt, user_prompt, techniques, node=NODE_PLAN)

        response = await gateway.acompletion(
            messages=messages,
            # Thread the *classified* complexity so a CRITICAL goal routes to a
            # stronger planning model instead of always SIMPLE (§5 C.1).
            complexity=plan_complexity,
            # Node identity → NODE_TIER_MAP: a COMPLEX/CRITICAL plan routes to
            # a MODERATE model (glm-4.7) per-node (findings-05 A).
            node=NODE_PLAN,
        )

        extractor = StructuredOutputManager()
        plan = await extractor.extract(
            response.content, GeneratedPlan, gateway=gateway, messages=messages
        )
        if plan is None or not plan.steps:
            return None

        steps: list[PlanStep] = []
        for gen_step in plan.steps:
            steps.append(PlanStep(
                id=uuid4().hex[:8],
                description=gen_step.description,
                tool_name=gen_step.tool_name,
                expected_output=gen_step.expected_output,
                depends_on=list(gen_step.depends_on),
                status=GoalStatus.PENDING,
            ))

        logger.info(f"LLM generated {len(steps)} plan steps")
        return steps
    except Exception as e:
        logger.debug(f"LLM planning failed, using heuristics: {e}")
        return None


def _generate_plan(goal_text: str, strategy: Strategy) -> list[PlanStep]:
    """Generate plan steps based on goal and strategy.

    This is a heuristic planner. In production, the LLM gateway would
    generate plans dynamically.
    """
    steps: list[PlanStep] = []

    if strategy == Strategy.DIRECT:
        # Single-step direct execution
        steps.append(PlanStep(
            description=f"Directly address: {goal_text}",
            status=GoalStatus.PENDING,
        ))

    elif strategy == Strategy.REACT:
        # Reasoning + Acting loop
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Analyze the task requirements and gather context",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description=f"Execute core task: {goal_text[:100]}",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Review results and verify against success criteria",
            status=GoalStatus.PENDING,
        ))

    elif strategy == Strategy.PLANNING:
        # Multi-step structured plan
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Break down the task into sub-components",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Execute each sub-component sequentially",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Integrate results from all sub-components",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Validate the integrated result",
            status=GoalStatus.PENDING,
        ))

    elif strategy == Strategy.REFLECTION:
        # Execute + reflect cycles
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Initial execution attempt",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Self-critique and identify improvements",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Refined execution based on critique",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Final review and validation",
            status=GoalStatus.PENDING,
        ))

    elif strategy == Strategy.TOT:
        # Tree of thought: explore multiple approaches
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Generate multiple solution approaches",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Evaluate and compare approaches",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Execute the best approach",
            status=GoalStatus.PENDING,
        ))

    elif strategy == Strategy.DEBATE:
        # Multi-perspective analysis
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Present arguments from perspective A",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Present arguments from perspective B",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Synthesize and resolve into final answer",
            status=GoalStatus.PENDING,
        ))

    else:
        # Fallback: simple 3-step plan
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description=f"Address: {goal_text[:100]}",
            status=GoalStatus.PENDING,
        ))

    return steps


# ── Feature C: per-step atomicity (pure heuristic) ────────────────────────

# Conjunctions / clause delimiters that signal a step bundles multiple actions.
_ATOMIC_CONJUNCTION_RE = re.compile(r"\b(?:and|then|also)\b|;", re.IGNORECASE)
# Splitter — consumes the delimiters + surrounding whitespace into clauses.
_COARSE_SPLIT_RE = re.compile(r"\s*(?:\b(?:and|then|also)\b|;)\s*", re.IGNORECASE)
_FINE_MAX_WORDS = 3
_COARSE_SPLIT_CAP = 4


def _validate_step_atomicity(steps: list[PlanStep]) -> PlanQuality:
    """Flag each plan step as atomic / too_coarse / too_fine (pure heuristic).

    - ``too_coarse``: >=2 conjunction/';' markers — the step bundles multiple
      actions and should be decomposed.
    - ``too_fine``: fewer than ``_FINE_MAX_WORDS`` AND no expected_output —
      the step is under-specified.
    - ``atomic``: otherwise.

    Returns a :class:`PlanQuality` (advisory; never mutates the steps).
    """
    from src.graph.schemas import PlanQuality, StepAtomicity

    per_step: list[StepAtomicity] = []
    coarse = 0
    fine = 0
    atomic_all = True
    for step in steps:
        desc = (step.description or "").strip()
        words = len(desc.split())
        conjunctions = len(_ATOMIC_CONJUNCTION_RE.findall(desc))
        flag = "atomic"
        reason = "single, well-scoped action"
        if conjunctions >= 2:
            flag = "too_coarse"
            reason = (
                f"{conjunctions} conjunction/clause markers — "
                "split into atomic sub-steps"
            )
            coarse += 1
            atomic_all = False
        elif words < _FINE_MAX_WORDS and not (step.expected_output or "").strip():
            flag = "too_fine"
            reason = (
                f"only {words} word(s) and no expected_output — under-specified"
            )
            fine += 1
            atomic_all = False
        per_step.append(StepAtomicity(
            step_id=step.id,
            description=desc[:120],
            flag=flag,
            reason=reason,
        ))
    return PlanQuality(
        per_step=per_step,
        atomic=atomic_all,
        too_coarse_count=coarse,
        too_fine_count=fine,
    )


def _split_coarse_step(step: PlanStep) -> list[PlanStep]:
    """Heuristically decompose a too_coarse step on its conjunctions.

    Bounded + deterministic (no LLM cost): 'fetch X and clean it then write to
    disk' becomes one ``PlanStep`` per clause, each inheriting the parent's
    ``tool_name`` / ``expected_output`` context. Returns ``[step]`` unchanged
    when the split yields fewer than 2 clauses (nothing to decompose).
    """
    parts = [
        p.strip() for p in _COARSE_SPLIT_RE.split(step.description or "") if p.strip()
    ]
    if len(parts) < 2:
        return [step]
    sub_steps: list[PlanStep] = []
    for clause in parts[:_COARSE_SPLIT_CAP]:
        sub_steps.append(PlanStep(
            id=uuid4().hex[:8],
            description=clause,
            tool_name=step.tool_name,
            expected_output=step.expected_output,
            status=step.status,
        ))
    return sub_steps
