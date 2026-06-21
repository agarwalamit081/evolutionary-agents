"""Conditional edge routing functions for the task execution graph.

Each router inspects state and returns the name of the next node.
"""

from __future__ import annotations

from loguru import logger

from src.config import get_settings
from src.graph.enums import Confidence
from src.graph.nodes.reflect import _ground_should_evolve
from src.graph.state import AgentState


def route_after_execute(state: AgentState) -> str:
    """Route after the execute node.

    Returns:
        "reflect" — plan exhausted or max iterations reached
        "execute" — more steps to execute (loop)
        "error_handler" — non-retriable error occurred
    """
    errors = state.get("errors", [])
    tool_results = state.get("tool_results", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations") or get_settings().agent.max_iterations
    plan_steps = state.get("plan_steps", [])
    step_index = state.get("current_step_index", 0)

    # Non-retriable: only auth/authorization errors abort the run.
    if errors:
        last_error = errors[-1] if isinstance(errors, list) else str(errors)
        if "authentication" in last_error.lower() or "authorization" in last_error.lower():
            logger.warning("Non-retriable auth error, routing to error_handler")
            return "error_handler"

    # BOUND (must precede the recoverable-tool-error retry below): once the
    # iteration budget is exhausted, stop retrying and reflect on what we have.
    # Without this ordering a persistently-failing tool would loop
    # execute→execute until LangGraph's recursion limit crashed the run.
    if iteration_count >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached, routing to reflect")
        return "reflect"

    # Recoverable: a failed tool call (timeout, rate limit, unknown tool,
    # permission error, …) loops back to execute so the LLM sees the failure
    # in tool_results_ctx / the message thread and retries with a valid tool or
    # different arguments. This MUST NOT route to error_handler — tool failures
    # live in ``tool_results``, not ``errors``, so error_handler would see
    # "nothing to handle" and falsely declare the task complete (F14: a
    # hallucinated tool name aborted the run with is_complete=True and no
    # deliverable). Retries are bounded by the max-iterations→reflect guard.
    if tool_results:
        last_result = tool_results[-1]
        if hasattr(last_result, "success") and not last_result.success:
            error_str = getattr(last_result, "error", "") or ""
            logger.info(f"Recoverable tool error, retrying execute: {error_str[:100]}")
            return "execute"

    # Plan exhausted → reflect
    if plan_steps and step_index >= len(plan_steps):
        logger.info("All plan steps executed, routing to reflect")
        return "reflect"

    # Memory folding checkpoint — give should_fold a mid-plan opportunity so
    # long single-plan runs still get context compression (otherwise reflect
    # is only reached at plan exhaustion). Conservative: fires at most once
    # per fold window — after a fold, last_fold_iteration advances and the
    # message list resets to ~1 (the summary), so both guards reset.
    #
    # CRITICAL: also gate on fold AVAILABILITY (len(fold_history) < max_folds).
    # Once max_folds is reached the fold never executes, so last_fold_iteration
    # stops advancing — without this guard the (iteration - last_fold) >= 6
    # condition stays permanently true and the checkpoint fires EVERY iteration,
    # churning reflect→verify→replan and starving multi-deliverable plans of the
    # uninterrupted execute steps they need to finish. (Observed on q4 under
    # deepseek-v4-flash: 3 folds consumed by iter 19, then the checkpoint fired
    # every iteration 25→34, and the agent rewrote test_q1_q3.py three times
    # without ever reaching the test_results.json / executive_summary.md steps.)
    messages = state.get("messages", [])
    last_fold = state.get("last_fold_iteration", 0) or 0
    fold_history = state.get("fold_history", [])
    try:
        max_folds = get_settings().agent.memory_folding_max_folds
    except AttributeError:
        # Defensive: a partial/mock settings object (e.g. unit tests) may omit
        # the knob. Fall back to the AgentSettings default rather than crash
        # routing — folding is a context-compression optimization, not
        # correctness-critical.
        max_folds = 3
    has_remaining_steps = bool(plan_steps) and step_index < len(plan_steps)
    if (
        has_remaining_steps
        and len(fold_history) < max_folds
        and len(messages) >= 10
        and (iteration_count - last_fold) >= 6
    ):
        logger.info(
            f"Folding checkpoint: {len(messages)} messages mid-plan "
            f"(iteration {iteration_count}), routing to reflect for fold evaluation"
        )
        return "reflect"

    # More steps to execute → loop back
    return "execute"


def route_after_reflect(state: AgentState) -> str:
    """Route after the reflect node.

    Returns:
        "agent_spawn" — sub-agent gaps detected, spawn new sub-agents
        "tool_create" — tool gaps detected, create missing tools
        "verify" — confidence is medium or higher
        "execute" — low confidence, retry execution
        "plan" — reflection suggests replanning
    """
    # HARD CAP GUARD: once the iteration budget is exhausted, never re-enter
    # execute (or gap resolution) from reflect. Without this, a low-confidence
    # reflection at the cap loops reflect↔execute forever — execute has no
    # remaining steps and bounces straight back to reflect — until LangGraph's
    # recursion limit crashes the run (GraphRecursionError). Route to verify,
    # which accepts the partial result and terminates. See F12 (T2 crash).
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations") or get_settings().agent.max_iterations
    if iteration_count >= max_iterations:
        logger.info(
            f"Max iterations ({max_iterations}) reached on reflect; "
            f"routing to verify to accept partial result"
        )
        return "verify"

    # Check for sub-agent gaps first — highest priority
    pending_agent_gaps = state.get("pending_agent_gaps", [])
    if pending_agent_gaps:
        # Guard: skip if sub-agents were already spawned for these gaps
        sub_agents_spawned = state.get("sub_agents_spawned", [])
        if not sub_agents_spawned:
            logger.info(f"Sub-agent gaps detected: {pending_agent_gaps}, routing to agent_spawn")
            return "agent_spawn"
        # Agents already spawned — don't re-route to agent_spawn
        logger.debug("Sub-agents already spawned, skipping agent_spawn")

    # Check for tool gaps — second priority
    pending_gaps = state.get("pending_tool_gaps", [])
    if pending_gaps:
        logger.info(f"Tool gaps detected: {pending_gaps}, routing to tool_create")
        return "tool_create"

    reflection = state.get("reflection")

    if reflection is None:
        return "verify"

    # Check for replanning trigger
    if hasattr(reflection, "should_replan") and reflection.should_replan:
        plan_steps = state.get("plan_steps", [])
        step_index = state.get("current_step_index", 0)
        completed_steps = state.get("completed_steps", [])
        errors = state.get("errors", [])
        has_remaining_steps = bool(plan_steps) and step_index < len(plan_steps)
        progressing = bool(completed_steps) and not errors

        # A replan discards the current plan. Guard two pathological loop
        # paths so a should_replan=True reflection cannot stall the run:
        # (a) Mid-plan replans: the memory-folding checkpoint fires ~every 6
        #     iterations DURING a plan and routes here. If steps are succeeding
        #     and there are no errors, replanning discards completed work and
        #     regenerates a fresh plan that gets interrupted the same way — the
        #     plan never finishes (Q9 looped at step 1/3 forever). Continue
        #     executing so the plan completes and verify can judge it.
        # (b) Exhausted-plan replans: route to verify for the objective
        #     deliverable check BEFORE regenerating. If the deliverable is
        #     present verify completes; if not, route_after_verify replans.
        #     Without this, a finished deliverable-producing goal replans
        #     forever because the reflect LLM rarely self-reports done.
        # (c) In-progress but genuinely stuck (no step completed, or errors) is
        #     the one case where a direct replan is warranted — keep it.
        if has_remaining_steps and progressing:
            logger.info(
                "Reflect suggests replan but plan is in-progress and "
                "progressing (steps remaining, steps completed, no errors); "
                "continuing execution instead of discarding completed work"
            )
            return "execute"
        if not has_remaining_steps:
            logger.info(
                "Reflect suggests replan with plan exhausted; routing to "
                "verify for objective deliverable check before replanning"
            )
            return "verify"
        logger.info("Reflection suggests replanning (plan in-progress but not progressing)")
        return "plan"

    # Check confidence level
    confidence = state.get("confidence", Confidence.MEDIUM)
    if isinstance(confidence, str):
        confidence = Confidence(confidence)

    low_confidence_levels = {Confidence.VERY_LOW, Confidence.LOW}
    if confidence in low_confidence_levels:
        # Pre-cap loop guard: if the plan is already exhausted, retrying
        # execute is futile — execute has no steps and bounces straight back
        # to reflect. Route to verify so the partial result is judged
        # (verify→plan will replan if it finds addressable gaps).
        plan_steps = state.get("plan_steps", [])
        step_index = state.get("current_step_index", 0)
        has_remaining_steps = bool(plan_steps) and step_index < len(plan_steps)
        if not has_remaining_steps:
            logger.info(
                "Low confidence but plan exhausted (no remaining steps); "
                "routing to verify instead of retrying execute"
            )
            return "verify"
        logger.info(f"Low confidence ({confidence.value}), routing back to execute")
        return "execute"

    return "verify"


def route_after_structure_analysis(state: AgentState) -> str:
    """Route after the structure_analysis node.

    Proactively dispatches to the spawn nodes when the goal stated tool-creation
    or sub-agent intent up front, bypassing the reactive reflect-driven path.

    Returns:
        "agent_spawn" — proactive sub-agent gaps seeded from the goal
        "tool_create" — proactive tool gaps seeded from the goal
        "execute" — no proactive gaps; proceed to the execute loop
    """
    # Sub-agent gaps take priority (mirrors route_after_reflect).
    pending_agent_gaps = state.get("pending_agent_gaps", [])
    if pending_agent_gaps and not state.get("sub_agents_spawned"):
        logger.info(
            f"Proactive sub-agent gaps from structure analysis: "
            f"{pending_agent_gaps}, routing to agent_spawn"
        )
        return "agent_spawn"

    pending_tool_gaps = state.get("pending_tool_gaps", [])
    if pending_tool_gaps:
        logger.info(
            f"Proactive tool gaps from structure analysis: {pending_tool_gaps}, "
            f"routing to tool_create"
        )
        return "tool_create"

    return "execute"


def route_after_verify(state: AgentState) -> str:
    """Route after the verify node.

    ``no_evolution`` (the CLI ``--no-evolution`` flag / API request field)
    is read from STATE, not from ``RunnableConfig``. LangGraph passes
    ``config=None`` to conditional-edge routers in this graph
    (AsyncPostgresSaver checkpointer + interrupt_before + subgraphs), so a
    config-based flag is silently dropped — Phase 4's live review (F4)
    proved the prior config path was a no-op. State is the single source
    of truth.

    Returns:
        "evolve" — verification passed + should_evolve flag (and not --no-evolution)
        "store_memory" — verification passed, budget exhausted, or evolution skipped
        "plan" — verification partial with no remaining steps (re-plan to address gaps)
        "execute" — verification partial with steps remaining (retry)
    """
    is_complete = state.get("is_complete", False)
    confidence = state.get("confidence", Confidence.MEDIUM)
    reflection = state.get("reflection")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations") or get_settings().agent.max_iterations
    plan_steps = state.get("plan_steps", [])
    step_index = state.get("current_step_index", 0)

    if is_complete:
        no_evolution = bool(state.get("no_evolution", False))

        # Ground should_evolve at the decision point. reflect computes this, but
        # its folding path early-returns before reflection runs (reflect.py:87-92),
        # so a successful multi-step run can reach verify with reflection unset and
        # should_evolve never computed — evolution then never fires (0 mutations
        # across the full battery). Re-derive from objective evidence (deliverable
        # on disk, else HIGH confidence + >=3 steps + no errors) so genuine
        # successes route to evolve regardless of which reflect path ran.
        proposed = bool(reflection.should_evolve) if (
            reflection is not None and hasattr(reflection, "should_evolve")
        ) else False
        goal = state.get("current_goal")
        goal_text = goal.text if goal is not None else ""
        should_evolve = _ground_should_evolve(
            proposed,
            goal_text,
            state.get("completed_steps", []),
            state.get("errors", []),
            confidence,
        )

        if should_evolve:
            if no_evolution:
                logger.info("Evolution skipped (--no-evolution in state), routing to store_memory")
                return "store_memory"
            logger.info(
                f"Verification passed (should_evolve proposed={proposed}, grounded), "
                f"triggering evolution"
            )
            return "evolve"
        return "store_memory"

    # Budget exhausted — accept the partial result rather than looping forever.
    # Without this guard, partial verification + execute's no-steps path
    # increments iteration_count unbounded while the cycle never terminates.
    if iteration_count >= max_iterations:
        logger.info(
            f"Max iterations ({max_iterations}) reached, accepting partial result"
        )
        return "store_memory"

    # Partial verification with no steps left to execute — re-plan so the
    # agent can make progress addressing the identified gaps instead of
    # spinning on an exhausted plan. (verify → plan edge is wired in the graph.)
    has_remaining_steps = bool(plan_steps) and step_index < len(plan_steps)
    if not has_remaining_steps:
        logger.info("Verification partial with no remaining steps, re-planning")
        return "plan"

    # Partial with steps remaining — retry execution (low or medium confidence)
    if isinstance(confidence, str):
        confidence = Confidence(confidence)

    if confidence in {Confidence.VERY_LOW, Confidence.LOW}:
        logger.info("Verification failed with low confidence, retrying execute")
        return "execute"

    logger.info("Verification partial, retrying execute")
    return "execute"


def route_after_evolve(state: AgentState) -> str:
    """Route after the evolve node.

    Returns:
        "execute" — a TOOL mutation was live-registered this cycle (one
            re-execution pass so the new tool is reachable in-run; Phase 4 E)
        "store_memory" — evolution succeeded (or nothing deployed/re-registered)
        "error_handler" — evolution failed
    """
    errors = state.get("errors", [])
    # If new errors were added during evolution, route to error handler
    if errors and "evolution" in str(errors[-1]).lower():
        logger.warning("Evolution failed, routing to error_handler")
        return "error_handler"

    # Phase 4 E: a TOOL mutation was live-registered in the ToolRegistry this
    # cycle → run ONE execute pass so the run can use the new tool before
    # terminating. Bounded once per run by ``evolve_reexecute_done`` (set at
    # registration time in the evolve node), so the next evolve cycle never
    # re-offers and this branch can't loop.
    if state.get("evolve_reexecute_offered"):
        logger.info(
            "Deployed TOOL mutation live-registered; routing to execute for "
            "one re-execution pass (Phase 4 E)"
        )
        return "execute"

    return "store_memory"


def route_after_store(state: AgentState) -> str:
    """Route after the store_memory node.

    Returns:
        "hitl_gate" — HITL required (high risk or explicit flag)
        "complete" — normal completion, or budget exhausted (accept partial)
        "execute" — not done yet and budget remains
    """
    is_complete = state.get("is_complete", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations") or get_settings().agent.max_iterations

    if is_complete:
        # Normal completion. HITL can be enabled via config.
        return "complete"

    # Budget exhausted — accept the partial result instead of bouncing back
    # to execute (which would re-enter the partial-verification cycle).
    if iteration_count >= max_iterations:
        logger.info(
            f"Max iterations ({max_iterations}) reached, "
            f"completing with partial result"
        )
        return "complete"

    # Not done yet, continue execution
    return "execute"


def route_after_hitl(state: AgentState) -> str:
    """Route after the HITL gate node.

    Returns:
        "complete" — human approved
        "execute" — human requested revision
    """
    is_complete = state.get("is_complete", True)
    if is_complete:
        return "complete"
    return "execute"


def route_after_error(state: AgentState) -> str:
    """Route after the error_handler node.

    Returns:
        "execute" — retriable error, retry
        "classify" — needs reclassification
        "hitl_gate" — needs human intervention
        "complete" — fatal error, abort
    """
    errors = state.get("errors", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations") or get_settings().agent.max_iterations

    if not errors:
        # No error to handle is an anomaly (genuine errors populate ``errors``;
        # tool failures live in ``tool_results`` and are recovered in
        # route_after_execute), not a success. Judge the actual state via verify
        # instead of falsely completing the run. (F14.)
        logger.warning("Error handler reached with no errors; routing to verify")
        return "verify"

    last_error = str(errors[-1]).lower()

    # Budget exhausted → human intervention
    if "budget" in last_error:
        logger.error("Budget exhausted, escalating to HITL")
        return "hitl_gate"

    # Rate limit → retry with backoff
    if "rate" in last_error or "429" in last_error:
        if iteration_count < max_iterations:
            return "execute"
        return "complete"

    # Auth errors → reclassify with different provider
    if "auth" in last_error or "401" in last_error or "403" in last_error:
        return "classify"

    # Max iterations exceeded → abort
    if iteration_count >= max_iterations:
        logger.error(f"Max iterations ({max_iterations}) exceeded with errors, aborting")
        return "complete"

    # Default: retry execution
    return "execute"


def route_after_tool_create(state: AgentState) -> str:
    """Route after the tool_create node.

    Returns:
        "plan" — new tools were created, replan to use them
        "execute" — no tools created, retry execution
    """
    tools_created = state.get("tools_created", [])
    if tools_created:
        logger.info(
            f"{len(tools_created)} tool(s) created, routing to plan for replanning"
        )
        return "plan"
    return "execute"


def route_after_agent_spawn(state: AgentState) -> str:
    """Route after the agent_spawn node.

    Returns:
        "delegate" — sub-agents were spawned, delegate subtasks to them
        "tool_create" — no agents spawned, but converted tool gaps exist
        "plan" — no sub-agents spawned and no tool gaps, replan without them
    """
    sub_agents_spawned = state.get("sub_agents_spawned", [])
    pending_tool_gaps = state.get("pending_tool_gaps", [])

    if sub_agents_spawned:
        # Always delegate first when agents exist. Converted tool gaps will be
        # handled in subsequent reflect → tool_create cycles after delegation.
        if pending_tool_gaps:
            logger.info(
                f"{len(sub_agents_spawned)} sub-agent(s) spawned + "
                f"{len(pending_tool_gaps)} tool gap(s), delegating first "
                f"(tool gaps handled in next reflect cycle)"
            )
        else:
            logger.info(
                f"{len(sub_agents_spawned)} sub-agent(s) spawned, "
                f"routing to delegate"
            )
        return "delegate"

    # No agents spawned — check if we can create tools as fallback
    if pending_tool_gaps:
        logger.info(
            f"No sub-agents spawned, {len(pending_tool_gaps)} tool gap(s), "
            f"routing to tool_create"
        )
        return "tool_create"

    logger.info("No sub-agents spawned, routing to plan")
    return "plan"


def route_after_delegate(state: AgentState) -> str:
    """Route after the delegate node.

    Returns:
        "verify" — all delegations succeeded, verify results
        "execute" — some delegations failed, retry execution
    """
    delegation_results = state.get("delegation_results", [])
    if not delegation_results:
        return "verify"

    all_success = all(r.get("success", False) for r in delegation_results)
    if all_success:
        logger.info(
            f"All {len(delegation_results)} delegation(s) succeeded, "
            f"routing to verify"
        )
        return "verify"

    failed = sum(1 for r in delegation_results if not r.get("success", False))
    logger.warning(
        f"{failed}/{len(delegation_results)} delegation(s) failed, "
        f"routing to execute"
    )
    return "execute"
