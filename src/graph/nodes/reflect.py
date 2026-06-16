"""Reflect node — self-reflection on execution progress."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from src.graph.enums import Confidence, Phase, TaskComplexity
from src.graph.models import ReflectionResult
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.memory.manager import MemoryManager
    from src.tools.registry import ToolRegistry


def _ground_should_evolve(
    proposed: bool,
    goal_text: str,
    completed_steps: list[Any],
    errors: list[str],
    confidence: Confidence,
) -> bool:
    """Force ``should_evolve=True`` on objective success.

    A cautious model returns ``should_evolve=False`` even on HIGH-confidence
    deliverable successes — the battery's Q7/Q8 hit ``confidence=high`` yet the
    LLM said ``should_evolve=False``, and every other successful query likewise
    — so evolution never fired (0 mutations across all 10 queries). Grounding it
    in objective evidence (no errors +, for deliverable goals, the artifact
    actually on disk) makes evolution fire on genuine successes while preserving
    a model's own ``should_evolve=True`` via OR.

    Step count is intentionally NOT a gate for deliverable goals: a
    delegate-style run hands all execution to a sub-agent, so the main graph's
    ``completed_steps`` can be empty (or <3) even though the deliverable was
    produced and verify marked the run complete (battery-02 N8 —
    ``repo_map_builder`` did the work; main ``completed_steps`` stayed empty;
    the <3-step guard wrongly suppressed evolution). The step-count +
    confidence bar still applies to non-deliverable goals, where there is no
    artifact to confirm success.

    The on-disk check reuses execute's deliverable helpers so reflect, verify,
    and the write-nudge all agree on what "produced" means.
    """
    if proposed:
        return True
    if errors:
        return False
    from src.graph.nodes.execute import _deliverable_on_disk, _extract_goal_deliverable

    goal_deliverable = _extract_goal_deliverable(goal_text)
    if goal_deliverable is not None:
        # Deliverable goal: the artifact on disk is the strongest objective
        # evidence of success (this branch is reached at route_after_verify only
        # when is_complete=True). Fire regardless of step count — see docstring.
        return _deliverable_on_disk(goal_deliverable)
    # Non-deliverable goal: require ≥3 completed steps AND high confidence so a
    # trivial/empty success does not evolve. (Reached at route_after_verify only
    # when is_complete=True, but the reflect call sites use it mid-run, where the
    # step-count guard still matters.)
    if len(completed_steps or []) < 3:
        return False
    return confidence in {Confidence.HIGH, Confidence.VERY_HIGH}


async def reflect_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
    memory: MemoryManager | None = None,
    folding_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform self-reflection on the execution so far.

    Evaluates completed steps, tool results, and progress toward the goal.
    When gateway is provided, uses LLM for deeper analysis.
    Otherwise falls back to heuristic reflection.

    Args:
        state: Current agent state with execution results.
        gateway: Optional LLM gateway for LLM-enhanced reflection.
        tools: Optional ToolRegistry for gap detection.
        memory: Optional MemoryManager for persisting folded summaries.
        folding_cfg: Optional memory-folding configuration sourced from
            ``AgentSettings`` (triggers, thresholds, enabled flag).

    Returns:
        Partial state update with reflection results.
    """
    goal = state.get("current_goal")
    completed_steps = state.get("completed_steps", [])
    errors = state.get("errors", [])

    goal_text = goal.text if goal else "Unknown goal"
    logger.info(f"Reflecting on execution of: {goal_text[:60]}...")

    # ── Memory Folding Check ────────────────────────────────────────────
    if gateway is not None:
        fold_update = await _check_and_fold(state, gateway, memory, folding_cfg or {})
        if fold_update is not None:
            # Folding happened — return fold result. The fold reduces the
            # conversation history (RemoveMessage + a compressed summary) and
            # persists the structured summaries to warm memory. The normal
            # reflection logic will run on the next iteration.
            return fold_update

    # Try LLM reflection first, fall back to heuristics
    if gateway is not None:
        result = await _llm_reflect(gateway, state, tools)
        if result is not None:
            # Always run heuristic gap detection — the LLM may miss
            # sub-agent gaps even when task completes successfully
            plan_steps = state.get("plan_steps", [])
            heuristic_gaps = _detect_agent_gaps_heuristic(state, goal_text, plan_steps)
            if heuristic_gaps:
                existing_gaps = result.get("pending_agent_gaps", [])
                # Deduplicate: merge heuristic gaps with LLM-identified gaps,
                # then filter against what is already accumulated in state
                state_gaps = state.get("pending_agent_gaps", [])
                merged = list(set(existing_gaps + heuristic_gaps))
                new_merged = [g for g in merged if g not in state_gaps]
                if new_merged:
                    result["pending_agent_gaps"] = new_merged
                    logger.info(f"Sub-agent gaps detected (heuristic): {new_merged}")
            return result

    return _heuristic_reflect(state, goal_text, completed_steps, errors, tools)


def _heuristic_reflect(
    state: AgentState,
    goal_text: str,
    completed_steps: list[Any],
    errors: list[str],
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Heuristic-based reflection using completion ratios."""
    plan_steps = state.get("plan_steps", [])
    total_steps = len(plan_steps) if plan_steps else 1
    completed_count = len(completed_steps) if completed_steps else 0
    completion_ratio = completed_count / total_steps if total_steps > 0 else 0.0

    has_errors = bool(errors)
    if has_errors:
        confidence = Confidence.LOW
    elif completion_ratio >= 0.8:
        confidence = Confidence.HIGH
    elif completion_ratio >= 0.5:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    lessons: list[str] = []
    memory_obs: list[str] = []

    if completed_steps:
        lessons.append(f"Completed {completed_count}/{total_steps} steps successfully")

    if has_errors:
        lessons.append(f"Encountered {len(errors)} errors during execution")
        memory_obs.append(f"Error pattern: {errors[-1][:100] if errors else 'none'}")

    # Detect tool gaps from "Unknown tool" errors in tool results
    tool_results = state.get("tool_results", [])
    missing_tools: list[str] = []
    for tr in tool_results:
        if hasattr(tr, "error") and tr.error and "Unknown tool" in str(tr.error):
            # Confirm the tool is truly absent from registry (not just a typo)
            if tools is not None and tools.get_handler(tr.tool_name) is None:
                missing_tools.append(f"tool matching '{tr.tool_name}' capability")

    if missing_tools:
        lessons.append(f"Missing tool capabilities detected: {', '.join(missing_tools)}")

    # Detect code_executor overuse as a tool gap signal — if the agent
    # relied on code_executor 3+ times, a dedicated tool may be better
    code_exec_count = sum(
        1 for tr in tool_results
        if hasattr(tr, "tool_name") and tr.tool_name == "code_executor"
    )
    if code_exec_count >= 3 and not missing_tools:
        missing_tools.append(
            "dedicated tool for recurring code_executor usage pattern"
        )
        lessons.append(
            f"code_executor used {code_exec_count} times — consider a dedicated tool"
        )

    should_replan = completion_ratio < 0.3 and has_errors
    should_evolve = (
        completion_ratio >= 0.5
        and confidence in {Confidence.HIGH, Confidence.VERY_HIGH}
        and len(completed_steps) >= 3
    )
    # Ground in objective deliverable evidence so a genuine success triggers
    # evolution even when the heuristic's confidence bar isn't quite met.
    should_evolve = _ground_should_evolve(
        should_evolve, goal_text, completed_steps, errors, confidence
    )

    reflection = ReflectionResult(
        summary=f"Executed {completed_count}/{total_steps} steps for: {goal_text[:80]}",
        lessons_learned=lessons,
        confidence=confidence,
        should_evolve=should_evolve,
        should_replan=should_replan,
        memory_observations=memory_obs,
        cost_efficiency=1.0,
    )

    logger.info(
        f"Reflection: confidence={confidence.value}, "
        f"complete={completion_ratio:.0%}, "
        f"should_evolve={should_evolve}, should_replan={should_replan}"
    )

    result: dict[str, Any] = {
        "phase": Phase.VERIFY,
        "reflection": reflection,
        "confidence": confidence,
        "memory_observations": memory_obs,
    }

    if missing_tools:
        # Deduplicate against gaps already attempted (prevents infinite retry loops)
        attempted = state.get("attempted_tool_gaps", [])
        new_tool_gaps = [g for g in missing_tools if g not in attempted]
        if new_tool_gaps:
            result["pending_tool_gaps"] = new_tool_gaps

    # Detect sub-agent gaps: complex multi-part tasks with 6+ steps suggest
    # the need for specialized sub-agents to handle independent subtask categories
    missing_agents = _detect_agent_gaps_heuristic(state, goal_text, plan_steps)
    if missing_agents:
        # Deduplicate against gaps already ATTEMPTED this run. attempted_agent_gaps
        # is an operator.add accumulator (survives across cycles); agent_spawn
        # clears pending_agent_gaps every cycle, so deduping against pending would
        # re-detect the same gap endlessly → agent_spawn re-fires → fails →
        # re-converts to a tool gap → endless spawn→tool_create churn
        # (battery-02 N6: 19 tool_create entries, ~56 generations, 764s).
        attempted = state.get("attempted_agent_gaps", [])
        new_agent_gaps = [g for g in missing_agents if g not in attempted]
        if new_agent_gaps:
            result["pending_agent_gaps"] = new_agent_gaps

    return result


def _format_available_tools(tools: ToolRegistry | None) -> str:
    """Render the registered tools as a compact ``- name: description`` list.

    Feeds the reflect LLM the real inventory so it does not re-flag a
    capability (read/find/...) already provided by a builtin or by a tool
    created earlier in the run. Best-effort: any registry access failure
    degrades to ``"none registered"``.
    """
    if tools is None:
        return "none registered"
    try:
        defs = tools.list_tools()
    except Exception:  # noqa: BLE001 — mock registries / unavailable list_tools
        return "none registered"

    lines: list[str] = []
    for d in defs:
        fn = d.get("function", {}) if isinstance(d, dict) else {}
        name = fn.get("name", "") if isinstance(fn, dict) else ""
        if not name:
            continue
        desc = (fn.get("description", "") or "").strip()
        # First line only, capped — keep the inventory compact.
        desc = desc.split("\n", 1)[0][:90].strip()
        lines.append(f"- {name}: {desc}" if desc else f"- {name}")

    if not lines:
        return "none registered"
    return "\n".join(sorted(lines))


def _format_step_results(completed_steps: list[Any] | None) -> str:
    """Render completed steps as ``- description → result snippet`` lines.

    The reflect LLM was previously fed only step *descriptions* — it could see
    that a step was marked done but never *what it produced*. With no evidence
    of success, a cautious model conservatively returns low confidence +
    should_replan=True even when every step completed and the deliverable was
    written (Q9: 3/3 steps done, file written, yet reflect kept replanning).
    Surfacing the result text grounds the confidence judgment in real output.
    """
    if not completed_steps:
        return "None yet"
    lines: list[str] = []
    for step in completed_steps[-5:]:
        desc = (getattr(step, "description", "") or "").strip()
        result = (getattr(step, "result", "") or "").strip()
        # Collapse whitespace + cap so the evidence stays compact.
        result = " ".join(result.split())[:160]
        if result:
            lines.append(f"- {desc} → {result}" if desc else f"- {result}")
        elif desc:
            lines.append(f"- {desc}")
    return "\n".join(lines) if lines else "None yet"


def _format_successful_tools(tool_results: list[Any] | None) -> str:
    """Render successful tool calls as ``- name: output snippet`` lines.

    Complements ``tool_errors`` (failures only): the reflect LLM must see what
    *succeeded* — especially ``file_writer`` paths — to recognize that a
    required deliverable was produced. Without this, a deliverable-producing
    goal has no success evidence to raise confidence above LOW, so it never
    stops replanning.
    """
    if not tool_results:
        return "None"
    lines: list[str] = []
    for tr in tool_results:
        if not getattr(tr, "success", False):
            continue
        output = (getattr(tr, "output", "") or "").strip()
        if not output:
            continue
        name = getattr(tr, "tool_name", "tool")
        output = " ".join(output.split())[:160]
        lines.append(f"- {name}: {output}")
    return "\n".join(lines[-8:]) if lines else "None"


async def _llm_reflect(
    gateway: LLMGateway,
    state: AgentState,
    tools: ToolRegistry | None = None,
) -> dict[str, Any] | None:
    """Attempt LLM-based reflection. Returns None on failure."""
    try:
        from src.graph.prompts import (
            NODE_REFLECT,
            REFLECT_SYSTEM,
            REFLECT_USER,
            build_messages,
            select_techniques_for_node,
        )
        from src.graph.schemas import ReflectionAnalysis
        from src.llm.structured_output import StructuredOutputManager

        goal = state.get("current_goal")
        completed_steps = state.get("completed_steps", [])
        plan_steps = state.get("plan_steps", [])
        errors = state.get("errors", [])
        tool_results = state.get("tool_results", [])

        goal_text = goal.text if goal else "Unknown goal"
        total_steps = len(plan_steps) if plan_steps else 1
        completed_count = len(completed_steps) if completed_steps else 0

        completed_summary = "\n".join(
            f"- {s.description}" for s in completed_steps[-5:]
        ) if completed_steps else "None yet"

        errors_summary = "\n".join(f"- {e[:100]}" for e in errors[-3:]) if errors else "None"

        # Include tool failure details for gap detection
        tool_errors = []
        for tr in tool_results:
            if hasattr(tr, "error") and tr.error:
                tool_errors.append(f"- {tr.tool_name}: {tr.error[:100]}")
        tool_errors_str = "\n".join(tool_errors[-3:]) if tool_errors else "None"

        # Available-tool inventory: ground the LLM's missing_tools judgement in
        # what is ACTUALLY registered. Without this the reflect LLM re-flags
        # read/find/etc. capabilities already satisfied by builtins or by tools
        # created earlier in the run — each new phrasing bypasses the
        # attempted_tool_gaps string-dedup and drives an endless
        # tool_create -> replan loop (Q9 never reached a deliverable).
        available_tools = _format_available_tools(tools)

        # Execution evidence: step results + successful tool calls. Without
        # these the reflect LLM sees only step descriptions and tool ERRORS —
        # never what a step actually produced or that file_writer succeeded —
        # so it cannot confirm the deliverable exists and conservatively
        # returns low confidence + should_replan=True forever (Q9 loop).
        step_results = _format_step_results(completed_steps)
        successful_tools = _format_successful_tools(tool_results)

        user_prompt = REFLECT_USER.format(
            goal_text=goal_text,
            completed_count=completed_count,
            total_steps=total_steps,
            completed_summary=completed_summary,
            step_results=step_results,
            successful_tools=successful_tools,
            error_count=len(errors),
            errors_summary=errors_summary,
            tool_errors=tool_errors_str,
            available_tools=available_tools,
        )

        reflect_complexity = (
            goal.complexity if goal and goal.complexity else TaskComplexity.COMPLEX
        )
        techniques = select_techniques_for_node(
            complexity=reflect_complexity,
            node=NODE_REFLECT,
            goal_text=goal.text if goal else None,
        )
        messages = build_messages(str(REFLECT_SYSTEM), user_prompt, techniques)

        response = await gateway.acompletion(
            messages=messages,
            # Thread the *classified* complexity (§5 C.1). Falls back to COMPLEX
            # when unclassified — reflection is inherently analytical.
            complexity=reflect_complexity,
        )

        extractor = StructuredOutputManager()
        analysis = await extractor.extract(response.content, ReflectionAnalysis)
        if analysis is None:
            return None

        # Map LLM confidence to enum
        conf = Confidence.HIGH if analysis.confidence >= 0.7 else (
            Confidence.MEDIUM if analysis.confidence >= 0.4 else Confidence.LOW
        )

        # Ground should_evolve in objective success (deliverable on disk, no
        # errors, ≥3 steps). The model conservatively returns False even on
        # HIGH-confidence deliverable successes (Q7/Q8), so evolution never
        # fired across the battery without this override.
        should_evolve = _ground_should_evolve(
            bool(analysis.should_evolve), goal_text, completed_steps, errors, conf
        )

        reflection = ReflectionResult(
            summary=analysis.progress_assessment[:200],
            lessons_learned=analysis.lessons_learned,
            confidence=conf,
            should_evolve=should_evolve,
            should_replan=analysis.should_replan,
            memory_observations=analysis.memory_observations,
            cost_efficiency=1.0,
        )

        logger.info(
            f"LLM Reflection: confidence={conf.value}, "
            f"should_evolve={should_evolve} (model={analysis.should_evolve}), "
            f"should_replan={analysis.should_replan}"
        )

        result: dict[str, Any] = {
            "phase": Phase.VERIFY,
            "reflection": reflection,
            "confidence": conf,
            "memory_observations": analysis.memory_observations,
        }

        # Propagate missing tool gaps identified by LLM
        if analysis.missing_tools:
            # Deduplicate against gaps already attempted (prevents infinite retry loops)
            attempted = state.get("attempted_tool_gaps", [])
            new_tool_gaps = [g for g in analysis.missing_tools if g not in attempted]
            if new_tool_gaps:
                result["pending_tool_gaps"] = new_tool_gaps

        # Propagate missing sub-agent gaps identified by LLM
        if analysis.missing_sub_agents:
            # Deduplicate against gaps already ATTEMPTED this run, not the
            # (cleared-each-cycle) pending_agent_gaps — see heuristic path above
            # for the spawn→tool_create churn this prevents (battery-02 N6).
            attempted = state.get("attempted_agent_gaps", [])
            new_agent_gaps = [g for g in analysis.missing_sub_agents if g not in attempted]
            if new_agent_gaps:
                result["pending_agent_gaps"] = new_agent_gaps

        return result
    except Exception as e:
        logger.debug(f"LLM reflection failed, using heuristics: {e}")
        return None


def _detect_agent_gaps_heuristic(
    state: AgentState,
    goal_text: str,
    plan_steps: list[Any],
) -> list[str]:
    """Detect heuristically whether sub-agents would help.

    Triggers when:
    - Goal contains multi-part indicators ("and", "then", "also", "combine")
    - Plan has 6+ steps suggesting multiple independent subtasks
    - Not already using sub-agents (checked via state)

    Returns:
        List of sub-agent gap descriptions.
    """
    # Skip if sub-agents already spawned this run
    if state.get("sub_agents_spawned"):
        return []

    # Skip if agent spawning was already attempted (prevents infinite loop
    # when spawn fails and heuristic re-detects the same gaps)
    if state.get("attempted_agent_gaps"):
        return []

    gaps: list[str] = []

    # Multi-part goal detection
    multi_part_indicators = [" and ", " then ", " also ", " as well as ", " combine "]
    goal_lower = goal_text.lower()
    multi_part_count = sum(1 for indicator in multi_part_indicators if indicator in goal_lower)

    if multi_part_count >= 2 and len(plan_steps) >= 6:
        gaps.append(
            f"multi-part task with {len(plan_steps)} steps — "
            f"specialized sub-agents for independent subtask categories"
        )

    return gaps


# ── Memory Folding ─────────────────────────────────────────────────────


def _derive_verified_actions(tool_results: list[Any]) -> dict[str, dict[str, int]]:
    """Ground the folded tool memory in real execution, not LLM speculation.

    Implements GenericAgent's "No Execution, No Memory" principle: only tool
    actions that actually ran and succeeded at least once count as verified.
    Returns ``{tool_name: {"calls": int, "successes": int}}`` restricted to
    verified tools, computed purely from the run's own ToolResult records
    (no extra LLM call). Dict-shaped results are handled for symmetry with
    ``_serialize_tool_history``.
    """
    stats: dict[str, dict[str, int]] = {}
    for tr in tool_results:
        if isinstance(tr, dict):
            name = tr.get("tool_name")
            success = bool(tr.get("success"))
        else:
            name = getattr(tr, "tool_name", None)
            success = bool(getattr(tr, "success", False))
        if not name:
            continue
        bucket = stats.setdefault(name, {"calls": 0, "successes": 0})
        bucket["calls"] += 1
        if success:
            bucket["successes"] += 1
    return {
        name: counts for name, counts in stats.items() if counts["successes"] >= 1
    }


async def _check_and_fold(
    state: AgentState,
    gateway: LLMGateway,
    memory: MemoryManager | None = None,
    folding_cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Check if memory folding is needed and perform it.

    Returns a partial state update if folding occurred, or None. The update
    *reduces* the message history: every existing message is wrapped in a
    ``RemoveMessage`` (so LangGraph's ``add_messages`` reducer deletes it) and a
    single compressed summary is appended. The structured episode/working/tool
    summaries are also persisted to warm memory so later runs can recall them.

    Args:
        state: Current agent state.
        gateway: LLM gateway for generating compressed memories.
        memory: Optional MemoryManager for persisting folded summaries.
        folding_cfg: Optional folding configuration (triggers, thresholds,
            ``enabled`` flag). Defaults to ``MemoryFolder`` defaults.

    Returns:
        Partial state update dict, or None if no folding needed.
    """
    cfg = folding_cfg or {}
    if not cfg.get("enabled", True):
        return None

    from src.memory.folding import MemoryFolder

    # Drop the "enabled" flag — MemoryFolder.__init__ does not accept it.
    folder_kwargs = {k: v for k, v in cfg.items() if k != "enabled"}
    folder = MemoryFolder(gateway, **folder_kwargs)

    if not folder.should_fold(cast("dict[str, Any]", state)):
        return None

    try:
        result = await folder.fold(cast("dict[str, Any]", state))
        state_dict = cast("dict[str, Any]", state)

        # Bug B fix: actually shrink the history. add_messages deletes by id
        # when handed RemoveMessage entries, then appends the summary.
        removal = folder.build_removal_messages(state_dict)
        summary_msg = folder.build_summary_message(result)
        new_messages = [*removal, summary_msg]

        # Bug C fix: persist the structured summaries to warm memory so future
        # runs can recall them via retrieve_memory_node. Best-effort: a store
        # failure must not break the in-run fold.
        #
        # GenericAgent "No Execution, No Memory" + minimum-sufficient pointer:
        # the tool summary is grounded against the run's real ToolResult
        # stats (verified_actions) and tagged honestly — "verified" only when
        # ≥1 tool actually succeeded, "unverified" otherwise. Episode/working
        # memories stay narrative LLM summaries (they capture the story, not
        # tool facts), so they keep the plain kind tag.
        if memory is not None:
            verified_actions = _derive_verified_actions(
                state.get("tool_results", [])
            )
            payloads = {
                "episode": result.episode_memory,
                "working": result.working_memory,
                "tool": result.tool_memory,
            }
            for kind, payload in payloads.items():
                try:
                    if kind == "tool":
                        # Embed the grounded execution anchor alongside the
                        # LLM summary and label by verification status.
                        store_payload: Any = {
                            "summary": payload,
                            "verified_actions": verified_actions,
                        }
                        tag = "verified" if verified_actions else "unverified"
                        tags = ["folded_memory", kind, tag]
                    else:
                        store_payload = payload
                        tags = ["folded_memory", kind]
                    await memory.store_skill(
                        name=f"fold_{result.fold_number}_{kind}",
                        content=json.dumps(store_payload, ensure_ascii=False),
                        skill_type="folded_memory",
                        tags=tags,
                    )
                except Exception as exc:
                    logger.debug(f"Failed to persist folded {kind} memory: {exc}")

        return {
            "phase": Phase.REFLECT,
            "messages": new_messages,
            # fold_history is operator.add-accumulated, so inject the iteration
            # into each record so every fold carries when it actually fired.
            "fold_history": [
                {**result.to_dict(), "iteration": state.get("iteration_count", 0)}
            ],
            "last_fold_iteration": state.get("iteration_count", 0),
        }
    except Exception as exc:
        logger.warning(f"Memory folding failed (non-fatal): {exc}")
        return None
