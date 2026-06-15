"""Verify node — validates execution results against success criteria."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config.settings import get_settings
from src.graph.enums import Confidence, Phase, TaskComplexity
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway


# ─── Deliverable evidence ─────────────────────────────────────────────
# The verify node must not rubber-stamp "complete" from the agent's own
# self-report. It independently checks the filesystem for the deliverables
# the agent declared (files written via ``file_writer`` or promised to
# "save/write/export to <path>"). ``verify_node`` is an unsandboxed graph
# node, so it reads ``results_root`` directly — unlike the ``file_reader``
# tool, whose ``workspace_root`` sandbox cannot see ``results/`` deliverables.
# Paths that appear after an INPUT verb (read/open/fetch/…) are excluded so a
# missing *input* file is never mistaken for a missing *deliverable*.
_INPUT_PATH_RE = re.compile(
    r"\b(?:read|open|fetch|load|parse|import|inspect)\b[^.]*?"
    r"\b([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+)\b",
    re.IGNORECASE,
)
_SAVE_TO_RE = re.compile(
    r"\b(?:save|write|export|store|dump|output)\b[^.]*?"
    r"\b([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+)\b",
    re.IGNORECASE,
)
_DIR_OUTPUT_RE = re.compile(
    r"\b(?:create|generate|produce|make)\b[^.]*?"
    r"\b(?:in(?:to)?|under|at)\s+([A-Za-z0-9_][A-Za-z0-9_./-]*?)"
    r"(?:\s|,|\.(?:\s|$)|$)",
    re.IGNORECASE,
)


async def verify_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
) -> dict[str, Any]:
    """Verify execution results against the goal's success criteria.

    Independently inspects the filesystem for declared deliverables (files the
    agent wrote via ``file_writer`` or promised to "save/write/export to
    <path>") so the verdict rests on evidence, not the agent's self-report.
    When gateway is provided, uses LLM for semantic verification; otherwise
    falls back to heuristics. In both paths, missing/empty declared
    deliverables force ``is_complete=False`` and append a state error.

    Args:
        state: Current agent state with reflection and execution results.
        gateway: Optional LLM gateway for LLM-enhanced verification.

    Returns:
        Partial state update with verification result.
    """
    goal = state.get("current_goal")
    reflection = state.get("reflection")
    completed_steps = state.get("completed_steps", [])
    errors = state.get("errors", [])
    plan_steps = state.get("plan_steps", [])
    step_index = state.get("current_step_index", 0)
    confidence = state.get("confidence", Confidence.MEDIUM)

    goal_text = goal.text if goal else "Unknown goal"
    logger.info(f"Verifying results for: {goal_text[:60]}...")

    # Independently verify declared deliverables exist on disk. This is the
    # guard against rubber-stamping: even if the LLM is optimistic, missing or
    # empty deliverables force an incomplete verdict and surface a state error.
    deliverable_paths = _extract_deliverable_paths(state)
    evidence_text, missing_deliverables, empty_deliverables = _check_deliverables(
        deliverable_paths
    )
    deliverable_problems = missing_deliverables + empty_deliverables

    if deliverable_problems:
        logger.warning(
            f"Deliverable evidence shows {len(deliverable_problems)} missing/empty "
            f"output(s): {', '.join(deliverable_problems[:5])}"
        )

    # Try LLM verification first, fall back to heuristics
    if gateway is not None:
        result = await _llm_verify(gateway, state, evidence_text)
        if result is not None:
            # First clamp: a missing/empty declared deliverable forces incomplete
            # (evidence beats the agent's self-report). Then the symmetric clamp:
            # a present, non-empty declared deliverable + all steps done + no
            # errors forces COMPLETE (evidence beats LLM pessimism). Without the
            # second clamp a deliverable-producing goal loops verify→plan until
            # the iteration hard-cap because the verify LLM rarely self-reports
            # 100% (Q9 wrote results/q9_onboarding.md yet verify returned 0%).
            result = _enforce_deliverables(result, deliverable_paths, deliverable_problems)
            return _force_complete_on_evidence(
                result, state, deliverable_paths, deliverable_problems
            )

    return _heuristic_verify(
        state,
        goal_text,
        reflection,
        completed_steps,
        errors,
        plan_steps,
        step_index,
        confidence,
        deliverable_problems,
    )


def _heuristic_verify(
    _state: AgentState,  # noqa: ARG001 — kept for interface consistency
    goal_text: str,
    reflection: Any,
    completed_steps: list[Any],
    errors: list[str],
    plan_steps: list[Any],
    step_index: int,
    confidence: Confidence,
    deliverable_problems: list[str],
) -> dict[str, Any]:
    """Heuristic verification based on step completion, errors, and deliverables."""
    total_steps = len(plan_steps)
    completed_count = len(completed_steps)
    has_errors = bool(errors)

    all_steps_done = step_index >= total_steps if total_steps > 0 else True
    no_errors = not has_errors
    high_confidence = confidence in {Confidence.HIGH, Confidence.VERY_HIGH}
    deliverable_ok = not deliverable_problems

    is_complete = all_steps_done and no_errors and high_confidence and deliverable_ok

    final_output = ""
    if is_complete:
        if reflection and hasattr(reflection, "summary"):
            final_output = reflection.summary
        else:
            completed_descriptions = [
                s.description for s in completed_steps
                if hasattr(s, "description")
            ]
            final_output = (
                f"Task completed successfully.\n"
                f"Goal: {goal_text}\n"
                f"Steps completed: {completed_count}/{total_steps}\n"
                f"Results: {'; '.join(completed_descriptions[:5])}"
            )
        logger.info("Verification PASSED — task complete")
    else:
        reasons = []
        if not all_steps_done:
            reasons.append(f"steps remaining ({step_index}/{total_steps})")
        if has_errors:
            reasons.append(f"{len(errors)} errors")
        if not high_confidence:
            reasons.append(f"low confidence ({confidence.value})")
        if not deliverable_ok:
            reasons.append(f"{len(deliverable_problems)} deliverable(s) missing/empty")
        logger.info(f"Verification incomplete: {', '.join(reasons)}")

    state_update: dict[str, Any] = {
        "phase": Phase.COMPLETE if is_complete else Phase.EXECUTE,
        "is_complete": is_complete,
        "final_output": final_output,
    }
    # Surface missing/empty deliverables as state errors (operator.add reducer
    # appends) so they reach the report instead of being silently swallowed.
    if deliverable_problems:
        state_update["errors"] = [
            f"verify: deliverable not present — {p}" for p in deliverable_problems[:5]
        ]
    return state_update


async def _llm_verify(
    gateway: LLMGateway,
    state: AgentState,
    evidence_text: str,
) -> dict[str, Any] | None:
    """Attempt LLM-based verification. Returns None on failure."""
    try:
        from src.graph.prompts import (
            NODE_VERIFY,
            VERIFY_SYSTEM,
            VERIFY_USER,
            TechniqueSelector,
            build_messages,
        )
        from src.graph.schemas import VerificationResult
        from src.llm.structured_output import StructuredOutputManager

        goal = state.get("current_goal")
        completed_steps = state.get("completed_steps", [])
        errors = state.get("errors", [])
        plan_steps = state.get("plan_steps", [])
        reflection = state.get("reflection")

        goal_text = goal.text if goal else "Unknown goal"
        success_criteria = ""
        if goal and hasattr(goal, "success_criteria") and goal.success_criteria:
            success_criteria = "; ".join(goal.success_criteria)
        else:
            success_criteria = "Goal is fully achieved"

        total_steps = len(plan_steps)
        completed_count = len(completed_steps)

        completed_summary = "\n".join(
            f"- {s.description}: {getattr(s, 'result', 'done')}" for s in completed_steps[-5:]
        ) if completed_steps else "None yet"

        final_output = ""
        if reflection and hasattr(reflection, "summary"):
            final_output = reflection.summary

        user_prompt = VERIFY_USER.format(
            goal_text=goal_text,
            success_criteria=success_criteria,
            completed_summary=completed_summary,
            completed_count=completed_count,
            total_steps=total_steps,
            error_count=len(errors),
            final_output=final_output or "In progress",
            evidence=evidence_text,
        )

        verify_complexity = (
            goal.complexity if goal and goal.complexity else TaskComplexity.SIMPLE
        )
        goal_pattern = TechniqueSelector.infer_goal_pattern(goal.text if goal else None)
        techniques = TechniqueSelector().select(
            complexity=verify_complexity,
            node=NODE_VERIFY,
            goal_pattern=goal_pattern,
        )
        messages = build_messages(str(VERIFY_SYSTEM), user_prompt, techniques)

        response = await gateway.acompletion(
            messages=messages,
            # Thread the *classified* complexity so verification of a CRITICAL
            # goal uses a stronger model rather than always SIMPLE (§5 C.1).
            complexity=verify_complexity,
        )

        extractor = StructuredOutputManager()
        verification = await extractor.extract(response.content, VerificationResult)
        if verification is None:
            return None

        is_complete = verification.is_complete

        output = (
            f"Verification: {verification.completion_percentage:.0f}% complete. "
            f"{verification.quality_assessment}"
        )
        if verification.gaps:
            output += f" Gaps: {'; '.join(verification.gaps[:3])}"

        logger.info(
            f"LLM Verification: complete={is_complete}, "
            f"progress={verification.completion_percentage:.0f}%"
        )

        return {
            "phase": Phase.COMPLETE if is_complete else Phase.EXECUTE,
            "is_complete": is_complete,
            "final_output": output,
        }
    except Exception as e:
        logger.debug(f"LLM verification failed, using heuristics: {e}")
        return None


def _enforce_deliverables(
    result: dict[str, Any],
    deliverable_paths: list[str],
    deliverable_problems: list[str],
) -> dict[str, Any]:
    """Override the (optimistic) LLM verdict when deliverables are missing.

    Independent filesystem evidence wins over the agent's self-report: if the
    agent declared deliverables that are absent or empty, the task is not
    complete. Records the gap in state errors (``operator.add`` appends) so it
    surfaces in the validation report.
    """
    if not deliverable_paths or not deliverable_problems:
        return result

    summary = ", ".join(deliverable_problems[:5])
    final_output = (result.get("final_output") or "").strip()
    if "Declared deliverables not verified" not in final_output:
        final_output = f"{final_output} Declared deliverables not verified present: {summary}.".strip()
    return {
        **result,
        "is_complete": False,
        "phase": Phase.EXECUTE,
        "final_output": final_output,
        "errors": [f"verify: deliverable not present — {p}" for p in deliverable_problems[:5]],
    }


def _force_complete_on_evidence(
    result: dict[str, Any],
    state: AgentState,
    deliverable_paths: list[str],
    deliverable_problems: list[str],
) -> dict[str, Any]:
    """Override a pessimistic LLM verdict when objective evidence shows success.

    Symmetric counterpart to ``_enforce_deliverables``: the latter clamps
    *down* (missing deliverable → incomplete). This clamps *up*. If the agent
    declared a concrete deliverable, every declared path is present and
    non-empty on disk, all plan steps have executed, and no errors remain, the
    task IS complete — a non-empty file on disk is stronger evidence than the
    verify LLM's self-reported confidence, which (for cheaper models) rarely
    crosses 100% even when the artifact is plainly there. Without this clamp a
    deliverable-producing goal loops ``verify → plan → execute → verify`` until
    the iteration hard-cap, burning budget without ever terminating.
    """
    # Already complete, or no declared deliverable to ground the verdict in →
    # trust the LLM / fall through. A missing/empty deliverable is handled by
    # _enforce_deliverables and must NOT be overridden here.
    if result.get("is_complete"):
        return result
    if not deliverable_paths or deliverable_problems:
        return result

    plan_steps = state.get("plan_steps", []) or []
    step_index = state.get("current_step_index", 0)
    errors = state.get("errors", []) or []

    all_steps_done = step_index >= len(plan_steps) if plan_steps else True
    if not all_steps_done or errors:
        return result

    logger.info(
        "Verification forced COMPLETE — all declared deliverables present and "
        "non-empty on disk, all steps executed, no errors "
        "(objective evidence overrides pessimistic LLM verdict)"
    )
    final_output = (result.get("final_output") or "").strip()
    note = (
        "Objective deliverable evidence: all declared outputs are present and "
        "non-empty on disk."
    )
    if "Objective deliverable evidence" not in final_output:
        final_output = f"{final_output} {note}".strip()
    return {
        **result,
        "is_complete": True,
        "phase": Phase.COMPLETE,
        "final_output": final_output,
    }


def _extract_deliverable_paths(state: AgentState) -> list[str]:
    """Collect deliverable paths the agent declared it would produce.

    Authoritative source first (``file_writer`` step ``tool_input``), then
    phrasal cues in the goal, success criteria, and plan descriptions. Paths
    named after an INPUT verb (read/open/fetch/…) are excluded so a missing
    *input* file is never mistaken for a missing *deliverable*.

    Returns:
        De-duplicated, order-preserved list of raw path strings.
    """
    paths: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str | None) -> None:
        if not candidate:
            return
        cleaned = candidate.strip().strip("\"'`,.;()")
        if not cleaned or cleaned in seen:
            return
        if cleaned.startswith(("http://", "https://")):
            return
        seen.add(cleaned)
        paths.append(cleaned)

    goal = state.get("current_goal")
    plan_steps = state.get("plan_steps", []) or []
    completed_steps = state.get("completed_steps", []) or []

    # 1. Authoritative: actual file_writer invocations carry the exact path.
    for step in completed_steps:
        if getattr(step, "tool_name", None) != "file_writer":
            continue
        tool_input = getattr(step, "tool_input", None) or {}
        if isinstance(tool_input, dict):
            _add(tool_input.get("file_path"))

    # Build a text blob from the goal, its success criteria, and the plan.
    blob_parts: list[str] = []
    if goal:
        if getattr(goal, "text", None):
            blob_parts.append(goal.text)
        criteria = getattr(goal, "success_criteria", None) or []
        blob_parts.extend(criteria)
    for step in plan_steps:
        desc = getattr(step, "description", "") or ""
        if desc:
            blob_parts.append(desc)
    blob = "\n".join(blob_parts)

    # Inputs to exclude (read/open/fetch/parse targets).
    excluded = set(_INPUT_PATH_RE.findall(blob))
    for match in _SAVE_TO_RE.findall(blob):
        if match not in excluded:
            _add(match)
    for match in _DIR_OUTPUT_RE.findall(blob):
        _add(match.rstrip("/"))

    return paths


def _resolve_deliverable(raw: str) -> Path | None:
    """Resolve a declared deliverable to its on-disk path, or None if absent.

    ``file_writer`` writes under ``results_root`` and de-nests a leading
    ``results/`` component; we mirror that, then fall back to
    ``workspace_root`` and a literal path. Returns the first existing match.
    """
    agent = get_settings().agent
    roots = [Path(agent.results_root), Path(agent.workspace_root)]
    strip_names = {
        n.lower()
        for n in ("results", Path(agent.results_root).name, Path(agent.workspace_root).name)
    }
    parts = Path(raw).parts
    while len(parts) > 1 and parts[0].lower() in strip_names:
        parts = parts[1:]
    if not parts:
        return None

    candidates: list[Path] = []
    for root in roots:
        candidates.append(root.joinpath(*parts))
    candidates.append(Path(*parts))
    candidates.append(Path(raw))

    seen: set[str] = set()
    for candidate in candidates:
        try:
            if candidate.exists():
                resolved = candidate.resolve()
                key = str(resolved)
                if key not in seen:
                    seen.add(key)
                    return resolved
        except OSError:
            continue
    return None


def _check_deliverables(
    paths: list[str],
) -> tuple[str, list[str], list[str]]:
    """Inspect declared deliverables on disk.

    Args:
        paths: Raw deliverable path strings extracted from goal/plan/writes.

    Returns:
        ``(evidence_text, missing, empty)`` — a human-readable evidence block
        for the verify prompt plus the missing/empty lists used to hard-override
        the completion verdict.
    """
    present: list[str] = []
    missing: list[str] = []
    empty: list[str] = []

    for raw in paths:
        resolved = _resolve_deliverable(raw)
        if resolved is None:
            missing.append(raw)
            continue
        try:
            if resolved.is_dir():
                count = sum(1 for _ in resolved.iterdir())
                if count == 0:
                    empty.append(f"{raw} (empty directory)")
                else:
                    present.append(f"{raw} -> {resolved} (dir, {count} entries)")
            else:
                size = resolved.stat().st_size
                if size == 0:
                    empty.append(f"{raw} (0 bytes)")
                else:
                    present.append(f"{raw} -> {resolved} ({size} bytes)")
        except OSError as exc:
            missing.append(f"{raw} (unreadable: {exc})")

    lines: list[str] = []
    if present:
        lines.append("Present: " + "; ".join(present[:15]))
    if missing:
        lines.append("MISSING: " + "; ".join(missing[:15]))
    if empty:
        lines.append("EMPTY/INCOMPLETE: " + "; ".join(empty[:15]))
    evidence_text = (
        "\n".join(lines)
        if lines
        else "No concrete deliverable paths were declared or detected."
    )
    return evidence_text, missing, empty
