"""Verify node — validates execution results against success criteria."""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config.settings import get_settings
from src.graph.enums import Confidence, Phase, TaskComplexity
from src.graph.state import AgentState
from src.tools._paths import resolve_existing, results_root, strip_results_prefix

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
    r"\b(?:read(?:s|er|ers|ing)?|open(?:s|ed|ing)?|fetch(?:es|ed|ing)?|"
    r"load(?:s|ed|ing)?|pars(?:e|es|ed|ing)|import(?:s|ed|ing)?|"
    r"inspect(?:s|ed|ing)?)\b[^.]*?"
    r"\b([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+)\b",
    re.IGNORECASE,
)
_SAVE_TO_RE = re.compile(
    r"\b(?:sav(?:e|es|ed|ing)|writ(?:e|es|ing|ten|er)|export(?:s|ed|ing)?|"
    r"stor(?:e|es|ed|ing)|dump(?:s|ed|ing)?|output(?:s|ted|ting)?)\b[^.]*?"
    r"\b([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+)\b",
    re.IGNORECASE,
)
_DIR_OUTPUT_RE = re.compile(
    r"\b(?:creat(?:e|es|ed|ing)|generat(?:e|es|ed|ing)|produc(?:e|es|ed|ing)|"
    r"mak(?:e|es|ing|de))\b[^.]*?"
    r"\b(?:in(?:to)?|under|at)\s+([A-Za-z0-9_][A-Za-z0-9_./-]*?)"
    r"(?:\s|,|\.(?:\s|$)|$)",
    re.IGNORECASE,
)
# Continuation of a deliverable list right after a captured path: "a.md and
# b.json", "a.md, b.json", "a.md, and b.json", "a.md / b.md", "a.md & b.md".
# _SAVE_TO_RE only captures the FIRST path following each save-verb; this walks
# the separator-joined tail so a single verb introducing multiple deliverables
# ("write the report to quality_report.md and scorecard.json") yields the whole
# list, not just the first (battery-04 q2 — otherwise goal_satisfied checked one
# file and a present-but-incomplete pair could read as satisfied).
_DELIVERABLE_CONTINUATION_RE = re.compile(
    r"\s*(?:,\s*(?:and\s+|or\s+)?|/\s*|&\s*|(?:and|or)\s+)"
    r"([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+)",
    re.IGNORECASE,
)

# Phrasal-cue captures that are clearly not deliverable paths. _DIR_OUTPUT_RE
# can grab the determiner/pronoun right after the preposition ("Create the tool
# in a module" → captures "a"; "Produce the report under the results dir" →
# "the"), or a plurality/quantifier adjective ("Create a tool that ... appears
# in multiple files" → captures "multiple"). Treated as a deliverable, such a
# token reads as missing and loops verify→plan until the iteration hard-cap
# (observed: N1 ran 455s looping on a phantom "a"; the post-fix N5 re-run looped
# 4+ verify cycles on a phantom "multiple"). Single chars and these prose tokens
# are rejected in _add() before a capture ever becomes an expected deliverable.
_PATH_NOISE_TOKENS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "another", "every",
    "our", "your", "its", "their", "his", "her", "one", "two", "each",
    "some", "any", "all", "both", "such", "more", "most", "other", "new",
    # Plurality/quantifier adjectives _DIR_OUTPUT_RE grabs from plan prose
    # ("...appears in multiple/several files", "produce across various
    # modules") — never a legitimate standalone deliverable path component.
    "multiple", "several", "various", "numerous", "many", "different",
    "separate", "distinct", "individual", "consecutive", "successive",
    "form", "way", "report", "summary", "function", "method", "module",
    "file", "directory", "subdirectory", "folder", "markdown", "table",
    "script", "string", "section", "block", "part", "note",
    # Abbreviations whose internal dot matches the [stem].[ext] capture group
    # so they masquerade as a file path ("e.g" → stem "e" + ext "g"; "i.e" →
    # "i" + "e"), plus prose words leaked from goal/plan text (e.g. "mixed
    # human formats" → "mixed"). Without these, verify treats the token as a
    # missing deliverable and forces is_complete=False on an otherwise-finished
    # run (observed leaking across battery-03 q2/q3).
    "e.g", "eg", "i.e", "ie", "etc", "mixed", "human",
})


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
    (
        evidence_text,
        missing_deliverables,
        empty_deliverables,
        malformed_deliverables,
    ) = _check_deliverables(deliverable_paths)
    deliverable_problems = (
        missing_deliverables + empty_deliverables + malformed_deliverables
    )

    if deliverable_problems:
        logger.warning(
            f"Deliverable evidence shows {len(deliverable_problems)} missing/empty "
            f"output(s): {', '.join(deliverable_problems[:5])}"
        )

    # F-h goal-sufficiency: the GOAL names the deliverables that define success.
    # When those are all present + non-empty, plan-declared INTERMEDIATE artifacts
    # (a script the agent intended to write, an input file wrongly listed) are an
    # implementation detail — the agent may reach the goal deliverables via a
    # different valid path (sub-agent delegation that writes the report directly
    # instead of the planned audit_quality.py script). So only goal-deliverable
    # gaps block completion; intermediate gaps are advisory. verify checks
    # PRESENCE; the Phase-3 eval_enforce layer checks CORRECTNESS, so this cannot
    # rubber-stamp a present-but-wrong goal artifact.
    goal_satisfied, goal_paths = _goal_deliverables_satisfied(state)
    if deliverable_problems and goal_satisfied:
        logger.info(
            f"Goal deliverables satisfied {goal_paths}; {len(deliverable_problems)} "
            f"plan-intermediate deliverable problem(s) treated as advisory: "
            f"{', '.join(deliverable_problems[:5])}"
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
            result = _enforce_deliverables(
                result, deliverable_paths, deliverable_problems, goal_satisfied
            )
            result = _force_complete_on_evidence(
                result, state, deliverable_paths, deliverable_problems, goal_satisfied
            )
            # Phase 3: correctness eval. Runs only when EVAL_ENABLED, a GoalSpec
            # is registered for this run, and the verdict is already complete —
            # so a normal goal (no spec) and incomplete runs are untouched, and
            # the LLM-judge adds at most ~one call per completion attempt.
            return await _run_correctness_checks(result, state, deliverable_paths, gateway)

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
            build_messages,
            select_techniques_for_node,
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

        # Deliverable honesty: feed the deliverable's own content + the real tool
        # outputs as ground truth so the verifier can flag fabricated numbers
        # (battery-02 N6's synthesized counts) rather than rubber-stamping a
        # well-structured but dishonest deliverable.
        deliverable_content = _load_deliverable_content(
            _extract_deliverable_paths(state)
        )
        tool_outputs = _summarize_data_tool_outputs(
            state.get("tool_results", []) or []
        )
        # Programmatic fabrication guard (advisory): cross-check cited counts/
        # paths against the actual filesystem, independent of the LLM.
        grounding_warning = _spot_check_cited_paths(
            deliverable_content, tool_outputs
        )

        user_prompt = VERIFY_USER.format(
            goal_text=goal_text,
            success_criteria=success_criteria,
            completed_summary=completed_summary,
            completed_count=completed_count,
            total_steps=total_steps,
            error_count=len(errors),
            final_output=final_output or "In progress",
            evidence=evidence_text,
            deliverable_content=deliverable_content or "(no readable deliverable content)",
            tool_outputs=tool_outputs or "(no data-producing tool outputs)",
            grounding_warning=grounding_warning or "(no grounding warnings)",
        )

        verify_complexity = (
            goal.complexity if goal and goal.complexity else TaskComplexity.SIMPLE
        )
        techniques = select_techniques_for_node(
            complexity=verify_complexity,
            node=NODE_VERIFY,
            goal_text=goal.text if goal else None,
        )
        messages = build_messages(str(VERIFY_SYSTEM), user_prompt, techniques, node=NODE_VERIFY)

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
    goal_satisfied: bool = False,
) -> dict[str, Any]:
    """Override the (optimistic) LLM verdict when deliverables are missing.

    Independent filesystem evidence wins over the agent's self-report: if the
    agent declared deliverables that are absent or empty, the task is not
    complete. Records the gap in state errors (``operator.add`` appends) so it
    surfaces in the validation report.

    F-h: when the goal's own named deliverables are present + non-empty
    (``goal_satisfied``), the gap must be in a plan-declared INTERMEDIATE
    artifact, not a goal deliverable. Those are advisory — the agent reached the
    goal via a different path — so the LLM verdict is preserved (not forced
    incomplete) and the intermediate gaps are logged, never added to state
    errors (which ``_force_complete_on_evidence`` treats as a blocker).
    """
    if not deliverable_paths or not deliverable_problems:
        return result

    if goal_satisfied:
        logger.info(
            f"_enforce_deliverables: goal satisfied; treating "
            f"{len(deliverable_problems)} plan-intermediate deliverable gap(s) as "
            f"advisory: {', '.join(deliverable_problems[:5])}"
        )
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
    goal_satisfied: bool = False,
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

    F-h: ``goal_satisfied`` widens the clamp. When the GOAL's own named
    deliverables are all present + non-empty, force-complete even if plan-
    declared INTERMEDIATE artifacts are missing — the agent reached the goal
    deliverables via a valid path (sub-agent delegation that wrote the report
    directly instead of the planned ``audit_quality.py`` script). The F4
    goal-match cross-check is auto-satisfied (the goal deliverables ARE present),
    and eval_enforce validates their correctness, so a present-but-wrong goal
    artifact is still caught.
    """
    if result.get("is_complete"):
        return result

    plan_steps = state.get("plan_steps", []) or []
    step_index = state.get("current_step_index", 0)
    errors = state.get("errors", []) or []

    all_steps_done = step_index >= len(plan_steps) if plan_steps else True

    if goal_satisfied:
        # F-h.2: the GOAL's own named deliverables are present + non-empty +
        # well-formed on disk AND every plan step executed → the run reached its
        # goal. ``state.errors`` accumulates via operator.add and is NEVER
        # cleared, so it still holds stale entries from EARLIER verify cycles —
        # recorded before the agent wrote the deliverable ("verify: deliverable
        # not present") or a write-nudge gap on a plan-intermediate INPUT file —
        # that no longer reflect reality. Honoring those here would loop an
        # objectively-complete goal verify→replan until the iteration hard-cap
        # (battery-04 q2). The Phase-3 eval_enforce layer independently validates
        # CORRECTNESS, so a present-but-wrong goal artifact is still caught; this
        # path only asserts PRESENCE. Require only that all steps executed.
        if not all_steps_done:
            return result
        reason = (
            "goal deliverables present and non-empty on disk; "
            "plan-intermediate gaps + stale errors advisory (battery-04 q2 F-h)"
        )
    else:
        # No goal-sufficiency: require EVERY declared deliverable present (no
        # problems), all steps done, and no errors before trusting the optimistic
        # clamp. A missing/empty declared deliverable is handled by
        # _enforce_deliverables and must NOT be overridden here.
        if not all_steps_done or errors:
            return result
        if not deliverable_paths or deliverable_problems:
            return result
        # F4: before trusting agent-declared deliverables, cross-check that at
        # least one of them matches the GOAL's expected deliverable. If the goal
        # names a specific file (e.g. ``q01/normalized.csv``) and none of the
        # on-disk deliverables match its basename, the agent produced only
        # intermediates — do NOT force-complete, or a half-finished run is done.
        goal_expected = _extract_goal_deliverables(state)
        if goal_expected:
            present_basenames = {Path(p).name for p in deliverable_paths}
            expected_basenames = [Path(g).name for g in goal_expected]
            if not any(name in present_basenames for name in expected_basenames):
                logger.info(
                    "Force-complete declined: goal expects %s but none of the present "
                    "deliverables (%s) match — agent produced intermediates only",
                    expected_basenames[:3],
                    sorted(present_basenames)[:5],
                )
                return result
        reason = (
            "all declared deliverables present and non-empty on disk "
            "(objective evidence overrides pessimistic LLM verdict)"
        )

    logger.info(f"Verification forced COMPLETE — {reason}; all steps executed")
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


async def _run_correctness_checks(
    result: dict[str, Any],
    state: AgentState,
    deliverable_paths: list[str],
    gateway: LLMGateway | None,
) -> dict[str, Any]:
    """Run the registered GoalSpec's correctness checks and fold them in.

    No-op (returns ``result`` unchanged) when eval is disabled, no spec is
    registered for the run, or the verdict is not yet complete — so an ordinary
    goal and an in-progress verify are both completely unaffected, and the
    LLM-judge adds at most ~one call per completion attempt. When checks run,
    the aggregate score + per-check breakdown are written to state and the
    durable eval store (non-fatal). With ``eval_enforce``, a failing check
    downgrades a "complete" verdict so the agent retries — but only while
    iterations remain, so a strict check can never loop a run past its
    iteration hard-cap (the final allowed verify completes regardless).
    """
    settings = get_settings()
    if not settings.eval.eval_enabled or not result.get("is_complete"):
        return result

    from src.eval.checks import run_checks
    from src.eval.golden import lookup_goal_spec

    spec = lookup_goal_spec(state.get("eval_goal_spec_id"))
    if spec is None:
        return result

    correctness = await run_checks(spec, deliverable_paths, state, gateway=gateway)
    result = {
        **result,
        "eval_correctness_score": correctness.overall_score,
        "eval_checks": [c.model_dump(mode="json") for c in correctness.checks],
        "eval_correctness_passed": correctness.passed,
    }

    # Persist to the durable store (non-fatal: a DB hiccup is logged inside).
    try:
        from src.eval.store import EvalStore

        await EvalStore().record_correctness(
            correctness, goal_id=spec.name, run_id=state.get("thread_id", "")
        )
    except Exception as exc:  # noqa: BLE001 — eval persistence must never break verify
        logger.debug("Eval store write skipped: {}", exc)

    if settings.eval.eval_enforce and not correctness.passed:
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 60)
        if iteration < max_iter - 1:
            final_output = (result.get("final_output") or "").strip()
            logger.info(
                "Correctness enforcement: downgrading complete→incomplete "
                "(score={:.2f}, iter={}/{})",
                correctness.overall_score,
                iteration,
                max_iter,
            )
            return {
                **result,
                "is_complete": False,
                "phase": Phase.EXECUTE,
                "final_output": f"{final_output} Correctness checks failed (score={correctness.overall_score:.2f}).".strip(),
                "errors": [
                    *result.get("errors", []),
                    f"verify: correctness checks failed (score={correctness.overall_score:.2f})",
                ],
            }
    return result


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
        # Dotfile basenames (".gitkeep", ".gitignore", ".keep", ".DS_Store") are
        # VCS/placeholder files the agent writes to create or preserve an output
        # directory — never a user-facing deliverable. Without this, a 0-byte
        # ".gitkeep" written via file_writer is treated as an empty declared
        # deliverable, forces is_complete=False, and loops verify→plan until the
        # iteration hard-cap despite the real deliverables being present
        # (battery-04 q2: the auditor wrote results/q02/.gitkeep to create the dir).
        if cleaned.rsplit("/", 1)[-1].startswith("."):
            return
        if len(cleaned) < 2 or cleaned.lower() in _PATH_NOISE_TOKENS:
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


def _extract_goal_deliverables(state: AgentState) -> list[str]:
    """Extract deliverable cues named by the GOAL itself (text + success criteria).

    Distinct from ``_extract_deliverable_paths``: that mixes authoritative
    ``file_writer`` writes with goal/plan cues and computes the input-exclusion
    set over the *whole* goal+plan blob. This isolates ONLY what the goal asks
    for, over a goal-only blob — so a deliverable the goal says to "write" is
    not dropped because some plan step later reads it as input. Used by
    ``_force_complete_on_evidence`` to cross-check the agent actually produced a
    GOAL deliverable (not merely intermediate outputs). (F4 fix.)
    """
    goal = state.get("current_goal")
    if not goal:
        return []

    blob_parts: list[str] = []
    if getattr(goal, "text", None):
        blob_parts.append(goal.text)
    criteria = getattr(goal, "success_criteria", None) or []
    blob_parts.extend(criteria)
    blob = "\n".join(blob_parts)
    if not blob.strip():
        return []

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
        # Dotfile basenames (".gitkeep", ".gitignore", ".keep", ".DS_Store") are
        # VCS/placeholder files the agent writes to create or preserve an output
        # directory — never a user-facing deliverable. Without this, a 0-byte
        # ".gitkeep" written via file_writer is treated as an empty declared
        # deliverable, forces is_complete=False, and loops verify→plan until the
        # iteration hard-cap despite the real deliverables being present
        # (battery-04 q2: the auditor wrote results/q02/.gitkeep to create the dir).
        if cleaned.rsplit("/", 1)[-1].startswith("."):
            return
        if len(cleaned) < 2 or cleaned.lower() in _PATH_NOISE_TOKENS:
            return
        seen.add(cleaned)
        paths.append(cleaned)

    excluded = set(_INPUT_PATH_RE.findall(blob))
    for m in _SAVE_TO_RE.finditer(blob):
        # A save-verb may introduce a LIST of deliverables ("write the report to
        # a.md and b.json, or c.csv"). _SAVE_TO_RE captures only the first path
        # after each verb; walk the separator-joined tail (via
        # _DELIVERABLE_CONTINUATION_RE) to collect the whole list (battery-04 q2
        # F-i: "writes ... to quality_report.md and scorecard.json" otherwise
        # yielded only quality_report.md, so goal_satisfied checked one file).
        deliverables = [m.group(1)]
        pos = m.end()
        while True:
            cm = _DELIVERABLE_CONTINUATION_RE.match(blob, pos)
            if not cm:
                break
            deliverables.append(cm.group(1))
            pos = cm.end()
        for cand in deliverables:
            if cand not in excluded:
                _add(cand)
    for match in _DIR_OUTPUT_RE.findall(blob):
        _add(match.rstrip("/"))

    return paths


def _goal_deliverables_satisfied(state: AgentState) -> tuple[bool, list[str]]:
    """F-h goal-sufficiency probe: are the GOAL's own named deliverables on disk?

    Returns ``(satisfied, goal_paths)`` where ``satisfied`` is True iff the goal
    names ≥1 concrete deliverable AND every goal-named deliverable is present,
    non-empty, and well-formed on disk. When the goal names no specific
    deliverable, returns ``(False, [])`` so callers fall back to whole-declared-
    set enforcement (prior behavior).

    Rationale: the goal's named deliverables (e.g. ``results/q02/quality_report.md``
    and ``results/q02/scorecard.json``) define what success MEANS. Plan-declared
    INTERMEDIATE artifacts (a script the agent intended to write, or an input
    file wrongly listed as a deliverable) are an implementation detail — the
    agent may reach the goal deliverables via a different valid path (sub-agent
    delegation that writes the report directly). Re-checks the goal paths on disk
    directly (no string parsing of the problems list).
    """
    goal_paths = _extract_goal_deliverables(state)
    if not goal_paths:
        return (False, [])
    _evidence, missing, empty, malformed = _check_deliverables(goal_paths)  # noqa: F841
    satisfied = not missing and not empty and not malformed
    return (satisfied, goal_paths)


def _resolve_deliverable(raw: str) -> Path | None:
    """Resolve a declared deliverable to its on-disk path, or None if absent.

    De-nesting is delegated to the shared resolver so the strip set matches
    ``file_writer`` exactly. We then check, in order: ``results_root`` (where
    ``file_writer`` lands), ``workspace_root`` (inputs/fixtures), and a literal
    path. Returns the first existing match.
    """
    candidates: list[Path] = []
    for base in ("results", "workspace"):
        try:
            candidates.append(resolve_existing(raw, base=base))
        except ValueError:
            continue
    parts = strip_results_prefix(Path(raw).parts)
    if parts:
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


# ─── Template-placeholder leak (F-j) ──────────────────────────────────
# A shipped text/markdown deliverable may contain unsubstituted template
# placeholders — the LLM wrote a .format()/Jinja template but the codegen never
# filled it in (battery-04 q2: quality_report.md shipped 13 prose tokens like
# "{uniqueness_pct}", "{total_rows}", "{dup_event_ids}"). Such a file is
# present and non-empty — so it passes the existence check and would otherwise
# be rubber-stamped complete — yet is not a usable deliverable. We scan the
# prose for residue of three forms (.format() {name}, Jinja {{ name }}, Jinja
# {% tag %}) after stripping fenced (```) and inline (`) code spans, so
# legitimate f-strings / JSON / Jinja shown *inside* code are not mistaken for
# leaks. >= _PLACEHOLDER_LEAK_MIN distinct residues in prose is treated as
# malformed: it forces is_complete=False and blocks the optimistic force-
# complete clamp, so the agent retries until the report is fully rendered.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
# .format() {name}: lowercase snake_case identifier (>=3 chars) in single
# braces. The lookarounds exclude {{ }} (Jinja) and {{{ }}} (escaped Jinja).
_PLACEHOLDER_LEAK_FORMAT_RE = re.compile(r"(?<!\{)\{([a-z_][a-z0-9_]{2,})\}(?!\})")
_PLACEHOLDER_LEAK_JINJA_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_.-]*)\s*\}\}")
_PLACEHOLDER_LEAK_JINJA_TAG_RE = re.compile(r"\{%[^}%]*%\}")
_PLACEHOLDER_LEAK_MIN = 2
_PLACEHOLDER_LEAK_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})


def _strip_markdown_code(text: str) -> str:
    """Remove fenced and inline code spans so placeholder tokens that are
    legitimate source *inside* code (f-strings, JSON, Jinja shown as docs) are
    not mistaken for unsubstituted leaks in the surrounding prose."""
    text = _FENCED_CODE_RE.sub("", text)
    return _INLINE_CODE_RE.sub("", text)


def _placeholder_leak_reason(path: Path) -> str | None:
    """Return a malformed reason if a text/markdown deliverable shipped with
    unsubstituted template-placeholder residue in its prose, else ``None``.

    A single incidental brace token in genuine prose is not flagged (the
    ``_PLACEHOLDER_LEAK_MIN`` threshold requires multiple distinct residues,
    which only a real template-substitution failure produces).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - filesystem race
        return f"unreadable ({exc})"
    if not text.strip():
        return None  # 0-byte files are reported via the size check
    prose = _strip_markdown_code(text)
    found: set[str] = set()
    for pat in (
        _PLACEHOLDER_LEAK_FORMAT_RE,
        _PLACEHOLDER_LEAK_JINJA_VAR_RE,
        _PLACEHOLDER_LEAK_JINJA_TAG_RE,
    ):
        for m in pat.finditer(prose):
            # Jinja {% tag %} has no capture group — use the whole match.
            found.add(m.group(1) if m.lastindex else m.group(0))
    if len(found) >= _PLACEHOLDER_LEAK_MIN:
        sample = ", ".join(sorted(found)[:6])
        return f"unsubstituted template placeholders ({len(found)}: {sample})"
    return None


def _classify_deliverable_format(path: Path) -> str | None:
    """Return a malformed-format reason if a present, non-empty deliverable does
    not parse as its declared format, else ``None``.

    Structured data formats (csv/json/jsonl) are parsed directly. Free-form
    text/markdown is scanned for unsubstituted template-placeholder residue
    (F-j). Catches an agent that overwrites a named deliverable (e.g.
    ``normalized.csv``) with report or diagnostic text, AND one that ships a
    report whose template variables were never rendered: in both cases the file
    is present and non-empty — so it passes the existence check and would
    otherwise be rubber-stamped complete — but is not a valid deliverable.
    Observed live (battery-04 q1: an 11-step "VERIFICATION REPORT" clobbered
    ``normalized.csv``; q2: ``quality_report.md`` shipped with 13 ``{var}``).
    """
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".json", ".jsonl"}:
        if suffix in _PLACEHOLDER_LEAK_TEXT_SUFFIXES:
            return _placeholder_leak_reason(path)
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - filesystem race
        return f"unreadable ({exc})"
    if not text.strip():
        return None  # 0-byte files are already reported via the size check

    if suffix == ".csv":
        # A valid CSV deliverable's first non-empty row must be comma-delimited
        # (>= 2 fields). A report dumped into a .csv has a prose first line with
        # no comma ("VERIFICATION REPORT - ...") -> not tabular -> malformed.
        reader = csv.reader(io.StringIO(text))
        for row in reader:
            if any(cell.strip() for cell in row):
                if len(row) < 2:
                    return "first row is not comma-delimited (not tabular CSV)"
                return None
        return "no non-empty rows (malformed CSV)"
    if suffix == ".json":
        try:
            json.loads(text)
        except ValueError as exc:
            reason = str(exc).splitlines()[0] if str(exc) else "parse error"
            return f"invalid JSON ({reason[:80]})"
        return None
    # suffix == ".jsonl"
    total = 0
    bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            json.loads(line)
        except ValueError:
            bad += 1
    if total == 0:
        return "no records (malformed JSONL)"
    if bad:
        return f"{bad}/{total} lines are not valid JSON"
    return None


def _check_deliverables(
    paths: list[str],
) -> tuple[str, list[str], list[str], list[str]]:
    """Inspect declared deliverables on disk.

    Args:
        paths: Raw deliverable path strings extracted from goal/plan/writes.

    Returns:
        ``(evidence_text, missing, empty, malformed)`` — a human-readable
        evidence block for the verify prompt plus the missing/empty/malformed
        lists used to hard-override the completion verdict. ``malformed`` holds
        present, non-empty files that do not parse as their declared format
        (e.g. report text written into a ``.csv``); like missing/empty it forces
        ``is_complete=False`` and blocks the optimistic force-complete clamp.
    """
    present: list[str] = []
    missing: list[str] = []
    empty: list[str] = []
    malformed: list[str] = []

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
                    reason = _classify_deliverable_format(resolved)
                    if reason is not None:
                        malformed.append(f"{raw} ({reason})")
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
    if malformed:
        lines.append(
            "MALFORMED (present but not valid format): "
            + "; ".join(malformed[:15])
        )
    evidence_text = (
        "\n".join(lines)
        if lines
        else "No concrete deliverable paths were declared or detected."
    )
    return evidence_text, missing, empty, malformed


# ─── Deliverable honesty ─────────────────────────────────────────────
# The verify LLM must be able to compare the deliverable's *claims* against the
# *real* tool outputs, so a well-structured but fabricated deliverable (battery-02
# N6: synthesized duplicate counts) is detected. We feed (a) the deliverable's
# own on-disk content and (b) a compact summary of the data-producing tool
# outputs as ground truth; the verify prompt instructs the LLM to flag any
# quantitative claim that is not present in or supported by that ground truth.
_DELIVERABLE_CONTENT_CAP = 6000  # total chars of deliverable text fed to verify
_TOOL_OUTPUT_CAP = 600  # per-tool chars
# Max data-tool outputs summarized as ground truth — operator-configurable via
# AgentSettings (VERIFY_MAX_DATA_TOOLS). Read at call-time in _summarize_tool_outputs.
# file_writer outputs ("Successfully wrote N bytes to <path>") carry no data, so
# they are excluded: nothing in them grounds or contradicts the deliverable's
# quantitative claims, and including them would only pad the verify prompt.
_NON_DATA_TOOLS = frozenset({"file_writer"})


def _load_deliverable_content(paths: list[str]) -> str:
    """Read the on-disk content of declared deliverables for honesty checking.

    Only *present* deliverables are read (a missing one is already a hard
    failure via ``_enforce_deliverables``). The combined text is capped so the
    verify prompt stays bounded. Returns ``""`` if nothing readable is found.
    """
    chunks: list[str] = []
    total = 0
    for raw in paths:
        if total >= _DELIVERABLE_CONTENT_CAP:
            break
        resolved = _resolve_deliverable(raw)
        if resolved is None or resolved.is_dir():
            continue
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        remaining = _DELIVERABLE_CONTENT_CAP - total
        if len(text) > remaining:
            text = text[:remaining] + "\n…[truncated]"
        chunks.append(f"--- {raw} ---\n{text}")
        total += len(text)
    return "\n\n".join(chunks)


def _summarize_data_tool_outputs(tool_results: list[Any]) -> str:
    """Summarize successful tool outputs as ground truth for the verify LLM.

    One compact line per data-producing tool (file_writer excluded), each
    truncated, so the verifier can check that numbers/claims in the deliverable
    appear in real tool output rather than being fabricated.
    """
    lines: list[str] = []
    for tr in tool_results:
        name = getattr(tr, "tool_name", "")
        if name in _NON_DATA_TOOLS or not getattr(tr, "success", False):
            continue
        output = (getattr(tr, "output", "") or "").strip()
        if not output:
            continue
        if len(output) > _TOOL_OUTPUT_CAP:
            output = output[:_TOOL_OUTPUT_CAP] + "…[truncated]"
        # collapse internal whitespace/newlines for a compact one-liner
        output = " ".join(output.split())
        lines.append(f"- {name}: {output}")
        if len(lines) >= get_settings().agent.verify_max_data_tools:
            break
    return "\n".join(lines)


# ─── Filesystem spot-check (defense-in-depth fabrication guard) ───────
# Programmatic cross-check that the numbers/paths the deliverable cites are
# consistent with the actual filesystem. Unlike the LLM honesty check above
# (which compares deliverable prose to tool-output prose), this catches a tool
# whose *output itself* is fabricated — e.g. a script that prints "found 12
# duplicates" while writing zero files (battery-02 N5's double-nest bug).
# Advisory only: the warning is interpolated into the verify prompt and NEVER
# touches ``_enforce_deliverables``/``_force_complete_on_evidence``, so it
# cannot loop or force a false-incomplete verdict.
_COUNT_NOUN_RE = re.compile(
    r"\b(\d+)\s+(files?|duplicates?|duplicate files?|matches?|"
    r"matching files?|documents?|artifacts?|markdown files?|md files?)\b",
    re.IGNORECASE,
)
# Nouns whose counts refer to on-disk files (rows/entries are excluded — they
# may live in a DB or in-memory and are not a reliable fabrication signal here).
_FILE_COUNT_NOUNS = frozenset(
    {"file", "files", "duplicate", "duplicates", "match", "matches",
     "document", "documents", "artifact", "artifacts"}
)
_CITED_RESULTS_PATH_RE = re.compile(r"results/[A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+")


def _spot_check_cited_paths(deliverable_content: str, tool_outputs: str) -> str:
    """Cross-check cited counts/paths against the filesystem — advisory only.

    Scans the deliverable + tool-output text for (a) explicit file-counts
    ("12 duplicate files", "3 matches") and (b) cited ``results/<file>`` paths,
    then verifies them against disk: cited paths must exist, and a file-count is
    suspicious when ``results/`` holds fewer files than claimed.

    Returns a short warning string to interpolate into the verify prompt, or
    ``""`` when everything checks out. This is defense-in-depth for the verifier
    LLM — never a hard override — so a false positive only adds an advisory
    line. Known false-positive risks: counts of non-file entities, and
    planned-but-not-yet-written paths; both merely add a caveat the LLM weighs.
    """
    blob = f"{deliverable_content}\n{tool_outputs}"
    warnings: list[str] = []

    # (a) Cited results/<file> paths must exist on disk.
    cited = sorted(set(_CITED_RESULTS_PATH_RE.findall(blob)))
    missing_cited: list[str] = []
    for rel in cited:
        try:
            target = resolve_existing(rel, base="results")
        except ValueError:
            continue
        if not target.exists():
            missing_cited.append(rel)
    if missing_cited:
        warnings.append(
            "Cited deliverable paths not found on disk: "
            + ", ".join(missing_cited[:8])
        )

    # (b) A file-count assertion cross-checked against results/ contents.
    file_counts = [
        int(n)
        for n, noun in _COUNT_NOUN_RE.findall(blob)
        if noun.lower() in _FILE_COUNT_NOUNS and int(n) > 0
    ]
    if file_counts:
        claimed = max(file_counts)
        try:
            root = results_root()
            actual = (
                sum(1 for p in root.rglob("*") if p.is_file())
                if root.exists()
                else 0
            )
        except OSError:  # noqa: BLE001 — unreadable results dir → treat as 0 files
            actual = 0
        if claimed > actual:
            warnings.append(
                f"Deliverable asserts ~{claimed} file(s) but results/ holds "
                f"{actual} — counts may be fabricated."
            )

    return "; ".join(warnings)
