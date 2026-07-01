"""Execute node — runs the current plan step with tool calling."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

from src.config.settings import get_settings
from src.graph.enums import GoalStatus, Phase
from src.graph.models import ToolResult
from src.graph.state import AgentState, objective_goal_text

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry
    from src.tools.result_cache import ToolResultCache

# Maximum concurrent tool calls per execute step — operator-configurable via
# AgentSettings (MAX_CONCURRENT_TOOLS). Read at call-time in _execute_tool_calls.

# Tools whose output produces an on-disk deliverable. Only ``file_writer``
# writes under ``results_root`` — verify treats its ``file_path`` as the
# authoritative deliverable path, so a write-step that never calls it leaves no
# artifact for verification to find.
FILE_OUTPUT_TOOLS: frozenset[str] = frozenset({"file_writer"})

# Tabular / row-data deliverables that must be PRODUCED BY CODE — the content is
# derived by reading + transforming/aggregating input data (e.g. normalize a
# JSONL into a CSV, compute aggregates), so a model cannot reliably hand-author
# it as text. Such write-steps are steered to code_executor (which writes the
# file from code), NOT file_writer. Text deliverables (.md/.txt/.json) the model
# can author directly stay on file_writer. Drives the write-step steering below.
COMPUTE_DELIVERABLE_EXTS: frozenset[str] = frozenset(
    {".csv", ".tsv", ".jsonl", ".jsonlines", ".xlsx", ".xls", ".parquet", ".feather"}
)


def _is_compute_deliverable(path: str) -> bool:
    """True if ``path`` is a data deliverable that must be produced by code."""
    lower = path.lower()
    return any(lower.endswith(ext) for ext in COMPUTE_DELIVERABLE_EXTS)

# How many EXTRA LLM turns a write-step gets to actually call a write tool after
# an explicit nudge. Cheaper models (haiku) frequently narrate the deliverable
# as prose on turn 1 and only call file_writer once told to. Bounded so a
# stubborn refusal degrades gracefully (mark complete) instead of looping;
# max_write_nudges + 1 total attempts per step. Operator-configurable via
# AgentSettings (MAX_WRITE_NUDGES). Read at call-time in _run_step.

# Detects a declared output path in a step description: "save/write/export …
# <file>". Mirrors verify._SAVE_TO_RE so the path we nudge toward is the same
# one verification will later check on disk.
_WRITE_INTENT_RE = re.compile(
    r"\b(?:save|write|export|store|dump|output)\b[^.]*?"
    # Require a >=2-char extension: real file extensions are always >=2 letters
    # (py/md/csv/json/txt/jsonl). A 1-char tail matches abbreviations ("e.g",
    # "i.e") and version/section fragments ("1.0", "3.2"), which were captured
    # as bogus deliverable paths and fed false missing-deliverable verify loops
    # (F-k).
    r"\b([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]{2,})\b",
    re.IGNORECASE,
)

# Detects the canonical deliverable path embedded in the GOAL text. Goals name
# their output as "results/<name>.<ext>" regardless of verb ("save/merge/
# combine ... to/into results/x.md"). This is the fall-back target for a
# producing step that declares no path of its own — without it, a "merge the
# results into a cohesive overview" step narrates the deliverable as prose and
# never calls file_writer (Q3 never produced q3_overview.md).
# An OUTPUT deliverable path named in the goal. Two shapes:
# (a) ``results/<name>.<ext>`` — a strong path signal (always a deliverable
#     candidate, subject only to input-skipping below).
# (b) a bare ``<name>.<ext>`` — natural-language goals often name outputs
#     without the results/ prefix ("write a CSV file named primes_demo.csv").
#     Restricted to known data/text extensions so decimals ("2.0") and version
#     strings ("v0.23.0") are not grabbed, AND (in _extract_goal_deliverable)
#     to a preceding output cue. Without (b) the objective-success evolution
#     gate never matched bare-filename goals, so evolution never fired on them
#     (Bug #5: covbench/primes goals returned None and a completed run's
#     success never crystallized into a mutation).
_GOAL_DELIVERABLE_RE = re.compile(
    r"\b(?:results/[A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+"
    r"|[A-Za-z0-9_][A-Za-z0-9_-]*\.(?:csv|json|jsonl|md|txt|py|xlsx|tsv|html|xml|yaml|yml))\b",
    re.IGNORECASE,
)

# Marks a ``results/`` path as an INPUT the goal reads from, not an output it
# writes. A goal commonly names its input first ("Reuse q01's normalizer output
# at results/q01/normalized.csv"), so the first match is the input; without this
# guard _extract_goal_deliverable returned that input as the deliverable (F-k).
# The signal is the word immediately before the path.
_GOAL_INPUT_CONTEXT_RE = re.compile(
    r"\b(?:at|from|using|against|reuse|input|source|read)\s*$",
    re.IGNORECASE,
)

# A bare deliverable (no results/ prefix) is only an OUTPUT when an output cue
# immediately precedes it; otherwise a known-extension token in the goal is
# incidental text (a doc mention, a reference). Checked in the same look-back
# window _extract_goal_deliverable uses for input-skipping.
_GOAL_OUTPUT_CONTEXT_RE = re.compile(
    r"\b(?:named|called|to|into|onto|as|file|filename|files|save\w*|writ\w*|produc\w*|generat\w*|output|export\w*|dump\w*)\s*$",
    re.IGNORECASE,
)

# A step that assembles the final output but names no concrete file (so
# _WRITE_INTENT_RE misses it). "merge/combine/assemble/integrate/finalize the
# results" is the reliable signal such a step is the deliverable producer. The
# last step of any plan is also treated as producing (catch-all for a final
# write the planner phrased without a bring-together verb).
_PRODUCING_STEP_RE = re.compile(
    r"\b(?:merge|merging|combined|combine|compil\w*|assembl\w*|integrat\w*|"
    r"consolidat\w*|finali\w*)",
    re.IGNORECASE,
)


def _messages_to_openai(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert LangChain messages (or OpenAI dicts) to OpenAI chat-format dicts.

    The gateway (litellm) expects OpenAI message dicts, while the graph state
    stores LangChain ``AnyMessage`` objects. This bridges the two so the
    execute node can feed the real conversation history into each LLM call —
    without it, the agent is stateless across steps and memory folding has no
    real context to compress.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict):
            out.append(m)
            continue
        mtype = getattr(m, "type", "")
        content = getattr(m, "content", "")
        if mtype == "human":
            out.append({"role": "user", "content": content})
        elif mtype == "system":
            out.append({"role": "system", "content": content})
        elif mtype == "ai":
            entry: dict[str, Any] = {"role": "assistant", "content": content}
            tcs = getattr(m, "tool_calls", None) or []
            if tcs:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(
                                tc.get("args", {}), default=str
                            ),
                        },
                    }
                    for tc in tcs
                ]
            out.append(entry)
        elif mtype == "tool":
            entry_t: dict[str, Any] = {
                "role": "tool",
                "content": content,
                "tool_call_id": getattr(m, "tool_call_id", ""),
            }
            name = getattr(m, "name", None)
            if name:
                entry_t["name"] = name
            out.append(entry_t)
        else:
            out.append({"role": "user", "content": str(content)})
    return out


def _build_ai_message(
    content: str | None,
    tool_calls: list[dict[str, Any]],
) -> AIMessage:
    """Build an AIMessage for the thread, attaching validated tool calls.

    Falls back to a content-only message if tool-call validation fails, so a
    malformed provider response never breaks the run.
    """
    text = content or ""
    if not tool_calls:
        return AIMessage(content=text)
    try:
        normalized = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                args = (
                    json.loads(raw_args)
                    if isinstance(raw_args, str)
                    else raw_args
                )
            except (json.JSONDecodeError, TypeError):
                args = {}
            normalized.append(
                {
                    "name": fn.get("name", ""),
                    "args": args,
                    "id": tc.get("id", ""),
                    "type": "tool_call",
                }
            )
        return AIMessage(content=text, tool_calls=normalized)
    except Exception as exc:  # noqa: BLE001 — never break the run on validation
        logger.debug(
            f"AIMessage tool_call validation failed, storing content only: {exc}"
        )
        return AIMessage(content=text)


def _extract_expected_file_path(step_description: str) -> str | None:
    """Return the deliverable path a write-step declares, or None.

    A step like "Write the onboarding guide to results/q9.md" declares
    ``results/q9.md`` as its output. Only steps naming a concrete output path
    are treated as write-steps — a vague "generate a report" step with no file
    target never triggers a nudge (and never risks a spurious loop).
    """
    match = _WRITE_INTENT_RE.search(step_description or "")
    return match.group(1) if match else None


def _extract_goal_deliverable(goal_text: str) -> str | None:
    """Return the canonical OUTPUT deliverable path named in the goal, or None.

    Two path shapes are recognized:
    - ``results/<name>.<ext>`` — a strong output signal; always a candidate
      (subject only to input-context skipping below).
    - a bare ``<name>.<ext>`` (no results/ prefix) — recognized ONLY when an
      output cue (named/called/to/into/as/file/save/write/produce/generate/
      output/export/dump) immediately precedes it. Natural-language goals name
      outputs without the prefix ("write a CSV file named primes_demo.csv");
      without bare-filename support such a goal's success never matched the
      evolution gate, so it never crystallized into a mutation (Bug #5). The cue
      gate keeps an incidental mention ("see perf.md") from being mistaken for a
      deliverable.

    A goal often names its INPUT first ("process results/q01/x.csv -> write
    ... to results/q03/y.csv"), so skip any path whose preceding word marks it
    as input (at/from/using/against/reuse/input/source/read) and return the LAST
    remaining candidate — outputs are stated last. If no candidate survives,
    return None (no confident output signal) rather than guessing.
    """
    text = goal_text or ""
    matches = list(_GOAL_DELIVERABLE_RE.finditer(text))
    if not matches:
        return None
    outputs: list[str] = []
    for match in matches:
        token = match.group(0)
        window = text[max(0, match.start() - 24): match.start()].lower()
        if _GOAL_INPUT_CONTEXT_RE.search(window):
            continue
        # A bare filename (no results/ prefix) is a deliverable only when an
        # output cue precedes it; a results/ path is always a candidate.
        if not token.lower().startswith("results/") and not _GOAL_OUTPUT_CONTEXT_RE.search(window):
            continue
        outputs.append(token)
    # No candidate survived input-skip + output-cue gating. Rather than guess an
    # incidental path (a bare-filename mention that the output cue filtered out
    # would be re-admitted by a naive "last match" fallback — reopening the very
    # false-positive hole the gate closes), report no confident output signal.
    # Both callers handle None: the execute nudge runs the step once, and the
    # evolution gate falls back to its non-deliverable success criterion.
    return outputs[-1] if outputs else None


def _is_producing_step(
    step_description: str, step_index: int, n_steps: int
) -> bool:
    """True if this step assembles the final deliverable.

    Either it uses a bring-together verb (merge/combine/assemble/integrate/
    finalize/…) or it is the last step of the plan (catch-all). Combined with
    the on-disk check this targets exactly the merge/finalize step without
    nudging read/enumerate/verify sub-steps.
    """
    if _PRODUCING_STEP_RE.search(step_description or ""):
        return True
    return n_steps > 0 and step_index + 1 >= n_steps


def _deliverable_on_disk(path: str) -> bool:
    """True if ``path`` already exists, non-empty, anywhere the agent writes.

    Mirrors verify's ``_resolve_deliverable`` candidate set so this check and the
    verify node agree on what "deliverable on disk" means. Without that parity a
    run could be marked complete (verify found the file) yet report no on-disk
    deliverable here — so the objective-success evolution gate never matched a
    genuinely-successful run (Bug #6: a ``code_executor``/``terminal`` subprocess
    wrote ``primes_demo.csv`` to the repo root, verify accepted it via the
    literal-path candidate, but this check only looked under ``results/`` and
    returned False, suppressing evolution).

    Candidates, in order: ``results`` (``file_writer``'s target — incl. the
    per-run subdir when active, with the flat-root read fallback), ``workspace``
    (fixtures), and a literal CWD-relative ``Path(path)`` (where subprocess
    writes land when the run-id contextvar does not cross the process boundary).
    """
    from src.tools._paths import resolve_existing

    candidates: list[Path] = []
    for base in ("results", "workspace"):
        try:
            candidates.append(resolve_existing(path, base=base))
        except Exception:  # noqa: BLE001 — traversal/settings failure must not abort a step
            continue
    candidates.append(Path(path))  # CWD-relative: subprocess writes land here

    for target in candidates:
        try:
            if target.is_file() and target.stat().st_size > 0:
                return True
        except (OSError, ValueError):  # noqa: BLE001 — unreadable/invalid candidate → skip
            continue
    return False


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    """Extract the function name from an OpenAI-format tool call."""
    fn = tool_call.get("function", {})
    name = fn.get("name") if isinstance(fn, dict) else None
    return name or str(tool_call.get("name", ""))


def _tool_call_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Parse the JSON arguments of an OpenAI-format tool call into a dict."""
    fn = tool_call.get("function", {})
    raw = fn.get("arguments", tool_call.get("args", "{}"))
    try:
        return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (json.JSONDecodeError, TypeError):
        return {}


def _called_file_output_tool(tool_calls: list[dict[str, Any]]) -> bool:
    """True if any tool call targets a deliverable-producing tool."""
    return any(_tool_call_name(tc) in FILE_OUTPUT_TOOLS for tc in tool_calls)


def _first_file_output_call(
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the first deliverable-producing tool call, or None."""
    for tc in tool_calls:
        if _tool_call_name(tc) in FILE_OUTPUT_TOOLS:
            return tc
    return None


def _write_nudge(path: str, *, compute: bool) -> str:
    """Build the user-role nudge that pushes the LLM to materialize the file.

    ``compute`` selects the right tool for the deliverable type: a data file
    (CSV/JSONL/…) that must be produced by code is nudged to code_executor; a
    text deliverable the model can author directly is nudged to file_writer.
    """
    if compute:
        return (
            "Your previous turn described the deliverable in text but did not "
            f"write it to disk. This step requires the data file '{path}', "
            "which must be produced by running code — you cannot reliably "
            "hand-author rows of normalized/transformed data as text. Call the "
            "code_executor tool now with a script that reads the input data, "
            f"processes it, and writes the result to '{path}'. Do not respond "
            "with text only — verification reads the filesystem, so the file "
            "must exist at that path to count as done."
        )
    return (
        "Your previous turn described the deliverable in text but did not "
        "write it to disk. This step requires producing the file deliverable "
        f"at '{path}'. Call the file_writer tool now with file_path set to "
        "that path, create_dirs set to true (so nested output folders are "
        "auto-created), and content set to your full deliverable text. Do not "
        "respond with text only — verification reads the filesystem, so the "
        "deliverable must be written via file_writer to count as done."
    )


def _offers_tool(tool_defs: list[dict[str, Any]], name: str) -> bool:
    """True if ``name`` is among the offered function-calling tool schemas."""
    for td in tool_defs:
        fn = td.get("function") if isinstance(td, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            return True
        if isinstance(td, dict) and td.get("name") == name:
            return True
    return False


async def execute_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
    result_cache: ToolResultCache | None = None,
) -> dict[str, Any]:
    """Execute the current plan step.

    When gateway and tools are provided, uses LLM tool calling.
    Otherwise falls back to simulated step execution.

    Args:
        state: Current agent state with plan and step index.
        gateway: Optional LLM gateway for tool-calling execution.
        tools: Optional tool registry for executing tool calls.
        result_cache: Optional Redis cache for idempotent tool results.
            Only tools flagged ``cacheable`` in the registry are routed
            through it; the cache is best-effort and never breaks a call.

    Returns:
        Partial state update with execution results.
    """
    plan_steps = state.get("plan_steps", [])
    step_index = state.get("current_step_index", 0)
    messages = state.get("messages", [])
    iteration_count = state.get("iteration_count", 0)

    # Guard: no plan or index out of range
    if not plan_steps or step_index >= len(plan_steps):
        logger.warning("Execute called with no remaining steps")
        return {
            "phase": Phase.REFLECT,
            "iteration_count": iteration_count + 1,
        }

    current_step = plan_steps[step_index]
    logger.info(
        f"Executing step {step_index + 1}/{len(plan_steps)}: "
        f"{current_step.description[:60]}..."
    )

    # Mark current step as active
    current_step.status = GoalStatus.ACTIVE

    # Build execution context
    goal_text = objective_goal_text(state) or "Unknown goal"

    # Try LLM + tool execution first, fall back to simulated
    if gateway is not None and tools is not None:
        result = await _llm_execute(
            gateway, tools, state, current_step.description, goal_text, result_cache
        )
        if result is not None:
            return result

    # Simulated execution fallback
    return await _simulated_execute(state, current_step, goal_text, messages, iteration_count)


async def _llm_execute(
    gateway: LLMGateway,
    tools: ToolRegistry,
    state: AgentState,
    step_description: str,
    goal_text: str,
    result_cache: ToolResultCache | None = None,
) -> dict[str, Any] | None:
    """Execute step via LLM with tool calling. Returns None on failure."""
    try:
        from src.graph.enums import TaskComplexity
        from src.graph.prompts import (
            EXECUTE_SYSTEM,
            NODE_EXECUTE,
            select_techniques_for_node,
            splice_evolved,
            splice_techniques,
        )

        plan_steps = state.get("plan_steps", [])
        step_index = state.get("current_step_index", 0)
        completed_steps = state.get("completed_steps", [])
        tool_results = state.get("tool_results", [])
        iteration_count = state.get("iteration_count", 0)
        memories = state.get("retrieved_memories", [])

        # Build memory context as a bare bulleted list — execute_system.j2 wraps it
        # in an explicit ADVISORY ONLY frame (objective-drift guard #254: recalled
        # memory must read as technique hints, not as the objective).
        memory_ctx = ""
        if memories:
            memory_ctx = "\n".join(f"- {m}" for m in memories[:3])

        tool_results_ctx = ""
        if tool_results:
            recent = tool_results[-3:]
            tool_results_ctx = "\nRecent tool results:\n" + "\n".join(
                f"- {r.tool_name}: {r.output[:100]}" for r in recent if hasattr(r, "tool_name")
            )

        system_prompt = EXECUTE_SYSTEM.format(
            goal_text=goal_text,
            completed_count=len(completed_steps),
            total_steps=len(plan_steps),
            step_description=step_description,
            memory_context=memory_ctx,
            tool_results_context=tool_results_ctx,
        )

        # §5: select prompting techniques for this execute call and splice their
        # bodies into the system prompt. The execute prompt has no JSON-schema
        # footer, so splice_techniques injects after the opening paragraph.
        # Gate on the classified complexity so the heuristic/LLM-failure path
        # (which carries no techniques) stays unchanged.
        goal = state.get("current_goal")
        execute_complexity = (
            goal.complexity if goal and goal.complexity else TaskComplexity.SIMPLE
        )
        # Phase 3 — per-step routing: the model for THIS execute call follows the
        # step's OWN nature (``step_nature``, stamped by the plan node), not the
        # goal-level complexity. A trivial step (single-tool lookup) routes to the
        # cheap tier (qwen3.6-flash); a complex step (code_executor / recompute /
        # multi-artifact) routes to glm-4.7. This makes execute routing REAL
        # (previously the gateway call passed no complexity → always the SIMPLE
        # tier, even though eval attribution tagged route(goal.complexity)) and
        # per-step. Falls back to the goal complexity when there is no current
        # step so routing never breaks.
        _cur_step = (
            plan_steps[step_index]
            if plan_steps and 0 <= step_index < len(plan_steps)
            else None
        )
        step_routing_complexity = (
            _cur_step.step_nature if _cur_step is not None else execute_complexity
        )
        techniques = select_techniques_for_node(
            complexity=execute_complexity, node=NODE_EXECUTE, goal_text=goal_text,
        )
        system_prompt = splice_techniques(system_prompt, techniques)
        # Phase 8: prepend any promoted [evolved] guidance for this node (no-op
        # unless evolution→live promotion is opted in AND a PROMPT mutation was
        # promoted for the execute node).
        system_prompt = splice_evolved(system_prompt, NODE_EXECUTE)

        # ── Stateful ReAct thread ─────────────────────────────────────────
        # state["messages"] is the canonical conversation history (seeded with
        # the goal by initial_state). Feed it to the LLM so the agent reasons
        # across steps, and append this step's user turn + assistant reply +
        # tool results so memory folding has real context to compress. Without
        # this, every step runs blind and folds save ~nothing.
        history = state.get("messages", [])
        step_label = (
            f"Execute step {step_index + 1}/{len(plan_steps)}: {step_description}"
        )

        # Get tool definitions for function calling. When tool retrieval is
        # enabled (findings-05), select_tools_for_query returns the built-ins
        # plus the top-k dynamic tools nearest this step instead of every active
        # tool; otherwise the full set (the default, unchanged behavior).
        from src.tools.selection import select_tools_for_query

        tool_defs = await select_tools_for_query(step_description, tools, get_settings())

        # A step that declares a concrete output file is a write-step: its
        # deliverable must be produced via a file-output tool (file_writer),
        # not narrated as text. Cheaper models frequently emit the deliverable
        # as prose on turn 1 and only call file_writer after an explicit nudge,
        # so a write-step gets up to MAX_WRITE_NUDGES extra turns — bounded so
        # a stubborn refusal degrades gracefully (mark complete) instead of
        # looping. A non-write step runs once, exactly as before.
        expected_path = _extract_expected_file_path(step_description)
        if expected_path is None:
            # A producing step (merge/combine/… or the final step) that names no
            # path of its own falls back to the goal's canonical deliverable, so
            # the final file still gets written to disk instead of narrated as
            # prose. Only while the deliverable is missing — once written, the
            # step runs once like any non-write step. (Q3's merge step produced a
            # text-only overview and q3_overview.md was never created.)
            goal_deliverable = _extract_goal_deliverable(goal_text)
            if (
                goal_deliverable is not None
                and _is_producing_step(step_description, step_index, len(plan_steps))
                and not _deliverable_on_disk(goal_deliverable)
            ):
                expected_path = goal_deliverable
                logger.debug(
                    f"Step has no declared output path; falling back to goal "
                    f"deliverable '{expected_path}' (producing step, not on disk)"
                )
        max_attempts = (get_settings().agent.max_write_nudges + 1) if expected_path else 1

        # Front-load the file_writer requirement on turn 1. Without this, cheaper
        # models narrate the deliverable as prose on the first attempt and only
        # call file_writer once the post-hoc nudge fires — burning 1-2 extra
        # turns (~10-20s) per write-step and often an extra verify→plan cycle.
        # Stating the exact path + tool up front lets the model write the file
        # on the first attempt; the nudge (below) remains as a bounded fallback.
        if expected_path:
            if _is_compute_deliverable(expected_path):
                # A data deliverable (CSV/JSONL/…) is produced by CODE that
                # reads + transforms input data and writes the file — a model
                # cannot reliably hand-author normalized/transformed rows as
                # text. Steer to code_executor on turn 1 so the file lands on
                # disk the first time (the disk-check + nudge remain as a
                # bounded fallback). file_writer cannot transform input data,
                # so it is actively wrong here (battery-04 q01 normalized.csv
                # never materialized → infinite verify loop).
                step_label = (
                    f"{step_label}\n\nThis step's deliverable is a DATA file "
                    f"that MUST be produced by running code: call the "
                    f"code_executor tool with a script that reads/processes "
                    f"the necessary input data and writes the result to "
                    f"'{expected_path}'. Do NOT hand-write the data as text — "
                    f"file_writer cannot transform input data. The file must "
                    f"exist at that path when this step ends — verification "
                    f"reads the filesystem."
                )
            else:
                step_label = (
                    f"{step_label}\n\nThis step's deliverable MUST be written to "
                    f"disk: call the file_writer tool with file_path='{expected_path}', "
                    f"create_dirs=true (so nested output folders are auto-created), "
                    f"and content set to your full deliverable text. A text-only "
                    f"reply does not count as done — verification reads the "
                    f"filesystem, so the file must exist at that path."
                )

        # Every LangChain message produced across this step's attempts. On a
        # nudge path this carries the text-only first turn + the nudge + the
        # tool-calling follow-up, so reflect/folding see the real trace.
        turn_messages: list[Any] = []
        nudge_text: str | None = None
        raw_tool_calls: list[dict[str, Any]] = []
        new_tool_results: list[ToolResult] = []
        response_content: str | None = ""

        for attempt in range(max_attempts):
            # Payload: system + history + the step turn + prior attempts in
            # this step + the nudge (when nudging). Re-derived each attempt so
            # the LLM sees its own prior text-only reply before the nudge.
            payload: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                *_messages_to_openai(history),
                {"role": "user", "content": step_label},
                *_messages_to_openai(turn_messages),
            ]
            if nudge_text is not None:
                payload.append({"role": "user", "content": nudge_text})

            # On a nudge turn (the model narrated instead of writing), force the
            # file_writer tool call structurally via a named tool_choice that
            # even narration-prone models cannot ignore. Only when file_writer is
            # actually offered (always true for write-steps). Turn 1 stays free
            # so a step that must compute (call code_executor, etc.) before
            # writing still can. Cures the q3/q4 "narrates through all nudges"
            # loop. OpenAI / DashScope-compat / NVIDIA-NIM all honor this.
            forced_tool_choice: dict[str, Any] | None = None
            # Force-lock to file_writer only for TEXT deliverables. A compute
            # deliverable (CSV/JSONL/…) nudge steers to code_executor — locking
            # it to file_writer here would contradict its own nudge and make the
            # instructed tool (code_executor) structurally un-callable, so the
            # agent would hand-write data and re-loop. Leave compute nudge turns
            # free to call code_executor (the disk check still bounds attempts).
            if (
                nudge_text is not None
                and _offers_tool(tool_defs, "file_writer")
                and not _is_compute_deliverable(expected_path or "")
            ):
                forced_tool_choice = {
                    "type": "function",
                    "function": {"name": "file_writer"},
                }

            response = await gateway.acompletion_with_tools(
                messages=payload,
                tools=tool_defs,
                tool_choice=forced_tool_choice,
                # Phase 3: route THIS step's model by its own nature
                # (``step_routing_complexity``) so a trivial step runs on the
                # cheap tier (qwen3.6-flash) and a complex step on glm-4.7.
                # ``node=NODE_EXECUTE`` still threads into NODE_TIER_MAP for the
                # per-(complexity,node) refine. This is the real per-step routing
                # (replacing the prior no-complexity → always-SIMPLE-tier path).
                complexity=step_routing_complexity,
                node=NODE_EXECUTE,
            )

            # Process tool calls if present. gather preserves order, so each
            # result zips 1:1 with its tool_call — needed for ToolMessage
            # correlation.
            raw_tool_calls = response.tool_calls or []
            new_tool_results = (
                await _execute_tool_calls_parallel(
                    raw_tool_calls, tools, result_cache
                )
                if raw_tool_calls
                else []
            )
            response_content = response.content

            # Append this attempt's real turn to the thread (Human → AI → Tool).
            turn_messages.append(
                HumanMessage(content=nudge_text if nudge_text is not None else step_label)
            )
            turn_messages.append(_build_ai_message(response.content, raw_tool_calls))
            for tc, tr in zip(raw_tool_calls, new_tool_results, strict=False):
                tr.metadata["tool_call_id"] = tc.get("id", "")
                turn_messages.append(
                    ToolMessage(
                        content=tr.output or tr.error or "",
                        tool_call_id=tc.get("id", ""),
                        name=tr.tool_name,
                    )
                )

            current_step = plan_steps[step_index]

            # A failed tool call (e.g. a hallucinated tool name) must not read
            # as progress. Keep the step ACTIVE, DO NOT advance the index, and
            # surface the failure in the thread + tool_results so the next
            # execute pass retries with feedback. route_after_execute routes
            # the failed result back to execute (recoverable, bounded by the
            # max-iterations guard). (F14: previously a failed tool call was
            # marked COMPLETED and the run aborted with is_complete=True.)
            if any(not tr.success for tr in new_tool_results):
                current_step.status = GoalStatus.ACTIVE
                logger.info(
                    f"Step {step_index + 1} had failed tool call(s); "
                    f"retrying without advancing"
                )
                return {
                    "phase": Phase.EXECUTE,
                    "messages": turn_messages,
                    "current_step_index": step_index,  # unchanged → retry same step
                    "iteration_count": iteration_count + 1,
                    "tool_results": new_tool_results,
                }

            # Write-step that did not call a file-output tool AND whose declared
            # deliverable is NOT already on disk → nudge and retry if attempts
            # remain; otherwise fall through to mark complete (verify then flags
            # the missing deliverable). The disk check is what lets a
            # code_executor-mediated write satisfy a write-step: code_executor is
            # not in FILE_OUTPUT_TOOLS (it has no file_path arg to record), but
            # when the agent's generated code DOES write the deliverable to disk
            # the step is genuinely done — nudging anyway false-fires and burns
            # max_attempts turns re-prompting for a file_writer call that isn't
            # needed (battery-04 q1+q3: the agent wrote results/<run>/*.csv via
            # code_executor, the nudge looped 3x → "marking complete (verify
            # will flag the gap)", verify flagged missing, run looped to
            # MAX_ITERATIONS). Resolves via the same resolve_existing() the goal-
            # deliverable fall-back uses, so it sees exactly where the file lands.
            if (
                expected_path
                and not _called_file_output_tool(raw_tool_calls)
                and not _deliverable_on_disk(expected_path)
            ):
                if attempt < max_attempts - 1:
                    nudge_text = _write_nudge(
                        expected_path, compute=_is_compute_deliverable(expected_path)
                    )
                    logger.info(
                        f"Step {step_index + 1} expects deliverable at "
                        f"{expected_path} but no file-output tool was called; "
                        f"nudging (attempt {attempt + 2}/{max_attempts})"
                    )
                    continue
                logger.warning(
                    f"Step {step_index + 1} expects deliverable at "
                    f"{expected_path} but file_writer was not called after "
                    f"{max_attempts} attempts; marking complete "
                    f"(verify will flag the gap)"
                )

            break  # success: tool called, or not a write-step, or budget spent

        # Mark step complete with the LLM result. Record the actual write-tool
        # call on the step so verify's deliverable detection has the
        # authoritative path (not just phrasal cues from the plan text).
        current_step = plan_steps[step_index]
        current_step.status = GoalStatus.COMPLETED
        result_text = response_content or step_description
        current_step.result = result_text[:500]
        file_tc = _first_file_output_call(raw_tool_calls)
        if file_tc is not None:
            current_step.tool_name = "file_writer"
            current_step.tool_input = _tool_call_args(file_tc)

        return {
            "phase": Phase.REFLECT,
            "messages": turn_messages,
            "current_step_index": step_index + 1,
            "iteration_count": iteration_count + 1,
            "completed_steps": [current_step],
            "tool_results": new_tool_results,
        }
    except Exception as e:
        logger.debug(f"LLM execution failed, using simulated: {e}")
        return None


async def _simulated_execute(
    state: AgentState,
    current_step: Any,
    goal_text: str,
    messages: list[Any],  # noqa: ARG001 — kept for future use
    iteration_count: int,
) -> dict[str, Any]:
    """Simulated step execution (heuristic fallback)."""
    step_index = state.get("current_step_index", 0)

    user_message = {
        "role": "user",
        "content": (
            f"Execute the following step:\n"
            f"Step: {current_step.description}\n"
            f"Goal context: {goal_text}\n"
            f"Tool: {current_step.tool_name or 'none specified'}\n"
            f"Proceed with execution."
        ),
    }

    current_step.status = GoalStatus.COMPLETED
    current_step.result = f"Executed: {current_step.description}"

    return {
        "phase": Phase.REFLECT,
        "messages": [user_message],
        "current_step_index": step_index + 1,
        "iteration_count": iteration_count + 1,
        "completed_steps": [current_step],
    }


async def _execute_tool_call(
    tc: dict[str, Any],
    tools: ToolRegistry,
    cache: ToolResultCache | None = None,
) -> ToolResult:
    """Execute a single tool call and return a ToolResult.

    For idempotent, cacheable tools (opt-in via the registry), successful
    results are served from / written to ``cache`` so repeated calls within
    and across runs skip the underlying work. Errors are never cached and any
    cache failure degrades to a transparent miss.

    Args:
        tc: Tool call dict with function.name and function.arguments.
        tools: Tool registry for handler lookup.
        cache: Optional result cache for cacheable tools.

    Returns:
        ToolResult with success/failure status.
    """
    tool_name = tc.get("function", {}).get("name", tc.get("name", ""))
    tool_args_str = tc.get("function", {}).get("arguments", tc.get("args", "{}"))

    handler = tools.get_handler(tool_name)
    if handler is None:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output="",
            error=f"Unknown tool: {tool_name}",
        )

    # Parse args up front: the cache key needs canonical args, and a clean
    # parse error is more useful than a generic handler exception.
    try:
        args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
    except (json.JSONDecodeError, TypeError) as exc:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output="",
            error=f"Invalid arguments for {tool_name}: {exc}",
        )

    # F3 — destructive-tool HITL gate (opt-in via DESTRUCTIVE_TOOL_HITL_ENABLED).
    # When the knob is on and the tool is flagged ``destructiveHint``
    # (terminal_command / http_request / index_corpus), pause for human approval
    # BEFORE invocation. A real compiled-graph ``interrupt()`` raises
    # ``GraphInterrupt`` (a ``GraphBubbleUp``) which we deliberately do NOT
    # catch — LangGraph catches it, checkpoints, and on resume re-invokes this
    # call so the same ``interrupt()`` returns the approval payload. Outside a
    # graph (headless worker with no resumer, or a direct unit call)
    # ``interrupt()`` raises ``RuntimeError``, which we DO catch → the safe
    # default is to BLOCK the destructive tool (an irreversible op never runs
    # without explicit approval). Knob off ⇒ the whole gate is skipped.
    from src.safety.pipeline import should_gate_destructive

    if get_settings().agent.destructive_tool_hitl_enabled and should_gate_destructive(tool_name, tools):
        try:
            from langgraph.types import interrupt

            decision = interrupt({
                "type": "destructive_tool_approval",
                "tool": tool_name,
                "args": args,
            })
        except (ImportError, TypeError, RuntimeError):
            decision = None  # no graph context / no resumer → blocked below

        approved = (
            bool(decision.get("approved", False))
            if isinstance(decision, dict)
            else bool(decision)
        )
        if not approved:
            # A policy block is not a tool-execution failure — don't pollute
            # tool_call_metrics (it feeds governance retirement + the E2 blend).
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Destructive tool '{tool_name}' blocked by HITL gate (not approved).",
            )

    # Cache lookup — only for opt-in cacheable tools.
    if cache is not None and tools.is_cacheable(tool_name):
        cached = await cache.get(tool_name, args)
        if cached is not None:
            logger.debug(f"Tool cache HIT: {tool_name}")
            return ToolResult(
                tool_name=tool_name,
                success=bool(cached.get("success", True)),
                output=str(cached.get("output", "")),
                error=cached.get("error"),
                metadata={"cached": True},
            )

    start = time.perf_counter()
    # Generated tools (``tool_create`` → registry) carry untrusted LLM
    # handler_code materialized via ``exec``. In a sandboxed code-exec mode
    # (docker/runner) route the invocation through that sandbox — the SAME
    # surface ``code_executor`` uses — so generated code never runs in-process
    # in the worker with full DB/Redis/FS access. Returns ``None`` in
    # subprocess mode (no isolation concept on a host-run agent) so the
    # in-process handler runs below, unchanged.
    from src.tools.dynamic.sandbox_dispatch import invoke_generated_tool

    isolated = await invoke_generated_tool(tool_name, tools, args)
    if isolated is not None:
        await _record_tool_metric(
            tool_name,
            success=isolated.success,
            empty_output=not (isolated.output or "").strip(),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
        return isolated

    try:
        result = await handler(**args)
        latency_ms = int((time.perf_counter() - start) * 1000)
        out = str(result)
        # #11 — evaluate the per-tool success contract (additive; None on the
        # tool = today's behavior). ``real_success`` feeds metrics/governance
        # ONLY; the model-facing ``ToolResult`` below is NEVER mutated — the
        # model still sees ``success=True`` + the full surface, so it can react
        # to an ``"ERROR: …"`` body itself (the agent's reflexes are unchanged).
        real_success = _evaluate_tool_success(tool_name, out, tools)
        tr = ToolResult(
            tool_name=tool_name,
            success=True,
            output=out[:2000],
        )
        # Record the REAL invocation outcome (non-fatal; gated by
        # TOOL_METRICS_ENABLED). ``real_success`` reflects the per-tool
        # contract; a blank success is an empty-output signal; latency is
        # captured for both the contract-pass and contract-fail paths.
        await _record_tool_metric(
            tool_name,
            success=real_success,
            empty_output=not out.strip(),
            latency_ms=latency_ms,
        )
        # Cache only successful results of cacheable tools (never errors). A
        # contract failure is a non-success surface (e.g. ``"ERROR: …"``) and is
        # NOT cached, mirroring the "never errors" intent — so a transient
        # failure is never served from the cache on a later identical call.
        if cache is not None and tools.is_cacheable(tool_name) and real_success:
            await cache.set(
                tool_name,
                args,
                {
                    "success": tr.success,
                    "output": tr.output,
                    "error": tr.error,
                },
            )
        return tr
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        await _record_tool_metric(
            tool_name, success=False, empty_output=False, latency_ms=latency_ms
        )
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output="",
            error=str(e)[:500],
        )


async def _record_tool_metric(
    tool_name: str, *, success: bool, empty_output: bool, latency_ms: int
) -> None:
    """Record a tool-invocation outcome (M4 success metrics).

    Thin, non-fatal wrapper around :meth:`ToolMetricsRecorder.record` so the
    execute chokepoint stays decoupled from the recorder/DB. A failure here (e.g.
    DB unavailable) is logged inside the recorder and never propagates — metrics
    are observability-only and must never break a tool call.
    """
    try:
        from src.tools.metrics import ToolMetricsRecorder

        await ToolMetricsRecorder().record(
            tool_name,
            success=success,
            empty_output=empty_output,
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 — metrics must never break a tool call
        logger.debug("Tool metric recording skipped for '{}': {}", tool_name, exc)


def _evaluate_tool_success(tool_name: str, output: str, tools: ToolRegistry) -> bool:
    """Return whether ``output`` satisfies the tool's success contract (#11).

    The contract (sourced from ``TOOL_ANNOTATIONS`` via
    :meth:`ToolRegistry.get_success_contract`) declares how to tell a REAL
    success from a handler that returned WITHOUT raising but produced an
    error/empty surface — e.g. ``git_clone`` returns ``"ERROR: …"`` on failure,
    which without a contract was recorded as ``success=True`` in
    ``tool_call_metrics`` (feeding governance retirement + the E2 selection
    blend). The recorded metric/governance signal now reflects this REAL outcome;
    the model-facing ``ToolResult`` is never mutated.

    Fail-open by design — the flag off, no contract on the tool, or any
    evaluation error ⇒ ``True`` (today's behavior, where a non-raising handler
    is a success). A malformed contract must NEVER break a tool call.
    """
    try:
        from src.config.settings import get_settings
        from src.tools.success import evaluate_success

        if not get_settings().agent.tool_success_contract_enabled:
            return True
        contract = tools.get_success_contract(tool_name)
        return evaluate_success(contract, output)
    except Exception:  # noqa: BLE001 — fail-open, never break a tool call
        return True


async def _execute_tool_calls_parallel(
    tool_calls: list[dict[str, Any]],
    tools: ToolRegistry,
    cache: ToolResultCache | None = None,
) -> list[ToolResult]:
    """Execute multiple tool calls concurrently with semaphore limiting.

    Uses asyncio.gather so independent tool calls run in parallel.
    A semaphore caps concurrency at MAX_CONCURRENT_TOOLS to avoid
    overwhelming external services.

    Args:
        tool_calls: List of tool call dicts from LLM response.
        tools: Tool registry for handler lookup.
        cache: Optional result cache forwarded to each call.

    Returns:
        List of ToolResult in the same order as tool_calls.
    """
    if not tool_calls:
        return []

    if len(tool_calls) == 1:
        # Single call — skip gather overhead
        return [await _execute_tool_call(tool_calls[0], tools, cache)]

    semaphore = asyncio.Semaphore(get_settings().agent.max_concurrent_tools)

    async def _limited(tc: dict[str, Any]) -> ToolResult:
        async with semaphore:
            return await _execute_tool_call(tc, tools, cache)

    results = await asyncio.gather(*[_limited(tc) for tc in tool_calls])
    return list(results)
