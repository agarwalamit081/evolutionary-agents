"""structure_analysis node — proactively detects capability needs from the goal.

Today tool_create and agent_spawn are *reactive*: they only fire from the
reflect node after an execution failure or a (6+ step + multi-connector)
heuristic. Goals that state their intent up front slip through — e.g. "Create
two custom tools" created zero tools, "Use specialized sub-agents" spawned zero
agents (see docs/e2e-validation-report.md).

This node runs once, right after retrieve_memory (before the execute loop), and
seeds ``pending_tool_gaps`` / ``pending_agent_gaps`` from explicit goal-text
signals so the existing tool_create / agent_spawn nodes fire proactively. It is
heuristic-only and deterministic (v1): it handles the explicit e2e signals
("create a tool", "use sub-agents", numbered "in parallel" lists) at zero LLM
cost. An LLM-assist refinement for ambiguous phrasings is deferred.

Loop safety: the node is single-shot — guarded by ``state.structure_analysis_done``,
which only this node writes. On any later reach (after tool_create -> plan or
agent_spawn -> delegate -> verify -> plan) the flag is True, so it passes through
without re-seeding. Combined with LangGraph's partial-state updates (tool_create /
agent_spawn only clear their *own* gaps), both ordering paths terminate.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config import get_settings
from src.graph.enums import Phase

if TYPE_CHECKING:
    from src.agents.registry import SubAgentRegistry
    from src.tools.registry import ToolRegistry

# ── Intent signals ──────────────────────────────────────────────────────
# Tool-creation intent: a build verb within ~40 chars of tool/utility/script.
# Accepts singular and plural ("tool"/"tools", "utility"/"utilities", …) so the
# real e2e goal "Create two custom tools" matches.
_TOOL_INTENT_RE = re.compile(
    r"\b(create|build|make|implement|define|write|develop|generate)\b[\w\s]{0,40}\b"
    r"(tools?|utilit(?:y|ies)|scripts?)\b",
    re.IGNORECASE,
)
# Quoted / backticked snake_case identifier, e.g. 'rss_aggregator' or `html_tool`.
# Requires at least one underscore — a real custom-tool name in this codebase is
# always multi-word snake_case (the generator and every existing example use it),
# whereas a bare quoted word is almost always a CSV column / data field / prose
# term mis-grabbed as a phantom tool. This mirrors _AGENT_NAME_RE's convention
# ("matches agent names while ignoring ordinary prose"). The negative lookahead
# skips a quoted identifier immediately followed by a colon — that is a JSON /
# Python-dict KEY (e.g. {"engineers": [...]} / {"capacity_hours": number} in a
# goal that specifies a deliverable schema), not a custom-tool name. Without
# EITHER guard a schema- or column-heavy goal floods tool_create with phantom
# tools named after JSON keys / CSV columns (battery-04 q09: the `amount` column
# was flagged "custom tool 'amount' described in the goal", wasting a replan).
_TOOL_NAME_RE = re.compile(r"['\"`]([a-z][a-z0-9]*(?:_[a-z0-9]+)+)['\"`](?!\s*:)", re.IGNORECASE)
_VALID_NAME_RE = re.compile(r"[a-z][a-z0-9_]{2,}")

# Explicit sub-agent / parallel phrasing (substring match, lowercased).
_AGENT_KEYWORDS = (
    "sub-agent", "sub agent", "specialized agent", "dedicated agent",
    "specialist agent", "sub_agents",
)
_PARALLEL_KEYWORDS = (
    "in parallel", "parallelize", "concurrently", "simultaneously",
)

# Cap how many words of a detected unit become the gap hint (keeps it concise).
_HINT_WORD_CAP = 8


async def structure_analysis_node(
    state: dict[str, Any],
    *,
    tools: ToolRegistry | None = None,
    sub_agent_registry: SubAgentRegistry | None = None,
) -> dict[str, Any]:
    """Proactively seed capability gaps from the goal before execution.

    Args:
        state: Current agent state (reads ``current_goal``, guard flags,
            ``attempted_*_gaps``). Writes ``pending_*_gaps`` and the
            ``structure_analysis_done`` single-shot flag.
        tools: Optional ToolRegistry — detected tool names already present are
            skipped so we never re-request an existing tool.
        sub_agent_registry: Optional SubAgentRegistry — recalled agent names are
            checked so a goal that references previously-created agents by name
            (e.g. "use the doc_outline sub-agent") does not proactively spawn a
            redundant helper (battery-02 N8 over-spawn).

    Returns:
        Partial state update. Always sets ``structure_analysis_done=True`` and
        the phase; includes ``pending_tool_gaps`` / ``pending_agent_gaps`` only
        when intent is detected.
    """
    # Single-shot guard — never re-seed (prevents loops regardless of reducers).
    if state.get("structure_analysis_done"):
        return {"phase": Phase.STRUCTURE_ANALYSIS, "structure_analysis_done": True}

    result: dict[str, Any] = {
        "phase": Phase.STRUCTURE_ANALYSIS,
        "structure_analysis_done": True,
    }

    # Respect the enable flag (best-effort: never fail the run on settings access).
    try:
        from src.config import get_settings

        if not get_settings().agent.structure_analysis_enabled:
            return result
    except Exception as exc:  # noqa: BLE001 — settings access must not break planning
        logger.debug(f"structure_analysis_enabled check skipped: {exc}")

    goal = state.get("current_goal")
    goal_text = goal.text if goal and hasattr(goal, "text") else ""
    if not goal_text:
        return result

    # Already spawned agents this run — don't add more proactively.
    if state.get("sub_agents_spawned"):
        return result

    attempted_tools = set(state.get("attempted_tool_gaps", []) or [])
    attempted_agents = set(state.get("attempted_agent_gaps", []) or [])

    tool_gaps = _detect_tool_gaps(goal_text, tools, attempted_tools)
    recalled_names = _recalled_agent_names(sub_agent_registry)
    agent_gaps = _detect_agent_gaps(goal_text, attempted_agents, recalled_names)

    if tool_gaps:
        result["pending_tool_gaps"] = tool_gaps
        logger.info(f"Structure analysis: proactive tool gaps -> {tool_gaps}")
    if agent_gaps:
        result["pending_agent_gaps"] = agent_gaps
        logger.info(f"Structure analysis: proactive sub-agent gaps -> {agent_gaps}")

    return result


# ── Tool-creation detection ─────────────────────────────────────────────


def _detect_tool_gaps(
    goal_text: str,
    tools: ToolRegistry | None,
    attempted: set[str],
) -> list[str]:
    """Return proactive tool-gap descriptions, or an empty list."""
    if attempted:
        return []
    if not _TOOL_INTENT_RE.search(goal_text):
        return []

    existing = _existing_tool_names(tools)
    names = _TOOL_NAME_RE.findall(goal_text)
    gaps: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.lower()
        if key in seen or key in existing:
            continue
        if not _VALID_NAME_RE.fullmatch(name):
            continue
        seen.add(key)
        gaps.append(f"custom tool '{name}' described in the goal")
        if len(gaps) >= get_settings().agent.max_tools_per_run:
            break

    if gaps:
        return [g for g in gaps if g not in attempted]

    # Intent present but no explicit names — let tool_create's LLM derive one.
    generic = "custom tool described in the goal"
    return [] if generic in attempted else [generic]


def _existing_tool_names(tools: ToolRegistry | None) -> set[str]:
    """Best-effort set of already-registered tool names (lowercased)."""
    if tools is None:
        return set()
    try:
        return {str(n).lower() for n in tools.list_names()}
    except Exception:  # noqa: BLE001 — mock registries / unavailable list_names
        return set()


# ── Sub-agent / parallel detection ──────────────────────────────────────


def _detect_agent_gaps(
    goal_text: str,
    attempted: set[str],
    recalled_names: set[str] | None = None,
) -> list[str]:
    """Return proactive sub-agent-gap descriptions, or an empty list."""
    if attempted:
        return []

    lower = goal_text.lower()
    has_agent_kw = any(k in lower for k in _AGENT_KEYWORDS)
    has_parallel_kw = any(k in lower for k in _PARALLEL_KEYWORDS)
    numbered = _extract_numbered_units(goal_text)

    units: list[str] = []
    if has_agent_kw:
        # Explicit ask — honor the roles named after the keyword IN THE SAME
        # SENTENCE (e.g. "sub-agents for data gathering and report
        # generation"). We deliberately do NOT fall back to an arbitrary
        # numbered list here: a goal that mentions "sub-agent" alongside an
        # unrelated "(1) … (2) … (3) …" list (battery-04 q07: the THREE HARD
        # constraints) would otherwise seed one phantom sub-agent per list
        # item. A numbered list only becomes parallel sub-agent work when the
        # goal also says "in parallel" (handled by the has_parallel_kw branch
        # below) — a bare constraint/requirement list does not.
        units = _extract_roles_after_keyword(goal_text)
        if not units:
            units = ["an independent subtask described in the goal"]
    elif has_parallel_kw and len(numbered) >= 2:
        # "do (1)…, (2)…, (3)… in parallel" — independent parallel units.
        units = numbered
    elif has_parallel_kw:
        # "do A, B, and C in parallel" without a numbered list.
        units = _extract_list_phrases(goal_text)

    # Suppress the GENERIC proactive spawn when the goal references recalled
    # sub-agents by name. The generic gap ("an independent subtask...") fires
    # whenever "sub-agents" appears without explicit roles; if the goal is in
    # fact naming previously-created agents to REUSE, spawning a new helper is
    # redundant — delegate reuses the recalled ones at zero spawn cost.
    # battery-02 N8: "Using the doc_outline and python_file_inventory
    # sub-agents (created earlier)..." matched the keyword with no explicit
    # roles, fell back to the generic gap, and needlessly spawned
    # repo_map_builder though both named agents were already recalled.
    # Explicit new roles ("sub-agents for X and Y") still spawn as today.
    if (
        recalled_names
        and has_agent_kw
        and units == ["an independent subtask described in the goal"]
    ):
        referenced = _named_existing_agents(goal_text, recalled_names)
        if referenced:
            logger.info(
                "Structure analysis: goal names existing recalled sub-agents "
                f"{sorted(referenced)}; suppressing generic proactive "
                "agent_spawn (delegate will reuse them)"
            )
            return []

    return _format_agent_gaps(units, attempted)


# Snake_case identifier with at least one underscore — matches agent names like
# ``doc_outline`` / ``python_file_inventory`` while ignoring ordinary prose.
_AGENT_NAME_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def _recalled_agent_names(registry: SubAgentRegistry | None) -> set[str]:
    """Best-effort lowercased set of recalled sub-agent names."""
    if registry is None:
        return set()
    try:
        return {str(n).lower() for n in registry.list_names()}
    except Exception:  # noqa: BLE001 — mock registries must not break planning
        logger.debug("SubAgentRegistry.list_names() failed; skipping reuse check")
        return set()


def _named_existing_agents(goal_text: str, recalled_names: set[str]) -> set[str]:
    """Recalled agent names that appear as identifiers in ``goal_text``.

    Catches goals that reference previously-created sub-agents by name so
    structure analysis does not proactively spawn a redundant helper for work an
    existing agent already covers.
    """
    if not recalled_names:
        return set()
    found = set(_AGENT_NAME_RE.findall(goal_text.lower()))
    return found & recalled_names


def _format_agent_gaps(units: list[str], attempted: set[str]) -> list[str]:
    """Dedupe, cap, and turn units into sub-agent gap descriptions."""
    seen: set[str] = set()
    gaps: list[str] = []
    for unit in units:
        hint = _truncate(unit)
        if not hint or hint in seen:
            continue
        seen.add(hint)
        gap = f"specialized sub-agent for: {hint}"
        if gap in attempted:
            continue
        gaps.append(gap)
        if len(gaps) >= get_settings().agent.max_sub_agents_per_run:
            break
    return gaps


def _extract_numbered_units(goal_text: str) -> list[str]:
    """Slice independent units from a "(1) … (2) … (3) …" numbered list.

    Captures the text between consecutive "(N)" markers (so a one-line list
    still splits correctly), trimmed at the first sentence boundary.
    """
    markers = list(re.finditer(r"\(\s*(\d+)\s*\)", goal_text))
    if len(markers) < 2:
        return []
    units: list[str] = []
    for i, marker in enumerate(markers):
        start = marker.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(goal_text)
        chunk = goal_text[start:end].strip().rstrip(".,;:")
        # Stop at the first sentence boundary within this unit.
        chunk = re.split(r"[.!?](?:\s|$)", chunk, maxsplit=1)[0].strip()
        if len(chunk) > 2:
            units.append(chunk)
    return units


def _extract_roles_after_keyword(goal_text: str) -> list[str]:
    """Extract the role list following an explicit sub-agent keyword.

    Matches a "for/to handle …" clause in the SAME SENTENCE as the keyword
    and splits it on commas / "and". E.g. "sub-agents for data gathering and
    report generation" -> ["data gathering", "report generation"].

    Scoped to the keyword's own sentence: a "for …" clause in a LATER
    sentence is unrelated prose, not a sub-agent role. battery-04 q07: the
    goal says "adversarial sub-agent that tries to BREAK the solution" then,
    in a separate sentence, "for each confirms the constraint checker WOULD
    flag it" — scanning the whole tail seeded two phantom sub-agents
    ("for each confirms…" / "then confirms…"), routing the run into a
    spurious delegate cycle.
    """
    lower = goal_text.lower()
    for kw in _AGENT_KEYWORDS:
        idx = lower.find(kw)
        if idx == -1:
            continue
        # A role list must follow "sub-agents for …" in the SAME sentence.
        # Cut the tail at the first sentence boundary so a "for …" clause in
        # a later, unrelated sentence is never captured as a sub-agent role.
        tail = goal_text[idx + len(kw):]
        sentence_tail = re.split(r"[.!?](?:\s|$)", tail, maxsplit=1)[0]
        clause_match = re.search(
            r"\b(?:for|to handle|to do|covering)\b\s+(.+)",
            sentence_tail,
            re.IGNORECASE,
        )
        if not clause_match:
            continue
        parts = re.split(r"\s+and\s+|\s*,\s+|\s*;\s+", clause_match.group(1))
        parts = [p.strip(" .:-") for p in parts if 3 <= len(p.strip()) <= 60]
        if parts:
            return parts
    return []


def _extract_list_phrases(goal_text: str) -> list[str]:
    """Fallback: comma/'and'-separated phrases for an unnumbered parallel list."""
    parts = re.split(r"\s*,\s+|\s+and\s+", goal_text)
    return [p.strip(" .:;-") for p in parts if 3 <= len(p.strip()) <= 50]


def _truncate(text: str) -> str:
    """Keep the first few words of a unit as a concise gap hint."""
    return " ".join(text.split()[:_HINT_WORD_CAP])
