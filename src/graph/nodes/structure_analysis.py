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

from src.agents.registry import MAX_SUB_AGENTS_PER_RUN
from src.graph.enums import Phase
from src.tools.dynamic.allowlist import MAX_TOOLS_PER_RUN

if TYPE_CHECKING:
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
_TOOL_NAME_RE = re.compile(r"['\"`]([a-z][a-z0-9_]*)['\"`]", re.IGNORECASE)
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
) -> dict[str, Any]:
    """Proactively seed capability gaps from the goal before execution.

    Args:
        state: Current agent state (reads ``current_goal``, guard flags,
            ``attempted_*_gaps``). Writes ``pending_*_gaps`` and the
            ``structure_analysis_done`` single-shot flag.
        tools: Optional ToolRegistry — detected tool names already present are
            skipped so we never re-request an existing tool.

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
    agent_gaps = _detect_agent_gaps(goal_text, attempted_agents)

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
        if len(gaps) >= MAX_TOOLS_PER_RUN:
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


def _detect_agent_gaps(goal_text: str, attempted: set[str]) -> list[str]:
    """Return proactive sub-agent-gap descriptions, or an empty list."""
    if attempted:
        return []

    lower = goal_text.lower()
    has_agent_kw = any(k in lower for k in _AGENT_KEYWORDS)
    has_parallel_kw = any(k in lower for k in _PARALLEL_KEYWORDS)
    numbered = _extract_numbered_units(goal_text)

    units: list[str] = []
    if has_agent_kw:
        # Explicit ask — honor the roles named after the keyword (e.g.
        # "sub-agents for data gathering and report generation").
        units = _extract_roles_after_keyword(goal_text)
        if not units and numbered:
            units = numbered
        if not units:
            units = ["an independent subtask described in the goal"]
    elif has_parallel_kw and len(numbered) >= 2:
        # "do (1)…, (2)…, (3)… in parallel" — independent parallel units.
        units = numbered
    elif has_parallel_kw:
        # "do A, B, and C in parallel" without a numbered list.
        units = _extract_list_phrases(goal_text)

    return _format_agent_gaps(units, attempted)


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
        if len(gaps) >= MAX_SUB_AGENTS_PER_RUN:
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

    Matches a "for/to handle …" clause after the keyword and splits it on
    commas / "and". E.g. "sub-agents for data gathering and report generation"
    -> ["data gathering", "report generation"].
    """
    lower = goal_text.lower()
    for kw in _AGENT_KEYWORDS:
        idx = lower.find(kw)
        if idx == -1:
            continue
        tail = goal_text[idx + len(kw):]
        clause_match = re.search(
            r"\b(?:for|to handle|to do|covering)\b\s+(.+?)(?:\.\s|\.$|$)",
            tail,
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
