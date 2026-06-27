"""Dynamic prompt construction — splice technique bodies into a base prompt (§5).

``build_messages`` assembles the ``[system, user]`` pair a node sends to the
gateway. When techniques are supplied, their bodies are injected into the
system prompt as a bulleted block:

- **Above** the JSON-schema footer marker when one is present (plan/verify/
  reflect), so the schema stays at the tail and ``StructuredOutputManager.extract``
  keeps working.
- **After the first paragraph** otherwise (e.g. the tool-calling execute
  prompt, which has no JSON schema), so the guidance still leads the response.

With no techniques the base prompt is passed through unchanged.

Phase 8: when evolution→live promotion is opted in and a PROMPT mutation has
been promoted for a node (``src.evolution.promote.PromotionGate``), that node's
promoted suffixes are prepended to its system prompt as a tagged ``[evolved]``
block — via ``splice_evolved`` / the ``node`` arg on ``build_messages``. An
in-process candidate override lets the eval canary trial a candidate mutation
without touching the on-disk pointer.
"""

from __future__ import annotations

from loguru import logger

from src.graph.enums import TaskComplexity
from src.graph.prompts.technique_selector import (
    JSON_SCHEMA_MARKER,
    Technique,
    TechniqueSelector,
)

_TECHNIQUE_HEADING = "Reasoning techniques to apply:"

# Pure/stateless selector — cheap to instantiate once and reuse across nodes.
_SELECTOR = TechniqueSelector()

# Phase 8: in-process candidate override. The eval canary
# (``src.evolution.promote.GoldenCanary``) binds a candidate (node → suffixes)
# for the duration of a canary run WITHOUT mutating the on-disk pointer, so a
# trial mutation is visible to ``evolved_suffixes_for_node`` then cleared.
# Empty unless a canary is mid-flight; promotion OFF → always empty.
_candidate_override: dict[str, list[str]] = {}


def set_evolved_candidate(node: str, suffixes: list[str]) -> None:
    """Bind a candidate (node → suffixes) for the canary run (in-process only)."""
    _candidate_override[node] = list(suffixes)


def clear_evolved_candidate(node: str | None = None) -> None:
    """Clear the candidate override (one node, or all when ``node`` is None)."""
    if node is None:
        _candidate_override.clear()
    else:
        _candidate_override.pop(node, None)


def evolved_suffixes_for_node(node: str | None) -> list[str]:
    """Active evolved suffixes for ``node`` (candidate override → on-disk pointer).

    Returns ``[]`` when promotion is disabled, no pointer exists, or the node has
    no promoted entry — so a node that never evolved (and the default OFF path)
    is completely unaffected. The candidate override takes precedence so a canary
    run sees its trial suffixes even before they are written to disk.
    """
    if not node:
        return []
    try:
        from src.config.settings import get_settings

        if not get_settings().evolution.evolution_promote_to_live:
            return []
    except Exception:  # noqa: BLE001 — promotion must never break prompt building
        return []
    if node in _candidate_override:
        return list(_candidate_override[node])
    try:
        from src.evolution.promote import PromotionGate

        return PromotionGate().current_suffixes(node)
    except Exception:  # noqa: BLE001 — pointer read must never break prompt building
        return []


# Phase 5 G3b: in-process AFlow technique-policy candidate override. The AFlow
# optimizer (``src.graph.search.aflow.AFlowOptimizer``) binds a candidate
# ((node, category) → technique names) for the duration of an evaluation run
# WITHOUT mutating the on-disk pointer, so a trial policy is visible to
# ``aflow_techniques_for`` then cleared. Empty unless AFlow is mid-evaluation;
# ``AFLOW_ENABLED`` off → the pointer read is skipped → byte-identical selection.
_aflow_candidate: dict[tuple[str, str], list[str]] = {}


def set_aflow_candidate(node: str, category: str, names: list[str]) -> None:
    """Bind a candidate ((node, category) → names) for an eval run (in-process only)."""
    _aflow_candidate[(node, category)] = list(names)


def clear_aflow_candidate(node: str | None = None, category: str | None = None) -> None:
    """Clear the candidate override (a matching node/category, or all when neither given).

    ``node``/``category`` filter the entries to drop (None matches all on that axis);
    both None clears everything. The optimizer clears per-(node, category) after each
    trial and clears-all before the baseline.
    """
    if node is None and category is None:
        _aflow_candidate.clear()
        return
    for key in [
        k
        for k in _aflow_candidate
        if (node is None or k[0] == node) and (category is None or k[1] == category)
    ]:
        _aflow_candidate.pop(key, None)


def aflow_candidate_for(node: str, category: str) -> list[str] | None:
    """Read the in-process candidate for (node, category) (None when none bound).

    Test/diagnostic accessor — mirrors how a fitness ``run_fn`` infers whether a
    candidate is currently active during an AFlow evaluation.
    """
    if (node, category) in _aflow_candidate:
        return list(_aflow_candidate[(node, category)])
    return None


def aflow_techniques_for(
    node: str, category: str, budget_tokens: int = 512
) -> list[Technique] | None:
    """Active AFlow technique policy for (node, category); None when AFlow is inactive.

    Precedence: in-process candidate → on-disk pointer (ONLY when ``AFLOW_ENABLED``)
    → None. Returns None (not []) when there is no policy so the caller falls
    through to the heuristic selector; returns a resolved Technique list (possibly
    empty) when a policy IS active. Names are resolved via the registry (unknown /
    off-node names dropped — a policy can never inject a technique the registry does
    not key on for that node) and budget-capped. Never raises — a pointer read error
    → None (AFlow must never break prompt building).
    """
    try:
        if (node, category) in _aflow_candidate:
            names = list(_aflow_candidate[(node, category)])
        else:
            from src.config.settings import get_settings

            if not get_settings().aflow.enabled:
                return None
            from src.graph.search.aflow import AflowPolicyStore

            names = AflowPolicyStore().current_policy(node, category)
            if names is None:
                return None
        from src.graph.search.aflow import resolve_policy

        return resolve_policy(names, node, budget_tokens)
    except Exception:  # noqa: BLE001 — AFlow must never break prompt building
        return None


def render_evolved_block(suffixes: list[str]) -> str:
    """Render promoted suffixes as a headed ``[evolved]`` block (empty when none)."""
    if not suffixes:
        return ""
    lines = ["[evolved] promoted guidance (apply unless it conflicts with the task):"]
    lines += [f"- {s}" for s in suffixes]
    return "\n".join(lines)


def splice_evolved(base_system: str, node: str | None) -> str:
    """Prepend the node's promoted ``[evolved]`` block; no-op when none promoted."""
    block = render_evolved_block(evolved_suffixes_for_node(node))
    if not block:
        return base_system
    return f"{block}\n\n{base_system}"


def select_techniques_for_node(
    complexity: TaskComplexity | None,
    node: str,
    goal_text: str | None = None,
    budget_tokens: int = 512,
    refined_intent: str | None = None,
) -> list[Technique]:
    """Select prompting techniques for a node's LLM call (§5 wiring helper).

    Centralizes the (complexity, node, goal-pattern) → technique-list lookup so
    each node's LLM branch stays a one-liner. Infers the goal pattern from the
    goal text, then delegates to :meth:`TechniqueSelector.select`.

    Feature D also infers the reader **audience** and answer **uncertainty** and
    threads them into selection. The richest available text is used: the
    classify node's ``refined_intent`` (Feature A — the real desired outcome)
    when present, else the literal ``goal_text``. Both inference helpers return
    ``None`` for generic text, so a goal with no audience/uncertainty markers
    yields identical selection to the pre-Feature-D behaviour.

    Degrades cleanly on missing inputs so a heuristic/fallback path never raises:
    a ``None`` complexity returns ``[]`` (techniques are only applied on the LLM
    path, where the classified complexity is always known). Returns ``[]`` for a
    node the registry does not key on.

    Args:
        complexity: The classified task complexity. ``None`` → no techniques
            (heuristic-fallback safety).
        node: Node identifier constant (``NODE_PLAN`` / ``NODE_EXECUTE`` / …).
        goal_text: The goal's raw text, used to infer a goal pattern.
        budget_tokens: Soft cap on total injected token cost.
        refined_intent: Optional refined intent (Feature A) used as the
            preferred signal for audience/uncertainty inference. Backward-
            compatible: existing callers omit it and get identical results.

    Returns:
        Ordered (priority desc) list of techniques to splice into the prompt;
        possibly empty.
    """
    if complexity is None:
        return []
    goal_pattern = TechniqueSelector.infer_goal_pattern(goal_text)
    # Feature D: prefer the refined intent for audience/uncertainty signals —
    # it captures the real desired outcome — else the literal goal text.
    signal_text = refined_intent or goal_text
    audience = TechniqueSelector.infer_audience(signal_text)
    uncertainty = TechniqueSelector.infer_uncertainty(signal_text)
    # Phase 5 G3b: an installed AFlow technique policy for (node, goal_pattern)
    # overrides the heuristic selection. Byte-identical when no policy exists —
    # ``aflow_techniques_for`` returns None (pointer read gated behind AFLOW_ENABLED,
    # short-circuited before any IO when off), so the selector runs as before.
    category = goal_pattern or "general"
    aflow_policy = aflow_techniques_for(node, category, budget_tokens)
    if aflow_policy is not None:
        logger.info(
            f"AFlow policy applied for {node}/{category}: "
            f"{[t.name for t in aflow_policy]}"
        )
        return aflow_policy
    selected = _SELECTOR.select(
        complexity=complexity,
        node=node,
        goal_pattern=goal_pattern,
        budget_tokens=budget_tokens,
        audience=audience,
        uncertainty=uncertainty,
    )
    logger.info(
        f"Techniques selected for {node}/{complexity.value}/"
        f"{goal_pattern or 'none'}/aud={audience or '-'}/"
        f"unc={uncertainty or '-'}: {[t.name for t in selected]}"
    )
    return selected


def render_technique_block(techniques: list[Technique]) -> str:
    """Render techniques as a headed bulleted block (empty string when none)."""
    if not techniques:
        return ""
    lines = [_TECHNIQUE_HEADING]
    lines += [f"- {technique.body}" for technique in techniques]
    return "\n".join(lines)


def splice_techniques(base_system: str, techniques: list[Technique]) -> str:
    """Inject technique bodies into a base system prompt.

    Above the JSON-schema marker when present; else after the first paragraph
    (or prepended if there is no paragraph break). No-op for an empty list.
    """
    block = render_technique_block(techniques)
    if not block:
        return base_system

    marker_pos = base_system.find(JSON_SCHEMA_MARKER)
    if marker_pos != -1:
        return base_system[:marker_pos] + block + "\n\n" + base_system[marker_pos:]

    # No schema marker: lead with the guidance after the opening paragraph.
    paragraph_break = base_system.find("\n\n")
    if paragraph_break != -1:
        head = base_system[:paragraph_break]
        rest = base_system[paragraph_break:]
        return f"{head}\n\n{block}{rest}"
    return f"{block}\n\n{base_system}"


def build_messages(
    base_system: str,
    user_content: str,
    techniques: list[Technique] | None = None,
    node: str | None = None,
) -> list[dict[str, str]]:
    """Build the ``[system, user]`` message pair for a gateway call.

    Args:
        base_system: The node's rendered system prompt (already ``.format``-ed).
        user_content: The rendered user prompt.
        techniques: Optional techniques whose bodies are spliced into the
            system prompt. ``None`` or empty → pass-through.
        node: Optional node id (``NODE_PLAN``/``NODE_EXECUTE``/…). When supplied
            AND a PROMPT mutation has been promoted for that node (Phase 8, opt-
            in), the promoted ``[evolved]`` guidance is prepended so the live
            agent loads it. ``None``/no promotion → unchanged behavior.

    Returns:
        A two-element message list ready for ``gateway.acompletion``.
    """
    system = splice_techniques(base_system, techniques or [])
    system = splice_evolved(system, node)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
