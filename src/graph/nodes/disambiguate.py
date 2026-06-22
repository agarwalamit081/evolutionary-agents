"""disambiguate node — graduated ambiguity-resolution cascade (Feature B).

Runs once between ``classify`` and ``plan`` (only when the clarifying gate is
on and classify flagged an ambiguous goal), and resolves the ambiguity through
an escalating ladder rather than driving forward blind:

  1. LLM self-resolve — one gateway call proposes the most-likely intended
     outcome + explicit assumptions, and decides whether web grounding would
     help (emitting 0..N search queries).
  2. Web grounding — if queries were emitted and grounding is enabled, the
     ``web_search`` tool is invoked (batched) and the evidence collected.
  3. LLM re-score — one gateway call consumes the evidence, refines the
     resolution, and judges whether the goal is now actionable.
  4. HITL last resort — only if the goal is still severely ambiguous AND HITL
     is enabled. There is no ``Command(resume=)`` surface in the worker/CLI
     today, so ``interrupt()`` degrades gracefully (carries the notes forward)
     rather than stalling the run.

The resolution is carried forward as ADVISORY planner context (rendered into
``plan_user.j2``); the literal goal text is NEVER replaced — it stays the
OBJECTIVE. ``disambiguate_node`` never rewrites ``current_goal``.

Loop safety: ``disambiguation_done`` is a single-shot guard (mirror
``structure_analysis_done``). The only classify re-entry is the error_handler
auth path, and the guard routes a re-reached disambiguate straight through.
Every step is wrapped so a failure degrades to carrying notes forward — a
gateway/tool hiccup never aborts the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config import get_settings
from src.graph.enums import Phase

if TYPE_CHECKING:
    from src.graph.schemas import DisambiguationResolution
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry


async def disambiguate_node(
    state: dict[str, Any],
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Resolve a flagged ambiguity before planning.

    Args:
        state: Current agent state (reads ``current_goal`` + the classify-side
            advisory fields; the ``disambiguation_done`` guard). Writes the
            disambiguation carry-forward fields + guard.
        gateway: Optional LLM gateway for the self-resolve / re-score passes.
            Without it, the cascade degrades to carrying the classify notes
            forward unchanged.
        tools: Optional ToolRegistry for the web-grounding step.

    Returns:
        Partial state update. Always sets ``disambiguation_done=True`` and the
        ``PLAN`` phase transition; includes the advisory carry-forward fields.
    """
    # Single-shot guard — never re-run (prevents classify↔disambiguate cycles).
    if state.get("disambiguation_done"):
        return {"phase": Phase.PLAN, "disambiguation_done": True}

    result: dict[str, Any] = {
        "phase": Phase.PLAN,
        "disambiguation_done": True,
        "disambiguation_resolution": "",
        "disambiguation_assumptions": [],
        "disambiguation_evidence": [],
        "disambiguation_context": "",
        "hitl_requested": False,
    }

    settings = get_settings().agent
    goal = state.get("current_goal")
    goal_text = goal.text if goal and hasattr(goal, "text") else ""
    if not goal_text or gateway is None:
        # No goal or no gateway → carry the classify notes forward, unchanged.
        return _finalize(result, state, settings.clarifying_hitl_threshold)

    refined_intent = str(state.get("refined_intent", "") or "")

    # ── Step 1: LLM self-resolve ────────────────────────────────────────
    resolution = await _resolve(
        gateway, goal_text, refined_intent, state, evidence=""
    )
    if resolution is None:
        logger.debug("Disambiguate: self-resolve failed; carrying notes forward")
        return _finalize(result, state, settings.clarifying_hitl_threshold)

    # ── Step 2: web grounding (only if queries emitted + enabled) ───────
    evidence: list[str] = []
    queries = list(resolution.grounding_queries or [])
    if queries and settings.clarifying_web_grounding_enabled and tools is not None:
        capped = queries[: settings.clarifying_max_queries]
        evidence = await _ground(tools, capped)
        result["disambiguation_evidence"] = evidence

    # ── Step 3: LLM re-score (consume evidence, refine) ─────────────────
    if evidence:
        evidence_block = "\n".join(f"- {e}" for e in evidence)
        rescored = await _resolve(
            gateway, goal_text, refined_intent, state, evidence=evidence_block
        )
        if rescored is not None:
            resolution = rescored

    # ── Step 4: HITL last resort (degrades gracefully — no resume path) ─
    hitl_requested = False
    if (
        settings.clarifying_hitl_enabled
        and resolution.remaining_severity >= settings.clarifying_hitl_threshold
        and not resolution.resolved
    ):
        hitl_requested = _request_hitl(goal_text, resolution)
        result["hitl_requested"] = hitl_requested

    # Carry the resolution forward as advisory metadata.
    result["disambiguation_resolution"] = resolution.proposed_interpretation
    result["disambiguation_assumptions"] = list(resolution.assumptions)

    logger.info(
        f"Disambiguate: resolved={resolution.resolved}, "
        f"severity={resolution.remaining_severity:.2f}, "
        f"evidence={len(evidence)}, hitl_requested={hitl_requested}"
    )
    return _finalize(result, state, settings.clarifying_hitl_threshold)


# ── Cascade helpers ────────────────────────────────────────────────────


async def _resolve(
    gateway: LLMGateway,
    goal_text: str,
    refined_intent: str,
    state: dict[str, Any],
    *,
    evidence: str,
) -> DisambiguationResolution | None:
    """One LLM resolution call. Returns None on any failure (never raises)."""
    try:
        from src.graph.prompts import DISAMBIGUATE_SYSTEM, DISAMBIGUATE_USER
        from src.graph.schemas import DisambiguationResolution
        from src.llm.structured_output import StructuredOutputManager

        notes = state.get("ambiguity_notes", []) or []
        user_prompt = DISAMBIGUATE_USER.format(
            goal_text=goal_text,
            refined_intent=refined_intent or "(none surfaced)",
            ambiguity_type=str(state.get("ambiguity_type", "none")),
            ambiguity_severity=state.get("ambiguity_severity", 0.0),
            ambiguity_notes="\n".join(f"- {n}" for n in notes) or "- (none)",
            evidence=evidence,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": str(DISAMBIGUATE_SYSTEM)},
            {"role": "user", "content": user_prompt},
        ]
        response = await gateway.acompletion(messages=messages, node="disambiguate")
        return await StructuredOutputManager().extract(
            response.content, DisambiguationResolution, gateway=gateway, messages=messages
        )
    except Exception as exc:  # noqa: BLE001 — a gateway/parse failure degrades, never aborts
        logger.debug(f"Disambiguate LLM call failed: {exc}")
        return None


async def _ground(tools: ToolRegistry, queries: list[str]) -> list[str]:
    """Run ``web_search`` (batched) for the queries; return evidence strings.

    Best-effort: a tool failure yields empty evidence, never an exception.
    """
    if not queries:
        return []
    try:
        handler = tools.get_handler("web_search")
        if handler is None:
            logger.debug("Disambiguate: web_search handler unavailable; skipping grounding")
            return []
        out = await handler(queries=queries, max_results=3)
        text = str(out)
        if not text.strip():
            return []
        # Keep each non-empty line as one evidence item (web_search returns
        # newline-joined title/snippet lines).
        return [ln.strip() for ln in text.splitlines() if ln.strip()][:12]
    except Exception as exc:  # noqa: BLE001 — grounding is opportunistic
        logger.debug(f"Disambiguate web grounding failed: {exc}")
        return []


def _request_hitl(goal_text: str, resolution: DisambiguationResolution) -> bool:
    """Attempt HITL; return True if a human turn was actually requested.

    Mirrors ``hitl_gate_node``: ``interrupt()`` raises outside a compiled
    interrupt context (the worker/CLI path today has no ``Command(resume=)``
    surface), so on ImportError/TypeError/RuntimeError we degrade to carrying
    the notes forward (return False) instead of stalling the run.
    """
    try:
        from langgraph.types import interrupt

        interrupt({
            "question": f"Clarify this ambiguous goal: {goal_text[:120]}",
            "proposed_interpretation": resolution.proposed_interpretation[:300],
            "assumptions": list(resolution.assumptions)[:5],
            "remaining_notes": list(resolution.notes)[:5],
        })
        return True
    except (ImportError, TypeError, RuntimeError):
        logger.debug("Disambiguate: HITL interrupt unavailable; carrying notes forward")
        return False


def _finalize(
    result: dict[str, Any],
    state: dict[str, Any],
    hitl_threshold: float,
) -> dict[str, Any]:
    """Assemble the advisory ``disambiguation_context`` string + return.

    The context is rendered into plan_user.j2's ADVISORY block — it explains
    the goal, never rewrites it. Falls back to the classify notes when no
    resolution was produced (e.g. no gateway) so the planner still sees the
    ambiguity flagged.
    """
    resolution = str(result.get("disambiguation_resolution", "") or "")
    assumptions: list[str] = list(result.get("disambiguation_assumptions", []) or [])
    evidence: list[str] = list(result.get("disambiguation_evidence", []) or [])
    notes = list(state.get("ambiguity_notes", []) or [])

    lines: list[str] = [
        "DISAMBIGUATION CONTEXT — the goal was ambiguous. Below is a proposed "
        "resolution + assumptions + evidence. Treat these as HYPOTHESES, not a "
        "changed objective; the literal goal above remains what you must deliver.",
    ]
    if resolution:
        lines.append(f"Proposed interpretation: {resolution}")
    if assumptions:
        lines.append("Assumptions:\n" + "\n".join(f"- {a}" for a in assumptions))
    if evidence:
        lines.append("Evidence:\n" + "\n".join(f"- {e}" for e in evidence))
    if not resolution and notes:
        lines.append("Unresolved points:\n" + "\n".join(f"- {n}" for n in notes))
    if result.get("hitl_requested"):
        lines.append(
            f"(HITL was requested at threshold {hitl_threshold:.2f} but no resume "
            "surface exists; proceeding with the resolution above as advisory.)"
        )

    result["disambiguation_context"] = "\n".join(lines)
    return result
