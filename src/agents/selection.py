"""Sub-agent semantic selection before the delegation fan-out (findings-05).

Default-OFF capability: instead of fanning out to EVERY spawned sub-agent,
keep the top ``agent_selection_top_k`` spawned agents whose capability
embeddings are nearest the current subtask. Reuses the existing
``sub_agent_definitions.capability_embedding`` index for RECALL — before this,
``find_similar`` was the index's only consumer and it gated dedup, never
ranked recall. Falls back to the full spawned set when selection is disabled,
the embedding provider is unavailable, or any step errors — so behavior is
unchanged until toggled on (``AGENT_SELECTION_ENABLED``) and a recall hiccup
can never drop a needed agent (the ``agent_spawn`` decision still stands).

Mirrors :mod:`src.tools.selection` (tool retrieval-before-selection): pure,
unit-tested rank primitive + a fail-safe async selection entry point that
degrades to the full spawned set on every error path.
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from src.config.settings import Settings
from src.memory.embeddings import embed_capability


class AgentRetriever(Protocol):
    """Structural type: recall spawned-agent similarities by capability vector.

    ``SubAgentPersister`` satisfies this structurally. Declared as a Protocol so
    the selection layer depends on the *capability* (rank named agents by vector
    similarity), not the concrete persister — and tests inject a lightweight
    fake (``retrieve_agents_with_scores`` is the single method on the Protocol).
    """

    async def retrieve_agents_with_scores(
        self, names: list[str], embedding: list[float], limit: int = 8
    ) -> list[tuple[str, float]]:
        ...


def _rank_names(
    scores: list[tuple[str, float]], top_k: int
) -> list[str]:
    """Best-first names from a cosine ``(name, similarity)`` list, top_k.

    Pure / unit-tested directly. ``scores`` is already cosine-ordered (the DB
    ``ORDER BY cosine_distance``), so taking the names preserves that order and
    truncates to ``top_k``. The selection caller decides *membership* (which
    agents survive) from this list; survivor order is the original spawn order
    (kept by :func:`select_subagents_for_subtask`), NOT this rank order, so
    tier-grouping / provider-spread in the delegate fan-out is unaffected.
    """
    return [name for name, _ in scores[:top_k]]


async def select_subagents_for_subtask(
    spawned: list[dict[str, Any]],
    subtask: str,
    settings: Settings,
    persister: AgentRetriever | None = None,
) -> list[dict[str, Any]]:
    """Return the spawned sub-agents to actually delegate to (F1).

    When ``settings.agent.agent_selection_enabled`` is False (the default),
    returns ``spawned`` unchanged — identical to pre-selection fan-out. When
    enabled: embed ``subtask``, recall the spawned agents' capability
    similarities, keep the top ``agent_selection_top_k``, and return those
    spawned info-dicts in their ORIGINAL spawn order (so a CRITICAL sibling is
    still grouped/pinned at its tier downstream).

    No-op short-circuits (return ``spawned`` unchanged): selection disabled,
    only one agent (nothing to prune), the cap already covers everyone, no API
    embedding available (the hash fallback is not semantically meaningful), or
    empty/failed recall. Fail-safe: any exception returns ``spawned`` — a
    recall hiccup can never drop a needed agent (``agent_spawn`` still stands).
    """
    agent = settings.agent
    if not agent.agent_selection_enabled or len(spawned) <= 1:
        return spawned

    top_k = max(1, agent.agent_selection_top_k)
    if len(spawned) <= top_k:
        return spawned

    try:
        embedding, source = await embed_capability(subtask)
        # Only real ("api") vectors are semantically rankable; the hash fallback
        # makes unrelated texts dissimilar but carries no meaning, so it cannot
        # rank agents — keep the full spawned set rather than pruning on noise.
        if embedding is None or source != "api":
            logger.debug("Sub-agent selection skipped: no API embedding available")
            return spawned

        from src.agents.persister import SubAgentPersister

        pers = persister or SubAgentPersister()
        names = [n for n in (info.get("name", "") for info in spawned) if n]
        scores = await pers.retrieve_agents_with_scores(
            names, embedding, limit=len(names)
        )
        keep = set(_rank_names(scores, top_k))
        if not keep:
            return spawned
        selected = [info for info in spawned if info.get("name", "") in keep]
        if not selected:
            return spawned
        logger.debug(
            f"Sub-agent selection kept {len(selected)}/{len(spawned)} agents "
            f"(top-{top_k} by similarity to subtask)"
        )
        return selected
    except Exception as e:
        logger.debug(f"Sub-agent selection failed, using all spawned agents: {e}")
        return spawned
