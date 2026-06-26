"""Tool retrieval-before-selection (findings-05).

Default-OFF capability: instead of injecting EVERY active tool into the
execute/plan prompt, keep the built-in tools (always) plus the top-k
dynamically-created tools whose capability embeddings are semantically nearest
the current goal/step. Reuses the existing ``capability_embedding`` index for
RECALL — before this, ``find_similar`` was the index's only consumer and it
gated dedup, never ranked recall. Falls back to the full set when retrieval is
disabled, the embedding provider is unavailable, or any step errors — so
behavior is unchanged until toggled on (``TOOL_RETRIEVAL_ENABLED``) and a
retrieval hiccup can never starve a run of tools.
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from src.config.settings import Settings
from src.memory.embeddings import embed_capability
from src.tools.builtin import ALL_TOOL_DEFINITIONS
from src.tools.dynamic.persister import ToolPersister
from src.tools.registry import ToolRegistry


class ToolRetriever(Protocol):
    """Structural type: anything that can RECALL tool names by capability embedding.

    ``ToolPersister`` satisfies this structurally. Declared as a Protocol so the
    selection layer depends on the *capability* (recall names by vector), not
    the concrete persister — and tests can inject a lightweight fake.

    The E2 blend path additionally reads ``retrieve_tools_with_scores`` and
    ``tool_success_metrics`` off the persister via :func:`getattr` — they are
    duck-typed (not on this Protocol) so a minimal fake that only implements
    ``retrieve_tools`` still satisfies the type, and the blend degrades to
    pure-cosine when those methods are absent.
    """

    async def retrieve_tools(self, query_embedding: list[float], top_k: int = 8) -> list[str]:
        ...


# Built-in tools are always available. They are NOT in the capability_embedding
# index (only ``tool_create``/``agent_spawn`` persist embeddings), so a naive
# "retrieve → filter" would silently drop web_search / file_writer /
# code_executor / etc. — the core tools. This set guarantees they survive
# retrieval. ``list_tools(names=...)`` silently skips any built-in not currently
# registered, so an over-broad set is harmless.
_BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(d["name"] for d in ALL_TOOL_DEFINITIONS)


# ─── E2 score-blend: cosine × success_rate (pure, unit-tested directly) ──────

# Base cosine weight. ``weight=0`` collapses the blend to ``cosine · base``
# which, with ``base=1.0``, is pure-cosine ranking — so the blend never makes a
# cold-start run worse than the pure-cosine path.
_BLEND_BASE: float = 1.0


def blend_score(
    similarity: float,
    success_rate: float,
    empty_output_rate: float,
    *,
    weight: float,
) -> float:
    """``cosine · (base + weight·success_rate·(1−empty_output_rate))`` (E2).

    A flaky near-match (low ``success_rate``) is penalized; a reliable one is
    boosted. ``weight=0`` ⇒ ``similarity`` (pure cosine) — the blend is then a
    no-op on ranking. ``success_rate`` defaults to 1.0 / ``empty_output_rate``
    to 0.0 when a tool has no metrics (cold start) so an untested tool is never
    starved (see :func:`blend_rank`).
    """
    success_factor = success_rate * (1.0 - empty_output_rate)
    return similarity * (_BLEND_BASE + weight * success_factor)


def blend_rank(
    pool: list[tuple[str, float]],
    metrics: dict[str, dict[str, float]],
    top_k: int,
    *,
    weight: float,
) -> list[str]:
    """Re-rank a ``(name, cosine_similarity)`` pool by :func:`blend_score`.

    The pool is already cosine-ranked (HNSW order); re-ranking by score with a
    stable sort preserves that order for ties and for ``weight=0`` (pure
    cosine). A name absent from ``metrics`` is a cold-start tool — defaulted to
    ``success_rate=1.0`` / ``empty_output_rate=0.0`` so the blend does not
    punish an untested tool. Returns the top-``top_k`` names (best-first).
    """
    scored: list[tuple[str, float]] = []
    for name, similarity in pool:
        m = metrics.get(name, {})
        sr = float(m.get("success_rate", 1.0))
        eor = float(m.get("empty_output_rate", 0.0))
        scored.append(
            (name, blend_score(similarity, sr, eor, weight=weight))
        )
    # Stable sort by score descending: ties keep the input (cosine) order.
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [name for name, _ in scored[:top_k]]


async def _retrieve_with_blend(
    persister: ToolRetriever,
    query_embedding: list[float],
    top_k: int,
    *,
    pool_multiplier: int,
    weight: float,
) -> list[str]:
    """Widen to a cosine pool, then re-rank by cosine×success and take top_k.

    The pool is ``max(top_k, top_k × pool_multiplier)`` — wider than the final
    top_k so a reliable tool just outside the top_k can be promoted. The
    ``retrieve_tools_with_scores`` / ``tool_success_metrics`` capabilities are
    duck-typed off ``persister``; if either is absent or raises, the blend
    degrades to the pure-cosine :meth:`retrieve_tools` path, so it can never
    starve the run (mirrors the surrounding full-set fallback).
    """
    score_fn = getattr(persister, "retrieve_tools_with_scores", None)
    if score_fn is None:
        return await persister.retrieve_tools(query_embedding, top_k=top_k)

    pool_n = max(top_k, top_k * pool_multiplier)
    try:
        pool = await score_fn(query_embedding, top_k=pool_n)
    except Exception as e:  # noqa: BLE001 — never starve the run
        logger.debug(f"Blend score-pool retrieval failed, using pure cosine: {e}")
        pool = []
    if not pool:
        return await persister.retrieve_tools(query_embedding, top_k=top_k)

    metrics: dict[str, dict[str, float]] = {}
    metrics_fn = getattr(persister, "tool_success_metrics", None)
    if metrics_fn is not None:
        try:
            metrics = await metrics_fn([name for name, _ in pool])
        except Exception as e:  # noqa: BLE001 — metrics are an enhancement, not a gate
            logger.debug(f"Blend success-metric lookup failed, pure-cosine ranking: {e}")
            metrics = {}
    return blend_rank(pool, metrics, top_k, weight=weight)





async def select_tools_for_query(
    query: str,
    registry: ToolRegistry,
    settings: Settings,
    persister: ToolRetriever | None = None,
) -> list[dict[str, Any]]:
    """Return the tool set to inject for a goal/step.

    When ``settings.agent.tool_retrieval_enabled`` is False (the default),
    returns the full registered set — identical to pre-retrieval behavior.
    When enabled: built-in tools (always) ∪ the top-k dynamically-created tools
    whose capability embeddings are nearest ``query``.

    The ``persister`` arg is optional purely for tests (inject a fake); the
    production callers omit it and a stateless ``ToolPersister()`` is built on
    demand, mirroring ``find_similar``'s own-session pattern.

    Graceful full-set fallback on: disabled flag, embedding unavailable (the
    hash fallback is not semantically meaningful), empty/failed retrieval, or
    any exception — so retrieval can never break a run.
    """
    agent = settings.agent
    full = registry.list_tools()
    if not agent.tool_retrieval_enabled:
        return full

    try:
        embedding, source = await embed_capability(query)
        # Only real ("api") vectors are semantically rankable; the hash fallback
        # makes unrelated texts dissimilar but carries no meaning, so it cannot
        # rank tools — fall back to the full set rather than ranking on noise.
        if embedding is None or source != "api":
            logger.debug("Tool retrieval skipped this call: no API embedding available")
            return full

        pers = persister or ToolPersister()
        if agent.tool_retrieval_blend_success:
            # E2 — widen to a cosine pool, then re-rank by cosine×success before
            # taking top_k. Any error inside degrades to pure-cosine (above) or
            # the full-set fallback (below), so the blend can never starve a run.
            retrieved = await _retrieve_with_blend(
                pers,
                embedding,
                agent.tool_retrieval_top_k,
                pool_multiplier=agent.tool_retrieval_blend_pool,
                weight=agent.tool_retrieval_blend_weight,
            )
        else:
            retrieved = await pers.retrieve_tools(
                embedding, top_k=agent.tool_retrieval_top_k
            )
        if not retrieved:
            return full

        # Built-ins always + retrieved dynamic tools. list_tools preserves
        # REGISTRY order regardless of the names arg's order, so output order
        # stays stable for the prompt.
        wanted = _BUILTIN_TOOL_NAMES | set(retrieved)
        selected = registry.list_tools(names=list(wanted))
        if not selected:
            return full
        rank_mode = "cosine×success blend" if agent.tool_retrieval_blend_success else "similarity"
        logger.debug(
            f"Tool retrieval selected {len(selected)}/{len(full)} tools "
            f"(built-ins + top-{agent.tool_retrieval_top_k} dynamic by {rank_mode})"
        )
        return selected
    except Exception as e:
        logger.debug(f"Tool retrieval failed, using full tool set: {e}")
        return full
