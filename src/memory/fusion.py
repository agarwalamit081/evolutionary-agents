"""Recall fusion — cross-tier Reciprocal Rank Fusion for retrieved memories (Q83).

``retrieve_memory_node`` assembles recall by concatenating per-tier lists whose
scores are HETEROGENEOUS (hot/warm = fitness; evolved/folded = fitness;
facts = confidence; skills = fitness; cold/error-episode = cosine-similarity;
some 0.0). Those scores are not on a common scale, so they cannot be meaningfully
compared or merged directly — today a low-rank item from the FIRST tier always
outranks a high-rank item from a LATER tier (plain concatenation order).

Reciprocal Rank Fusion (RRF, Cormack et al.) sidesteps that: it ranks items
purely by their per-tier POSITION (each tier's list is already internally
ordered by that tier's semantic / recency / fitness ranking) and fuses a
single score ``1/(k + rank)``. The result is a cross-tier ordering that
demotes a low-rank item from a long tier relative to a top-rank item from
another tier — without ever trusting the incomparable scores.

Opt-in (``MEMORY_RECALL_RRF_ENABLED``, default off); off ⇒ the node keeps
today's plain concatenation. Default ``MEMORY_RECALL_TOP_K`` (20) ≥ the
~19-item recall total, so the first enable REORDERS but never DROPS.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def fuse_rrf(
    items: Sequence[dict[str, Any]],
    *,
    k: int = 60,
    top_k: int = 20,
    tier_key: str = "tier",
) -> list[dict[str, Any]]:
    """Fuse heterogeneous per-tier recall lists by Reciprocal Rank Fusion.

    Items are grouped by their ``tier_key`` value; within a tier, rank is the
    item's append order (1-based), which already reflects that tier's internal
    semantic / recency / fitness ordering. The fused score is the per-tier
    contribution ``1/(k + rank)`` (each item belongs to exactly one tier, so its
    fused score is that single contribution). Items are then sorted by fused
    score descending; ties break by original append order (stable/deterministic).

    Args:
        items: Recall item dicts, each carrying at least ``content`` and the
            ``tier_key`` field. A missing/empty tier groups under ``"default"``.
        k: RRF smoothing constant (standard 60). Larger ⇒ flatter ranking.
        top_k: Maximum items to return. ``<= 0`` ⇒ empty list.
        tier_key: The dict key holding the tier label.

    Returns:
        Up to ``top_k`` item dicts in fused order (the SAME dict objects,
        reordered — inputs are never mutated).
    """
    if top_k <= 0 or not items:
        return []

    per_tier_rank: dict[str, int] = {}
    # (fused_score, original_index, item) — sort by score desc, then index asc
    # for a deterministic stable tie-break (Python's sort is stable, so ties
    # preserve original append order).
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for orig_idx, item in enumerate(items):
        tier = str(item.get(tier_key) or "default")
        rank = per_tier_rank.get(tier, 0) + 1
        per_tier_rank[tier] = rank
        scored.append((1.0 / (k + rank), orig_idx, item))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [item for _, _, item in scored[:top_k]]
