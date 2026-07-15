"""src.memory.fusion — Reciprocal Rank Fusion across recall tiers (Q83).

``fuse_rrf`` reorders the ``retrieve_memory_node`` concatenation by per-tier
RANK alone (fused score = Σ 1/(k+rank)), sidestepping the heterogeneous per-
tier scores (fitness / confidence / cosine-similarity / 0.0). Each item
belongs to exactly one tier, so its fused score is the single per-tier
contribution; ties break by original append order (stable sort).

These tests lock the math on fixed-rank fixtures (no DB, no settings) and the
core invariants: deterministic interleaving, top-k truncation, empty/zero
edge cases, input non-mutation, default-tier grouping, and — crucially — that
RRF is rank-based (a 0.0-score rank-1 item still interleaves with a 0.99-score
rank-1 item from another tier).
"""

from __future__ import annotations

from src.memory.fusion import fuse_rrf


def _item(content: str, tier: str, score: float = 0.0) -> dict[str, object]:
    return {"content": content, "tier": tier, "score": score}


class TestFuseRRFKnownOrder:
    def test_interleaves_across_tiers_on_fixed_ranks(self) -> None:
        # tier A = [a1,a2,a3], tier B = [b1,b2], k=60.
        # Scores: a1/b1 = 1/61, a2/b2 = 1/62, a3 = 1/63.
        # Sorted desc (stable by original index) → [a1, b1, a2, b2, a3].
        items = [
            _item("a1", "A"),
            _item("a2", "A"),
            _item("a3", "A"),
            _item("b1", "B"),
            _item("b2", "B"),
        ]
        out = fuse_rrf(items, k=60, top_k=20)
        assert [d["content"] for d in out] == ["a1", "b1", "a2", "b2", "a3"]

    def test_rank_based_not_score_based(self) -> None:
        # A rank-1 item with score 0.0 ties a rank-1 item with score 0.99 — RRF
        # ignores the scores entirely. Tie breaks by original order (A before B).
        items = [
            _item("high_score", "A", score=0.99),
            _item("zero_score", "B", score=0.0),
        ]
        out = fuse_rrf(items, k=60, top_k=20)
        # Both are rank-1 in their tier → 1/61 each → original order preserved.
        assert [d["content"] for d in out] == ["high_score", "zero_score"]

    def test_single_tier_preserves_internal_order(self) -> None:
        items = [_item("x1", "T"), _item("x2", "T"), _item("x3", "T")]
        out = fuse_rrf(items, k=60, top_k=20)
        # One tier → ranks 1,2,3 → monotonically decreasing → order unchanged.
        assert [d["content"] for d in out] == ["x1", "x2", "x3"]


class TestFuseRRFEdgeCases:
    def test_empty_input_returns_empty(self) -> None:
        assert fuse_rrf([], k=60, top_k=20) == []

    def test_top_k_zero_returns_empty(self) -> None:
        out = fuse_rrf([_item("a", "A")], k=60, top_k=0)
        assert out == []

    def test_negative_top_k_returns_empty(self) -> None:
        out = fuse_rrf([_item("a", "A")], k=60, top_k=-1)
        assert out == []

    def test_top_k_truncates_highest_rank_first(self) -> None:
        # [a1,b1,a2,b2,a3] (see known-order) truncated to 3 → [a1,b1,a2].
        items = [
            _item("a1", "A"),
            _item("a2", "A"),
            _item("a3", "A"),
            _item("b1", "B"),
            _item("b2", "B"),
        ]
        out = fuse_rrf(items, k=60, top_k=3)
        assert [d["content"] for d in out] == ["a1", "b1", "a2"]

    def test_missing_tier_uses_default_group(self) -> None:
        # Items without a "tier" key all land in the "default" group (one RRF
        # tier) so their relative order is preserved.
        items = [
            {"content": "p1"},
            {"content": "p2"},
            {"content": "p3"},
        ]
        out = fuse_rrf(items, k=60, top_k=20)
        assert [d["content"] for d in out] == ["p1", "p2", "p3"]

    def test_none_tier_value_uses_default_group(self) -> None:
        # A present-but-None tier falls into "default" rather than KeyError.
        items = [{"content": "p1", "tier": None}, {"content": "p2", "tier": None}]
        out = fuse_rrf(items, k=60, top_k=20)
        assert [d["content"] for d in out] == ["p1", "p2"]


class TestFuseRRFDoesNotMutate:
    def test_returns_same_objects_not_copies(self) -> None:
        a, b = _item("a", "A"), _item("b", "B")
        out = fuse_rrf([a, b], k=60, top_k=20)
        # The SAME dict objects are returned (reordered), never cloned/edited.
        assert out[0] is a or out[0] is b
        assert a in out and b in out
        assert all(o in (a, b) for o in out)

    def test_input_list_length_unchanged(self) -> None:
        items = [_item("a", "A"), _item("b", "B"), _item("c", "A")]
        _ = fuse_rrf(items, k=60, top_k=20)
        assert len(items) == 3  # original list not truncated
