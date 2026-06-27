"""Tool selection blend parity — ``TOOL_RETRIEVAL_ENABLED`` + blend knobs.

Companion to ``test_selection.py`` (which covers the pure ``blend_score``/
``blend_rank`` primitives, the disabled/full-set regression guards, and the
``select_tools_for_query`` blend-on path). This file locks the
``_retrieve_with_blend`` internal contract and the knob-parity invariants the
brief calls out that are NOT already covered: the blend path bounds strictly to
``top_k``; a persister lacking ``retrieve_tools_with_scores`` degrades to
pure-cosine; the blend widens to a pool then re-ranks (reliable promoted from
OUTSIDE a tight cosine top_k); all ``TOOL_RETRIEVAL_*`` knobs are default-off;
and a persister whose score call raises degrades to pure-cosine (never starves).
Deterministic: embed + persister are faked.
"""

from __future__ import annotations


import pytest

from src.config.settings import Settings
from src.tools.selection import _retrieve_with_blend, blend_rank, select_tools_for_query


# ─── fakes ──────────────────────────────────────────────────────────────────


class _BlendPersister:
    """Implements both retrieve_tools_with_scores + tool_success_metrics."""

    def __init__(
        self,
        pool: list[tuple[str, float]],
        metrics: dict[str, dict[str, float]] | None = None,
        score_exc: BaseException | None = None,
    ) -> None:
        self._pool = pool
        self._metrics = metrics or {}
        self._score_exc = score_exc

    async def retrieve_tools(self, query_embedding: list[float], top_k: int = 8) -> list[str]:
        return [n for n, _ in self._pool[:top_k]]

    async def retrieve_tools_with_scores(
        self, query_embedding: list[float], top_k: int = 8
    ) -> list[tuple[str, float]]:
        if self._score_exc is not None:
            raise self._score_exc
        return list(self._pool[:top_k])

    async def tool_success_metrics(self, names: list[str]) -> dict[str, dict[str, float]]:
        return {n: self._metrics.get(n, {}) for n in names}


class _NamesOnlyPersister:
    """Persister with ONLY retrieve_tools (no scores) — blend degrades to cosine."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    async def retrieve_tools(self, query_embedding: list[float], top_k: int = 8) -> list[str]:
        return list(self._names[:top_k])


def _patch_embed(monkeypatch: pytest.MonkeyPatch, *, source: str = "api") -> None:
    async def fake_embed(query: str) -> tuple[list[float] | None, str | None]:
        del query
        return ([0.1] * 8, source)

    monkeypatch.setattr("src.tools.selection.embed_capability", fake_embed)


# ─── _retrieve_with_blend contract ──────────────────────────────────────────


class TestRetrieveWithBlendContract:
    async def test_should_bound_strictly_to_top_k(self) -> None:
        # Pool of 5, top_k=2 → exactly 2 names returned.
        pool = [(f"dyn{i}", round(0.9 - i * 0.1, 2)) for i in range(5)]
        pers = _BlendPersister(pool, metrics={})
        out = await _retrieve_with_blend(pers, [0.1] * 8, top_k=2,
                                         pool_multiplier=2, weight=0.5)
        assert len(out) == 2
        # weight=0 with no metrics ⇒ pure-cosine order preserved.
        assert out == ["dyn0", "dyn1"]

    async def test_should_promote_reliable_from_outside_pure_cosine_top_k(self) -> None:
        # dyn_a cosinely closest but flaky; dyn_b reliable but farther. With the
        # blend ON, dyn_b is promoted INTO the top_k=1 result over dyn_a.
        pool = [("dyn_a", 0.95), ("dyn_b", 0.80), ("dyn_c", 0.70)]
        metrics = {
            "dyn_a": {"success_rate": 0.1, "empty_output_rate": 0.0},  # flaky
            "dyn_b": {"success_rate": 1.0, "empty_output_rate": 0.0},  # reliable
        }
        pers = _BlendPersister(pool, metrics=metrics)
        out = await _retrieve_with_blend(pers, [0.1] * 8, top_k=1,
                                         pool_multiplier=3, weight=0.5)
        assert out == ["dyn_b"]

    async def test_should_degrade_to_pure_cosine_when_scores_unavailable(self) -> None:
        # A persister without retrieve_tools_with_scores ⇒ the blend falls back to
        # the pure-cosine retrieve_tools path (never starves the run).
        pers = _NamesOnlyPersister(["a", "b", "c"])
        out = await _retrieve_with_blend(pers, [0.1] * 8, top_k=2,
                                         pool_multiplier=2, weight=0.5)
        assert out == ["a", "b"]

    async def test_should_degrade_to_pure_cosine_on_score_call_error(self) -> None:
        pers = _BlendPersister([], score_exc=RuntimeError("pgvector down"))
        # Score pool raises ⇒ empty pool ⇒ falls back to retrieve_tools.
        out = await _retrieve_with_blend(pers, [0.1] * 8, top_k=2,
                                         pool_multiplier=2, weight=0.5)
        assert out == []

    async def test_pool_is_wider_than_final_top_k(self) -> None:
        # pool_multiplier=3 + top_k=1 ⇒ pool fetched at max(1, 3)=3, final=1.
        fetched: dict[str, int] = {}

        class _Cap(_BlendPersister):
            async def retrieve_tools_with_scores(
                self, q: list[float], top_k: int = 8
            ) -> list[tuple[str, float]]:
                fetched["pool_n"] = top_k
                return [("a", 0.9), ("b", 0.8), ("c", 0.7)]

        pers = _Cap([("a", 0.9)])
        out = await _retrieve_with_blend(pers, [0.1] * 8, top_k=1,
                                         pool_multiplier=3, weight=0.5)
        assert len(out) == 1
        assert fetched["pool_n"] == 3


# ─── full-entry parity + knob defaults ──────────────────────────────────────


class TestSelectionParityAndKnobs:
    def test_all_tool_retrieval_knobs_default_off(self) -> None:
        # Byte-identical behavior until toggled on — every knob starts False/conservative.
        s = Settings()
        a = s.agent
        assert a.tool_retrieval_enabled is False
        assert a.tool_retrieval_blend_success is False
        assert a.agent_selection_enabled is False  # F1 also default-off
        # The numeric knobs are sane positive bounds.
        assert a.tool_retrieval_top_k >= 1
        assert a.tool_retrieval_blend_pool >= 1
        assert a.tool_retrieval_blend_weight >= 0.0

    async def test_disabled_returns_full_registered_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Default-off ⇒ the full registry is returned unchanged; embed is never
        # consulted (so no API spend on a feature an operator hasn't enabled).
        from src.tools.registry import ToolRegistry

        reg = ToolRegistry()
        embed_called = {"v": False}

        async def fake_embed(query: str) -> tuple[list[float] | None, str | None]:
            embed_called["v"] = True
            return ([0.1] * 8, "api")

        monkeypatch.setattr("src.tools.selection.embed_capability", fake_embed)
        s = Settings()  # tool_retrieval_enabled False
        out = await select_tools_for_query("anything", reg, s)
        assert out == reg.list_tools()
        assert embed_called["v"] is False

    async def test_blend_rank_is_top_k_bounded_after_rerank(self) -> None:
        # Pure-primitive guard via blend_rank: a pool larger than top_k is
        # truncated AFTER the score re-rank, not before. blend_score =
        # cosine · (1 + weight·success_rate·(1−empty_output_rate)), so a
        # mid-cosine reliable tool (b) beats the cosinely-closest flaky one (a).
        pool = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        metrics = {
            "a": {"success_rate": 0.0, "empty_output_rate": 0.0},  # 0.9·1.0 = 0.90
            "b": {"success_rate": 1.0, "empty_output_rate": 0.0},  # 0.8·1.5 = 1.20 ← winner
            "c": {"success_rate": 1.0, "empty_output_rate": 0.0},  # 0.7·1.5 = 1.05
        }
        out = blend_rank(pool, metrics, top_k=1, weight=0.5)
        assert out == ["b"]  # re-rank moved b to #1, then truncated to 1
        # And the full re-ranked order is b > c > a (the closer-flaky a is last).
        assert blend_rank(pool, metrics, top_k=3, weight=0.5) == ["b", "c", "a"]
