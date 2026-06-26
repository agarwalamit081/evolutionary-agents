"""Tests for src.agents.selection — sub-agent semantic selection (F1, findings-05).

Covers the pure rank primitive and the async selection entry point over a fake
AgentRetriever (mirrors tests/test_tools/test_selection.py for tool retrieval).
The DB-backed query is exercised by tests/test_agents/test_persister_recall.py
(``retrieve_agents_with_scores``); here the recall is stubbed so ranking and
the fail-safe fallbacks are deterministic and hermetic.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.selection import (
    _rank_names,
    select_subagents_for_subtask,
)
from src.config.settings import Settings


class _FakeAgentRetriever:
    """Stand-in AgentRetriever: returns canned (name, similarity) scores."""

    def __init__(
        self,
        scores: list[tuple[str, float]],
        raise_exc: BaseException | None = None,
    ) -> None:
        self._scores = scores
        self._raise = raise_exc

    async def retrieve_agents_with_scores(
        self, names: list[str], embedding: list[float], limit: int = 8
    ) -> list[tuple[str, float]]:
        del names, embedding, limit  # canned response
        if self._raise is not None:
            raise self._raise
        return list(self._scores)


def _enabled_settings(top_k: int = 1) -> Settings:
    s = Settings()
    s.agent.agent_selection_enabled = True
    s.agent.agent_selection_top_k = top_k
    return s


def _spawned(*names: str) -> list[dict[str, Any]]:
    """Spawned-agent info dicts in spawn order (the state shape delegate reads)."""
    return [{"name": n, "id": f"id-{n}"} for n in names]


def _api_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch embed_capability to return a meaningful 'api' vector."""

    async def fake_embed(query: str) -> tuple[list[float] | None, str | None]:
        del query
        return ([0.1] * 8, "api")

    monkeypatch.setattr("src.agents.selection.embed_capability", fake_embed)


class TestRankNames:
    """The pure primitive: cosine-ordered names truncated to top_k."""

    def test_returns_names_in_score_order(self) -> None:
        scores = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        assert _rank_names(scores, top_k=3) == ["a", "b", "c"]

    def test_truncates_to_top_k(self) -> None:
        scores = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        assert _rank_names(scores, top_k=1) == ["a"]

    def test_empty_scores(self) -> None:
        assert _rank_names([], top_k=3) == []


class TestSelectSubagentsForSubtask:
    async def test_disabled_returns_spawned_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard: default-off is unchanged — all spawn, embed never called."""
        s = Settings()
        assert s.agent.agent_selection_enabled is False
        spawned = _spawned("a", "b", "c")
        embed_called = {"v": False}

        async def fake_embed(query: str) -> tuple[list[float] | None, str | None]:
            del query
            embed_called["v"] = True
            return ([0.1] * 8, "api")

        monkeypatch.setattr("src.agents.selection.embed_capability", fake_embed)
        out = await select_subagents_for_subtask(spawned, "x", s)
        assert out == spawned
        assert not embed_called["v"]

    async def test_single_spawned_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing to prune with one agent — full set, embed never called."""
        _api_embed(monkeypatch)
        spawned = _spawned("only")
        out = await select_subagents_for_subtask(spawned, "x", _enabled_settings(top_k=1))
        assert out == spawned

    async def test_cap_covers_everyone_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When top_k >= len(spawned), no reduction — full set returned."""
        _api_embed(monkeypatch)
        spawned = _spawned("a", "b")
        out = await select_subagents_for_subtask(spawned, "x", _enabled_settings(top_k=5))
        assert out == spawned

    async def test_keeps_top_k_by_similarity_preserving_spawn_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DoD: with selection ON, only the top-K most-relevant agents survive;
        survivors keep SPAWN order (not rank order) so tier-grouping is intact."""
        _api_embed(monkeypatch)
        spawned = _spawned("agent1", "agent2", "agent3")
        # Recall ranks agent3 most relevant, then agent1, then agent2.
        pers = _FakeAgentRetriever(
            [("agent3", 0.95), ("agent1", 0.80), ("agent2", 0.40)]
        )
        out = await select_subagents_for_subtask(
            spawned, "x", _enabled_settings(top_k=2), persister=pers
        )
        names = [d["name"] for d in out]
        # Top-2 by similarity are agent3 + agent1; agent2 (least similar) is pruned.
        assert set(names) == {"agent3", "agent1"}
        assert "agent2" not in names
        # Survivor order is SPAWN order (agent1 before agent3), not rank order.
        assert names == ["agent1", "agent3"]

    async def test_no_api_embedding_falls_back_to_full(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hash fallback is not semantically meaningful — can't rank on it."""

        async def fake_embed(query: str) -> tuple[list[float] | None, str | None]:
            del query
            return ([0.1] * 8, "hash")

        monkeypatch.setattr("src.agents.selection.embed_capability", fake_embed)
        spawned = _spawned("a", "b", "c")
        pers = _FakeAgentRetriever([("c", 0.9)])
        out = await select_subagents_for_subtask(
            spawned, "x", _enabled_settings(top_k=1), persister=pers
        )
        assert out == spawned  # full set, no pruning

    async def test_empty_recall_falls_back_to_full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty recall result (no embeddings stored) ⇒ all spawn."""
        _api_embed(monkeypatch)
        spawned = _spawned("a", "b", "c")
        pers = _FakeAgentRetriever([])
        out = await select_subagents_for_subtask(
            spawned, "x", _enabled_settings(top_k=1), persister=pers
        )
        assert out == spawned

    async def test_recall_error_falls_back_to_full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A DB error during recall must not drop a needed agent — all spawn then."""
        _api_embed(monkeypatch)
        spawned = _spawned("a", "b", "c")
        pers = _FakeAgentRetriever([], raise_exc=RuntimeError("db down"))
        out = await select_subagents_for_subtask(
            spawned, "x", _enabled_settings(top_k=1), persister=pers
        )
        assert out == spawned

    async def test_all_pruned_falls_back_to_full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If recall returns only unknown names (none match spawned), keep all."""
        _api_embed(monkeypatch)
        spawned = _spawned("a", "b")
        pers = _FakeAgentRetriever([("zzz_not_spawned", 0.99)])
        out = await select_subagents_for_subtask(
            spawned, "x", _enabled_settings(top_k=1), persister=pers
        )
        # No spawned name survives the filter ⇒ full-set fallback.
        assert out == spawned
