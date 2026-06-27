"""Sub-agent semantic selection ranking (F1) — ``AGENT_SELECTION_ENABLED`` parity.

Companion to ``test_selection.py`` (which covers the pure ``_rank_names``
primitive + the disabled-returns-unchanged regression guard). This file locks
the RANKING + fail-safe scenarios the brief calls out: a matching sub-agent is
ranked #1; the survivor ORDER is the original spawn order (not the rank order,
so tier/provider grouping downstream is unaffected); top_k bounds the result;
single-agent / cap-covered short-circuits; the hash-fallback and retriever-error
paths degrade to all-spawn. Deterministic: embed + persister are faked.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.selection import select_subagents_for_subtask
from src.config.settings import Settings


# ─── fakes ──────────────────────────────────────────────────────────────────


class _FakeRetriever:
    """Returns a canned cosine-ordered ``(name, similarity)`` list."""

    def __init__(self, scores: list[tuple[str, float]]) -> None:
        self._scores = scores

    async def retrieve_agents_with_scores(
        self, names: list[str], embedding: list[float], limit: int = 8
    ) -> list[tuple[str, float]]:
        return list(self._scores)


def _settings(top_k: int = 2, enabled: bool = True) -> Settings:
    s = Settings()
    s.agent.agent_selection_enabled = enabled
    s.agent.agent_selection_top_k = top_k
    return s


def _spawned(*names: str) -> list[dict[str, Any]]:
    return [{"name": n, "id": f"id-{n}"} for n in names]


def _patch_embed(
    monkeypatch: pytest.MonkeyPatch, *, source: str = "api"
) -> dict[str, bool]:
    """Patch embed_capability; track whether it was called."""
    called = {"v": False}

    async def fake_embed(query: str) -> tuple[list[float] | None, str | None]:
        del query
        called["v"] = True
        return ([0.1] * 8, source)

    monkeypatch.setattr("src.agents.selection.embed_capability", fake_embed)
    return called


# ─── semantic ranking ───────────────────────────────────────────────────────


class TestSemanticRanking:
    async def test_should_rank_matching_subagent_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 'researcher' is the top cosine hit → it must survive the top-1 cut.
        called = _patch_embed(monkeypatch)
        spawned = _spawned("writer", "researcher", "coder")
        scores = [("researcher", 0.95), ("coder", 0.6), ("writer", 0.4)]
        out = await select_subagents_for_subtask(
            spawned, "research the topic", _settings(top_k=1),
            persister=_FakeRetriever(scores),
        )
        assert called["v"] is True
        assert [d["name"] for d in out] == ["researcher"]

    async def test_should_preserve_spawn_order_not_rank_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Rank puts coder first, but survivors come back in ORIGINAL spawn order
        # so downstream tier-grouping / provider-spread is unaffected.
        _patch_embed(monkeypatch)
        spawned = _spawned("writer", "researcher", "coder")
        scores = [("coder", 0.9), ("researcher", 0.8), ("writer", 0.7)]
        out = await select_subagents_for_subtask(
            spawned, "x", _settings(top_k=2), persister=_FakeRetriever(scores),
        )
        names = [d["name"] for d in out]
        # writer ranked lowest → dropped; researcher + coder survive in spawn order.
        assert names == ["researcher", "coder"]

    async def test_should_bound_result_to_top_k(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_embed(monkeypatch)
        spawned = _spawned("a", "b", "c", "d")
        scores = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
        out = await select_subagents_for_subtask(
            spawned, "x", _settings(top_k=2), persister=_FakeRetriever(scores),
        )
        assert len(out) == 2
        assert {d["name"] for d in out} == {"a", "b"}


class TestShortCircuits:
    async def test_should_return_all_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = _patch_embed(monkeypatch)
        spawned = _spawned("a", "b", "c")
        out = await select_subagents_for_subtask(
            spawned, "x", _settings(enabled=False),
            persister=_FakeRetriever([("a", 0.9)]),
        )
        assert out == spawned
        assert called["v"] is False  # embed never called when disabled

    async def test_should_short_circuit_single_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = _patch_embed(monkeypatch)
        spawned = _spawned("only")
        out = await select_subagents_for_subtask(
            spawned, "x", _settings(top_k=1),
            persister=_FakeRetriever([("only", 0.9)]),
        )
        assert out == spawned
        assert called["v"] is False  # nothing to prune

    async def test_should_short_circuit_when_top_k_covers_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # top_k >= len(spawned) → no pruning needed → all returned, no embed.
        called = _patch_embed(monkeypatch)
        spawned = _spawned("a", "b")
        out = await select_subagents_for_subtask(
            spawned, "x", _settings(top_k=5),
            persister=_FakeRetriever([("a", 0.9)]),
        )
        assert out == spawned
        assert called["v"] is False


class TestFailSafe:
    async def test_should_return_all_on_hash_fallback_embedding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The hash fallback carries no semantic meaning → selection must NOT
        # prune on noise → returns the full spawned set.
        _patch_embed(monkeypatch, source="hash")
        spawned = _spawned("a", "b", "c", "d")
        out = await select_subagents_for_subtask(
            spawned, "x", _settings(top_k=1),
            persister=_FakeRetriever([("a", 0.9)]),
        )
        assert out == spawned

    async def test_should_return_all_on_retriever_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_embed(monkeypatch)

        class _Boom:
            async def retrieve_agents_with_scores(
                self, names: list[str], embedding: list[float], limit: int = 8
            ) -> list[tuple[str, float]]:
                raise RuntimeError("pgvector down")

        spawned = _spawned("a", "b", "c", "d")
        out = await select_subagents_for_subtask(
            spawned, "x", _settings(top_k=1), persister=_Boom(),
        )
        # A recall hiccup can never drop a needed agent.
        assert out == spawned

    async def test_should_return_all_when_recall_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_embed(monkeypatch)
        spawned = _spawned("a", "b", "c")
        out = await select_subagents_for_subtask(
            spawned, "x", _settings(top_k=1), persister=_FakeRetriever([]),
        )
        assert out == spawned
