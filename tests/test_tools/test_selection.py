"""Tests for src.tools.selection — tool retrieval-before-selection (findings-05).

Also covers the two primitives it composes: ``ToolRegistry.list_tools(names=)``
(the name filter) and ``ToolPersister.retrieve_tools`` (the recall wrapper over
the dedup-only ``find_similar``)."""

from __future__ import annotations

import pytest

from src.config.settings import Settings
from src.tools.registry import ToolRegistry
from src.tools.selection import select_tools_for_query


async def _noop_handler(**_: object) -> str:
    """Dummy tool handler — never invoked by these selection tests."""
    return ""


def _registry_with(*names: str) -> ToolRegistry:
    """Build a registry with dummy tools of the given names (registry order)."""
    reg = ToolRegistry()
    for name in names:
        reg.register(name=name, handler=_noop_handler, description=f"dummy {name}", parameters={})
    return reg


def _names(defs: list[dict[str, object]]) -> list[str]:
    return [t["function"]["name"] for t in defs]  # type: ignore[index]


class _FakePersister:
    """Stand-in ToolPersister: returns canned names, optionally raises."""

    def __init__(self, retrieved: list[str], raise_exc: BaseException | None = None) -> None:
        self._retrieved = retrieved
        self._raise = raise_exc

    async def retrieve_tools(self, query_embedding: list[float], top_k: int = 8) -> list[str]:
        del query_embedding, top_k  # unused — canned response
        if self._raise is not None:
            raise self._raise
        return list(self._retrieved)


def _enabled_settings(top_k: int = 8) -> Settings:
    s = Settings()
    s.agent.tool_retrieval_enabled = True
    s.agent.tool_retrieval_top_k = top_k
    return s


def _api_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch embed_capability to return a meaningful 'api' vector."""

    async def fake_embed(query: str) -> tuple[list[float] | None, str | None]:
        del query
        return ([0.1] * 8, "api")

    monkeypatch.setattr("src.tools.selection.embed_capability", fake_embed)


class TestListToolsNamesFilter:
    def test_filters_to_named_in_registry_order(self) -> None:
        reg = _registry_with("web_search", "file_writer", "dyn_a", "dyn_b", "dyn_c")
        defs = reg.list_tools(names=["dyn_c", "web_search", "nope"])
        # registry order preserved; unknown name silently skipped
        assert _names(defs) == ["web_search", "dyn_c"]

    def test_none_returns_all(self) -> None:
        reg = _registry_with("a", "b")
        assert _names(reg.list_tools()) == ["a", "b"]
        assert _names(reg.list_tools(None)) == ["a", "b"]

    def test_empty_names_returns_empty(self) -> None:
        reg = _registry_with("a", "b")
        assert reg.list_tools(names=[]) == []


class TestRetrieveToolsWrapper:
    async def test_returns_names_at_threshold_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """retrieve_tools is the RECALL counterpart to dedup-only find_similar:
        threshold=0.0 (every embedded tool eligible, HNSW order ranks) and it
        projects to names only."""
        from src.tools.dynamic.persister import ToolPersister

        captured: dict[str, object] = {}

        async def fake_find_similar(
            self: ToolPersister,
            embedding: list[float],
            threshold: float = 0.85,
            limit: int = 5,
        ) -> list[dict[str, object]]:
            captured["threshold"] = threshold
            captured["limit"] = limit
            return [
                {"tool_name": "dyn_a", "description": "d", "similarity": 0.9},
                {"tool_name": "dyn_b", "description": "d", "similarity": 0.5},
            ]

        monkeypatch.setattr(ToolPersister, "find_similar", fake_find_similar)
        names = await ToolPersister().retrieve_tools([0.1] * 8, top_k=4)
        assert names == ["dyn_a", "dyn_b"]
        assert captured["threshold"] == 0.0  # recall gate, not the 0.85 dedup gate
        assert captured["limit"] == 4


class TestSelectToolsForQuery:
    async def test_disabled_flag_returns_full_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression guard: default-off is unchanged — full set, embed never called."""
        s = Settings()
        assert s.agent.tool_retrieval_enabled is False
        reg = _registry_with("web_search", "dyn_a", "dyn_b")
        embed_called = {"v": False}

        async def fake_embed(query: str) -> tuple[list[float] | None, str | None]:
            embed_called["v"] = True
            return ([0.1] * 8, "api")

        monkeypatch.setattr("src.tools.selection.embed_capability", fake_embed)
        defs = await select_tools_for_query("x", reg, s, persister=_FakePersister(retrieved=["dyn_a"]))
        assert _names(defs) == ["web_search", "dyn_a", "dyn_b"]
        assert not embed_called["v"]

    async def test_enabled_builtins_always_plus_retrieved_dynamic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Core invariant: built-ins survive retrieval (they're not in the
        embedding index) AND only the retrieved dynamic tools are kept."""
        _api_embed(monkeypatch)
        reg = _registry_with("web_search", "file_writer", "dyn_a", "dyn_b", "dyn_c")
        defs = await select_tools_for_query(
            "do thing", reg, _enabled_settings(top_k=8), persister=_FakePersister(retrieved=["dyn_a"])
        )
        names = _names(defs)
        assert "web_search" in names and "file_writer" in names and "dyn_a" in names
        # unretrieved dynamic tools are dropped (the whole point of retrieval)
        assert "dyn_b" not in names and "dyn_c" not in names

    async def test_no_api_embedding_falls_back_to_full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_embed(query: str) -> tuple[list[float] | None, str | None]:
            return (None, None)

        monkeypatch.setattr("src.tools.selection.embed_capability", fake_embed)
        reg = _registry_with("web_search", "dyn_a", "dyn_b")
        defs = await select_tools_for_query(
            "x", reg, _enabled_settings(), persister=_FakePersister(retrieved=["dyn_a"])
        )
        assert _names(defs) == ["web_search", "dyn_a", "dyn_b"]

    async def test_hash_embedding_falls_back_to_full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The hash fallback is not semantically meaningful — can't rank on it."""

        async def fake_embed(query: str) -> tuple[list[float] | None, str | None]:
            return ([0.1] * 8, "hash")

        monkeypatch.setattr("src.tools.selection.embed_capability", fake_embed)
        reg = _registry_with("web_search", "dyn_a")
        defs = await select_tools_for_query(
            "x", reg, _enabled_settings(), persister=_FakePersister(retrieved=["dyn_a"])
        )
        assert _names(defs) == ["web_search", "dyn_a"]

    async def test_empty_retrieval_falls_back_to_full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _api_embed(monkeypatch)
        reg = _registry_with("web_search", "dyn_a", "dyn_b")
        defs = await select_tools_for_query(
            "x", reg, _enabled_settings(), persister=_FakePersister(retrieved=[])
        )
        assert _names(defs) == ["web_search", "dyn_a", "dyn_b"]

    async def test_retrieval_error_falls_back_to_full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _api_embed(monkeypatch)
        reg = _registry_with("web_search", "dyn_a")
        defs = await select_tools_for_query(
            "x",
            reg,
            _enabled_settings(),
            persister=_FakePersister(retrieved=[], raise_exc=RuntimeError("db down")),
        )
        assert _names(defs) == ["web_search", "dyn_a"]

    async def test_stale_retrieved_name_skipped_builtins_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retrieved name with no live registry entry (stale DB row / retired
        tool) is skipped — and built-ins still come through."""
        _api_embed(monkeypatch)
        reg = _registry_with("web_search")  # dyn_x is NOT registered
        defs = await select_tools_for_query(
            "x", reg, _enabled_settings(), persister=_FakePersister(retrieved=["dyn_x"])
        )
        assert _names(defs) == ["web_search"]
