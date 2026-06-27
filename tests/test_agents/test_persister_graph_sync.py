"""Tests for SubAgentPersister._sync_subagent_graph — the I3 mirror hook.

The persister builds a fresh Neo4jGraph per call via the lazy get_settings()
idiom, so the suite patches:
  - src.config.settings.get_settings  → controls .neo4j.enabled
  - src.memory.graph.Neo4jGraph       → a capturing stand-in (no driver/server)

DoD: the hook mirrors the structured sub-agent def when the graph is enabled,
is a no-op when off, and never raises (graph hiccups can't abort persistence).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.agents.persister import SubAgentPersister
from src.graph.enums import TaskComplexity
from src.graph.models import SubAgentSpec


class _CapturingGraph:
    """Stand-in for Neo4jGraph: records sync_subagent args + close()."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.synced: list[tuple[Any, ...]] = []
        self.closed = False

    async def sync_subagent(
        self,
        name: str,
        purpose: str,
        *,
        tool_scope: list[str] | None = None,
        model_tier: str | None = None,
    ) -> None:
        self.synced.append((name, purpose, tool_scope, model_tier))

    async def close(self) -> None:
        self.closed = True


class _RaisingGraph(_CapturingGraph):
    async def sync_subagent(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("graph write failed")


def _spec(
    *,
    name: str = "csv-agent",
    description: str = "Cleans CSV inputs",
    tool_subset: list[str] | None = None,
    model_tier: TaskComplexity = TaskComplexity.COMPLEX,
) -> SubAgentSpec:
    return SubAgentSpec(
        goal="normalize a CSV",
        parent_thread_id="thread-1",
        name=name,
        description=description,
        tool_subset=tool_subset if tool_subset is not None else ["code_executor"],
        model_tier=model_tier,
    )


class TestSyncSubagentGraph:
    async def test_disabled_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GRAPH_ENABLED=False ⇒ no Neo4jGraph constructed, no sync."""
        built: list[Any] = []

        def _fake_get_settings() -> SimpleNamespace:
            return SimpleNamespace(neo4j=SimpleNamespace(enabled=False))

        monkeypatch.setattr(
            "src.config.settings.get_settings", _fake_get_settings
        )
        monkeypatch.setattr(
            "src.memory.graph.Neo4jGraph",
            lambda s: built.append(s) or _CapturingGraph(s),
        )

        persister = SubAgentPersister()
        await persister._sync_subagent_graph(_spec())

        assert built == []  # never constructed (early return on disabled)

    async def test_enabled_syncs_structured_def(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enabled ⇒ sync_subagent called with name/desc/tools/tier, then closed."""
        captured: dict[str, Any] = {}

        def _fake_get_settings() -> SimpleNamespace:
            return SimpleNamespace(neo4j=SimpleNamespace(enabled=True))

        graph = _CapturingGraph(SimpleNamespace(enabled=True))

        def _ctor(settings: Any) -> _CapturingGraph:
            captured["settings"] = settings
            return graph

        monkeypatch.setattr(
            "src.config.settings.get_settings", _fake_get_settings
        )
        monkeypatch.setattr("src.memory.graph.Neo4jGraph", _ctor)

        persister = SubAgentPersister()
        await persister._sync_subagent_graph(
            _spec(tool_subset=["code_executor", "file_writer"])
        )

        assert captured["settings"].enabled is True  # built from get_settings().neo4j
        assert graph.synced == [
            (
                "csv-agent",
                "Cleans CSV inputs",
                ["code_executor", "file_writer"],
                "complex",  # TaskComplexity.COMPLEX.value
            )
        ]
        assert graph.closed is True  # driver torn down after the sync

    async def test_never_raises_on_sync_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A graph write failure must not abort the persist call."""
        monkeypatch.setattr(
            "src.config.settings.get_settings",
            lambda: SimpleNamespace(neo4j=SimpleNamespace(enabled=True)),
        )
        monkeypatch.setattr(
            "src.memory.graph.Neo4jGraph", lambda s: _RaisingGraph(s)
        )

        persister = SubAgentPersister()
        await persister._sync_subagent_graph(_spec())  # must not raise
