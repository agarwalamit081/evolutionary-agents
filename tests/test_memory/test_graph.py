"""Tests for src.memory.graph — Neo4j structured-mirror store (Phase 5 I3).

Runs entirely against a FAKE neo4j driver (no Neo4j install, no live server).
The fake records every (cypher, params) pair the store writes and returns
canned rows for reads, so the suite is deterministic and hermetic.

Covers the three I3 guarantees (mirror the CostTracker-resilience pattern):
default-off / byte-identical-when-off, lazy+optional driver, never raises.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import Neo4jSettings
from src.memory.graph import Neo4jGraph
from src.memory.manager import MemoryManager


# ── fake driver stack ──────────────────────────────────────────────────────


class _FakeResult:
    """A neo4j result: ``consume()`` for writes, ``data()`` for reads."""

    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store

    async def consume(self) -> None:
        return None

    async def data(self) -> list[dict[str, Any]]:
        return self._store["read_rows"]


class _FakeSession:
    """Records writes; returns the shared fake result for any query."""

    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def run(self, cypher: str, **params: Any) -> _FakeResult:
        self._store["writes"].append((cypher, params))
        return _FakeResult(self._store)


class _FakeDriver:
    """An injectable async neo4j driver stand-in."""

    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store
        self.closed = False

    def session(self) -> _FakeSession:
        return _FakeSession(self._store)

    async def verify_connectivity(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class _ExplodingSession(_FakeSession):
    async def run(self, _cypher: str, **_params: Any) -> _FakeResult:
        raise RuntimeError("driver write blew up")


class _ExplodingDriver(_FakeDriver):
    def session(self) -> _ExplodingSession:  # type: ignore[override]
        return _ExplodingSession(self._store)


def _store() -> dict[str, Any]:
    return {"writes": [], "read_rows": []}


def _enabled_settings() -> Neo4jSettings:
    # _env_file=None keeps the suite hermetic (never reads the live .env); the
    # field carries a validation_alias, so the init-kwarg is ignored — flip the
    # instance attribute directly (Neo4jSettings is not frozen).
    settings = Neo4jSettings(_env_file=None)  # type: ignore[call-arg]
    settings.enabled = True
    return settings


def _writes(store: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return store["writes"]


# ── default-off / byte-identical-when-off ──────────────────────────────────


class TestDefaultOff:
    async def test_off_short_circuits_no_driver_touch(self) -> None:
        """enabled=False ⇒ _write/_read never open a session."""
        store = _store()
        # `enabled` carries a validation_alias (GRAPH_ENABLED), so an `enabled=False`
        # init-kwarg is ignored by pydantic-settings (it'd map to the alias name).
        # Flip the instance attribute directly — mirrors `_enabled_settings` — so this
        # default-off test stays hermetic to the live .env value.
        settings = Neo4jSettings(_env_file=None)  # type: ignore[call-arg]
        settings.enabled = False
        graph = Neo4jGraph(settings, driver=_FakeDriver(store))

        await graph.sync_skill("normalize", "content", skill_type="skill")
        assert _writes(store) == []  # nothing synced
        assert await graph.skills_depending_on("x") == []  # read no-op too


# ── sync hooks write the expected nodes + edges ────────────────────────────


class TestSyncWrites:
    async def test_sync_skill_writes_node_and_dependency_edges(self) -> None:
        store = _store()
        graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))

        await graph.sync_skill(
            "normalize_csv",
            "def normalize(): ...",
            skill_type="skill",
            tags=["data", "csv"],
            depends_on=["read_csv"],
        )
        w = _writes(store)
        # Node block: MERGE the Skill + SET its content/type/tags.
        node_cypher, node_params = w[0]
        assert ":Skill" in node_cypher and "MERGE" in node_cypher
        assert node_params == {
            "name": "normalize_csv",
            "content": "def normalize(): ...",
            "type": "skill",
            "tags": ["data", "csv"],
        }
        # Dependency block: a DEPENDS_ON edge onto a :Skill dependency.
        dep_cypher, dep_params = w[1]
        assert "-[:DEPENDS_ON]->" in dep_cypher
        assert dep_params == {"dep": "read_csv", "name": "normalize_csv"}

    async def test_sync_skill_procedure_label_used(self) -> None:
        store = _store()
        graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))

        await graph.sync_skill("p", "c", skill_type="procedure")
        assert ":Procedure" in _writes(store)[0][0]

    async def test_sync_skill_skips_folded_memory(self) -> None:
        """folded_memory (not graph-worthy) is skipped via the label whitelist."""
        store = _store()
        graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))

        await graph.sync_skill("fold-1", "summary", skill_type="folded_memory")
        assert _writes(store) == []

    async def test_sync_fact_writes_entity_about_fact(self) -> None:
        store = _store()
        graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))

        await graph.sync_fact("row_count", "1024", entity="events", confidence=0.9)
        cypher, params = _writes(store)[0]
        assert ":Entity" in cypher and ":Fact" in cypher and "-[:ABOUT]->" in cypher
        assert params == {
            "entity": "events",
            "key": "row_count",
            "value": "1024",
            "conf": 0.9,
        }

    async def test_sync_fact_defaults_entity_to_key(self) -> None:
        store = _store()
        graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))

        await graph.sync_fact("k", "v")  # no entity, no confidence
        _, params = _writes(store)[0]
        assert params["entity"] == "k"
        assert params["conf"] == 0.5  # default confidence floor

    async def test_sync_subagent_writes_node_with_tier_and_tools(self) -> None:
        store = _store()
        graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))

        await graph.sync_subagent(
            "csv-agent",
            "Cleans CSV inputs",
            tool_scope=["code_executor", "file_writer"],
            model_tier="complex",
        )
        cypher, params = _writes(store)[0]
        assert ":SubAgent" in cypher and "MERGE" in cypher
        assert params == {
            "name": "csv-agent",
            "purpose": "Cleans CSV inputs",
            "tier": "complex",
            "tools": ["code_executor", "file_writer"],
        }

    async def test_sync_subagent_bad_tier_falls_back(self) -> None:
        """A non-identifier model_tier cannot be a Cypher label → 'unspecified'."""
        store = _store()
        graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))

        await graph.sync_subagent("a", "p", model_tier="bad tier!")
        _, params = _writes(store)[0]
        assert params["tier"] == "unspecified"


# ── recall returns driver-produced rows ────────────────────────────────────


class TestRecall:
    async def test_skills_depending_on_returns_driver_rows(self) -> None:
        store = _store()
        rows = [{"skill": "a", "type": "skill"}]
        store["read_rows"] = rows
        graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))

        out = await graph.skills_depending_on("read_csv")
        assert out == rows
        # The recall Cypher targets the dependency relationship.
        assert "DEPENDS_ON" in _writes(store)[0][0]

    async def test_subagents_handling_returns_driver_rows(self) -> None:
        store = _store()
        rows = [{"name": "csv-agent", "purpose": "Cleans CSV", "tier": "complex"}]
        store["read_rows"] = rows
        graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))

        out = await graph.subagents_handling("CSV")
        assert out == rows
        assert "SubAgent" in _writes(store)[0][0]

    async def test_facts_about_returns_driver_rows(self) -> None:
        store = _store()
        rows = [{"key": "row_count", "value": "1024", "confidence": 0.9}]
        store["read_rows"] = rows
        graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))

        out = await graph.facts_about("events")
        assert out == rows
        assert "-[:ABOUT]->" in _writes(store)[0][0]

    async def test_arbitrary_query_returns_rows(self) -> None:
        store = _store()
        store["read_rows"] = [{"c": 7}]
        graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))

        out = await graph.query("MATCH (n) RETURN count(n) AS c")
        assert out == [{"c": 7}]


# ── never raises ───────────────────────────────────────────────────────────


class TestNeverRaises:
    async def test_write_swallows_driver_error(self) -> None:
        store = _store()
        graph = Neo4jGraph(_enabled_settings(), driver=_ExplodingDriver(store))

        # Must not raise even though session.run blows up.
        await graph.sync_skill("n", "c", skill_type="skill")

    async def test_read_returns_empty_on_driver_error(self) -> None:
        store = _store()
        graph = Neo4jGraph(_enabled_settings(), driver=_ExplodingDriver(store))

        assert await graph.skills_depending_on("x") == []

    async def test_missing_neo4j_package_marks_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing neo4j install → store unavailable, ops become no-ops."""
        # ``None`` in sys.modules makes ``import neo4j`` raise ImportError.
        monkeypatch.setitem(sys.modules, "neo4j", None)
        store = _store()
        graph = Neo4jGraph(_enabled_settings())  # no injected driver → lazy path

        await graph.sync_skill("n", "c", skill_type="skill")  # no raise
        assert _writes(store) == []  # no driver ⇒ nothing synced
        assert graph._unavailable is True  # sticky

    async def test_connection_failure_marks_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unreachable Neo4j → verify_connectivity fails → unavailable."""

        class _BadDriver:
            async def verify_connectivity(self) -> None:
                raise ConnectionError("refused")

            async def close(self) -> None:
                return None

        class _FakeAsyncGraphDatabase:
            @staticmethod
            def driver(*_a: Any, **_k: Any) -> _BadDriver:
                return _BadDriver()

        fake = types.ModuleType("neo4j")
        fake.AsyncGraphDatabase = _FakeAsyncGraphDatabase  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "neo4j", fake)

        store = _store()
        graph = Neo4jGraph(_enabled_settings())  # lazy driver path

        await graph.sync_subagent("a", "p")  # no raise
        assert _writes(store) == []
        assert graph._unavailable is True

    async def test_close_is_safe(self) -> None:
        store = _store()
        driver = _FakeDriver(store)
        graph = Neo4jGraph(_enabled_settings(), driver=driver)

        await graph.close()
        assert driver.closed is True
        assert graph._driver is None

    async def test_close_noop_without_driver(self) -> None:
        graph = Neo4jGraph(_enabled_settings())  # no driver
        await graph.close()  # must not raise


# ── MemoryManager wiring: store_skill/store_fact mirror when enabled ────────


def _manager_with_graph(store: dict[str, Any]) -> MemoryManager:
    """Build a MemoryManager whose warm tier is mocked and whose _graph is a
    fake-driver-backed Neo4jGraph (the real hook call-site exercised)."""
    settings = MagicMock()
    settings.redis.cache_ttl_seconds = 3600
    settings.llm.embedding_dim = 768
    settings.neo4j.enabled = False  # __init__ leaves _graph=None; injected below
    mgr = MemoryManager(
        redis_client=MagicMock(),  # type: ignore[arg-type]
        db_session=MagicMock(),  # type: ignore[arg-type]
        settings=settings,  # type: ignore[arg-type]
    )
    warm = MagicMock()
    warm.store = AsyncMock(return_value="skill-uuid")
    warm.store_fact = AsyncMock(return_value="fact-uuid")
    mgr.warm = warm  # type: ignore[assignment]
    # Inject the enabled graph the hook will actually call.
    mgr._graph = Neo4jGraph(_enabled_settings(), driver=_FakeDriver(store))
    return mgr


class TestManagerHook:
    async def test_store_skill_mirrors_to_graph(self) -> None:
        store = _store()
        mgr = _manager_with_graph(store)

        uid = await mgr.store_skill(
            "normalize_csv", "content", skill_type="procedure", tags=["csv"]
        )
        assert uid == "skill-uuid"  # warm write still returned
        cypher, _params = _writes(store)[0]
        assert ":Procedure" in cypher  # graph mirror fired

    async def test_store_fact_mirrors_to_graph(self) -> None:
        store = _store()
        mgr = _manager_with_graph(store)

        uid = await mgr.store_fact("row_count", "1024", confidence=0.8)
        assert uid == "fact-uuid"
        cypher, params = _writes(store)[0]
        assert ":Fact" in cypher and "-[:ABOUT]->" in cypher
        assert params["conf"] == 0.8

    async def test_graph_off_when_settings_disabled(self) -> None:
        """Default-off: a plain mock-settings MemoryManager builds NO graph."""
        settings = MagicMock()
        settings.redis.cache_ttl_seconds = 3600
        settings.llm.embedding_dim = 768
        settings.neo4j.enabled = False
        mgr = MemoryManager(
            redis_client=MagicMock(),  # type: ignore[arg-type]
            db_session=MagicMock(),  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
        )
        assert mgr._graph is None
