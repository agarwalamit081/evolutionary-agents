"""Neo4j entity/relation graph — structured mirror (Phase 5 I3).

An ADDITIVE, opt-in relationship substrate. When ``GRAPH_ENABLED``, the
``MemoryManager`` write hooks mirror STRUCTURED records (sub-agent defs,
skills/procedures/workflows, facts) into Neo4j nodes/edges — relationships the
relational + pgvector stores cannot express (which skills depend on X, which
sub-agent handles Y, which facts are about an entity). Pure structured sync —
NO LLM extraction (a later option).

Three guarantees (mirror the CostTracker-resilience pattern):

1. **Default-off / byte-identical-when-off.** Every public method short-circuits
   before any driver work when ``settings.enabled`` is False — nothing syncs,
   identical to pre-I3.
2. **Lazy driver, never a hard dependency.** ``neo4j`` is imported INSIDE
   ``_ensure_driver``; a missing install or an unreachable server marks the
   store unavailable and every op becomes a silent no-op.
3. **Never raises.** Every op is wrapped — a graph hiccup (import error,
   connection refused, write error) is logged WARNING and swallowed. A graph
   store can NEVER abort a run.

The driver is injectable (``driver=``) so the unit suite runs against a fake
driver with no Neo4j install and no live server.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from src.config.settings import Neo4jSettings


# Neo4j labels / relationship types are IDENTIFIERS interpolated into Cypher
# (they cannot be parameterized). Validate them strictly to prevent Cypher
# injection — anything that is not a clean identifier falls back to a default.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(value: str | None, default: str) -> str:
    """Return ``value`` if it is a safe Neo4j identifier, else ``default``."""
    if value and _IDENT_RE.match(value):
        return value
    return default


# Whitelisted skill_type → node label. Keeps arbitrary memory_type strings (e.g.
# "folded_memory") out of the graph — only genuine skills/procedures/workflows
# become graph nodes; the rest are skipped (the caller never mirrors them).
_SKILL_LABELS: dict[str, str] = {
    "skill": "Skill",
    "procedure": "Procedure",
    "workflow": "Workflow",
    "technique": "Technique",
}


class Neo4jGraph:
    """Structured-mirror graph store. Lazy, default-off, never raises."""

    def __init__(self, settings: Neo4jSettings, *, driver: Any = None) -> None:
        self._settings = settings
        # Injected (tests) or lazily built on first use (prod). Typed Any because
        # ``neo4j.AsyncDriver`` is imported lazily and may be absent.
        self._driver: Any = driver
        # Sticky once a connection/import is known to be impossible: every op
        # short-circuits without re-trying the (failing) driver construction.
        self._unavailable: bool = False

    # ── driver lifecycle ───────────────────────────────────────────────────

    async def _ensure_driver(self) -> Any:
        """Return a usable driver, or None (sets ``_unavailable`` on failure)."""
        if self._driver is not None:
            return self._driver
        if self._unavailable:
            return None
        try:
            from neo4j import AsyncGraphDatabase  # lazy: never a hard dependency
        except ImportError:
            logger.warning(
                "neo4j driver not installed — entity/relation graph sync "
                "disabled (run continues)"
            )
            self._unavailable = True
            return None
        try:
            driver = AsyncGraphDatabase.driver(
                self._settings.uri,
                auth=(self._settings.user, self._settings.password),
            )
            await driver.verify_connectivity()
        except Exception as exc:  # noqa: BLE001 — non-fatal observability-only
            logger.warning(
                f"Neo4j unreachable at {self._settings.uri} — entity/relation "
                f"graph sync disabled (run continues): {exc}"
            )
            self._unavailable = True
            return None
        self._driver = driver
        return driver

    async def _write(self, blocks: list[tuple[str, dict[str, Any]]]) -> None:
        """Run idempotent MERGE blocks under one session; never raises."""
        if not self._settings.enabled:
            return
        driver = await self._ensure_driver()
        if driver is None:
            return
        try:
            async with driver.session() as session:
                for cypher, params in blocks:
                    result = await session.run(cypher, **params)
                    await result.consume()
        except Exception as exc:  # noqa: BLE001 — non-fatal observability-only
            logger.warning(f"Neo4j write failed (run continues): {exc}")

    async def _read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Run a read query; returns [] on any failure or when off."""
        if not self._settings.enabled:
            return []
        driver = await self._ensure_driver()
        if driver is None:
            return []
        try:
            async with driver.session() as session:
                result = await session.run(cypher, **params)
                return await result.data()
        except Exception as exc:  # noqa: BLE001 — non-fatal observability-only
            logger.warning(f"Neo4j query failed (run continues): {exc}")
            return []

    # ── sync hooks (called from MemoryManager write methods) ───────────────

    async def sync_skill(
        self,
        name: str,
        content: str,
        *,
        skill_type: str = "skill",
        tags: list[str] | None = None,
        depends_on: list[str] | None = None,
    ) -> None:
        """Mirror a skill/procedure/workflow node + its DEPENDS_ON edges."""
        label = _SKILL_LABELS.get(skill_type, "")
        if not label:
            # Not a graph-worthy record (e.g. folded_memory) — skip, never raise.
            return
        blocks: list[tuple[str, dict[str, Any]]] = [
            (
                f"MERGE (s:{label} {{name: $name}}) "
                "SET s.content = $content, s.type = $type, s.tags = $tags",
                {
                    "name": name,
                    "content": content,
                    "type": skill_type,
                    "tags": tags or [],
                },
            ),
        ]
        for dep in depends_on or []:
            blocks.append(
                (
                    "MERGE (d:Skill {name: $dep}) "
                    f"WITH d MATCH (s:{label} {{name: $name}}) "
                    "MERGE (s)-[:DEPENDS_ON]->(d)",
                    {"dep": dep, "name": name},
                )
            )
        await self._write(blocks)

    async def sync_fact(
        self,
        key: str,
        value: str,
        *,
        entity: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Mirror a Fact node linked :ABOUT an Entity node (keyed by entity)."""
        ent = entity or key
        blocks: list[tuple[str, dict[str, Any]]] = [
            (
                "MERGE (e:Entity {name: $entity}) "
                "MERGE (f:Fact {key: $key}) "
                "SET f.value = $value, f.confidence = $conf "
                "MERGE (f)-[:ABOUT]->(e)",
                {
                    "entity": ent,
                    "key": key,
                    "value": value,
                    "conf": confidence if confidence is not None else 0.5,
                },
            ),
        ]
        await self._write(blocks)

    async def sync_subagent(
        self,
        name: str,
        purpose: str,
        *,
        tool_scope: list[str] | None = None,
        model_tier: str | None = None,
    ) -> None:
        """Mirror a SubAgent node (purpose + tool scope + model tier)."""
        tier = _ident(model_tier, "unspecified")
        await self._write(
            [
                (
                    "MERGE (a:SubAgent {name: $name}) "
                    "SET a.purpose = $purpose, a.model_tier = $tier, "
                    "a.tools = $tools",
                    {
                        "name": name,
                        "purpose": purpose,
                        "tier": tier,
                        "tools": tool_scope or [],
                    },
                ),
            ]
        )

    # ── recall / query API (used by graph_query + retrieve_memory_node) ─────

    async def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Arbitrary read-only Cypher. Returns [] off / on failure."""
        return await self._read(cypher, **params)

    async def skills_depending_on(self, target: str) -> list[dict[str, Any]]:
        """Which skills depend on ``target`` (the dependency node name)."""
        return await self._read(
            "MATCH (s)-[:DEPENDS_ON]->(:Skill {name: $target}) "
            "RETURN s.name AS skill, s.type AS type",
            target=target,
        )

    async def subagents_handling(self, keyword: str) -> list[dict[str, Any]]:
        """Which sub-agents' purpose mentions ``keyword`` (which handles Y)."""
        return await self._read(
            "MATCH (a:SubAgent) WHERE a.purpose CONTAINS $keyword "
            "RETURN a.name AS name, a.purpose AS purpose, a.model_tier AS tier",
            keyword=keyword,
        )

    async def facts_about(self, entity: str) -> list[dict[str, Any]]:
        """Facts linked :ABOUT the named entity."""
        return await self._read(
            "MATCH (f:Fact)-[:ABOUT]->(:Entity {name: $entity}) "
            "RETURN f.key AS key, f.value AS value, f.confidence AS confidence",
            entity=entity,
        )

    # ── teardown ───────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the driver if we built one; never raises."""
        if self._driver is None:
            return
        try:
            await self._driver.close()
        except Exception as exc:  # noqa: BLE001 — non-fatal observability-only
            logger.warning(f"Neo4j driver close failed (run continues): {exc}")
        finally:
            self._driver = None
