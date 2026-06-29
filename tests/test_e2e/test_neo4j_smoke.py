"""#9 / Phase-5 I3 — live Neo4j entity/relation graph E2E smoke.

Exercises the REAL ``Neo4jGraph`` (``src/memory/graph.py``) against a live Neo4j,
proving the structured-mirror write hooks (``sync_subagent`` / ``sync_skill`` /
``sync_fact``) and the recall API (``subagents_handling`` /
``skills_depending_on`` / ``facts_about``) round-trip end to end.

Neo4j is an OPT-IN, profile-gated dependency, so this test SKIPS (never fails)
when the graph is not reachable / the driver is not installed — mirroring the
store's "never a hard dependency" contract. It never edits ``graph.py``.

Operator bring-up (the profile gate):

    docker compose --profile graph up -d        # neo4j service + Bolt host mirror on 17687

For HOST-run pytest, point the gateway at the host mirror in ``.env``:
``NEO4J_URI=bolt://localhost:17687`` + ``NEO4J_PASSWORD=<NEO4J_AUTH password>``
(default ``turing-graph``). Inside the worker the internal ``bolt://neo4j:7687``
resolves over ``turing-net``. Requires the ``self-evolving-agent:latest`` image
to carry the ``neo4j`` driver (rebuild if it was added since the last build).

Run (opt-in):

    python -m pytest tests/test_e2e/test_neo4j_smoke.py -v -m e2e
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.config import get_settings
from src.memory.graph import Neo4jGraph

# Class-level: this is an E2E test (excluded from the default ``not e2e`` gate).
pytestmark = pytest.mark.e2e


def _graph() -> Neo4jGraph:
    """A Neo4jGraph pointed at the live server, with the master switch forced ON.

    Reads uri/user/password from the live settings (.env) so it respects the
    operator's deployment (host mirror vs internal turing-net name); does NOT
    mutate the cached ``get_settings()`` singleton (model_copy).
    """
    live = get_settings().neo4j.model_copy()
    return Neo4jGraph(live.model_copy(update={"enabled": True}))


class TestNeo4jGraphSmoke:
    @pytest.mark.asyncio
    async def test_live_graph_write_and_recall_round_trip(self) -> None:
        graph = _graph()
        tag = f"__smoke_{uuid.uuid4().hex[:8]}"
        try:
            # Availability gate: a live ``RETURN 1`` probe. [] ⇒ the opt-in graph
            # is off / driver missing / server unreachable ⇒ SKIP (never fail).
            probe = await graph.query("RETURN 1 AS ok")
            if not probe:
                pytest.skip(
                    "Neo4j not reachable / driver not installed — "
                    "profile-gated optional dependency "
                    "(bring up with `docker compose --profile graph up -d`)"
                )

            sa = f"{tag}-subagent"
            skill = f"{tag}-skill"
            dep = f"{tag}-dep"
            entity = f"{tag}-entity"
            fact_key = f"{tag}-fact"

            # ── Write hooks: SubAgent node, Skill node + DEPENDS_ON edge, Fact ──
            await graph.sync_subagent(
                sa,
                purpose=f"handles the {tag} domain",
                tool_scope=["code_executor"],
                model_tier="cheap",
            )
            await graph.sync_skill(
                skill,
                content="smoke skill body",
                skill_type="skill",
                depends_on=[dep],
            )
            await graph.sync_fact(
                fact_key,
                "smoke fact value",
                entity=entity,
                confidence=0.9,
            )

            # ── Recall API: each query returns the node/edge we just mirrored ──
            handling = await graph.subagents_handling(tag)
            assert any(h["name"] == sa for h in handling), (
                f"sync_subagent node not recalled: {handling}"
            )

            depending = await graph.skills_depending_on(dep)
            assert any(d["skill"] == skill for d in depending), (
                f"DEPENDS_ON edge not recalled: {depending}"
            )

            facts = await graph.facts_about(entity)
            assert any(f["key"] == fact_key for f in facts), (
                f"Fact/Entity ABOUT edge not recalled: {facts}"
            )
        finally:
            # Clean up ONLY our tagged nodes/edges (DETACH DELETE drops edges).
            # Reached via the lazily-built driver (no public write/delete API on
            # Neo4jGraph by design — graph.py is not edited for this test).
            driver: Any = getattr(graph, "_driver", None)
            if driver is not None:
                try:
                    async with driver.session() as session:
                        result = await session.run(
                            "MATCH (n) WHERE n.name STARTS WITH $p "
                            "OR n.key STARTS WITH $p DETACH DELETE n",
                            p=tag,
                        )
                        await result.consume()
                except Exception:  # noqa: BLE001 — cleanup is best-effort
                    pass
            await graph.close()
