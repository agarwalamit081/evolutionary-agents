"""Integration tests for the agent API routes over the worker seam (Phase 2b).

``POST /run`` no longer executes inline — it enqueues a ``RunJob`` and returns
``202``; ``GET /runs/{run_id}`` polls the status store. The routes build their
Redis client from settings via ``aioredis.from_url``; we monkeypatch that to a
shared fakeredis server so the 202/200/404 contract is exercised hermetically
(no live Redis, no agent execution).
"""

from __future__ import annotations

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

import src.api.routes.agent as agent_mod
from src.api.app import create_app
from src.api.routes.agent import API_PREFIX


@pytest.fixture
async def fakeredis_app(monkeypatch):
    """Pin ``aioredis.from_url`` in the agent module to ONE fakeredis server so
    enqueue (POST) and status (GET) observe the same store within a test."""
    server = fakeredis.FakeServer()
    shared: dict[str, object] = {}

    def fake_from_url(_url: str, **_kw: object):
        if "client" not in shared:
            shared["client"] = fakeredis.FakeAsyncRedis(server=server)
        return shared["client"]

    monkeypatch.setattr(agent_mod.aioredis, "from_url", fake_from_url)
    return create_app()


class TestAgentRoutes:
    async def test_post_run_returns_202_enqueue(self, fakeredis_app) -> None:
        transport = ASGITransport(app=fakeredis_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(f"{API_PREFIX}/run", json={"goal": "hello"})
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "queued"
        assert body["run_id"]
        assert body["thread_id"] == f"api-{body['run_id']}"
        assert body["status_url"] == f"{API_PREFIX}/runs/{body['run_id']}"

    async def test_get_status_after_enqueue(self, fakeredis_app) -> None:
        transport = ASGITransport(app=fakeredis_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            post = await ac.post(
                f"{API_PREFIX}/run", json={"goal": "g", "run_id": "r42"}
            )
            get = await ac.get(f"{API_PREFIX}/runs/r42")
        assert post.status_code == 202
        assert get.status_code == 200
        body = get.json()
        assert body["run_id"] == "r42"
        assert body["status"] == "queued"
        assert body["thread_id"] == "api-r42"

    async def test_get_status_404_when_unknown(self, fakeredis_app) -> None:
        transport = ASGITransport(app=fakeredis_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(f"{API_PREFIX}/runs/does-not-exist")
        assert resp.status_code == 404

    async def test_post_run_rejects_empty_goal(self, fakeredis_app) -> None:
        """Validation (Pydantic min_length=1) → 422, not enqueue."""
        transport = ASGITransport(app=fakeredis_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(f"{API_PREFIX}/run", json={"goal": ""})
        assert resp.status_code == 422
