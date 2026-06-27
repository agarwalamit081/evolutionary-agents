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
from src.worker.schema import JobStatus, RunStatus


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

    async def test_get_status_surfaces_results_dir(self, monkeypatch) -> None:
        """The surfaced output folder (``results/<run_id>/``) is readable from the
        API — a caller discovers where the artifacts landed without guessing."""
        server = fakeredis.FakeServer()
        client = fakeredis.FakeAsyncRedis(server=server)

        def fake_from_url(_url: str, **_kw: object) -> object:
            return client

        monkeypatch.setattr(agent_mod.aioredis, "from_url", fake_from_url)
        app = create_app()

        # Stamp a RUNNING record with a results_dir (as the worker does at run
        # start) straight into the status hash the route reads — same store the
        # route's get_run_status polls.
        record = RunStatus(
            run_id="r77",
            thread_id="api-r77",
            status=JobStatus.RUNNING,
            results_dir="/vol/results/r77",
        )
        await client.hset("turing:run:r77", mapping=record.to_hash())

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            get = await ac.get(f"{API_PREFIX}/runs/r77")
        assert get.status_code == 200
        assert get.json()["results_dir"] == "/vol/results/r77"

    async def test_post_run_rejects_empty_goal(self, fakeredis_app) -> None:
        """Validation (Pydantic min_length=1) → 422, not enqueue."""
        transport = ASGITransport(app=fakeredis_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(f"{API_PREFIX}/run", json={"goal": ""})
        assert resp.status_code == 422


class TestCancelRoute:
    async def test_cancel_returns_202_when_run_exists(self, fakeredis_app) -> None:
        transport = ASGITransport(app=fakeredis_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            post = await ac.post(
                f"{API_PREFIX}/run", json={"goal": "g", "run_id": "r9"}
            )
            cancel = await ac.post(f"{API_PREFIX}/runs/r9/cancel")
        assert post.status_code == 202
        assert cancel.status_code == 202
        body = cancel.json()
        assert body["run_id"] == "r9"
        assert body["status"] == "cancel_requested"

    async def test_cancel_404_when_unknown(self, fakeredis_app) -> None:
        transport = ASGITransport(app=fakeredis_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(f"{API_PREFIX}/runs/never-existed/cancel")
        assert resp.status_code == 404

    async def test_cancel_idempotent(self, fakeredis_app) -> None:
        """A repeat POST is a no-op (the flag's presence is the signal)."""
        transport = ASGITransport(app=fakeredis_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post(f"{API_PREFIX}/run", json={"goal": "g", "run_id": "r7"})
            first = await ac.post(f"{API_PREFIX}/runs/r7/cancel")
            second = await ac.post(f"{API_PREFIX}/runs/r7/cancel")
        assert first.status_code == 202
        assert second.status_code == 202  # still accepted — no-op

    async def test_cancel_sets_flag_in_shared_redis(self, monkeypatch) -> None:
        """Integration: the route sets the cancel flag on the SAME Redis the
        status store reads, so the worker's progress poll observes it. Pins the
        wiring end-to-end (``aioredis.from_url`` → ``request_cancel`` → Redis)."""
        server = fakeredis.FakeServer()
        shared: dict[str, object] = {}

        def fake_from_url(_url: str, **_kw: object):
            if "client" not in shared:
                shared["client"] = fakeredis.FakeAsyncRedis(server=server)
            return shared["client"]

        monkeypatch.setattr(agent_mod.aioredis, "from_url", fake_from_url)
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.post(f"{API_PREFIX}/run", json={"goal": "g", "run_id": "rF"})
            resp = await ac.post(f"{API_PREFIX}/runs/rF/cancel")
        assert resp.status_code == 202
        # Read the flag back over the SAME shared client the route wrote to.
        from src.config import get_settings
        from src.worker.status import RunStatusStore

        store = RunStatusStore(shared["client"], get_settings().worker)  # type: ignore[arg-type]
        assert await store.is_cancelled("rF") is True

    async def test_cancel_deletes_stream_entry_no_redelivery(self, monkeypatch) -> None:
        """P1 regression — cancel must delete the pending stream entry
        (XACK+XDEL) the instant the flag is set, so ``reclaim_stale`` cannot
        hand it to a peer worker that would resume the run from its checkpoint
        (the respawn / token-burn vector). Also pins that enqueue captured
        ``entry_id`` onto the status record so cancel knows what to delete."""
        from src.config import get_settings
        from src.worker.queue import RunsQueue
        from src.worker.status import RunStatusStore

        server = fakeredis.FakeServer()
        client = fakeredis.FakeAsyncRedis(server=server)
        monkeypatch.setattr(
            agent_mod.aioredis, "from_url", lambda _url, **_kw: client
        )
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            post = await ac.post(f"{API_PREFIX}/run", json={"goal": "g", "run_id": "rE"})
            cancel = await ac.post(f"{API_PREFIX}/runs/rE/cancel")
        assert post.status_code == 202
        assert cancel.status_code == 202

        ws = get_settings().worker
        # entry_id was captured on the status record at enqueue time.
        store = RunStatusStore(client, ws)  # type: ignore[arg-type]
        record = await store.get("rE")
        assert record is not None
        assert record.entry_id  # captured + non-empty

        # The pending entry is GONE — no peer can reclaim it (no respawn).
        q = RunsQueue(client, ws)  # type: ignore[arg-type]
        assert int(await client.xlen(ws.runs_stream)) == 0  # XDEL: stream body empty
        assert await q.reclaim_stale() == []  # XACK: removed from the PEL


class TestEnqueueDedup:
    """P1 — a run_id IS the run's identity (thread_id / checkpoint-resume key),
    so a repeated ``POST /run`` for an already-QUEUED/RUNNING run is refused
    (409): a second entry would resume the SAME checkpoint (double-spend until
    the lease lock serializes them). Terminal runs may re-enqueue."""

    async def test_post_run_rejects_duplicate_queued_run_id(
        self, fakeredis_app
    ) -> None:
        transport = ASGITransport(app=fakeredis_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            first = await ac.post(
                f"{API_PREFIX}/run", json={"goal": "g", "run_id": "rD"}
            )
            second = await ac.post(
                f"{API_PREFIX}/run", json={"goal": "g", "run_id": "rD"}
            )
        assert first.status_code == 202
        assert second.status_code == 409
        assert "already" in second.json()["detail"]

    async def test_post_run_allows_reenqueue_after_terminal(
        self, monkeypatch
    ) -> None:
        """A TERMINAL run_id (COMPLETED/FAILED/CANCELLED/...) may be re-enqueued
        (resume-by-run_id stays available); only QUEUED/RUNNING are blocked."""
        from src.config import get_settings
        from src.worker.schema import JobStatus
        from src.worker.status import RunStatusStore

        server = fakeredis.FakeServer()
        client = fakeredis.FakeAsyncRedis(server=server)
        monkeypatch.setattr(
            agent_mod.aioredis, "from_url", lambda _url, **_kw: client
        )
        # Seed a terminal status so the dedup check sees COMPLETED, not QUEUED.
        store = RunStatusStore(client, get_settings().worker)  # type: ignore[arg-type]
        await store.mark("rT", "api-rT", JobStatus.COMPLETED, is_complete=True)

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                f"{API_PREFIX}/run", json={"goal": "g", "run_id": "rT"}
            )
        assert resp.status_code == 202
