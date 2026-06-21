"""Phase 3 role-split integration: api POST → worker drain → deliverable.

The api route and the worker consumer are the two halves of the role-split; the
existing suites cover each in isolation (``test_api_routes`` the 202/200/404
HTTP contract; ``test_runner`` the consumer's at-least-once drain). This file
wires them together over ONE shared fakeredis server: a real ``POST /run`` lands
a RunJob on the stream, a real ``RunConsumer`` (with a fake executor — no LLM,
no cost) drains it, and the ``GET /runs/{id}`` status reflects the outcome.

This is the deterministic proof that the two roles compose. The live e2e (real
``execute_run`` + evolution + checkpoint resume against the compose stack) lives
in ``test_e2e/`` behind ``@pytest.mark.e2e`` and is the cost-bounded counterpart.
"""

from __future__ import annotations

from typing import Any

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

import src.api.routes.agent as agent_mod
from src.api.app import create_app
from src.api.routes.agent import API_PREFIX
from src.config import get_settings
from src.worker.queue import RunsQueue
from src.worker.runner import RunConsumer
from src.worker.schema import JobStatus, RunJob
from src.worker.status import RunStatusStore


@pytest.fixture
async def role_split_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Shared fakeredis: the api enqueues and the worker drains the same store.

    Pins ``aioredis.from_url`` in the agent module to ONE fakeredis server and
    hands back that client + the consumer settings. The consumer settings MUST
    share the api's stream/group — the api builds its queue from
    ``get_settings().worker`` (agent.py), so the consumer copies that exact
    object (only ``block_ms``/``reclaim_min_idle_ms`` lowered for a snappy test,
    never the stream/group, or the two roles would talk past each other).
    """
    server = fakeredis.FakeServer()
    client = fakeredis.FakeAsyncRedis(server=server)

    def fake_from_url(_url: str, **_kw: object) -> Any:
        return client

    monkeypatch.setattr(agent_mod.aioredis, "from_url", fake_from_url)

    consumer_settings = get_settings().worker.model_copy(
        update={"block_ms": 10, "reclaim_min_idle_ms": 0}
    )
    return {
        "app": create_app(),
        "client": client,
        "consumer_settings": consumer_settings,
    }


def _build_consumer(
    stack: dict[str, Any], executor: Any
) -> RunConsumer:
    """A RunConsumer wired to the same store the api wrote to."""
    client = stack["client"]
    settings = stack["consumer_settings"]
    queue = RunsQueue(client, settings)
    store = RunStatusStore(client, settings)
    return RunConsumer(queue, store, executor, settings)


async def _post_run(app: Any, goal: str, run_id: str) -> dict[str, Any]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            f"{API_PREFIX}/run", json={"goal": goal, "run_id": run_id}
        )
    assert resp.status_code == 202, resp.text
    return resp.json()


async def _get_status(app: Any, run_id: str) -> dict[str, Any]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(f"{API_PREFIX}/runs/{run_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestApiWorkerRoleSplit:
    async def test_post_then_drain_completes_and_acks(self, role_split_stack) -> None:
        """api enqueue → worker drain → status COMPLETED + deliverable + acked."""
        seen: dict[str, RunJob] = {}

        async def fake_executor(job: RunJob) -> dict[str, Any]:
            seen["job"] = job
            return {"final_output": "DELIVERABLE", "is_complete": True, "iteration_count": 1}

        consumer = _build_consumer(role_split_stack, fake_executor)

        # 1. api enqueues via the real HTTP route (touches the shared fakeredis).
        body = await _post_run(role_split_stack["app"], "write a one-line report", "p3-rs-ok")
        assert body["status"] == "queued"
        assert body["thread_id"] == "api-p3-rs-ok"

        # 2. worker drains exactly the enqueued job (no LLM — fake executor).
        acked = await consumer.run_once()
        assert acked == 1
        assert seen["job"].run_id == "p3-rs-ok"

        # 3. status reflects the run; the entry is acked (terminal, no pending).
        status = await _get_status(role_split_stack["app"], "p3-rs-ok")
        assert status["status"] == JobStatus.COMPLETED.value
        assert status["final_output"] == "DELIVERABLE"
        assert status["iteration_count"] == 1
        queue = RunsQueue(role_split_stack["client"], role_split_stack["consumer_settings"])
        assert await queue.pending_count() == 0

    async def test_executor_failure_marks_failed_and_redelivers(self, role_split_stack) -> None:
        """api enqueue → worker raises → FAILED + NOT acked → re-drain redelivers.

        The at-least-once contract through the api path: a failed run is left
        pending, so a later drain (a different worker, or a retry) reclaims it.
        """
        calls = {"n": 0}

        async def flaky_executor(job: RunJob) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient worker failure")
            return {"final_output": "OK-ON-RETRY", "is_complete": True, "iteration_count": 1}

        consumer = _build_consumer(role_split_stack, flaky_executor)
        await _post_run(role_split_stack["app"], "do a thing", "p3-rs-flaky")

        # First drain: executor raises → FAILED, entry stays pending (NOT acked).
        assert await consumer.run_once() == 0
        status = await _get_status(role_split_stack["app"], "p3-rs-flaky")
        assert status["status"] == JobStatus.FAILED.value
        assert "transient worker failure" in status["error"]
        queue = RunsQueue(role_split_stack["client"], role_split_stack["consumer_settings"])
        assert await queue.pending_count() == 1  # still pending → redeliverable

        # Second drain: reclaim_stale (reclaim_min_idle_ms=0) hands it back → succeeds.
        assert await consumer.run_once() == 1
        status = await _get_status(role_split_stack["app"], "p3-rs-flaky")
        assert status["status"] == JobStatus.COMPLETED.value
        assert status["final_output"] == "OK-ON-RETRY"
        assert await queue.pending_count() == 0
