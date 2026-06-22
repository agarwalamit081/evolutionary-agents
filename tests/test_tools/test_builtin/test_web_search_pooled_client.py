"""Unit tests for the pooled ``httpx.AsyncClient`` across a search batch (S16).

``_fetch_batch`` opens ONE shared ``httpx.AsyncClient`` and threads it into every
``_fetch_results`` in the batch, so TCP connections + TLS handshakes are reused
across queries (and across each query's provider-fallback chain) instead of being
built/torn down per query. The single-query path still opens its own.

These tests are deterministic — no network: the HTTP layer (``_search_with_fallback``)
is mocked, and we assert client *identity* (same object across the batch, distinct
objects across single calls, an explicit client reused verbatim).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src.tools.builtin import web_search as ws


async def _noop_pace() -> None:
    """No-op stand-in for ``_pace_search`` (the spacer isn't under test here)."""
    return None


def _patch_http(
    monkeypatch: pytest.MonkeyPatch,
    sink: list[httpx.AsyncClient],
    *,
    concurrency: int = 2,
    fail_on: str | None = None,
) -> None:
    """Mock the HTTP layer + spacer + batch-concurrency so only client identity is on trial."""

    async def fake_fallback(client: httpx.AsyncClient, query: str, *_args: Any) -> list[dict[str, str]]:
        sink.append(client)
        if fail_on is not None and query == fail_on:
            raise RuntimeError("simulated engine 500")
        return [{"title": query, "href": "http://x", "body": "b"}]

    monkeypatch.setattr(ws, "_search_with_fallback", fake_fallback)
    monkeypatch.setattr(ws, "_pace_search", _noop_pace)
    monkeypatch.setattr(
        ws,
        "_search_settings",
        lambda: SimpleNamespace(search_batch_concurrency=concurrency),
    )


# ─── batch: one shared client ──────────────────────────────────────────────


class TestPooledBatchClient:
    """A batch must funnel every query through ONE shared httpx.AsyncClient."""

    @pytest.mark.asyncio
    async def test_batch_shares_one_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[httpx.AsyncClient] = []
        _patch_http(monkeypatch, seen)

        queries = ["alpha", "beta", "gamma"]
        await ws._fetch_batch(queries, 3, "us-en", "", False)

        assert len(seen) == len(queries)  # one fallback call per query
        # Every query got the SAME client object → connection reuse across the batch.
        first = seen[0]
        assert all(c is first for c in seen), "batch queries did not share one client"
        assert isinstance(first, httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_batch_shares_one_client_when_serialized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrency=1 (fully serialized) still shares the single pooled client."""
        seen: list[httpx.AsyncClient] = []
        _patch_http(monkeypatch, seen, concurrency=1)

        await ws._fetch_batch(["a", "b", "c", "d"], 3, "us-en", "", False)

        assert len(seen) == 4
        assert all(c is seen[0] for c in seen)

    @pytest.mark.asyncio
    async def test_batch_failure_is_contained_keeps_shared_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One query raising yields an empty slot; the shared client serves the rest."""
        seen: list[httpx.AsyncClient] = []
        _patch_http(monkeypatch, seen, fail_on="boom")

        results = await ws._fetch_batch(["ok1", "boom", "ok2"], 3, "us-en", "", False)

        # Failed slot → []; others returned their results.
        assert results == [
            [{"title": "ok1", "href": "http://x", "body": "b"}],
            [],
            [{"title": "ok2", "href": "http://x", "body": "b"}],
        ]
        # All three still went through the SAME client (failure was contained).
        assert len(seen) == 3
        assert all(c is seen[0] for c in seen)

    @pytest.mark.asyncio
    async def test_empty_batch_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty query list is a no-op (no client opened, no fallback calls)."""
        seen: list[httpx.AsyncClient] = []
        _patch_http(monkeypatch, seen)

        assert await ws._fetch_batch([], 3, "us-en", "", False) == []
        assert seen == []


# ─── single-query: own client, unless one is handed in ─────────────────────


class TestSingleQueryClient:
    """The single-query path owns its client; an explicit client is reused verbatim."""

    @pytest.mark.asyncio
    async def test_single_query_opens_its_own_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[httpx.AsyncClient] = []
        _patch_http(monkeypatch, seen)

        await ws._fetch_results("solo", 3, "us-en", "", False)
        assert len(seen) == 1
        assert isinstance(seen[0], httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_two_single_queries_get_distinct_clients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The single-query path does NOT pool across calls (each opens its own)."""
        seen: list[httpx.AsyncClient] = []
        _patch_http(monkeypatch, seen)

        await ws._fetch_results("solo1", 3, "us-en", "", False)
        await ws._fetch_results("solo2", 3, "us-en", "", False)

        assert len(seen) == 2
        assert seen[0] is not seen[1]  # distinct clients — no cross-call pooling

    @pytest.mark.asyncio
    async def test_explicit_client_is_reused_not_owned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client passed via kwarg is used verbatim (caller owns its lifecycle)."""
        seen: list[httpx.AsyncClient] = []
        _patch_http(monkeypatch, seen)

        async with httpx.AsyncClient() as explicit:
            await ws._fetch_results("x", 3, "us-en", "", False, client=explicit)
            assert seen[0] is explicit  # the exact object, reused — not wrapped
            assert not explicit.is_closed  # caller owns lifecycle; still open
