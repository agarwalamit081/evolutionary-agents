"""Tests for src.tools.builtin.corpus — hybrid corpus index/search (Phase 1).

Unit-level + MockTransport tests. No live Meilisearch or DB: the Meilisearch
REST layer is exercised via httpx.MockTransport, and the semantic/keyword legs
are patched at the internal-helper seam so RRF fusion + degradation paths are
deterministic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.tools.builtin import corpus
from src.tools.builtin.corpus import (
    CorpusUnavailable,
    TransientCorpusError,
    _fusion_key,
    _meili_add_documents,
    _meili_request,
    _meili_search,
    corpus_search,
    index_corpus,
    rrf_fuse,
)


@pytest.fixture(autouse=True)
def _reset_index_ready() -> Iterator[None]:
    """Each test starts with a clean index-init flag (module global)."""
    corpus._INDEX_READY = False
    yield
    corpus._INDEX_READY = False


# ── Reciprocal Rank Fusion ──────────────────────────────────────────────


class TestRRFFusion:
    """RRF merge: cross-leg boost, single-leg ranking, truncation, dedup."""

    def test_empty_inputs(self) -> None:
        assert rrf_fuse([], []) == []

    def test_cross_leg_merge_boosts_score(self) -> None:
        """A doc in BOTH legs merges into one entry with a boosted score."""
        keyword = [{"url": "u1", "title": "A", "content": "ka"}]
        semantic = [{"url": "u1", "content": "sa", "content_hash": "h1"}]
        out = rrf_fuse(keyword, semantic, k=60, final_limit=5)
        assert len(out) == 1  # deduped across legs by url
        merged = out[0]
        assert merged["keyword_rank"] == 1
        assert merged["semantic_rank"] == 1
        # Boosted: 1/(60+1) + 1/(60+1) > a single-leg score of 1/(60+1).
        single = rrf_fuse(keyword, [], k=60, final_limit=5)[0]["rrf_score"]
        assert merged["rrf_score"] > single

    def test_both_legs_doc_outranks_single_leg(self) -> None:
        """A doc in both legs ranks above a keyword-only doc at a worse rank."""
        keyword = [
            {"url": "u1", "title": "both", "content": "x"},
            {"url": "u2", "title": "kw-only", "content": "y"},
        ]
        semantic = [{"url": "u1", "content": "sx", "content_hash": "h1"}]
        out = rrf_fuse(keyword, semantic, k=60, final_limit=5)
        # u1 (both legs) must rank first despite u2 being keyword-rank 2 only.
        assert out[0]["url"] == "u1"
        assert out[1]["url"] == "u2"

    def test_final_limit_truncates(self) -> None:
        keyword = [{"url": f"u{i}", "title": str(i), "content": "c"} for i in range(10)]
        out = rrf_fuse(keyword, [], k=60, final_limit=3)
        assert len(out) == 3
        # Ordered by score descending (rank 1 first).
        assert out[0]["url"] == "u0"
        assert out[2]["url"] == "u2"

    def test_within_leg_dedup_by_url(self) -> None:
        """Two keyword hits sharing a url collapse to one entry."""
        keyword = [
            {"url": "dup", "title": "first", "content": "a"},
            {"url": "dup", "title": "second", "content": "b"},
        ]
        out = rrf_fuse(keyword, [], k=60, final_limit=5)
        assert len(out) == 1
        # Score accumulates from both ranks (1/(61) + 1/(62)).
        assert out[0]["rrf_score"] > 1 / 61


class TestFusionKey:
    """Key precedence: url → content_hash → id → fallback."""

    def test_url_takes_precedence(self) -> None:
        hit = {"url": "u", "content_hash": "h", "id": "i"}
        assert _fusion_key(hit, "fb") == "u"

    def test_content_hash_when_no_url(self) -> None:
        hit = {"content_hash": "h", "id": "i"}
        assert _fusion_key(hit, "fb") == "h"

    def test_id_when_no_url_or_hash(self) -> None:
        hit = {"id": "i"}
        assert _fusion_key(hit, "fb") == "i"

    def test_fallback_when_nothing(self) -> None:
        assert _fusion_key({}, "fb") == "fb"


# ── Meilisearch REST (httpx.MockTransport) ─────────────────────────────


def _search_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/search"):
        return httpx.Response(
            200,
            json={
                "hits": [
                    {"title": "Hit One", "url": "http://a.example", "content": "alpha"},
                    {"title": "Hit Two", "url": "http://b.example", "content": "beta"},
                ]
            },
        )
    return httpx.Response(404)


class TestMeiliSearch:
    def test_returns_hits(self) -> None:
        transport = httpx.MockTransport(_search_handler)
        async def run() -> list[dict[str, object]]:
            async with httpx.AsyncClient(transport=transport) as client:
                return await _meili_search(client, "query", 5)

        hits = asyncio.run(run())
        assert len(hits) == 2
        assert hits[0]["title"] == "Hit One"


class TestMeiliAddDocuments:
    def test_returns_task_uid(self) -> None:
        """POST /documents returns the task uid; the task is awaited (GET /tasks/{uid})."""
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path.endswith("/documents"):
                return httpx.Response(202, json={"taskUid": 42})
            if request.url.path.endswith("/tasks/42"):
                return httpx.Response(200, json={"status": "succeeded"})
            return httpx.Response(200, json={})

        corpus._INDEX_READY = True  # skip _ensure_index for a focused test
        transport = httpx.MockTransport(handler)

        async def run() -> str | None:
            async with httpx.AsyncClient(transport=transport) as client:
                return await _meili_add_documents(
                    client, [{"id": "1", "url": "u", "title": "t", "content": "c", "content_hash": "h"}]
                )

        task = asyncio.run(run())
        assert task == "42"
        assert any(p.endswith("/documents") for p in calls)
        assert any(p.endswith("/tasks/42") for p in calls)  # task polled to completion (S11)


_SAMPLE_DOC = {"id": "1", "url": "u", "title": "t", "content": "c", "content_hash": "h"}


class TestMeiliWaitForTask:
    """S11: _meili_add_documents awaits the async index task so a search in the
    SAME coroutine sees the docs. Previously it was fire-and-forget — the search
    raced the still-enqueued task and saw nothing. httpx.MockTransport; no live
    Meilisearch."""

    def test_polls_until_succeeded(self) -> None:
        """The task is polled through enqueued -> succeeded (>=2 GETs)."""
        corpus._INDEX_READY = True
        statuses = iter(["enqueued", "succeeded"])
        polled: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            polled.append(request.url.path)
            if request.url.path.endswith("/documents"):
                return httpx.Response(202, json={"taskUid": 7})
            if request.url.path.endswith("/tasks/7"):
                return httpx.Response(200, json={"status": next(statuses)})
            return httpx.Response(200, json={})

        transport = httpx.MockTransport(handler)

        async def run() -> str | None:
            async with httpx.AsyncClient(transport=transport) as client:
                return await _meili_add_documents(client, [_SAMPLE_DOC])

        task = asyncio.run(run())
        assert task == "7"
        assert sum(1 for p in polled if p.endswith("/tasks/7")) >= 2

    def test_index_then_search_in_same_coroutine_sees_docs(self) -> None:
        """The S11 fix: after add_documents returns, a search finds the doc.

        The fake Meilisearch only serves the doc on /search once the index task
        reached 'succeeded' — proving the wait closes the index->search race.
        """
        corpus._INDEX_READY = True
        indexed = {"done": False}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/documents"):
                return httpx.Response(202, json={"taskUid": 1})
            if path.endswith("/tasks/1"):
                indexed["done"] = True
                return httpx.Response(200, json={"status": "succeeded"})
            if path.endswith("/search"):
                hits = [{"title": "T", "content": "C", "url": "U"}] if indexed["done"] else []
                return httpx.Response(200, json={"hits": hits})
            return httpx.Response(200, json={})

        transport = httpx.MockTransport(handler)

        async def run() -> list[dict[str, object]]:
            async with httpx.AsyncClient(transport=transport) as client:
                await _meili_add_documents(client, [_SAMPLE_DOC])
                return await _meili_search(client, "C", 5)

        hits = asyncio.run(run())
        assert hits and hits[0]["title"] == "T"

    def test_wait_is_bounded_and_does_not_hang(self, monkeypatch) -> None:
        """A task that never completes returns after max_polls (no infinite loop)."""
        corpus._INDEX_READY = True
        monkeypatch.setattr(corpus, "_meili_task_max_polls", lambda: 3)
        monkeypatch.setattr(corpus, "_meili_task_poll_interval", lambda: 0.0)
        polled: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            polled.append(request.url.path)
            if request.url.path.endswith("/documents"):
                return httpx.Response(202, json={"taskUid": 9})
            if request.url.path.endswith("/tasks/9"):
                return httpx.Response(200, json={"status": "enqueued"})
            return httpx.Response(200, json={})

        transport = httpx.MockTransport(handler)

        async def run() -> None:
            async with httpx.AsyncClient(transport=transport) as client:
                await _meili_add_documents(client, [_SAMPLE_DOC])

        asyncio.run(run())  # must return, not hang
        assert sum(1 for p in polled if p.endswith("/tasks/9")) == 3

    def test_failed_task_returns_without_raising(self, monkeypatch) -> None:
        """A failed task is logged + treated as a soft miss (index leg never raises)."""
        corpus._INDEX_READY = True
        monkeypatch.setattr(corpus, "_meili_task_poll_interval", lambda: 0.0)
        polled: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            polled.append(request.url.path)
            if request.url.path.endswith("/documents"):
                return httpx.Response(202, json={"taskUid": 5})
            if request.url.path.endswith("/tasks/5"):
                return httpx.Response(200, json={"status": "failed", "error": {"message": "boom"}})
            return httpx.Response(200, json={})

        transport = httpx.MockTransport(handler)

        async def run() -> str | None:
            async with httpx.AsyncClient(transport=transport) as client:
                return await _meili_add_documents(client, [_SAMPLE_DOC])

        task = asyncio.run(run())
        assert task == "5"
        assert sum(1 for p in polled if p.endswith("/tasks/5")) == 1  # terminal → no extra polls


class TestMeiliRetry:
    """Retry/raise semantics live in _meili_request (the keyword leg degrades)."""

    def test_transient_5xx_retried_then_raises(self) -> None:
        """A persistent 503 is retried up to the cap, then TransientCorpusError."""
        attempts = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(503, text="unavailable")

        transport = httpx.MockTransport(handler)

        async def run() -> None:
            async with httpx.AsyncClient(transport=transport) as client:
                await _meili_request(
                    client, "POST", "/indexes/turing_corpus/search", json_body={"q": "q", "limit": 5}
                )

        with pytest.raises(TransientCorpusError):
            asyncio.run(run())
        assert attempts["n"] == 3  # retried to the cap (stop_after_attempt(3))

    def test_4xx_not_retried(self) -> None:
        """A 400 is CorpusUnavailable and is NOT retried (one attempt only)."""
        attempts = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(400, text="bad request")

        transport = httpx.MockTransport(handler)

        async def run() -> None:
            async with httpx.AsyncClient(transport=transport) as client:
                await _meili_request(
                    client, "POST", "/indexes/turing_corpus/search", json_body={"q": "q", "limit": 5}
                )

        with pytest.raises(CorpusUnavailable):
            asyncio.run(run())
        assert attempts["n"] == 1


# ── corpus_search orchestration (legs mocked at the seam) ───────────────


class TestCorpusSearch:
    @pytest.mark.asyncio
    async def test_formats_fused_results(self) -> None:
        """Fused hits are formatted as 'N. title / snippet / url [rrf=...]'."""
        keyword = [{"url": "http://a", "title": "Alpha", "content": "first hit"}]
        semantic: list[dict[str, object]] = []
        with (
            patch("src.tools.builtin.corpus._meili_search", new=AsyncMock(return_value=keyword)),
            patch("src.tools.builtin.corpus._cold_semantic_search", new=AsyncMock(return_value=semantic)),
        ):
            out = await corpus_search(query="alpha", top_k=3)
        assert "Alpha" in out
        assert "[rrf=" in out
        assert "http://a" in out

    @pytest.mark.asyncio
    async def test_degrades_to_zero_results(self) -> None:
        """Both legs empty → a clear '0 results' message (never raises)."""
        with (
            patch("src.tools.builtin.corpus._meili_search", new=AsyncMock(return_value=[])),
            patch("src.tools.builtin.corpus._cold_semantic_search", new=AsyncMock(return_value=[])),
        ):
            out = await corpus_search(query="nothing", top_k=3)
        assert "0 results" in out

    @pytest.mark.asyncio
    async def test_batch_one_block_per_query(self) -> None:
        """Batch mode yields one '## <query>' block per query."""
        with patch(
            "src.tools.builtin.corpus._search_single",
            new=AsyncMock(side_effect=lambda q, _k: f"hits-for-{q}"),
        ):
            out = await corpus_search(queries=["a", "b"], top_k=2)
        assert "## a" in out and "hits-for-a" in out
        assert "## b" in out and "hits-for-b" in out

    @pytest.mark.asyncio
    async def test_empty_query_returns_hint(self) -> None:
        out = await corpus_search(query="   ")
        assert "needs" in out.lower() or "query" in out.lower()


# ── index_corpus orchestration ──────────────────────────────────────────


class TestIndexCorpus:
    @pytest.mark.asyncio
    async def test_single_no_content_returns_hint(self) -> None:
        """Single mode with no content/url returns the needs-content message."""
        out = await index_corpus()
        assert "needs" in out.lower() or "nothing" in out.lower()

    @pytest.mark.asyncio
    async def test_single_success_message(self) -> None:
        """A real index attempt (legs degrade gracefully offline) reports status."""
        # Both stores are unavailable in CI (no Meilisearch/DB) → _index_single
        # returns False; the message reflects nothing was indexed.
        out = await index_corpus(url="http://x.example", content="hello world", title="X")
        assert isinstance(out, str) and out  # never raises; returns a status line

    @pytest.mark.asyncio
    async def test_batch_counts_successes(self) -> None:
        """Batch reports 'Indexed N/M' based on _index_single results."""
        with patch(
            "src.tools.builtin.corpus._index_single",
            new=AsyncMock(side_effect=[True, False, True]),
        ):
            out = await index_corpus(
                documents=[
                    {"url": "u1", "content": "a"},
                    {"url": "u2", "content": "b"},
                    {"url": "u3", "content": "c"},
                ]
            )
        assert "Indexed 2/3" in out

    @pytest.mark.asyncio
    async def test_batch_concurrency_capped(self) -> None:
        """Batch fan-out respects SEARCH_BATCH_CONCURRENCY (peak ≤ cap)."""
        peak = {"n": 0, "cur": 0}

        async def fake_index(_url, _content, _title, _chash, _metadata) -> bool:
            peak["cur"] += 1
            peak["n"] = max(peak["n"], peak["cur"])
            await asyncio.sleep(0.01)
            peak["cur"] -= 1
            return True

        with (
            patch("src.tools.builtin.corpus._index_single", new=AsyncMock(side_effect=fake_index)),
            patch(
                "src.tools.builtin.corpus._search_settings",
                return_value=SimpleNamespace(search_batch_concurrency=2),
            ),
        ):
            out = await index_corpus(
                documents=[{"url": f"u{i}", "content": f"c{i}"} for i in range(6)]
            )
        assert "Indexed 6/6" in out
        assert peak["n"] <= 2  # concurrency never exceeded the cap

    @pytest.mark.asyncio
    async def test_batch_one_failure_does_not_sink_batch(self) -> None:
        """A raising _index_single is absorbed (return_exceptions) — batch survives."""
        with patch(
            "src.tools.builtin.corpus._index_single",
            new=AsyncMock(side_effect=[RuntimeError("boom"), True]),
        ):
            out = await index_corpus(
                documents=[{"url": "u1", "content": "a"}, {"url": "u2", "content": "b"}]
            )
        assert "Indexed 1/2" in out  # one raised (counted as not-ok), one succeeded
