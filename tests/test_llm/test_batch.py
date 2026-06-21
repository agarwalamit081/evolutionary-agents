"""Request batching (2B): ``LLMGateway.abatch`` (asyncio.gather + semaphore).

No async litellm batch primitive exists in this build, so ``abatch`` wraps many
``acompletion`` calls in ``asyncio.gather`` bounded by an ``asyncio.Semaphore``
(``BatchingSettings.max_concurrency``). The feature is OFF by default → when
disabled, the semaphore collapses to size 1 (sequential) while preserving the
isolation contract. These tests lock: empty input, the concurrency cap
(timing), partial-failure isolation, sequential behavior when off, and that
each request routes through ``acompletion`` (reusing the rate limiter).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.config.settings import BatchingSettings, Settings
from src.llm.gateway import LLMGateway
from src.llm.models import BatchRequest, BatchResponse


def _make_settings(*, batch_on: bool = False, max_concurrency: int = 5) -> Settings:
    return Settings(
        batching=BatchingSettings(enabled=batch_on, max_concurrency=max_concurrency)
    )


def _make_gateway(settings: Settings) -> LLMGateway:
    with patch.object(LLMGateway, "_configure_litellm"):
        gw = LLMGateway(settings)
    gw._rate_limiter.acquire = AsyncMock(return_value=None)  # type: ignore[assignment]
    return gw


def _req(content: str = "x", model: str = "gpt-4o-mini-2024-07-18") -> BatchRequest:
    return BatchRequest(messages=[{"role": "user", "content": content}], model=model)


class TestAbatchBasics:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self) -> None:
        gw = _make_gateway(_make_settings(batch_on=True))
        assert await gw.abatch([]) == []

    @pytest.mark.asyncio
    async def test_returns_one_per_request_in_order(self) -> None:
        gw = _make_gateway(_make_settings(batch_on=True))

        async def _fake_acompletion(**kwargs: Any) -> Any:
            user_msg = kwargs["messages"][0]["content"]
            resp = type("R", (), {"content": f"echo:{user_msg}", "model": kwargs.get("model"),
                                  "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0})()
            return resp

        with patch.object(gw, "acompletion", side_effect=_fake_acompletion):
            out = await gw.abatch([_req("a"), _req("b"), _req("c")])

        assert len(out) == 3
        assert all(isinstance(r, BatchResponse) for r in out)
        assert [r.content for r in out] == ["echo:a", "echo:b", "echo:c"]

    @pytest.mark.asyncio
    async def test_routes_through_acompletion_per_request(self) -> None:
        gw = _make_gateway(_make_settings(batch_on=True))
        mock_acompletion = AsyncMock(
            side_effect=[
                type("R", (), {"content": "1", "model": "m", "input_tokens": 1,
                               "output_tokens": 1, "cost_usd": 0.0})(),
                type("R", (), {"content": "2", "model": "m", "input_tokens": 1,
                               "output_tokens": 1, "cost_usd": 0.0})(),
            ]
        )
        with patch.object(gw, "acompletion", mock_acompletion):
            await gw.abatch([_req("a"), _req("b")])
        assert mock_acompletion.await_count == 2


class TestAbatchConcurrency:
    @pytest.mark.asyncio
    async def test_concurrency_cap_parallelizes(self) -> None:
        """With max_concurrency >= N, N slow calls finish in ~1 delay, not N."""
        gw = _make_gateway(_make_settings(batch_on=True, max_concurrency=4))
        delay = 0.1

        async def _slow(**kwargs: Any) -> Any:
            await asyncio.sleep(delay)
            return type("R", (), {"content": "ok", "model": "m", "input_tokens": 1,
                                  "output_tokens": 1, "cost_usd": 0.0})()

        with patch.object(gw, "acompletion", side_effect=_slow):
            loop = asyncio.get_event_loop()
            start = loop.time()
            await gw.abatch([_req(str(i)) for i in range(4)])
            elapsed = loop.time() - start

        # 4 concurrent → ~1 delay; sequential would be ~4*delay. Use a generous
        # midpoint band to stay CI-stable.
        assert elapsed < delay * 2.5, f"expected parallel (~{delay}s), got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_disabled_runs_sequentially(self) -> None:
        """When batching is OFF, the semaphore is size 1 → requests serialize."""
        gw = _make_gateway(_make_settings(batch_on=False, max_concurrency=5))
        delay = 0.1

        async def _slow(**kwargs: Any) -> Any:
            await asyncio.sleep(delay)
            return type("R", (), {"content": "ok", "model": "m", "input_tokens": 1,
                                  "output_tokens": 1, "cost_usd": 0.0})()

        with patch.object(gw, "acompletion", side_effect=_slow):
            loop = asyncio.get_event_loop()
            start = loop.time()
            await gw.abatch([_req(str(i)) for i in range(3)])
            elapsed = loop.time() - start

        # Sequential → ~3*delay; concurrent would be ~delay.
        assert elapsed >= delay * 2.5, f"expected sequential (~{3*delay}s), got {elapsed:.2f}s"


class TestAbatchIsolation:
    @pytest.mark.asyncio
    async def test_partial_failure_isolated(self) -> None:
        """One failing request yields an error-marked BatchResponse; the rest
        still complete and order is preserved."""
        gw = _make_gateway(_make_settings(batch_on=True))

        async def _flaky(**kwargs: Any) -> Any:
            user_msg = kwargs["messages"][0]["content"]
            if user_msg == "boom":
                raise RuntimeError("upstream 500")
            return type("R", (), {"content": f"ok:{user_msg}", "model": "m",
                                  "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0})()

        with patch.object(gw, "acompletion", side_effect=_flaky):
            out = await gw.abatch([_req("a"), _req("boom"), _req("c")])

        assert [r.content for r in out] == ["ok:a", "", "ok:c"]
        assert (out[1].metadata or {}).get("error") == "upstream 500"
        assert out[1].cost_usd == 0.0
