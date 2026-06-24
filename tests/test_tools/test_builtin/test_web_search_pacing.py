"""Unit tests for the web_search inter-request spacer (S10).

Wires the previously-dead WEB_SEARCH_DELAY_MIN/MAX knobs: concurrent/batched
queries must be spaced so they don't burst-fire and trip an engine rate limit.
Mocks the HTTP layer (_search_with_fallback) so no network is touched.
"""

from __future__ import annotations

import asyncio

from src.tools.builtin import web_search as ws


class _MiniLimits:
    """Stand-in for ToolLimitsSettings exposing only the delay knobs the spacer reads."""

    def __init__(self, lo: float, hi: float) -> None:
        self.web_search_delay_min = lo
        self.web_search_delay_max = hi


def _patch_pacing(monkeypatch, lo: float, hi: float):
    """Point the spacer's settings source at (lo, hi) and drop any cached spacer."""
    monkeypatch.setattr(ws, "_tool_limits", lambda: _MiniLimits(lo, hi))
    ws._reset_search_spacer()


async def test_concurrent_fetches_respect_min_and_max_spacing(monkeypatch):
    """Every concurrent dispatch is paced; each wait respects [MAX-MIN, MAX].

    Asserts the delay was *applied* (via a monkeypatched ``asyncio.sleep``) rather
    than measuring wall-clock gaps — the latter flakes on slow/scheduled CI runners
    where event-loop latency swamps a 50ms floor. ``random.uniform`` is pinned to its
    upper bound so jitter is exactly ``MAX - MIN`` (deterministic + reproducible);
    the spacer then chooses ``wait = max(0, MIN - elapsed) + (MAX - MIN)``. Because
    ``elapsed`` is non-negative, ``wait`` is always in ``[MAX - MIN, MAX]`` and is
    always > 0 (so every fetch is paced) — the real contract the knob encodes, with
    no wall-clock dependency. (The wall-clock *floor* — that dispatches land >= MIN
    apart — is the spacer's runtime effect, observable only by timing and so left
    out of this deterministic unit assertion.)
    """
    MIN, MAX = 0.05, 0.08
    _patch_pacing(monkeypatch, MIN, MAX)

    waits: list[float] = []

    async def fake_sleep(delay: float) -> None:
        # Record the delay the spacer chose instead of actually sleeping.
        waits.append(delay)

    # The spacer resolves ``asyncio.sleep`` via the module's imported asyncio, so
    # patch it there. monkeypatch scopes the patch to this test only.
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    # Pin jitter to its max (MAX - MIN) so waits are reproducible, not random.
    monkeypatch.setattr(ws.random, "uniform", lambda lo, hi: hi)

    async def fake_fallback(_client, query, *_args):
        return [{"title": "t", "href": f"http://x/{query}", "body": "b"}]

    monkeypatch.setattr(ws, "_search_with_fallback", fake_fallback)

    n = 5
    await asyncio.gather(
        *(ws._fetch_results(f"q{i}", 3, "us-en", "", False) for i in range(n))
    )

    # Every fetch was paced (jitter pinned to MAX - MIN > 0 ⇒ wait always > 0).
    assert len(waits) == n, f"expected {n} paced sleeps, got {len(waits)}: {waits}"
    # wait = max(0, MIN - elapsed) + (MAX - MIN) ∈ [MAX - MIN, MAX] — the absolute
    # ceiling (MAX) holds, deterministically, under concurrency.
    assert all((MAX - MIN) - 1e-9 <= w <= MAX + 1e-9 for w in waits), (
        f"wait out of [MAX-MIN, MAX]={[MAX - MIN, MAX]}: {waits}"
    )


async def test_disabled_delay_is_a_noop(monkeypatch):
    """min=max=0 → spacer short-circuits before the lock; never sleeps."""
    _patch_pacing(monkeypatch, 0.0, 0.0)

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    fired: list[str] = []

    async def fake_fallback(_client, query, *_args):
        fired.append(query)
        return [{"title": "t", "href": "http://x", "body": "b"}]

    monkeypatch.setattr(ws, "_search_with_fallback", fake_fallback)

    await asyncio.gather(
        *(ws._fetch_results(f"q{i}", 3, "us-en", "", False) for i in range(8))
    )
    assert len(fired) == 8  # all fired, none delayed
    # The disabled spacer short-circuits at ``if self._max <= 0.0: return`` — it
    # must never reach asyncio.sleep.
    assert sleeps == [], f"disabled spacer should not sleep, got: {sleeps}"


async def test_reset_rebuilds_spacer_from_new_knobs(monkeypatch):
    """_reset_search_spacer() makes the next fetch re-read the delay knobs."""
    _patch_pacing(monkeypatch, 0.0, 0.0)
    s0 = await ws._get_search_spacer()
    assert (s0._min, s0._max) == (0.0, 0.0)

    # Change the knobs + reset → a fresh spacer reflects them.
    monkeypatch.setattr(ws, "_tool_limits", lambda: _MiniLimits(0.04, 0.06))
    ws._reset_search_spacer()
    s1 = await ws._get_search_spacer()
    assert (s1._min, s1._max) == (0.04, 0.06)
    assert s1 is not s0
