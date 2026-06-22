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
    """N concurrent _fetch_results dispatches land >= MIN and <= MAX apart."""
    MIN, MAX = 0.05, 0.08
    _patch_pacing(monkeypatch, MIN, MAX)

    timestamps: list[float] = []

    async def fake_fallback(_client, query, *_args):
        timestamps.append(asyncio.get_running_loop().time())
        return [{"title": "t", "href": f"http://x/{query}", "body": "b"}]

    monkeypatch.setattr(ws, "_search_with_fallback", fake_fallback)

    n = 5
    await asyncio.gather(
        *(ws._fetch_results(f"q{i}", 3, "us-en", "", False) for i in range(n))
    )

    assert len(timestamps) == n
    ordered = sorted(timestamps)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    # Floor: every gap >= MIN (small slack for event-loop scheduling granularity).
    assert all(g >= MIN - 0.012 for g in gaps), f"gap below floor: {gaps}"
    # Ceiling: no gap exceeds MAX (the floor is min_delay + at-most (max-min) jitter).
    assert all(g <= MAX + 0.012 for g in gaps), f"gap above ceiling: {gaps}"


async def test_disabled_delay_is_a_noop(monkeypatch):
    """min=max=0 → spacer short-circuits; dispatches are NOT serialized."""
    _patch_pacing(monkeypatch, 0.0, 0.0)

    fired: list[str] = []

    async def fake_fallback(_client, query, *_args):
        fired.append(query)
        return [{"title": "t", "href": "http://x", "body": "b"}]

    monkeypatch.setattr(ws, "_search_with_fallback", fake_fallback)

    await asyncio.gather(
        *(ws._fetch_results(f"q{i}", 3, "us-en", "", False) for i in range(8))
    )
    assert len(fired) == 8  # all fired, none delayed


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
