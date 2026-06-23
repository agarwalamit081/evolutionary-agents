"""CurveRegressionGate (Phase 2 C1c): regression verdict → metrics + opt-in rollback.

The gate consumes ``CapabilityCurve.detect_regression`` and orchestrates rollback
via the EXISTING ``PromotionGate`` (no new rollback code). These tests pin the
control flow with a fake curve + fake/real gate:

* gauge is ALWAYS set (the detection signal, even when not regressed);
* counter + telemetry fire ONLY on a regression;
* auto-rollback reverts ONLY suspect promotions (active, within lookback_days);
* a regression with no active promotion is alert-only (model/provider drift);
* ``run`` never raises on a rollback failure (observability-only, like CostTracker).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.config.settings import CapabilityCurveSettings
from src.evolution import curve_gate
from src.evolution.curve_gate import CurveRegressionGate
from src.evolution.promote import PromotionGate


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeCurve:
    """Stand-in for CapabilityCurve: returns a configured regression verdict."""

    def __init__(self, verdict: dict[str, Any]) -> None:
        self._verdict = verdict

    async def detect_regression(self) -> dict[str, Any]:
        return dict(self._verdict)


class _FakeGauge:
    def __init__(self) -> None:
        self.values: list[float] = []

    def set(self, value: float) -> None:
        self.values.append(float(value))


class _FakeCounter:
    def __init__(self) -> None:
        self.count = 0

    def inc(self) -> None:
        self.count += 1


class _TelemetrySink:
    """Records (event_type, event_data) calls; quacks like the real sink."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, event_type: str, event_data: dict[str, Any]) -> None:
        self.events.append((event_type, dict(event_data)))


class _FakeGate:
    """Records rollback calls; optional raise + configurable active promotions."""

    def __init__(
        self,
        *,
        active: list[dict[str, Any]] | None = None,
        rollback_raises: bool = False,
    ) -> None:
        self._active = active or []
        self.rollback_calls: list[str] = []
        self._rollback_raises = rollback_raises

    def active_promotions(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self._active]

    def rollback(self, node: str) -> dict[str, Any]:
        self.rollback_calls.append(node)
        if self._rollback_raises:
            raise RuntimeError("boom")
        return {"rolled_back": True, "node": node}


# --------------------------------------------------------------------------- #
# Verdict builders
# --------------------------------------------------------------------------- #


def _regressed_verdict(*, current: float = 0.30, best_prior: float = 0.85) -> dict[str, Any]:
    return {
        "regressed": True,
        "inconclusive": False,
        "current": current,
        "best_prior": best_prior,
        "delta": round(best_prior - current, 4),
        "n_points": 5,
        "floor": 0.5,
        "delta_floor": 0.1,
        "scope": "battery",
    }


def _ok_verdict(*, current: float = 0.92) -> dict[str, Any]:
    return {
        "regressed": False,
        "inconclusive": False,
        "current": current,
        "best_prior": 0.95,
        "delta": 0.03,
        "n_points": 5,
        "floor": 0.5,
        "delta_floor": 0.1,
        "scope": "battery",
    }


def _recent_active(node: str = "execute", days_ago: int = 4) -> dict[str, Any]:
    recent = datetime(2026, 6, 24, tzinfo=timezone.utc) - timedelta(days=days_ago)
    return {"node": node, "active": f"{node}.recent.json", "promoted_at": recent.isoformat(), "canary_score": 0.9}


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_no_regression_sets_gauge_no_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not regressed → gauge set, counter untouched, no rollback even when auto-rollback is on."""
    gauge, counter = _FakeGauge(), _FakeCounter()
    monkeypatch.setattr(curve_gate, "CAPABILITY_CURVE_SCORE", gauge)
    monkeypatch.setattr(curve_gate, "CAPABILITY_CURVE_REGRESSIONS", counter)

    gate_node = _FakeGate(active=[_recent_active()])  # a promotion exists; must NOT be touched
    telemetry = _TelemetrySink()
    gate = CurveRegressionGate(
        _FakeCurve(_ok_verdict(current=0.92)),
        gate_node,
        telemetry=telemetry,
        settings=CapabilityCurveSettings(auto_rollback=True, lookback_days=30),
    )

    verdict = await gate.run(now=datetime(2026, 6, 24, tzinfo=timezone.utc))

    assert gauge.values == [0.92]            # detection signal always emitted
    assert counter.count == 0                # no regression → counter not incremented
    assert gate_node.rollback_calls == []    # never reached the suspect scan
    assert telemetry.events == []            # no telemetry on a clean curve
    assert verdict["rollbacks"] == []


@pytest.mark.asyncio
async def test_run_regression_alerts_without_autorollback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regressed + auto_rollback OFF → counter + telemetry fire, no rollback."""
    gauge, counter = _FakeGauge(), _FakeCounter()
    monkeypatch.setattr(curve_gate, "CAPABILITY_CURVE_SCORE", gauge)
    monkeypatch.setattr(curve_gate, "CAPABILITY_CURVE_REGRESSIONS", counter)

    gate_node = _FakeGate(active=[_recent_active()])  # suspect exists; auto-rollback off → left alone
    telemetry = _TelemetrySink()
    gate = CurveRegressionGate(
        _FakeCurve(_regressed_verdict(current=0.30)),
        gate_node,
        telemetry=telemetry,
        settings=CapabilityCurveSettings(auto_rollback=False, lookback_days=30),
    )

    verdict = await gate.run(now=datetime(2026, 6, 24, tzinfo=timezone.utc))

    assert gauge.values == [0.3]
    assert counter.count == 1
    assert [e for e in telemetry.events if e[0] == "curve_regression"], "regression event recorded"
    assert gate_node.rollback_calls == []    # alert-only path
    assert verdict["rollbacks"] == []


@pytest.mark.asyncio
async def test_run_regression_autorollback_reverts_suspect_promotion(tmp_path: Any) -> None:
    """Regressed + auto_rollback ON → only the within-lookback promotion is reverted.

    Uses a REAL PromotionGate with a seeded pointer (recent + old node) so both
    ``active_promotions`` and ``rollback`` are exercised faithfully: the recent
    node is rolled back (dropped from the pointer) and the old node is untouched.
    """
    real_gate = PromotionGate(handlers_dir=tmp_path)
    recent_iso = datetime(2026, 6, 20, tzinfo=timezone.utc).isoformat()  # 4 days ago → within 30
    old_iso = datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat()       # way outside lookback
    pointer = {
        "execute": {  # suspect: recent + has history to revert to
            "active": "execute.abc12345.json",
            "active_sha": "abc12345",
            "suffixes": ["..."],
            "canary_score": 0.9,
            "promoted_at": recent_iso,
            "history": [
                {"sha": "abc12345", "version": "execute.abc12345.json", "canary_score": 0.9, "promoted_at": recent_iso}
            ],
        },
        "plan": {  # NOT a suspect: outside lookback; must survive the run
            "active": "plan.xyz98765.json",
            "active_sha": "xyz98765",
            "suffixes": ["..."],
            "canary_score": 0.8,
            "promoted_at": old_iso,
            "history": [
                {"sha": "xyz98765", "version": "plan.xyz98765.json", "canary_score": 0.8, "promoted_at": old_iso}
            ],
        },
    }
    (real_gate.prompts_dir).mkdir(parents=True, exist_ok=True)
    (real_gate.prompts_dir / "current.json").write_text(json.dumps(pointer), encoding="utf-8")

    telemetry = _TelemetrySink()
    gate = CurveRegressionGate(
        _FakeCurve(_regressed_verdict(current=0.30)),
        real_gate,
        telemetry=telemetry,
        settings=CapabilityCurveSettings(auto_rollback=True, lookback_days=30),
    )

    verdict = await gate.run(now=datetime(2026, 6, 24, tzinfo=timezone.utc))

    # Exactly the recent node was reverted.
    assert len(verdict["rollbacks"]) == 1
    assert verdict["rollbacks"][0]["node"] == "execute"
    assert verdict["rollbacks"][0]["result"]["rolled_back"] is True

    # Rollback telemetry fired for the reverted node only.
    assert [e for e in telemetry.events if e[0] == "curve_regression_rollback"]

    # Pointer state: recent node dropped (single-entry history → node removed);
    # the old, out-of-lookback node is untouched.
    active_nodes = {e["node"] for e in real_gate.active_promotions()}
    assert "execute" not in active_nodes   # reverted
    assert "plan" in active_nodes          # left alone


@pytest.mark.asyncio
async def test_run_regression_no_active_promotion_alerts_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regressed + auto_rollback ON but NO active promotion → drift; alert only, no rollback."""
    gauge, counter = _FakeGauge(), _FakeCounter()
    monkeypatch.setattr(curve_gate, "CAPABILITY_CURVE_SCORE", gauge)
    monkeypatch.setattr(curve_gate, "CAPABILITY_CURVE_REGRESSIONS", counter)

    gate_node = _FakeGate(active=[])  # nothing promotion-side to revert
    telemetry = _TelemetrySink()
    gate = CurveRegressionGate(
        _FakeCurve(_regressed_verdict(current=0.30)),
        gate_node,
        telemetry=telemetry,
        settings=CapabilityCurveSettings(auto_rollback=True, lookback_days=30),
    )

    verdict = await gate.run(now=datetime(2026, 6, 24, tzinfo=timezone.utc))

    assert counter.count == 1
    assert [e for e in telemetry.events if e[0] == "curve_regression"]
    assert gate_node.rollback_calls == []   # no suspect → no rollback call
    assert verdict["rollbacks"] == []


@pytest.mark.asyncio
async def test_run_never_raises_on_rollback_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rollback that raises is swallowed; the failure is recorded, run() never aborts."""
    gauge, counter = _FakeGauge(), _FakeCounter()
    monkeypatch.setattr(curve_gate, "CAPABILITY_CURVE_SCORE", gauge)
    monkeypatch.setattr(curve_gate, "CAPABILITY_CURVE_REGRESSIONS", counter)

    gate_node = _FakeGate(active=[_recent_active()], rollback_raises=True)
    telemetry = _TelemetrySink()
    gate = CurveRegressionGate(
        _FakeCurve(_regressed_verdict(current=0.30)),
        gate_node,
        telemetry=telemetry,
        settings=CapabilityCurveSettings(auto_rollback=True, lookback_days=30),
    )

    verdict = await gate.run(now=datetime(2026, 6, 24, tzinfo=timezone.utc))  # must not raise

    assert counter.count == 1
    assert gate_node.rollback_calls == ["execute"]   # attempt was made
    assert len(verdict["rollbacks"]) == 1
    assert verdict["rollbacks"][0]["node"] == "execute"
    assert "error" in verdict["rollbacks"][0]        # failure captured, not raised
