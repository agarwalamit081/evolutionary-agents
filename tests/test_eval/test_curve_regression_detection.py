"""Curve-regression detection (Phase 2 C1) — ``CAPABILITY_CURVE_GATE_ENABLED`` parity.

The existing ``test_curve.py`` locks the pure ``CapabilityCurve`` verdict logic.
This companion file locks the ``CurveRegressionGate`` that CONSUMES that verdict:
(a) a current-below-floor + delta-above-threshold verdict yields REGRESSED and
increments the regression counter + emits telemetry; (b) a passing run yields OK
(no rollback, no telemetry); (c) auto-rollback fires on REGRESSED only when the
opt-in ``auto_rollback`` is on AND a suspect active promotion is within the
lookback window; (d) a regression with no suspect = model drift (alert only);
(e) ``run`` never raises on a rollback failure. Deterministic: the curve and the
``PromotionGate`` are faked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock


from src.config.settings import CapabilityCurveSettings
from src.evolution.curve_gate import CurveRegressionGate, _parse_iso


# ─── fakes ──────────────────────────────────────────────────────────────────


class _FakeCurve:
    def __init__(self, verdict: dict[str, Any]) -> None:
        self._v = verdict

    async def detect_regression(self) -> dict[str, Any]:
        return dict(self._v)


class _FakeGate:
    """PromotionGate-shaped: serves active_promotions + rollback capture."""

    def __init__(self, active: list[dict[str, Any]]) -> None:
        self._active = active
        self.rolled_back: list[str] = []
        self.rollback_exc: Exception | None = None

    def active_promotions(self) -> list[dict[str, Any]]:
        return list(self._active)

    def rollback(self, node: str) -> dict[str, Any]:
        if self.rollback_exc is not None:
            raise self.rollback_exc
        self.rolled_back.append(node)
        return {"rolled_back": True, "node": node}


def _settings(**kw: Any) -> CapabilityCurveSettings:
    base = dict(auto_rollback=True, lookback_days=30)
    base.update(kw)
    return CapabilityCurveSettings(**base)  # type: ignore[arg-type]


def _verdict(regressed: bool, **extra: Any) -> dict[str, Any]:
    base = {
        "regressed": regressed, "inconclusive": False,
        "current": 0.4, "best_prior": 0.9, "delta": 0.5, "n_points": 3,
        "floor": 0.5, "delta_floor": 0.1,
    }
    base.update(extra)
    return base


def _suspect(days_ago: int, node: str = "execute") -> dict[str, Any]:
    return {
        "node": node, "active": f"{node}.abc123.txt",
        "promoted_at": (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(),
        "canary_score": 0.9,
    }


# ─── verdict consumption ────────────────────────────────────────────────────


class TestVerdictConsumption:
    async def test_should_emit_telemetry_and_count_on_regressed(self) -> None:
        gate = _FakeGate(active=[_suspect(days_ago=1)])
        events: list[tuple[str, dict[str, Any]]] = []

        async def _tel(event: str, data: dict[str, Any]) -> None:
            events.append((event, data))

        cg = CurveRegressionGate(
            _FakeCurve(_verdict(True)), gate, telemetry=_tel, settings=_settings()
        )
        out = await cg.run()
        assert out["regressed"] is True
        # The regression telemetry event fired.
        types = [e for e, _ in events]
        assert "curve_regression" in types

    async def test_should_skip_telemetry_and_rollback_on_ok_run(self) -> None:
        gate = _FakeGate(active=[_suspect(days_ago=1)])
        events: list[tuple[str, dict[str, Any]]] = []

        async def _tel(event: str, data: dict[str, Any]) -> None:
            events.append((event, data))

        cg = CurveRegressionGate(
            _FakeCurve(_verdict(False)), gate, telemetry=_tel, settings=_settings()
        )
        out = await cg.run()
        assert out["regressed"] is False
        assert out["rollbacks"] == []
        assert events == []  # no regression → no telemetry
        assert gate.rolled_back == []


# ─── auto-rollback trigger ──────────────────────────────────────────────────


class TestAutoRollbackTrigger:
    async def test_should_rollback_suspect_when_auto_rollback_on_and_within_lookback(self) -> None:
        gate = _FakeGate(active=[_suspect(days_ago=5, node="execute")])
        cg = CurveRegressionGate(
            _FakeCurve(_verdict(True)), gate,
            telemetry=AsyncMock(), settings=_settings(auto_rollback=True),
        )
        out = await cg.run()
        assert gate.rolled_back == ["execute"]
        assert out["rollbacks"] == [{"node": "execute", "result": {"rolled_back": True, "node": "execute"}}]

    async def test_should_not_rollback_when_auto_rollback_off(self) -> None:
        # Default-off auto_rollback: detect + log only, never revert.
        gate = _FakeGate(active=[_suspect(days_ago=1)])
        cg = CurveRegressionGate(
            _FakeCurve(_verdict(True)), gate,
            telemetry=AsyncMock(), settings=_settings(auto_rollback=False),
        )
        out = await cg.run()
        assert out["regressed"] is True
        assert out["rollbacks"] == []
        assert gate.rolled_back == []

    async def test_should_skip_suspect_outside_lookback_window(self) -> None:
        # A promotion older than lookback_days is not a suspect → no rollback.
        gate = _FakeGate(active=[_suspect(days_ago=60)])  # > 30d lookback
        cg = CurveRegressionGate(
            _FakeCurve(_verdict(True)), gate,
            telemetry=AsyncMock(), settings=_settings(auto_rollback=True, lookback_days=30),
        )
        out = await cg.run()
        assert out["rollbacks"] == []
        assert gate.rolled_back == []

    async def test_regression_with_no_suspect_is_alert_only(self) -> None:
        # A regression with NO active promotion within the window is model/provider
        # drift, not a mutation → alert only (no rollback, no error).
        gate = _FakeGate(active=[])
        cg = CurveRegressionGate(
            _FakeCurve(_verdict(True)), gate,
            telemetry=AsyncMock(), settings=_settings(auto_rollback=True),
        )
        out = await cg.run()
        assert out["regressed"] is True
        assert out["rollbacks"] == []

    async def test_should_never_raise_on_rollback_failure(self) -> None:
        # A rollback failure is swallowed (observability-only) — recorded as an
        # error entry, never re-raised.
        gate = _FakeGate(active=[_suspect(days_ago=1, node="plan")])
        gate.rollback_exc = RuntimeError("pointer gone")
        cg = CurveRegressionGate(
            _FakeCurve(_verdict(True)), gate,
            telemetry=AsyncMock(), settings=_settings(auto_rollback=True),
        )
        out = await cg.run()
        assert len(out["rollbacks"]) == 1
        assert "error" in out["rollbacks"][0]
        assert out["rollbacks"][0]["node"] == "plan"


# ─── helpers ────────────────────────────────────────────────────────────────


class TestParseIsoHelper:
    def test_should_parse_aware_iso(self) -> None:
        dt = _parse_iso("2026-06-01T00:00:00+00:00")
        assert dt is not None and dt.tzinfo is not None

    def test_should_treat_naive_as_utc(self) -> None:
        dt = _parse_iso("2026-06-01T00:00:00")
        assert dt is not None and dt.tzinfo is not None

    def test_should_parse_trailing_z(self) -> None:
        dt = _parse_iso("2026-06-01T00:00:00Z")
        assert dt is not None

    def test_should_return_none_on_garbage(self) -> None:
        assert _parse_iso("not a date") is None
        assert _parse_iso("") is None
        assert _parse_iso(None) is None  # type: ignore[arg-type]


# ─── opt-in gate ────────────────────────────────────────────────────────────


class TestGateOptIn:
    def test_gate_settings_default_off(self) -> None:
        s = CapabilityCurveSettings()
        assert s.gate_enabled is False
        assert s.auto_rollback is False
