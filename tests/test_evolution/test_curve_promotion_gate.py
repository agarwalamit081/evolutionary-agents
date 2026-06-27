"""Tests for the Phase-2 C1 curve regression-guard on PROMPT promotion.

Locks the ``run_cycle`` Phase-8 behavior: a *regressed* capability-curve verdict
skips ``promotion_gate.promote``; ``inconclusive`` / guard-off / curve-error do
not. All four cases drive the REAL ``run_cycle`` — the analyze→generate phases
run for real (heuristic; no LLM) and the validate/sandbox/ab/deploy phases are
mocked so the cycle reaches ``effective_deploy`` — so the guard WIRING (not just
the helper) is exercised. No live LLM / DB (the curve read is monkeypatched).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.evolution.engine import SelfEvolutionEngine
from src.graph.enums import MutationType


def _settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    curve_clear: bool,
    promote_on: bool = True,
) -> None:
    """Point the engine's lazy ``from src.config import get_settings`` at a fake.

    The engine resolves ``get_settings`` via the ``src.config`` package
    re-export (``src/config/__init__.py``), so patch THAT binding.
    """
    fake = SimpleNamespace(
        evolution=SimpleNamespace(
            evolution_promote_to_live=promote_on,
            evolution_require_curve_clear=curve_clear,
        ),
    )
    monkeypatch.setattr("src.config.get_settings", lambda: fake)


def _engine_at_promotion_phase() -> tuple[SelfEvolutionEngine, MagicMock]:
    """An engine whose middle phases are mocked so run_cycle reaches deploy.

    analyze→generate run for real (heuristic PROMPT proposal from a failure
    pattern, no gateway); validate/sandbox/ab/deploy return passing results so
    ``effective_deploy`` is True and Phase 8 (promotion) is reached. The PROMPT
    mutation_type means the G1 invariant verify + post-deploy sandbox smoke are
    skipped (non-executable), so nothing else can short-circuit before Phase 8.
    """
    engine = SelfEvolutionEngine()  # no gateway → heuristic; no persister
    engine.validate = AsyncMock(return_value={"passed": True})
    engine.sandbox_test = AsyncMock(return_value={"passed": True})
    engine.ab_test = AsyncMock(return_value={"is_significant": True})
    engine.deploy = AsyncMock(
        return_value={
            "deployed": True,
            "pre_deploy_hash": None,
            "commit_hash": None,
            "mutation_type": MutationType.PROMPT,
            "description": "prompt fix",
            "target_path": None,
            "rationale": "",
            "ab_result": None,
        }
    )
    gate = MagicMock()
    gate.promote = AsyncMock(return_value={"promoted": True, "reason": "canary passed"})
    return engine, gate


class TestCurvePromotionGuard:
    async def test_regressed_blocks_promotion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _settings(monkeypatch, curve_clear=True)
        engine, gate = _engine_at_promotion_phase()
        engine._curve_verdict = AsyncMock(
            return_value={
                "regressed": True,
                "inconclusive": False,
                "current": 0.4,
                "delta": 0.2,
                "n_points": 5,
            }
        )

        result = await engine.run_cycle(
            [], failure_patterns=["bad json"], promotion_gate=gate
        )

        assert gate.promote.await_count == 0  # promotion skipped
        assert result["promotion"]["promoted"] is False
        assert "curve guard" in result["promotion"]["reason"]

    async def test_inconclusive_allows_promotion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _settings(monkeypatch, curve_clear=True)
        engine, gate = _engine_at_promotion_phase()
        engine._curve_verdict = AsyncMock(
            return_value={
                "regressed": False,
                "inconclusive": True,
                "current": 0.4,
                "best_prior": None,
                "delta": None,
                "n_points": 1,
            }
        )

        result = await engine.run_cycle(
            [], failure_patterns=["bad json"], promotion_gate=gate
        )

        assert gate.promote.await_count == 1  # cold-start promotion proceeds
        assert result["promotion"]["promoted"] is True

    async def test_guard_off_is_byte_identical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _settings(monkeypatch, curve_clear=False)
        engine, gate = _engine_at_promotion_phase()
        # A regressed verdict that MUST be ignored because the guard is off.
        engine._curve_verdict = AsyncMock(return_value={"regressed": True})

        result = await engine.run_cycle(
            [], failure_patterns=["bad json"], promotion_gate=gate
        )

        assert gate.promote.await_count == 1
        assert engine._curve_verdict.await_count == 0  # curve never consulted
        assert result["promotion"]["promoted"] is True

    async def test_fail_open_on_curve_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _settings(monkeypatch, curve_clear=True)
        engine, gate = _engine_at_promotion_phase()
        engine._curve_verdict = AsyncMock(side_effect=RuntimeError("eval DB down"))

        result = await engine.run_cycle(
            [], failure_patterns=["bad json"], promotion_gate=gate
        )

        # Curve read failed → fail-open → the GoldenCanary (gate) is authoritative.
        assert gate.promote.await_count == 1
        assert result["promotion"]["promoted"] is True
