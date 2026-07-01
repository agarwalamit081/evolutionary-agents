"""``--capability-curve`` CLI: window parsing + the read-only trend/verdict handler.

Pins:
* ``_parse_window`` — date-only → UTC midnight; ``--until`` date-only → end-of-day
  (inclusive); ISO datetime passthrough; ``None``; bad input → ``BadParameter``.
* ``_run_capability_curve`` — prints the battery table + per-goal table + verdict
  (OK / REGRESSED / INCONCLUSIVE); ``--export`` dispatches by suffix (.json/.csv)
  and rejects a .png; ``--plot`` renders a PNG.

The curve is faked by subclassing the real ``CapabilityCurve`` and overriding
only ``snapshot`` (deterministic bundle, no DB) — the real ``export_json`` /
``export_csv`` / ``plot_png`` are exercised against that snapshot, so the export
round-trips are real.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import click
import pytest

import main as main_mod
import src.eval.curve as curve_mod
from src.eval.curve import CapabilityCurve


def _sample_snapshot(*, regressed: bool = False, inconclusive: bool = False) -> dict:
    """Deterministic snapshot matching the real CapabilityCurve.snapshot shape."""
    current = None if inconclusive else (0.30 if regressed else 0.90)
    return {
        "battery_trend": [
            {"date": "2026-06-20", "mean_score": 0.85, "n_goals": 9},
            {"date": "2026-06-21", "mean_score": 0.30 if regressed else 0.90, "n_goals": 9},
        ],
        "latest_per_goal": [
            {
                "goal_id": "battery04_q01",
                "date": "2026-06-21",
                "mean_score": 0.95,
                "n_checks": 4,
            }
        ],
        "verdict": {
            "regressed": regressed,
            "inconclusive": inconclusive,
            "current": current,
            "best_prior": None if inconclusive else 0.85,
            "delta": None if inconclusive else (0.55 if regressed else 0.05),
            "n_points": 1 if inconclusive else 2,
            "floor": 0.5,
            "delta_floor": 0.1,
        },
    }


class _FakeCurve(CapabilityCurve):
    """Real export methods; ``snapshot`` returns a deterministic bundle (no DB)."""

    async def snapshot(self, *, since=None, until=None, model=None):  # noqa: ANN001 — matches parent signature
        return _sample_snapshot()


def _patch_curve(monkeypatch: pytest.MonkeyPatch, curve_cls: type) -> None:
    """Point the handler's lazy ``from src.eval.curve import CapabilityCurve`` at ``curve_cls``."""
    monkeypatch.setattr(curve_mod, "CapabilityCurve", curve_cls)


class TestParseWindow:
    def test_none_returns_none(self) -> None:
        assert main_mod._parse_window(None) is None

    def test_date_only_is_utc_midnight(self) -> None:
        assert main_mod._parse_window("2026-06-01") == datetime(2026, 6, 1, tzinfo=timezone.utc)

    def test_until_date_only_widens_to_end_of_day(self) -> None:
        dt = main_mod._parse_window("2026-06-01", end_of_day=True)
        assert dt == datetime(2026, 6, 1, 23, 59, 59, 999999, tzinfo=timezone.utc)

    def test_iso_datetime_passthrough(self) -> None:
        assert main_mod._parse_window("2026-06-01T12:00:00Z") == datetime(
            2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc
        )

    def test_bad_input_raises_bad_parameter(self) -> None:
        with pytest.raises(click.BadParameter):
            main_mod._parse_window("not-a-date")


class TestRunCapabilityCurve:
    @pytest.mark.asyncio
    async def test_prints_battery_table_per_goal_and_ok_verdict(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _patch_curve(monkeypatch, _FakeCurve)
        await main_mod._run_capability_curve(since=None, until=None, export=None, plot=None)
        out = capsys.readouterr().out
        assert "Capability curve" in out
        assert "2026-06-21" in out            # battery table row
        assert "battery04_q01" in out         # per-goal row
        assert "Regression verdict: OK" in out

    @pytest.mark.asyncio
    async def test_regressed_verdict_marked(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class _RegCurve(_FakeCurve):
            async def snapshot(self, *, since=None, until=None, model=None):  # noqa: ANN001
                return _sample_snapshot(regressed=True)

        _patch_curve(monkeypatch, _RegCurve)
        await main_mod._run_capability_curve(since=None, until=None, export=None, plot=None)
        assert "REGRESSED" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_inconclusive_verdict_shows_min_points(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class _IncCurve(_FakeCurve):
            async def snapshot(self, *, since=None, until=None, model=None):  # noqa: ANN001
                return _sample_snapshot(inconclusive=True)

        _patch_curve(monkeypatch, _IncCurve)
        await main_mod._run_capability_curve(since=None, until=None, export=None, plot=None)
        assert "INCONCLUSIVE" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_export_json_roundtrip(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: object
    ) -> None:
        from pathlib import Path

        _patch_curve(monkeypatch, _FakeCurve)
        out_path = Path(str(tmp_path)) / "curve.json"
        await main_mod._run_capability_curve(since=None, until=None, export=str(out_path), plot=None)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "battery_trend" in data and "verdict" in data
        assert "Exported JSON" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_export_csv_writes_header_and_rows(self, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
        from pathlib import Path

        _patch_curve(monkeypatch, _FakeCurve)
        out_path = Path(str(tmp_path)) / "curve.csv"
        await main_mod._run_capability_curve(since=None, until=None, export=str(out_path), plot=None)
        text = out_path.read_text(encoding="utf-8")
        assert "date,mean_score,n_goals" in text
        assert "2026-06-21" in text

    @pytest.mark.asyncio
    async def test_export_rejects_png_suffix(self, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
        from pathlib import Path

        _patch_curve(monkeypatch, _FakeCurve)
        with pytest.raises(click.BadParameter):
            await main_mod._run_capability_curve(
                since=None, until=None, export=str(Path(str(tmp_path)) / "x.png"), plot=None
            )

    @pytest.mark.asyncio
    async def test_plot_renders_png(self, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
        from pathlib import Path

        _patch_curve(monkeypatch, _FakeCurve)
        out_path = Path(str(tmp_path)) / "curve.png"
        await main_mod._run_capability_curve(since=None, until=None, export=None, plot=str(out_path))
        assert out_path.exists()  # real CapabilityCurve.plot_png renders (matplotlib pinned)
