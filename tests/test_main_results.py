"""Phase 7 — per-run results subfolders: ``--clean`` + ``--results-dir``.

``_clean_run_results`` is the testable unit behind ``--clean``: it removes ONLY
``<results_root>/<run_id>``, refuses an unsafe/missing run_id, and is a no-op
when the subfolder is already absent. The CLI test wires ``--results-dir`` +
``--clean`` + ``--run-id`` end-to-end (monkeypatching ``_run_agent`` so no live
run fires) and asserts the run subdir is cleared before the run starts.

``get_settings()`` is an ``lru_cache`` singleton, so the ``--results-dir``
override (which mutates ``settings.agent.results_root``) is restored via
``monkeypatch`` to avoid leaking across the suite.
"""

from __future__ import annotations

from pathlib import Path

import click.testing
import pytest

import main as main_mod


def _pin_results_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a teardown restore for the singleton's results_root.

    main() mutates ``settings.agent.results_root`` when ``--results-dir`` is
    passed; registering the current value now means monkeypatch restores it
    after the test even though the value changed mid-run.
    """
    from src.config import get_settings

    agent = get_settings().agent
    monkeypatch.setattr(agent, "results_root", agent.results_root)


class TestCleanRunResults:
    """``_clean_run_results`` targets exactly one run's subfolder, safely."""

    def test_removes_only_the_run_subfolder(self, tmp_path: Path) -> None:
        root = tmp_path / "res"
        (root / "q01").mkdir(parents=True)
        (root / "q01" / "x.md").write_text("d")
        (root / "other.md").write_text("keep")  # sibling must survive

        main_mod._clean_run_results(str(root), "q01")

        assert not (root / "q01").exists()
        assert (root / "other.md").exists()
        assert root.exists()  # results_root itself never removed

    def test_refuses_missing_run_id(self) -> None:
        with pytest.raises(SystemExit):
            main_mod._clean_run_results("results", None)

    @pytest.mark.parametrize("bad", ["..", ".", "../etc", "a/b", "a\\b", ""])
    def test_refuses_unsafe_run_id(self, tmp_path: Path, bad: str) -> None:
        with pytest.raises(SystemExit):
            main_mod._clean_run_results(str(tmp_path), bad)

    def test_noop_when_subfolder_absent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main_mod._clean_run_results(str(tmp_path / "res"), "q01")
        assert "no prior subfolder" in capsys.readouterr().out


class TestCleanResultsCli:
    """``--results-dir`` + ``--clean`` + ``--run-id`` wire through the CLI."""

    def test_clean_clears_run_subfolder_then_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _pin_results_root(monkeypatch)
        monkeypatch.setattr(
            main_mod,
            "_run_agent",
            _noop_run_agent,
        )
        run_root = tmp_path / "res"
        (run_root / "q01").mkdir(parents=True)
        (run_root / "q01" / "stale.md").write_text("d")

        runner = click.testing.CliRunner()
        result = runner.invoke(
            main_mod.main,
            ["--results-dir", str(run_root), "--clean", "--run-id", "q01", "--goal", "x"],
        )

        assert result.exit_code == 0, result.output
        assert "Cleared prior results subfolder" in result.output
        assert not (run_root / "q01").exists()  # cleared before the run
        assert run_root.exists()  # results_root untouched

    def test_clean_without_run_id_exits_clean_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _pin_results_root(monkeypatch)
        runner = click.testing.CliRunner()
        result = runner.invoke(
            main_mod.main,
            ["--results-dir", str(tmp_path / "res"), "--clean", "--goal", "x"],
        )
        assert result.exit_code == 1
        assert "--clean requires --run-id" in result.output


async def _noop_run_agent(*_args: object, **_kwargs: object) -> dict[str, object]:
    """Stand-in for _run_agent so the CLI test never fires a live run."""
    return {"final_output": "ok", "iteration_count": 1, "is_complete": True}
