"""Unit tests for scripts/measure_write_friction.py — the write-hint parser.

Hermetic: feeds synthetic log excerpts to the script's ``_measure`` and
``_group_total`` and asserts the nudge counts and the extra-turns arithmetic
(extra turns = Σ(attempt−1) per nudge). No real logs, no DB, no keys.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_NAME = "measure_write_friction"
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / f"{_SCRIPT_NAME}.py"
_spec = importlib.util.spec_from_file_location(_SCRIPT_NAME, _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
mwf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mwf)  # type: ignore[union-attr]


def _nudge(step: int, attempt: int) -> str:
    return (
        f"_llm_execute:516 | Step {step} expects deliverable at results/x.md "
        f"but no file-output tool was called; nudging (attempt {attempt}/3)"
    )


def _write(path: str) -> str:
    return f"file_writer:file_writer:72 | Writing file: {path} (100 bytes)"


class TestMeasureWriteFriction:
    def test_counts_nudges_and_extra_turns(self, tmp_path: Path) -> None:
        """2× attempt-2 (1 turn each) + 1× attempt-3 (2 turns) = 4 extra turns."""
        log = tmp_path / "n_sample.log"
        log.write_text(
            "\n".join([
                _nudge(1, 2), _nudge(1, 3), _nudge(2, 2),
                _write("results/a.md"), _write("results/b.md"),
            ]),
            encoding="utf-8",
        )
        r = mwf._measure(log)
        assert r["nudges"] == 3
        assert r["attempt3"] == 1
        assert r["extra_turns"] == 4  # (2-1) + (3-1) + (2-1)
        assert r["writes"] == 2

    def test_missing_file_is_zeros(self, tmp_path: Path) -> None:
        r = mwf._measure(tmp_path / "does_not_exist.log")
        assert r == {"nudges": 0, "attempt3": 0, "extra_turns": 0, "writes": 0}

    def test_group_total_sums_across_queries(self) -> None:
        rows = {
            "n3": {"nudges": 3, "attempt3": 1, "extra_turns": 4, "writes": 2},
            "n4": {"nudges": 2, "attempt3": 1, "extra_turns": 3, "writes": 3},
        }
        total = mwf._group_total(rows, ("n3", "n4"))
        assert total == {"nudges": 5, "attempt3": 2, "extra_turns": 7, "writes": 5}
