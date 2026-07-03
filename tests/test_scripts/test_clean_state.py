"""Unit tests for scripts/clean_state.py — the G0 reset instrument.

Hermetic tests of the PURE reset-set logic (select_channels + the canonical
RESET_CHANNELS invariants) and a monkeypatched exercise of the file-clear path.
The DB/reset execution is exercised by the live ``--dry-run`` smoke, not here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the standalone script (lives in scripts/). Register in sys.modules BEFORE
# exec so Python 3.12 ``@dataclass(frozen=True, slots=True)`` resolves __module__.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "clean_state.py"
_spec = importlib.util.spec_from_file_location("clean_state", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
cs = importlib.util.module_from_spec(_spec)
sys.modules["clean_state"] = cs
_spec.loader.exec_module(cs)


# ─── reset-set invariants ────────────────────────────────────────────────────


def test_reset_channel_names_are_unique() -> None:
    names = [c.name for c in cs.RESET_CHANNELS]
    assert len(names) == len(set(names))


def test_canonical_channels_present() -> None:
    """The exact G0 reset set the plan specifies."""
    names = {c.name for c in cs.RESET_CHANNELS}
    assert names == {
        "prompts", "warm", "embeddings", "cold", "tools", "subagents", "redis",
        "results",
    }


def test_all_scope_excludes_results() -> None:
    """'all' resets the 7 leak channels but NOT results (DAG input-pinning)."""
    chans = cs.select_channels("all")
    assert {c.name for c in chans} == set(cs.DEFAULT_SCOPE)
    assert "results" not in {c.name for c in chans}
    assert len(chans) == 7


def test_results_is_opt_in_separate() -> None:
    chans = cs.select_channels("results")
    assert {c.name for c in chans} == {"results"}


def test_all_plus_results_is_full_wipe() -> None:
    chans = cs.select_channels("all,results")
    assert {c.name for c in chans} == {
        "prompts", "warm", "embeddings", "cold", "tools", "subagents", "redis",
        "results",
    }


def test_subset_scope() -> None:
    chans = cs.select_channels("warm,cold")
    assert [c.name for c in chans] == ["warm", "cold"]  # stable RESET_CHANNELS order


def test_scope_order_follows_reset_channels() -> None:
    """Output order is always RESET_CHANNELS order, regardless of input order."""
    chans = cs.select_channels("redis,prompts,warm")
    assert [c.name for c in chans] == ["prompts", "warm", "redis"]


def test_unknown_channel_raises() -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        cs.select_channels("warm,bogus")


def test_empty_scope_raises() -> None:
    with pytest.raises(ValueError, match="empty scope"):
        cs.select_channels(",, ")
    with pytest.raises(ValueError):
        cs.select_channels("")


def test_kinds_are_valid() -> None:
    valid = {"db_delete", "db_update", "file_glob", "file_tree", "redis"}
    for c in cs.RESET_CHANNELS:
        assert c.kind in valid, c


# ─── file-clear path (hermetic via monkeypatch of module paths) ──────────────


def test_prompt_dir_clear_and_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a fake prompts dir with two files + a subdir (subdir left alone).
    fake = tmp_path / "prompts"
    fake.mkdir()
    (fake / "execute.abc123.txt").write_text("evolved prompt", encoding="utf-8")
    (fake / "current.json").write_text("{}", encoding="utf-8")
    (fake / "subdir").mkdir()

    monkeypatch.setattr(cs, "_PROMPT_DIR", fake)
    assert cs._count_prompt_dir() == 2
    n = cs._clear_prompt_dir()
    assert n == 2
    assert cs._count_prompt_dir() == 0
    assert (fake / "subdir").exists()  # dirs untouched


def test_prompt_dir_clear_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_PROMPT_DIR", tmp_path / "nope")
    assert cs._count_prompt_dir() == 0
    assert cs._clear_prompt_dir() == 0  # idempotent no-op


def test_results_subdirs_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "results"
    fake.mkdir()
    (fake / "run-a").mkdir()
    (fake / "run-a").joinpath("out.csv").write_text("x", encoding="utf-8")
    (fake / "orphan.txt").write_text("y", encoding="utf-8")  # file at root, kept

    monkeypatch.setattr(cs, "_RESULTS_DIR", fake)
    assert cs._count_results_subdirs() == 1
    assert cs._clear_results_subdirs() == 1
    assert cs._count_results_subdirs() == 0
    assert (fake / "orphan.txt").exists()  # root files kept
