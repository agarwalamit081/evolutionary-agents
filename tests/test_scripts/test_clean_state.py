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


def test_host_prompt_dir_clear_and_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a fake HOST prompts dir with two files + a subdir (subdir left alone).
    fake = tmp_path / "prompts"
    fake.mkdir()
    (fake / "execute.abc123.txt").write_text("evolved prompt", encoding="utf-8")
    (fake / "current.json").write_text("{}", encoding="utf-8")
    (fake / "subdir").mkdir()

    monkeypatch.setattr(cs, "_PROMPT_DIR", fake)
    assert cs._count_host_prompt_dir() == 2
    n = cs._clear_host_prompt_dir()
    assert n == 2
    assert cs._count_host_prompt_dir() == 0
    assert (fake / "subdir").exists()  # dirs untouched


def test_host_prompt_dir_clear_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_PROMPT_DIR", tmp_path / "nope")
    assert cs._count_host_prompt_dir() == 0
    assert cs._clear_host_prompt_dir() == 0  # idempotent no-op


# ─── worker named-volume clear (the REAL store in docker mode) ───────────────
# The worker writes promotions to the turing-workspace volume at
# EVOLVED_HANDLERS_DIR/prompts — NOT to the host filesystem. These tests mock
# subprocess.run so they NEVER touch a live volume.


class _FakeProc:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_worker_prompts_count_and_clear_when_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **_kw: object) -> _FakeProc:
        seen.append(cmd)
        return _FakeProc(returncode=0, stdout="2\n")

    monkeypatch.setattr(cs.subprocess, "run", fake_run)
    assert cs._count_worker_prompts() == 2
    assert cs._clear_worker_prompts() == 2
    # Every call targets the worker service (not redis / not host).
    assert all("exec" in c and "worker" in c for c in seen)
    # The path is resolved INSIDE the container from its own env, not interpolated host-side.
    assert all("EVOLVED_HANDLERS_DIR" in c[-1] for c in seen)


def test_worker_prompts_unreachable_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host-run mode / stack down → worker exec fails → return 0, do not raise."""
    monkeypatch.setattr(
        cs.subprocess,
        "run",
        lambda _cmd, **_kw: _FakeProc(returncode=1, stdout="", stderr="no such service: worker"),
    )
    assert cs._count_worker_prompts() == 0
    assert cs._clear_worker_prompts() == 0


def test_worker_prompts_garbage_output_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cs.subprocess, "run", lambda _cmd, **_kw: _FakeProc(returncode=0, stdout="not-a-number\n")
    )
    assert cs._count_worker_prompts() == 0


def test_prompt_dir_combined_sums_host_and_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public clear/count funcs clear BOTH stores (host-run + docker volume)."""
    monkeypatch.setattr(cs, "_clear_host_prompt_dir", lambda: 2)
    monkeypatch.setattr(cs, "_clear_worker_prompts", lambda: 3)
    monkeypatch.setattr(cs, "_count_host_prompt_dir", lambda: 2)
    monkeypatch.setattr(cs, "_count_worker_prompts", lambda: 3)
    assert cs._clear_prompt_dir() == 5
    assert cs._count_prompt_dir() == 5


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
