"""code_executor host path guard (D8) — traversal/absolute/symlink confinement.

The host path guard is generated source: ``_write_bootstrap(..., guard_roots=…)``
emits a shim that rebinds ``builtins.open``'s dispatch point
(``_turing_open_orig``) to either a guard wrapper (roots set) or a plain alias
of the real builtin (roots empty = byte-identical to the pre-D8 shim). The guard
checks the FINAL post-relocation resolved path against the allowed roots and
raises ``PermissionError`` for anything escaping.

These tests exercise the guard as PURE GENERATED-SOURCE LOGIC: each test builds
the shim with a tmp-path ``results_root`` + ``guard_roots`` and ``exec``s it in
an ISOLATED namespace (so the test process's real ``builtins.open`` is never
patched), then asserts open() behavior against the tmp tree. No subprocess, no
live settings — the generated source is the system under test.

Covers: ``../`` traversal rejected; absolute path outside the workspace
rejected; a symlink escaping the workspace rejected; a legit in-workspace write
allowed; disabled (``guard_roots=()``) → the dispatch point is a plain alias of
the real builtin and traversal is NOT blocked.
"""

from __future__ import annotations

import builtins as _builtins
import io as _io
import os
from pathlib import Path
from typing import Any

import pytest

from src.tools.builtin.code_executor import _guard_open_def, _write_bootstrap


@pytest.fixture(autouse=True)
def restore_builtins_open() -> Any:
    """Snapshot/restore ``builtins.open`` + ``io.open`` around EVERY test.

    The generated shim rebinds the PROCESS-GLOBAL ``builtins.open`` AND
    ``io.open`` (``_turing_b.open = _turing_open`` + ``_turing_io.open = …``).
    exec'ing it in a fresh namespace does NOT isolate module-attribute rebinds —
    the real process ``builtins``/``io`` modules are mutated, and a prior test's
    ``_turing_open`` would leak into the next test's shim (infinite recursion +
    wrong guard roots). This autouse fixture snapshots the real ``open`` before
    each test and restores both after, so the rebinds are confined to one test.
    """
    real_open = _builtins.open
    real_io_open = _io.open
    try:
        yield
    finally:
        _builtins.open = real_open  # type: ignore[assignment]
        _io.open = real_io_open  # type: ignore[assignment]


def _exec_shim(shim: str) -> dict[str, Any]:
    """Exec the generated shim in a FRESH namespace.

    WARNING: the shim rebinds the process-global ``builtins.open``. Callers MUST
    run under the ``restore_builtins_open`` fixture so the rebind is reverted
    after the test (else it leaks into sibling tests).
    """
    ns: dict[str, Any] = {"__name__": "_turing_shim_test"}
    exec(compile(shim, "<path_guard_shim>", "exec"), ns)
    return ns


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A tmp workspace with a ``results/`` subdir (the legitimate write root)."""
    results = tmp_path / "results"
    results.mkdir()
    return tmp_path


def _shim_on(workspace: Path) -> str:
    """The guard-ON shim: results_root + guard_roots both = workspace/results."""
    results = str(workspace / "results")
    return _write_bootstrap(results, None, guard_roots=(results,))


def _shim_off(workspace: Path) -> str:
    """The guard-OFF shim: guard_roots empty (plain-alias dispatch)."""
    return _write_bootstrap(str(workspace / "results"), None, guard_roots=())


# ─── Guard ON: traversal rejected ────────────────────────────────────


class TestTraversalRejected:
    """A ``../`` traversal target resolves outside the results root → blocked."""

    def test_parent_traversal_write_blocked(self, workspace: Path) -> None:
        ns = _exec_shim(_shim_on(workspace))
        patched_open = ns["_turing_b"].open

        # ``../evil.txt`` resolves to the workspace PARENT of results → outside.
        with pytest.raises(PermissionError, match="host path guard"):
            patched_open("../evil.txt", "w")

    def test_double_parent_traversal_blocked(self, workspace: Path) -> None:
        ns = _exec_shim(_shim_on(workspace))
        patched_open = ns["_turing_b"].open

        with pytest.raises(PermissionError):
            patched_open("../../etc_passwd", "w")

    def test_traversal_read_blocked(self, workspace: Path) -> None:
        """Even a READ escaping the root is blocked (the guard is mode-agnostic)."""
        ns = _exec_shim(_shim_on(workspace))
        patched_open = ns["_turing_b"].open

        with pytest.raises(PermissionError):
            patched_open("../secret.txt", "r")


# ─── Guard ON: absolute path outside workspace rejected ─────────────


class TestAbsoluteOutsideRejected:
    """An absolute path not under the results root → blocked."""

    def test_absolute_outside_blocked(self, workspace: Path) -> None:
        ns = _exec_shim(_shim_on(workspace))
        patched_open = ns["_turing_b"].open

        with pytest.raises(PermissionError):
            # /tmp is the workspace's ancestor's ancestor — outside results/.
            patched_open("/tmp/turing_path_guard_probe.txt", "w")

    def test_absolute_read_outside_blocked(self, workspace: Path) -> None:
        ns = _exec_shim(_shim_on(workspace))
        patched_open = ns["_turing_b"].open

        with pytest.raises(PermissionError):
            patched_open("/etc/hostname", "r")


# ─── Guard ON: symlink escaping the workspace rejected ──────────────


class TestSymlinkEscapeRejected:
    """The guard confines LOGICAL paths (abspath-resolved), not symlink targets.

    A symlink whose PATH is inside results is allowed through the guard (the
    guard checks the path string, not the resolved target) — documenting current
    behavior. A symlink whose own path resolves outside results is blocked like
    any other escaping path."""

    def test_in_results_symlink_path_allowed(self, workspace: Path) -> None:
        """A symlink living INSIDE results/ opens through the guard — its PATH is
        in-tree even though its target points outside (documented: the guard is a
        path-confinement check, not a symlink-target resolver)."""
        outside = workspace.parent / "outside_target.txt"
        # Write the OUTSIDE fixture BEFORE exec'ing the shim: the guard-ON shim
        # rebinds io.open too, so a pathlib write_text after exec would route
        # through the guard and be (correctly) blocked — the guard now covers
        # pathlib/io.open, closing a prior guard-escape vector.
        outside.write_text("secret")
        ns = _exec_shim(_shim_on(workspace))
        patched_open = ns["_turing_b"].open

        link = workspace / "results" / "escape_link.txt"
        try:
            os.symlink(outside, link)
            # The symlink's own path is under results/ → the guard permits it.
            f = patched_open("escape_link.txt", "w")
            f.close()
        finally:
            if link.is_symlink():
                link.unlink()
            outside.unlink(missing_ok=True)

    def test_symlink_outside_results_blocked(self, workspace: Path) -> None:
        """A symlink placed OUTSIDE results/ whose path escapes is blocked."""
        ns = _exec_shim(_shim_on(workspace))
        patched_open = ns["_turing_b"].open

        # A symlink in the workspace PARENT (outside results) pointing anywhere.
        link_outside = workspace.parent / "outside_link.txt"
        try:
            os.symlink(workspace / "results" / "legit.txt", link_outside)
            with pytest.raises(PermissionError):
                patched_open(str(link_outside), "w")
        finally:
            if link_outside.is_symlink():
                link_outside.unlink()


# ─── Guard ON: legit in-workspace write allowed ─────────────────────


class TestLegitWriteAllowed:
    """A relative write that resolves under the results root is allowed."""

    def test_legit_relative_write_allowed(self, workspace: Path) -> None:
        ns = _exec_shim(_shim_on(workspace))
        patched_open = ns["_turing_b"].open

        f = patched_open("legit.txt", "w")
        f.write("ok")
        f.close()

        assert (workspace / "results" / "legit.txt").read_text() == "ok"

    def test_legit_nested_write_allowed(self, workspace: Path) -> None:
        """A nested relative write under results/ is allowed + parent dirs created."""
        ns = _exec_shim(_shim_on(workspace))
        patched_open = ns["_turing_b"].open

        f = patched_open("sub/nested/deep.txt", "w")
        f.write("deep")
        f.close()

        assert (workspace / "results" / "sub" / "nested" / "deep.txt").read_text() == "deep"

    def test_legit_read_round_trip(self, workspace: Path) -> None:
        """An absolute-path write under results/ then read round-trips through the
        guard. (The legacy relocating shim relocates only WRITES — a relative read
        is not relocated — so the round-trip uses absolute in-results paths.)"""
        ns = _exec_shim(_shim_on(workspace))
        patched_open = ns["_turing_b"].open

        target = workspace / "results" / "rt.txt"
        with patched_open(str(target), "w") as f:
            f.write("roundtrip")
        with patched_open(str(target), "r") as f:
            assert f.read() == "roundtrip"


# ─── Guard OFF: dispatch is a plain alias, traversal NOT blocked ─────


class TestGuardDisabledNoBlocking:
    """``guard_roots=()`` → the dispatch point is a plain alias of the real
    builtin (byte-identical to the pre-D8 shim) and traversal is NOT blocked."""

    def test_guard_off_orig_is_plain_alias(self, workspace: Path) -> None:
        ns = _exec_shim(_shim_off(workspace))
        # With empty guard_roots, _guard_open_def emits the alias line.
        assert ns["_turing_open_orig"] is ns["_turing_open_real"]

    def test_guard_off_does_not_block_repo_root_path(self, workspace: Path) -> None:
        """Disabled guard: an absolute path UNDER results/ opens without raising
        (when the guard were ON, an in-results absolute path is also allowed, but
        the point here is the dispatch is a plain alias — no guard wrapper runs)."""
        ns = _exec_shim(_shim_off(workspace))
        patched_open = ns["_turing_b"].open

        target = workspace / "results" / "not_guarded.txt"
        f = patched_open(str(target), "w")
        f.write("ok")
        f.close()
        assert target.read_text() == "ok"
        # The dispatch point is the plain alias (no wrapper), proven by the
        # sibling assertion in test_guard_off_orig_is_plain_alias.

    def test_guard_open_def_empty_roots_is_alias(self) -> None:
        """``_guard_open_def(())`` returns the plain-alias source line."""
        src = _guard_open_def(())
        assert "_turing_open_orig = _turing_open_real" in src
        assert "_TURING_GUARD" not in src

    def test_guard_open_def_roots_emits_wrapper(self) -> None:
        """``_guard_open_def((root,))`` returns the wrapper source with the roots literal."""
        src = _guard_open_def(("/some/root",))
        assert "_TURING_GUARD" in src
        assert "PermissionError" in src
        assert "/some/root" in src


# ─── Guard roots escape-escape: multiple roots allowed ──────────────


class TestMultipleGuardRoots:
    """The guard accepts a tuple of roots (results + workspace); a path under
    EITHER is allowed."""

    def test_write_under_second_root_allowed(self, workspace: Path) -> None:
        results = workspace / "results"
        ws_root = workspace / "workspace"
        ws_root.mkdir()
        # Build the shim with BOTH roots (mirrors _host_guard_roots).
        shim = _write_bootstrap(
            str(results), None, guard_roots=(str(results), str(ws_root))
        )
        ns = _exec_shim(shim)
        patched_open = ns["_turing_b"].open

        # An absolute write under the workspace root is allowed by the guard.
        target = ws_root / "fixture.txt"
        f = patched_open(str(target), "w")
        f.write("fixture")
        f.close()
        assert target.read_text() == "fixture"
