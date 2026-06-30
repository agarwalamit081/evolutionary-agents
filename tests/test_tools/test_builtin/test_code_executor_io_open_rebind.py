"""code_executor ``io.open`` rebind — the CWD-leak fix.

The ``_write_bootstrap`` shim historically rebound ONLY ``builtins.open`` to its
relocating wrapper. But ``pathlib.Path.write_text`` / ``write_bytes`` and a direct
``io.open`` call dispatch through ``io.open`` — a SEPARATE name that was never
rebound — so a generated script doing ``pathlib.Path("v.txt").write_text("x")``
or ``io.open("v.txt", "w")`` wrote straight to the host subprocess cwd (the
project/repo root), polluting the repo (``rel.txt`` / ``shared.txt`` style leaks)
and tripping ``ruff check .``. The fix rebinds ``io.open`` to the SAME wrapper as
``builtins.open`` in every variant, so those bare relative writes relocate into
the results tree like a plain ``open()`` does.

These tests:

- pin that ``io.open`` IS rebound in all three variants (no-root / legacy /
  run-subdir) via a generated-source string assertion;
- run the REAL confinement path end-to-end (a subprocess mirroring
  ``_run_host_subprocess``: shim + vector code, ``cwd = project_root``) and assert
  the previously-leaking vectors — ``pathlib.Path.write_text`` and ``io.open`` —
  now land INSIDE ``results/`` (legacy mode) / ``results/<run_id>/`` (run-subdir
  mode), never in the project root;
- non-regress the ``results/<file>`` flat-write contract (a write already under
  ``results/`` is NOT relocated and NOT double-nested) so the leak fix does not
  break the cross-tool resolution every file-touching tool shares.

Residual (documented, not asserted-confined here): ``pathlib.Path.touch`` /
``os.open`` / shell redirects bypass Python file APIs and resolve to the cwd;
confining those needs ``code_executor_host_cwd="results_subdir"`` + a run-id
(cwd = the results cell) or the docker/runner sandbox — covered by
``TestCodeExecutorHostCwd`` in ``test_builtin_tools.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.tools.builtin.code_executor import _write_bootstrap


def _run_in_subprocess(shim: str, code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Mirror ``_run_host_subprocess``: shim + agent code, run python in ``cwd``.

    The shim's module rebinds are confined to the child process, so (unlike the
    exec-in-namespace path-guard tests) no process-global restore fixture is
    needed here. Returns the completed process so a caller can assert on the
    filesystem side-effects (where the write actually landed).
    """
    script = cwd / "_turing_probe.py"
    script.write_text(shim + "\n" + code, encoding="utf-8")
    try:
        return subprocess.run(
            [sys.executable, str(script)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        script.unlink(missing_ok=True)


# ─── io.open IS rebound in every variant (generated-source assertion) ───────


class TestIoOpenReboundInAllVariants:
    """All three shim variants rebind ``io.open`` to the wrapper."""

    def test_no_root_variant_rebinds_io_open(self) -> None:
        src = _write_bootstrap("", None)
        assert "import io as _turing_io" in src
        assert "_turing_io.open = _turing_open" in src

    def test_legacy_variant_rebinds_io_open(self) -> None:
        src = _write_bootstrap("/tmp/rr", None)
        assert "import io as _turing_io" in src
        assert "_turing_io.open = _turing_open" in src

    def test_run_subdir_variant_rebinds_io_open(self) -> None:
        src = _write_bootstrap("/tmp/rr", "runA")
        assert "import io as _turing_io" in src
        assert "_turing_io.open = _turing_open" in src

    def test_io_open_rebind_after_builtins_open(self) -> None:
        """The rebind must come AFTER ``_turing_open`` is defined + assigned."""
        src = _write_bootstrap("/tmp/rr", None)
        assert src.index("_turing_b.open = _turing_open") < src.index(
            "_turing_io.open = _turing_open"
        )


# ─── legacy mode (run_subdir=None, cwd=project_root) confinement ────────────


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "with open('v.txt', 'w') as f:\n    f.write('X')\n",
            id="builtin_open",
        ),
        pytest.param(
            "import pathlib\npathlib.Path('v.txt').write_text('X')\n",
            id="pathlib_write_text",
        ),
        pytest.param(
            "import pathlib\npathlib.Path('v.txt').write_bytes(b'X')\n",
            id="pathlib_write_bytes",
        ),
        pytest.param(
            "import io\nwith io.open('v.txt', 'w') as f:\n    f.write('X')\n",
            id="io_open",
        ),
    ],
)
def test_bare_python_write_relocates_into_results_not_project_root(
    code: str, tmp_path: Path
) -> None:
    """A bare relative Python write (open / pathlib / io.open) relocates into
    ``results/`` — NOT the project root — in the default legacy mode.

    Before the ``io.open`` rebind, the pathlib + io.open vectors wrote to the cwd
    (repo root). This is the direct regression for the CWD-leak fix.
    """
    results = tmp_path / "results"
    results.mkdir()
    shim = _write_bootstrap(str(results), None)  # legacy variant, cwd=project_root
    proc = _run_in_subprocess(shim, code, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr

    assert (results / "v.txt").read_text() == "X"  # relocated INTO results/
    assert not (tmp_path / "v.txt").exists()  # NOT leaked to the project root


def test_results_prefixed_write_not_double_nested_legacy(tmp_path: Path) -> None:
    """Non-regression: a write already under ``results/`` stays flat in the legacy
    mode — not relocated, not double-nested to ``results/results/<file>``. The leak
    fix (rebinding ``io.open``) must not disturb the cross-tool ``results/<file>``
    resolution contract."""
    results = tmp_path / "results"
    results.mkdir()
    shim = _write_bootstrap(str(results), None)
    code = "with open('results/report.md', 'w') as f:\n    f.write('body')\n"
    proc = _run_in_subprocess(shim, code, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr

    assert (results / "report.md").read_text() == "body"  # flat, contract preserved
    assert not (results / "results" / "report.md").exists()  # no double-nest
    assert not (tmp_path / "report.md").exists()  # no project-root leak


# ─── run-subdir mode (run_subdir set, cwd=project_root) confinement ─────────


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "import pathlib\npathlib.Path('v.txt').write_text('X')\n",
            id="pathlib_write_text",
        ),
        pytest.param(
            "import io\nwith io.open('v.txt', 'w') as f:\n    f.write('X')\n",
            id="io_open",
        ),
    ],
)
def test_bare_python_write_isolates_under_run_cell(
    code: str, tmp_path: Path
) -> None:
    """In run-subdir mode the bare write isolates under ``results/<run_id>/`` —
    the pathlib + io.open vectors now round-trip into the per-run cell like a
    plain ``open()`` instead of leaking to the project root."""
    results = tmp_path / "results"
    results.mkdir()
    shim = _write_bootstrap(str(results), "run42")
    proc = _run_in_subprocess(shim, code, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr

    assert (results / "run42" / "v.txt").read_text() == "X"  # in the run cell
    assert not (results / "v.txt").exists()  # not flat
    assert not (tmp_path / "v.txt").exists()  # not the project root


# ─── import-safety: rebinding io.open must not break module loading ──────────


def test_io_open_rebind_does_not_break_importlib(tmp_path: Path) -> None:
    """``importlib`` uses the C-level ``_io.FileIO``, NOT ``io.open``, so
    rebinding ``io.open`` cannot break module loading. A script that imports a
    stdlib module (exercising the import machinery) AFTER the shim runs must
    succeed — guarding against an accidental regression that wraps the importer."""
    results = tmp_path / "results"
    results.mkdir()
    shim = _write_bootstrap(str(results), None)
    code = (
        "import json, csv, statistics\n"  # stdlib imports exercise importlib
        "import pathlib\n"
        "pathlib.Path('v.txt').write_text(json.dumps({'ok': statistics.mean([1, 2, 3])}))\n"
    )
    proc = _run_in_subprocess(shim, code, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = results / "v.txt"
    assert out.exists(), "pathlib write did not relocate into results/"
    # statistics.mean([1,2,3]) is int 2 on Py3.12 (float on others) — assert
    # structurally; the point is importlib worked + the write landed in results.
    assert '"ok"' in out.read_text()
    assert not (tmp_path / "v.txt").exists()
