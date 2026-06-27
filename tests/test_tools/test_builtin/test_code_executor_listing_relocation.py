"""code_executor listing relocation — glob.glob/iglob + os.listdir/scandir now
resolve a results-prefixed target through the SAME subdir-first + flat-fallback
relocation as the open() read-branch.

The run-subdir bootstrap patched ``builtins.open`` only, so an in-code_executor
script doing ``glob.glob('results/*.csv')`` / ``os.listdir('results')`` scanned
the FLAT ``results/`` root (which holds per-run SUBDIRS, not files) and found
nothing — even though the agent is *told* (code_executor docstring) to read
deliverables that way. These tests lock the listing relocation added in
``_listing_relocation_shim`` (run-subdir variant only):

- the agent's own write→list round-trip finds its file under ``results/<run_id>/``
  via ``glob.glob`` / ``os.listdir`` / ``os.scandir``;
- a legacy FLAT file (no subdir) still recalls via the flat fallback;
- a bare (non-results-prefixed) pattern stays on the real CWD (never relocated);
- a ``root_dir=`` glob passes through unchanged (caller already scoped the dir);
- the legacy (``run_subdir=None``) + degenerate (``results_root_abs=""``)
  variants are byte-identical — no listing rebinds.

Pure generated-source logic: each test builds the shim with a tmp ``results_root``
+ ``run_subdir`` and ``exec``s it in an isolated namespace, then asserts listing
behavior against the tmp tree. No subprocess, no live settings.
"""

from __future__ import annotations

import builtins as _builtins
import glob as _glob
import os
from pathlib import Path
from typing import Any

import pytest

from src.tools.builtin.code_executor import _write_bootstrap


@pytest.fixture(autouse=True)
def restore_listings() -> Any:
    """Snapshot/restore builtins.open + os.listdir/scandir + glob.glob/iglob.

    The run-subdir shim rebinds all five on the REAL process modules (builtins,
    os, glob). exec in a fresh namespace does NOT isolate module-attribute
    rebinds — they are process-global — so each test MUST restore them or the
    patched versions leak into sibling tests (incl. the path-guard module).
    """
    real_open = _builtins.open
    real_listdir = os.listdir
    real_scandir = os.scandir
    real_glob = _glob.glob
    real_iglob = _glob.iglob
    try:
        yield
    finally:
        _builtins.open = real_open
        os.listdir = real_listdir  # type: ignore[assignment]
        os.scandir = real_scandir  # type: ignore[assignment]
        _glob.glob = real_glob  # type: ignore[assignment]
        _glob.iglob = real_iglob  # type: ignore[assignment]


def _exec_subdir_shim(results_root: Path, run_subdir: str) -> dict[str, Any]:
    """Exec the run-subdir bootstrap (open + listing relocation) in a fresh ns.

    Must run under ``restore_listings`` so the module rebinds are reverted after.
    """
    ns: dict[str, Any] = {"__name__": "_turing_listing_test"}
    shim = _write_bootstrap(str(results_root), run_subdir)
    exec(compile(shim, "<listing_shim>", "exec"), ns)
    return ns


def _run(ns: dict[str, Any], code: str) -> None:
    """Exec a snippet of "agent" code in the shim namespace.

    The snippet uses bare ``open``/``glob.glob``/``os.listdir`` — exactly the
    shapes an LLM-generated code_executor script emits — which resolve to the
    REBOUND process-global functions (the agent never sees the ``_turing_``
    internals).
    """
    exec(compile(code, "<agent>", "exec"), ns)


# ─── core: write→list round-trip finds the file under results/<run_id>/ ───


def test_glob_and_listdir_find_self_written_file_under_subdir(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()  # <root>; <root>/<sub> is created by the relocated write
    ns = _exec_subdir_shim(results, "runA")

    _run(
        ns,
        "import glob, os\n"
        "open('results/report.csv', 'w').write('region,sales\\nnorth,100\\n')\n"
        "hits = glob.glob('results/*.csv')\n"
        "entries = sorted(os.listdir('results'))\n",
    )

    # glob relocates the results-prefixed pattern -> absolute path under the subdir.
    assert ns["hits"] == [str(results / "runA" / "report.csv")]
    # os.listdir('results') lists the run subdir (the lone leading marker is
    # stripped), not the flat root.
    assert ns["entries"] == ["report.csv"]
    # And the file really did land in the subdir, not the flat root.
    assert (results / "runA" / "report.csv").is_file()
    assert not (results / "report.csv").exists()


def test_scandir_finds_self_written_file_under_subdir(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    ns = _exec_subdir_shim(results, "runB")

    _run(
        ns,
        "import os\n"
        "open('results/data.json', 'w').write('{\"ok\": 1}')\n"
        "names = sorted(e.name for e in os.scandir('results'))\n",
    )

    assert ns["names"] == ["data.json"]
    assert (results / "runB" / "data.json").is_file()


# ─── flat fallback: a legacy flat file still recalls ─────────────────────


def test_flat_fallback_recalls_legacy_flat_file(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    # A legacy FLAT deliverable at the root, and NO run subdir yet — so the
    # subdir-first probe misses and the flat fallback engages.
    (results / "legacy.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    ns = _exec_subdir_shim(results, "runC")

    _run(
        ns,
        "import glob, os\n"
        "hits = glob.glob('results/*.csv')\n"
        "entries = os.listdir('results')\n",
    )

    assert ns["hits"] == [str(results / "legacy.csv")]
    assert "legacy.csv" in ns["entries"]


# ─── conservative: bare patterns stay on the real CWD ───────────────────


def test_bare_pattern_stays_on_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "note.csv").write_text("x\n1\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()

    ns = _exec_subdir_shim(results, "runD")
    monkeypatch.chdir(cwd)  # bare pattern globs the REAL cwd, not results/

    _run(ns, "import glob\nhits = glob.glob('*.csv')\n")

    # No results/ prefix -> pattern unchanged -> real CWD glob finds note.csv.
    assert ns["hits"] == ["note.csv"]


def test_non_results_prefixed_dir_is_not_relocated(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "t.txt").write_text("hi", encoding="utf-8")

    ns = _exec_subdir_shim(results, "runE")
    _run(ns, "import os\nentries = os.listdir(" + repr(str(other)) + ")\n")

    # A non-results/ absolute dir is passed straight through (isabs guard).
    assert ns["entries"] == ["t.txt"]


# ─── root_dir= glob passes through unchanged ────────────────────────────


def test_root_dir_glob_passes_through_unchanged(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    scoped = tmp_path / "scoped"
    scoped.mkdir()
    (scoped / "keep.csv").write_text("k\n1\n", encoding="utf-8")

    ns = _exec_subdir_shim(results, "runF")
    _run(
        ns,
        "import glob\n"
        "hits = glob.glob('*.csv', root_dir=" + repr(str(scoped)) + ")\n",
    )

    # root_dir= means the caller already scoped the dir — no relocation.
    assert ns["hits"] == ["keep.csv"]


# ─── legacy + degenerate variants are byte-identical (no listing shim) ────


def test_legacy_and_degenerate_variants_have_no_listing_rebind() -> None:
    legacy = _write_bootstrap("/tmp/rr", None)
    degenerate = _write_bootstrap("", None)
    for shim in (legacy, degenerate):
        assert "_turing_glob_mod" not in shim
        assert "_turing_listdir_real" not in shim
        assert "_turing_scandir_real" not in shim
    # And the run-subdir variant DOES carry the listing rebinds.
    subdir = _write_bootstrap("/tmp/rr", "runG")
    assert "_turing_glob_mod.glob = _turing_glob" in subdir
    assert "_turing_os.listdir = _turing_listdir" in subdir
    assert "_turing_os.scandir = _turing_scandir" in subdir
