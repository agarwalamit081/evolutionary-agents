"""O3: honesty on self-fabricated tool output — deliverable provenance spot-checks.

The verify node's deliverable-grounding check verifies deliverable↔tool-output
CONSISTENCY, not whether the tool output itself is real. A self-written tool that
emits hardcoded demo data passes grounding because its fabricated tool-output
prose agrees with its fabricated file — neither the LLM honesty check nor the
consistency check can see the fabrication. O3 closes that gap with two advisory
cross-checks added to ``_spot_check_cited_paths``:

(c) **data-row provenance** — a deliverable asserting "~42 customers" whose cited
    CSV/JSONL holds only 12 rows is fabricating; the count is recomputed from the
    cited data deliverable on disk (independent of any tool prose).
(d) **demo-data fingerprints** — obvious placeholder/fixture markers
    (``file1.md`` / ``lorem ipsum`` / ``sample_data``) embedded in the deliverable.

Both are advisory — they interpolate a caveat into the verify prompt and never
force completion. A rounding/merge difference of 1–2 rows never fires (the signal
is hardcoded-small-fixture fabrication, not off-by-one). These tests lock the
helper's row math, both advisory branches, and the ``results_dir`` anchoring a
generated tool can use to write its deliverable deterministically.
"""

from __future__ import annotations

import csv as csv_mod
import json as json_mod
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.graph.nodes.verify import _count_cited_data_rows, _spot_check_cited_paths
from src.tools.dynamic.allowlist import get_materializer_namespace


def _patch_results_root(monkeypatch: pytest.MonkeyPatch, results: Path) -> None:
    """Point the shared resolver at a tmp results tree.

    ``_paths.py`` reads ``src.config.settings.get_settings().agent`` for the root,
    so patching the single source propagates to ``resolve_existing`` /
    ``results_root`` used inside the O3 helpers. ``workspace_root`` is required by
    ``_strip_names``; ``results_per_run_subdir`` is intentionally absent so
    per-run subfoldering stays off (no run_id is bound).
    """
    fake = SimpleNamespace(
        agent=SimpleNamespace(
            results_root=str(results),
            workspace_root=str(results / "ws"),
        ),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv_mod.writer(fh).writerows(rows)


class TestCountCitedDataRows:
    """The recomputation backbone: row math per deliverable shape."""

    def test_csv_excludes_header(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        results = tmp_path / "results"
        _patch_results_root(monkeypatch, results)
        _write_csv(results / "data.csv", [["id", "name"], ["1", "a"], ["2", "b"], ["3", "c"]])
        assert _count_cited_data_rows(["results/data.csv"]) == 3

    def test_tsv_excludes_header(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        results = tmp_path / "results"
        _patch_results_root(monkeypatch, results)
        target = results / "data.tsv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("id\tname\n1\ta\n2\tb\n", encoding="utf-8")
        assert _count_cited_data_rows(["results/data.tsv"]) == 2

    def test_jsonl_counts_nonblank_lines(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        results = tmp_path / "results"
        _patch_results_root(monkeypatch, results)
        target = results / "events.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        # 3 payload lines + 1 blank line → 3 data rows.
        target.write_text(
            '{"id": 1}\n{"id": 2}\n\n{"id": 3}\n', encoding="utf-8"
        )
        assert _count_cited_data_rows(["results/events.jsonl"]) == 3

    def test_json_list_counts_elements(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        results = tmp_path / "results"
        _patch_results_root(monkeypatch, results)
        target = results / "summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json_mod.dumps([1, 2, 3, 4]), encoding="utf-8")
        assert _count_cited_data_rows(["results/summary.json"]) == 4

    def test_no_data_deliverable_returns_minus_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # -1 is the "nothing to count" sentinel (distinct from "checked, zero rows").
        results = tmp_path / "results"
        _patch_results_root(monkeypatch, results)
        assert _count_cited_data_rows([]) == -1
        target = results / "notes.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# notes", encoding="utf-8")
        assert _count_cited_data_rows(["results/notes.md"]) == -1

    def test_cited_but_missing_data_file_returns_minus_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A cited .csv that does not exist contributes nothing readable → sentinel.
        results = tmp_path / "results"
        _patch_results_root(monkeypatch, results)
        assert _count_cited_data_rows(["results/ghost.csv"]) == -1


class TestSpotCheckRowProvenance:
    """O3 check (c): claimed row-count vs the cited data deliverable on disk."""

    def test_fabricated_row_count_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = tmp_path / "results"
        _patch_results_root(monkeypatch, results)
        # The deliverable claims 42 customers but the cited CSV holds 12 rows — a
        # hardcoded stub masquerading as derived output.
        _write_csv(
            results / "customers.csv",
            [["customer_id", "name"], *[[f"c{i}", f"n{i}"] for i in range(12)]],
        )
        deliverable = (
            "Reconciliation complete: processed 42 customers and wrote "
            "results/customers.csv."
        )
        out = _spot_check_cited_paths(deliverable, "")
        assert "data rows/records" in out
        assert "42" in out
        assert "hardcoded" in out or "fabricated" in out

    def test_honest_row_count_not_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Claim matches the cited deliverable → no provenance caveat at all.
        results = tmp_path / "results"
        _patch_results_root(monkeypatch, results)
        _write_csv(
            results / "customers.csv",
            [["customer_id", "name"], *[[f"c{i}", f"n{i}"] for i in range(12)]],
        )
        deliverable = (
            "Reconciliation complete: processed 12 customers and wrote "
            "results/customers.csv."
        )
        assert _spot_check_cited_paths(deliverable, "") == ""

    def test_off_by_one_not_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A 1-row rounding/merge difference is below the flag gap — never fires.
        results = tmp_path / "results"
        _patch_results_root(monkeypatch, results)
        _write_csv(
            results / "customers.csv",
            [["customer_id", "name"], *[[f"c{i}", f"n{i}"] for i in range(12)]],
        )
        deliverable = "Processed 13 customers → results/customers.csv."
        assert _spot_check_cited_paths(deliverable, "") == ""


class TestSpotCheckDemoFingerprints:
    """O3 check (d): obvious placeholder/fixture markers in the deliverable."""

    def test_demo_fingerprint_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = tmp_path / "results"
        _patch_results_root(monkeypatch, results)
        target = results / "data.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("id,v\n1,a\n", encoding="utf-8")
        deliverable = (
            "Summary written. Note this used lorem ipsum filler text. "
            "See results/data.csv."
        )
        out = _spot_check_cited_paths(deliverable, "")
        assert "demo/placeholder" in out
        assert "lorem ipsum" in out

    def test_real_content_not_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = tmp_path / "results"
        _patch_results_root(monkeypatch, results)
        target = results / "data.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("id,v\n1,a\n", encoding="utf-8")
        deliverable = "Summary written from the real source data. See results/data.csv."
        assert _spot_check_cited_paths(deliverable, "") == ""


class TestMaterializerResultsDirAnchor:
    """A generated tool can anchor writes under results_root via ``results_dir``."""

    def test_results_dir_bound_to_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.config.settings.get_settings",
            lambda: SimpleNamespace(
                agent=SimpleNamespace(
                    results_root=str(tmp_path),
                    workspace_root=str(tmp_path / "ws"),
                ),
            ),
        )
        namespace = get_materializer_namespace()
        # O3 path-anchoring: the materializer namespace exposes a results_root-
        # bound Path so a generated tool writes deterministically under results/
        # (e.g. ``pathlib.Path(results_dir, "out.csv")``) instead of the CWD,
        # where verify cannot find it.
        assert "results_dir" in namespace
        assert Path(namespace["results_dir"]) == Path(tmp_path).resolve()
