"""IdempotencyCheck (determinism methodology): run a transform twice, compare.

Hermetic: the transform runs in the subprocess sandbox over a tmp-rooted
deliverable. The deterministic case passes; wall-clock / unseeded-RNG drift,
missing input, and a raising transform all fail. This locks the
"recompute-and-compare" methodology the q06 capstone relies on.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.checks import IdempotencyCheck
from src.eval.models import CheckConfig


def _patch_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Point the shared settings source at isolated tmp roots (mirrors verify)."""
    results = tmp_path / "results"
    workspace = tmp_path / "workspace"
    results.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    fake = SimpleNamespace(
        agent=SimpleNamespace(results_root=str(results), workspace_root=str(workspace)),
        eval=SimpleNamespace(
            eval_enabled=False,
            eval_enforce=False,
            eval_llm_judge_enabled=False,
            eval_canary_min_score=0.8,
            eval_store_enabled=False,
        ),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    return str(results)


def _write_input(root: str) -> None:
    """A small dated revenue CSV the transforms roll up."""
    Path(root, "in.csv").write_text(
        "date,revenue\n2026-01-01,10\n2026-01-01,5\n2026-01-02,7\n",
        encoding="utf-8",
    )


# Deterministic: sorted, no wall-clock / RNG → two runs are byte-identical.
_DETERMINISTIC = (
    "import csv\n"
    "rows = []\n"
    "with open(_INPUT, newline='') as _f:\n"
    "    for _r in csv.DictReader(_f):\n"
    "        rows.append((_r.get('date', ''), _r.get('revenue', '0')))\n"
    "for _d, _r in sorted(rows):\n"
    "    print(_d, _r)\n"
)

# Non-deterministic: wall-clock + unseeded RNG drift across two subprocess runs.
_NONDETERMINISTIC = (
    "import csv, time, random\n"
    "with open(_INPUT, newline='') as _f:\n"
    "    _ = list(csv.DictReader(_f))\n"
    "print('run_id', time.time(), random.random())\n"
)


class TestIdempotencyCheck:
    @pytest.mark.asyncio
    async def test_deterministic_transform_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _write_input(root)
        res = await IdempotencyCheck().check(
            CheckConfig(
                check_type="idempotency",
                name="det",
                params={"transform_code": _DETERMINISTIC, "input_deliverable": "results/in.csv"},
            ),
            ["results/in.csv"],
            {},
        )
        assert res.passed is True
        assert res.score == 1.0
        assert res.evidence["identical"] is True
        assert res.evidence["output_sha256"][0] == res.evidence["output_sha256"][1]

    @pytest.mark.asyncio
    async def test_nondeterministic_transform_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _write_input(root)
        res = await IdempotencyCheck().check(
            CheckConfig(
                check_type="idempotency",
                name="nondet",
                params={"transform_code": _NONDETERMINISTIC, "input_deliverable": "results/in.csv"},
            ),
            ["results/in.csv"],
            {},
        )
        assert res.passed is False
        # Ran cleanly both times but produced different output → half score.
        assert res.evidence["identical"] is False
        assert res.evidence["output_sha256"][0] != res.evidence["output_sha256"][1]

    @pytest.mark.asyncio
    async def test_missing_input_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_roots(monkeypatch, tmp_path)
        res = await IdempotencyCheck().check(
            CheckConfig(
                check_type="idempotency",
                name="missing",
                params={"transform_code": _DETERMINISTIC, "input_deliverable": "results/missing.csv"},
            ),
            ["results/missing.csv"],
            {},
        )
        assert res.passed is False
        assert res.evidence["reason"] == "input deliverable not on disk"

    @pytest.mark.asyncio
    async def test_raising_transform_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _write_input(root)
        res = await IdempotencyCheck().check(
            CheckConfig(
                check_type="idempotency",
                name="raises",
                params={"transform_code": "raise RuntimeError('boom')\n", "input_deliverable": "results/in.csv"},
            ),
            ["results/in.csv"],
            {},
        )
        assert res.passed is False
        # Both runs exited nonzero → neither condition holds.
        assert res.score == 0.0

    @pytest.mark.asyncio
    async def test_no_transform_code_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_roots(monkeypatch, tmp_path)
        res = await IdempotencyCheck().check(
            CheckConfig(check_type="idempotency", name="nocode", params={}),
            ["results/in.csv"],
            {},
        )
        assert res.passed is False
        assert "transform_code" in res.error

    @pytest.mark.asyncio
    async def test_defaults_to_first_deliverable_when_input_unnamed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _patch_roots(monkeypatch, tmp_path)
        _write_input(root)
        res = await IdempotencyCheck().check(
            CheckConfig(
                check_type="idempotency",
                name="default",
                params={"transform_code": _DETERMINISTIC},
            ),
            ["results/in.csv"],
            {},
        )
        assert res.passed is True
