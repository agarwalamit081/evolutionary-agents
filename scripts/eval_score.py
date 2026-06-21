"""Score on-disk deliverables against a golden ``GoalSpec`` (ground truth).

The live agent's LLM verify narrative can under-report (it sometimes trusts a
stale self-report instead of re-reading files). The eval harness's
recomputation checks are the authoritative score: they re-derive every hard
constraint and the objective from the raw deliverables. Use this to get the
true score for a finished run without re-running the agent.

Reads ``DATABASE_URL`` from ``.env`` via the app's own config (never printed).
No API key required — checks are deterministic recomputations.

Usage::

    python scripts/eval_score.py battery04_q07 \\
        results/q07/instance.json \\
        results/q07/solution.json \\
        results/q07/constraint_report.json \\
        results/q07/adversarial_report.json

    python scripts/eval_score.py battery04_q08 results/q08/final_report.md   # positional deliverables
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Make ``src`` importable when run directly as ``python scripts/eval_score.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.checks import run_checks  # noqa: E402
from src.eval.golden import GOLDEN_SPECS  # noqa: E402


def _evidence_str(ev: object) -> str:
    """Compact one-line rendering of a check's evidence payload."""
    if ev is None:
        return ""
    if isinstance(ev, str):
        return ev[:120].replace("\n", " ")
    try:
        return json.dumps(ev, default=str)[:120]
    except (TypeError, ValueError):
        return str(ev)[:120]


async def _score(spec_id: str, deliverables: list[str]) -> int:
    if spec_id not in GOLDEN_SPECS:
        print(f"unknown spec id: {spec_id!r}")
        print(f"available: {', '.join(sorted(GOLDEN_SPECS))}")
        return 2

    spec = GOLDEN_SPECS[spec_id]
    # ``run_checks`` merges the spec's ``expected_deliverables`` with the paths
    # supplied here, so every spec-named file is scored even if the caller omits
    # it. Missing files surface as a failed structural check, not a crash.
    result = await run_checks(spec, deliverables, {})

    print(f"\n>>> {spec_id}: OVERALL SCORE {result.overall_score:.3f}  passed={result.passed}\n")
    for c in result.checks:
        flag = "PASS" if c.passed else "FAIL"
        skip = " (skipped)" if getattr(c, "skipped", False) else ""
        err = f"  ERR={c.error}" if getattr(c, "error", None) else ""
        print(f"  [{flag}]{skip} {c.check_name:38s} score={c.score:.2f}{err}")
        line = _evidence_str(c.evidence)
        if line:
            print(f"        {line}")
    print()
    return 0 if result.passed else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score on-disk deliverables against a golden GoalSpec (recomputation ground truth).",
    )
    ap.add_argument("spec_id", help="GoalSpec id in GOLDEN_SPECS (e.g. battery04_q07)")
    ap.add_argument(
        "deliverables", nargs="+",
        help="deliverable file paths to score (mapped to their declared aliases)",
    )
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_score(args.spec_id, args.deliverables)))


if __name__ == "__main__":
    main()
