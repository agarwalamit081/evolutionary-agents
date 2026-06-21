"""Measure write-hint friction across the battery per-query logs.

The execute node nudges a step whose declared deliverable was not produced by a
``file_writer`` call: ``"...no file-output tool was called; nudging (attempt k/3)"``.
Each nudge burns an extra execute turn (attempt 2 = 1 extra turn, attempt 3 = 2).
``docs/validation-battery-02.md`` records the front-load write-hint fix (put the
``file_writer`` requirement in the step LABEL on turn 1) being applied mid-battery:
N3-N6 ran pre-fix, N7-N10 post-fix. This script measures the actual pre/post delta.

CAVEAT: the per-query ``logs/nN.log`` captures the main graph AND any delegated
sub-agent subgraphs, so delegated queries (N6, N8) can double-count sub-agent
nudges. Read single-agent queries (N7, N9, N10) for the cleanest signal.

Usage::

    python scripts/measure_write_friction.py            # markdown table to stdout
    python scripts/measure_write_friction.py --json      # machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

_NUDGE_RE = re.compile(r"no file-output tool was called; nudging \(attempt (\d+)/3\)")
_WRITE_RE = re.compile(r"file_writer:file_writer:\d+ \| Writing file:")

# Pre/post split as documented in docs/validation-battery-02.md (§Write-hint).
_PRE_FIX = ("n3", "n4", "n5", "n6")
_POST_FIX = ("n7", "n8", "n9", "n10")


def _measure(log_path: Path) -> dict[str, int]:
    """Nudge attempts, extra turns burned, and file_writer calls for one query."""
    attempts: list[int] = []
    writes = 0
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"nudges": 0, "extra_turns": 0, "attempt3": 0, "writes": 0}
    for m in _NUDGE_RE.finditer(text):
        attempts.append(int(m.group(1)))
    writes = len(_WRITE_RE.findall(text))
    # Each nudge at attempt k cost (k-1) extra turns before the write landed.
    extra_turns = sum(k - 1 for k in attempts)
    return {
        "nudges": len(attempts),
        "attempt3": sum(1 for k in attempts if k >= 3),
        "extra_turns": extra_turns,
        "writes": writes,
    }


def _collect() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for stem in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10"):
        out[stem] = _measure(LOG_DIR / f"{stem}.log")
    return out


def _group_total(rows: dict[str, dict[str, int]], keys: tuple[str, ...]) -> dict[str, int]:
    return {
        "nudges": sum(rows[k]["nudges"] for k in keys),
        "attempt3": sum(rows[k]["attempt3"] for k in keys),
        "extra_turns": sum(rows[k]["extra_turns"] for k in keys),
        "writes": sum(rows[k]["writes"] for k in keys),
    }


def _to_markdown(rows: dict[str, dict[str, int]]) -> str:
    lines = [
        "## Write-hint friction (battery-02 per-query logs)",
        "",
        "| Query | nudges | attempt-3 | extra turns | file_writer calls |",
        "|-------|-------:|----------:|------------:|------------------:|",
    ]
    for stem in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10"):
        r = rows[stem]
        lines.append(
            f"| {stem.upper()} | {r['nudges']} | {r['attempt3']} | "
            f"{r['extra_turns']} | {r['writes']} |"
        )
    pre = _group_total(rows, _PRE_FIX)
    post = _group_total(rows, _POST_FIX)
    lines += [
        "",
        f"**Pre-fix (N3-N6):** {pre['nudges']} nudges, {pre['attempt3']} hit attempt-3, "
        f"**{pre['extra_turns']} extra turns** burned, {pre['writes']} writes.",
        f"**Post-fix (N7-N10):** {post['nudges']} nudges, {post['attempt3']} hit attempt-3, "
        f"**{post['extra_turns']} extra turns** burned, {post['writes']} writes.",
    ]
    # Single-agent queries carry the cleanest signal (no sub-agent double-count).
    single = _group_total(rows, ("n7", "n9", "n10"))
    lines.append(
        f"**Single-agent post-fix (N7, N9, N10 — cleanest signal):** "
        f"{single['nudges']} nudges, {single['extra_turns']} extra turns."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args()

    rows = _collect()
    if args.json:
        print(json.dumps({"rows": rows,
                          "pre_fix": _group_total(rows, _PRE_FIX),
                          "post_fix": _group_total(rows, _POST_FIX)}, indent=2))
    else:
        print(_to_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
