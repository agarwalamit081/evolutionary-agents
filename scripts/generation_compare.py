#!/usr/bin/env python3
"""Generation-over-generation comparison table — the thesis artifact.

The self-improvement thesis ("does a prior run's crystallized skill/tool/prompt
improve a later run") is answered by comparing G0 → G1 → G2: each generation
inherits the prior's run-behavior state (tools/skills/facts/cold/prompts) and is
scored. This script produces that comparison: for N generation suffixes it calls
``scripts/run_metrics.py``'s report builder per suffix and prints the delta
table — score (higher is better) alongside every efficiency metric (lower is
better for cost/tokens/calls/span), per goal + battery mean.

The score signal is the decisive one for goals that FAIL at G0; for goals
already passing at G0 the signal is EFFICIENCY (cost/LLM-calls/verify-cycles
should fall as accumulated capability is recalled). Both directions are shown so
either kind of improvement is visible.

Read-only (no writes). Connects via the app's ``DATABASE_URL``.

Usage::

    python scripts/generation_compare.py gen0-20260701 gen1-20260702 gen2-20260703
    python scripts/generation_compare.py 20260706 20260707 --json cmp.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Project root on sys.path so ``src.*`` imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

# Reuse run_metrics' report builder + report_to_dict (both live in scripts/).
_RM_PATH = Path(__file__).resolve().parent / "run_metrics.py"
_rm_spec = importlib.util.spec_from_file_location("run_metrics", _RM_PATH)
assert _rm_spec is not None and _rm_spec.loader is not None
rm = importlib.util.module_from_spec(_rm_spec)
sys.modules["run_metrics"] = rm  # dataclass(slots=True) resolves __module__ via this
_rm_spec.loader.exec_module(rm)


# ─── comparison core (pure; unit-tested with fake reports) ────────────────────


@dataclass(frozen=True, slots=True)
class GenSummary:
    """One generation's headline metrics, flattened for the delta table."""

    suffix: str
    battery_mean: float | None
    n_goals_ran: int
    total_cost_usd: float | None
    input_tokens: int | None
    cached_tokens: int | None
    cache_hit_rate: float | None
    llm_calls: int | None
    llm_span_seconds: float | None
    subagents_delegated: int | None
    tools_created: int
    subagents_created: int


@dataclass(slots=True)
class Comparison:
    suffixes: list[str]
    summaries: list[GenSummary]
    # goal_id → [score-per-generation-or-None]; rows of the per-goal matrix.
    per_goal: dict[str, list[float | None]] = field(default_factory=dict)
    ran_flags: dict[str, list[bool]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suffixes": self.suffixes,
            "summaries": [asdict(s) for s in self.summaries],
            "per_goal": self.per_goal,
            "per_goal_ran": self.ran_flags,
        }


def summarize(suffix: str, report: Any) -> GenSummary:
    """Flatten a RunMetricsReport into the headline metrics for the delta table."""
    score = report.score
    cost = report.cost
    subs = report.subagents
    return GenSummary(
        suffix=suffix,
        battery_mean=score.battery_mean if score else None,
        n_goals_ran=score.n_goals_ran if score else 0,
        total_cost_usd=cost.total_cost_usd if cost else None,
        input_tokens=cost.input_tokens if cost else None,
        cached_tokens=cost.cached_tokens if cost else None,
        cache_hit_rate=cost.cache_hit_rate if cost else None,
        llm_calls=cost.total_calls if cost else None,
        llm_span_seconds=report.llm_span_seconds,
        subagents_delegated=subs.delegated if subs else None,
        tools_created=report.created.tools,
        subagents_created=report.created.subagents,
    )


def build_per_goal(reports: list[Any]) -> tuple[
    dict[str, list[float | None]], dict[str, list[bool]]
]:
    """goal_id → [score per generation]. A goal with no rows in a generation is
    None (excluded from that generation's battery mean, mirroring run_metrics)."""
    # Union of goal ids in generation order, then per-generation lookup.
    goal_order: list[str] = []
    per_gen: list[dict[str, tuple[float, bool]]] = []
    for report in reports:
        score = report.score
        gmap: dict[str, tuple[float, bool]] = {}
        if score:
            for g in score.per_goal:
                gmap[g.goal_id] = (g.score, g.score is not None and g.n_checks > 0)
                if g.goal_id not in goal_order:
                    goal_order.append(g.goal_id)
        per_gen.append(gmap)

    matrix: dict[str, list[float | None]] = {}
    ran: dict[str, list[bool]] = {}
    for goal_id in goal_order:
        row: list[float | None] = []
        rrow: list[bool] = []
        for gmap in per_gen:
            entry = gmap.get(goal_id)
            if entry is None:
                row.append(None)
                rrow.append(False)
            else:
                row.append(entry[0])
                rrow.append(entry[1])
        matrix[goal_id] = row
        ran[goal_id] = rrow
    return matrix, ran


def build_comparison(suffixes: list[str], reports: list[Any]) -> Comparison:
    """Compose the full comparison from one report per suffix (pure)."""
    summaries = [summarize(s, r) for s, r in zip(suffixes, reports, strict=True)]
    per_goal, ran_flags = build_per_goal(reports)
    return Comparison(
        suffixes=list(suffixes), summaries=summaries, per_goal=per_goal, ran_flags=ran_flags
    )


# ─── rendering ────────────────────────────────────────────────────────────────


def _fmt(v: float | int | None, prec: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def _delta(cur: float | int | None, prev: float | int | None) -> str:
    if cur is None or prev is None:
        return ""
    d = cur - prev
    sign = "+" if d > 0 else ("" if d == 0 else "")
    if isinstance(cur, float) or isinstance(prev, float):
        return f"{sign}{d:.4f}"
    return f"{sign}{d}"


def render_table(comp: Comparison) -> None:
    gens = comp.suffixes
    n = len(gens)
    print(f"\n═ generation_compare · {n} generations: {' → '.join(gens)}")

    # ── headline delta table ──
    # metric key → (label, getter, is_score_like_higher_better, precision)
    metrics: list[tuple[str, str, bool, int]] = [
        ("battery_mean", "score", True, 4),
        ("total_cost_usd", "cost($)", False, 4),
        ("input_tokens", "tok_in", False, 0),
        ("cached_tokens", "tok_cache", False, 0),
        ("cache_hit_rate", "hit_rate", True, 3),
        ("llm_calls", "llm_calls", False, 0),
        ("llm_span_seconds", "span_s", False, 1),
        ("subagents_delegated", "subs_used", False, 0),
        ("tools_created", "tools_new", True, 0),
        ("subagents_created", "subs_new", True, 0),
    ]
    col_w = 14
    header = f"  {'metric':<12}" + "".join(f"{g[:col_w]:>{col_w}}" for g in gens)
    if n >= 2:
        header += f"{'Δ(last-first)':>{col_w}}"
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for key, label, higher_better, prec in metrics:
        vals = [getattr(s, key) for s in comp.summaries]
        line = f"  {label:<12}"
        for v in vals:
            line += f"{_fmt(v, prec):>{col_w}}"
        if n >= 2:
            first, last = vals[0], vals[-1]
            d = _delta(last, first)
            arrow = ""
            if d and first is not None and last is not None:
                better = (last > first) if higher_better else (last < first)
                worse = (last < first) if higher_better else (last > first)
                arrow = " ✓" if better and last != first else (" ✗" if worse else "")
            line += f"{(d + arrow):>{col_w}}"
        print(line)

    # ── per-goal score matrix ──
    if comp.per_goal:
        print("\n  PER-GOAL SCORE (terminal-state; '—' = goal did not run that gen)")
        header2 = f"  {'goal':<26}" + "".join(f"{g[8:16]:>{col_w}}" for g in gens)
        if n >= 2:
            header2 += f"{'Δ(last-first)':>{col_w}}"
        print(header2)
        print("  " + "-" * (len(header2) - 2))
        for goal_id, row in sorted(comp.per_goal.items()):
            line = f"  {goal_id:<26}"
            for v in row:
                line += f"{_fmt(v, 4):>{col_w}}"
            if n >= 2:
                first, last = row[0], row[-1]
                line += f"{_delta(last, first):>{col_w}}"
            print(line)
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────


async def _async_compare(suffixes: list[str]) -> Comparison:
    """Fetch one run_metrics report per suffix concurrently → comparison."""
    reports = await asyncio.gather(*(rm.build_report(s) for s in suffixes))
    return build_comparison(suffixes, list(reports))


async def _async_main(args: argparse.Namespace) -> int:
    comp = await _async_compare(args.suffixes)
    render_table(comp)
    if args.json:
        Path(args.json).write_text(
            json.dumps(comp.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("generation_compare JSON written → {}", args.json)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "suffixes",
        nargs="+",
        help="two or more generation suffixes in order (G0 → G1 → G2 …)",
    )
    parser.add_argument("--json", help="write the comparison as JSON to this path")
    args = parser.parse_args()
    if len(args.suffixes) < 2:
        parser.error("need at least two suffixes to compare")
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
