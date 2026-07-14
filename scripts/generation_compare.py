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
from typing import Any, cast

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


def _bootstrap_mean_ci(
    diffs: list[float], *, seed: int = 12345, n_boot: int = 10000, alpha: float = 0.05
) -> tuple[float, float] | None:
    """Percentile bootstrap 95% CI on the mean of ``diffs``.

    Deterministic (fixed ``seed``) so the verdict is reproducible run-to-run.
    Returns ``None`` when there are fewer than 2 diffs (no resample). Pure
    aside from the seeded RNG; no network/DB. numpy is a core dep
    (requirements.txt) so the import is unguarded.
    """
    if len(diffs) < 2:
        return None
    import numpy as np  # noqa: PLC0415 — core dep

    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_significance(
    scores_baseline: list[float],
    scores_candidate: list[float],
    *,
    label_baseline: str = "G0",
    label_candidate: str = "Glast",
) -> dict[str, Any]:
    """Paired per-goal test of one generation's scores against another's.

    The thesis unit is a GOAL paired across two generations (G0 score vs G2 score
    for the same goal). Both lists must be equal-length and already aligned by
    goal (the caller supplies only MATCHED goals — present in both gens — so no
    imputation). ``delta = candidate - baseline`` (positive = the candidate gen
    scored higher → improvement under evolution).

    Reports, with NO overclaim at small n (R3):
      - ``n`` matched pairs, ``mean_{baseline,candidate}``, ``mean_delta``,
        ``median_delta``.
      - ``ci95`` percentile-bootstrap 95% CI on the mean delta (fixed seed).
      - ``wilcoxon_stat`` / ``p_value``: two-sided Wilcoxon signed-rank
        (scipy). ``None`` when undefined (n<2 or all-zero diffs).
      - ``effect_size``: matched-pairs rank-biserial correlation ∈ [-1, 1]
        (positive = candidate tends higher). Computed from the same signed ranks
        as the test so the two are consistent.
      - ``caveat``: an explicit underpowered-n note when n < ~10 (Wilcoxon at
        small n cannot reach the conventional α=0.05 two-sided for any effect —
        the minimum attainable p grows with 1/2^n).

    Pure (seeded RNG aside); unit-tested on a fixed score matrix. Never raises —
    any stats-library failure degrades gracefully to the descriptive fields.
    """
    n = min(len(scores_baseline), len(scores_candidate))
    out: dict[str, Any] = {
        "label_baseline": label_baseline,
        "label_candidate": label_candidate,
        "n": n,
        "mean_baseline": None,
        "mean_candidate": None,
        "mean_delta": None,
        "median_delta": None,
        "ci95": None,
        "wilcoxon_stat": None,
        "p_value": None,
        "effect_size": None,
        "caveat": "",
    }
    if n == 0:
        out["caveat"] = "no matched goals"
        return out

    a = [float(x) for x in scores_baseline[:n]]
    b = [float(x) for x in scores_candidate[:n]]
    diffs = [b_i - a_i for a_i, b_i in zip(a, b, strict=True)]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    mean_d = sum(diffs) / n
    sdiffs = sorted(diffs)
    mid = n // 2
    median_d = sdiffs[mid] if n % 2 else (sdiffs[mid - 1] + sdiffs[mid]) / 2
    out["mean_baseline"] = round(mean_a, 4)
    out["mean_candidate"] = round(mean_b, 4)
    out["mean_delta"] = round(mean_d, 4)
    out["median_delta"] = round(median_d, 4)
    out["ci95"] = _bootstrap_mean_ci(diffs)

    # Signed ranks (drop zeros — the "wilcox" zero-method). The rank-biserial r
    # and the Wilcoxon T are derived from the SAME ranking for consistency.
    nz = [d for d in diffs if d != 0.0]
    abs_sorted = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    # Average ranks for ties (rank mid-assignment over tied |diff| groups).
    ranks = [0.0] * len(nz)
    i = 0
    while i < len(abs_sorted):
        j = i
        while (
            j + 1 < len(abs_sorted)
            and abs(nz[abs_sorted[j + 1]]) == abs(nz[abs_sorted[i]])
        ):
            j += 1
        avg_rank = (i + 1 + j + 1) / 2  # 1-indexed average over [i..j]
        for k in range(i, j + 1):
            ranks[abs_sorted[k]] = avg_rank
        i = j + 1
    w_pos = sum(r for r, d in zip(ranks, nz, strict=True) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nz, strict=True) if d < 0)
    denom = w_pos + w_neg
    out["effect_size"] = round((w_pos - w_neg) / denom, 4) if denom else 0.0

    # Wilcoxon signed-rank via scipy (two-sided). Undefined for n<2 or all-zero
    # diffs — degrade to None rather than raise. scipy's bundled stubs type the
    # return as ``_`` (unknown), so cast to a typed (statistic, pvalue) pair and
    # index — that keeps pyright happy without a # type: ignore.
    if len(nz) >= 2:
        try:
            from scipy.stats import wilcoxon  # noqa: PLC0415 — core dep

            res = cast("tuple[float, float]", wilcoxon(nz, alternative="two-sided", zero_method="wilcox"))
            stat_val = float(res[0])
            p_val = float(res[1])
            out["wilcoxon_stat"] = stat_val
            # NaN guard: scipy returns nan for degenerate inputs.
            out["p_value"] = None if p_val != p_val else round(p_val, 6)
        except ValueError:
            # All-zero or constant after tie-collapse → no movement signal.
            pass

    if n < 10:
        out["caveat"] = (
            f"n={n} is below ~10 — Wilcoxon is underpowered (the smallest "
            "two-sided p it can reach grows with 1/2^n); treat the Δ, CI, "
            "and effect size as descriptive, not a confirmatory p-value."
        )
    return out


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
    # goal_id → [converged-per-generation-or-None]; lets the paired test run a
    # CONVERGED-only variant (matched goals converged in BOTH first+last gen).
    converged_flags: dict[str, list[bool | None]] = field(default_factory=dict)
    # Paired per-goal significance of first-vs-last generation. ``all`` uses every
    # matched goal; ``converged`` uses only goals converged in both endpoint gens
    # (the clean experiment-design signal). ``None`` when fewer than 2 generations.
    paired: dict[str, dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "suffixes": self.suffixes,
            "summaries": [asdict(s) for s in self.summaries],
            "per_goal": self.per_goal,
            "per_goal_ran": self.ran_flags,
            "per_goal_converged": self.converged_flags,
            "paired": self.paired,
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
    dict[str, list[float | None]],
    dict[str, list[bool]],
    dict[str, list[bool | None]],
]:
    """goal_id → [score per generation]. A goal with no rows in a generation is
    None (excluded from that generation's battery mean, mirroring run_metrics).

    Also returns ``ran_flags`` (goal had rows that gen) and ``converged_flags``
    (goal was converged that gen — None when absent) so the paired test can run a
    converged-only variant without re-reading the reports.
    """
    # Union of goal ids in generation order, then per-generation lookup.
    goal_order: list[str] = []
    per_gen: list[dict[str, tuple[float, bool, bool]]] = []
    for report in reports:
        score = report.score
        gmap: dict[str, tuple[float, bool, bool]] = {}
        if score:
            for g in score.per_goal:
                gmap[g.goal_id] = (
                    g.score,
                    g.score is not None and g.n_checks > 0,
                    bool(g.converged),
                )
                if g.goal_id not in goal_order:
                    goal_order.append(g.goal_id)
        per_gen.append(gmap)

    matrix: dict[str, list[float | None]] = {}
    ran: dict[str, list[bool]] = {}
    converged: dict[str, list[bool | None]] = {}
    for goal_id in goal_order:
        row: list[float | None] = []
        rrow: list[bool] = []
        crow: list[bool | None] = []
        for gmap in per_gen:
            entry = gmap.get(goal_id)
            if entry is None:
                row.append(None)
                rrow.append(False)
                crow.append(None)
            else:
                row.append(entry[0])
                rrow.append(entry[1])
                crow.append(entry[2])
        matrix[goal_id] = row
        ran[goal_id] = rrow
        converged[goal_id] = crow
    return matrix, ran, converged


def _endpoint_paired(
    per_goal: dict[str, list[float | None]],
    converged_flags: dict[str, list[bool | None]],
    suffixes: list[str],
) -> dict[str, dict[str, Any]] | None:
    """First-vs-last-generation paired test (all-matched + converged-matched).

    The decisive thesis contrast is G0 → Glast. ``all`` pairs every goal present
    in both endpoint generations; ``converged`` further requires the goal to be
    converged in BOTH (so a budget-exhausted q06 can't drag the signal either
    way). Returns ``None`` when there are fewer than 2 generations. Pure.
    """
    if len(suffixes) < 2:
        return None
    first_i, last_i = 0, len(suffixes) - 1
    base_all: list[float] = []
    cand_all: list[float] = []
    base_conv: list[float] = []
    cand_conv: list[float] = []
    for goal_id, row in per_goal.items():
        a, b = row[first_i], row[last_i]
        if a is None or b is None:
            continue
        base_all.append(a)
        cand_all.append(b)
        cflags = converged_flags.get(goal_id, [])
        c_first = cflags[first_i] if first_i < len(cflags) else None
        c_last = cflags[last_i] if last_i < len(cflags) else None
        if c_first and c_last:
            base_conv.append(a)
            cand_conv.append(b)
    out: dict[str, dict[str, Any]] = {
        "all": paired_significance(
            base_all, cand_all, label_baseline=suffixes[first_i], label_candidate=suffixes[last_i]
        )
    }
    out["converged"] = paired_significance(
        base_conv,
        cand_conv,
        label_baseline=suffixes[first_i],
        label_candidate=suffixes[last_i],
    )
    return out


def build_comparison(suffixes: list[str], reports: list[Any]) -> Comparison:
    """Compose the full comparison from one report per suffix (pure)."""
    summaries = [summarize(s, r) for s, r in zip(suffixes, reports, strict=True)]
    per_goal, ran_flags, converged_flags = build_per_goal(reports)
    paired = _endpoint_paired(per_goal, converged_flags, suffixes)
    return Comparison(
        suffixes=list(suffixes),
        summaries=summaries,
        per_goal=per_goal,
        ran_flags=ran_flags,
        converged_flags=converged_flags,
        paired=paired,
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

    # ── paired significance (first vs last generation) ──
    if comp.paired:
        print("\n  PAIRED SIGNIFICANCE (per-goal, first → last generation)")
        for variant in ("all", "converged"):
            p = comp.paired.get(variant)
            if not p:
                continue
            label = "all matched goals" if variant == "all" else "converged in both"
            ci = p.get("ci95")
            ci_s = (
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci is not None else "—"
            )
            pv = p.get("p_value")
            pv_s = "—" if pv is None else f"{pv:.4f}"
            eff = p.get("effect_size")
            eff_s = "—" if eff is None else f"{eff:+.4f}"
            print(
                f"    {label:<20} n={p['n']:<3} "
                f"Δ={_fmt(p.get('mean_delta'), 4):>8}  "
                f"CI95={ci_s:<24}  p={pv_s:<8}  r={eff_s}"
            )
        # Echo the underpowered caveat once (it's the same for both variants at
        # small n) so a reader does not over-read a small-n p-value.
        caveat = (comp.paired.get("all") or {}).get("caveat") or ""
        if caveat:
            print(f"    caveat: {caveat}")
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
