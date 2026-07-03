#!/usr/bin/env python3
"""Per-run / per-generation metrics from the persisted observation tables.

The thesis experiment ("does a prior run's crystallized skill/tool/prompt improve
a later run") needs ONE canonical, scripted read of the metrics that matter — no
ad-hoc SQL, no copy-paste variation. This is that instrument: point it at a run
(``--run-id``) or a whole curve/generation (``--suffix``) and it derives score,
tokens, cost, model calls, LLM span, verify-cycle estimate, subagent delegations
and created tool/subagent counts from the tables the agent already writes.

Run ids are stored in their thread-id form (``api-battery04_q01-20260707``,
``cli-<id>``, ``bench-<id>``), so selectors are matched as **ends-with**: a
``--run-id q01-20260707`` or ``--suffix 20260707`` resolves across the
``api-``/``cli-``/``bench-`` prefixes. Score uses the SAME terminal-state logic
as ``scripts/curve_score.py`` (latest attempt → latest row per check → mean;
self-correction is not penalized).

**Attribution reality (grounded against the live DB):**
- cost_ledger.run_id, eval_results.run_id, sub_agent_runs.parent_thread_id are
  populated → tokens/cost/models/score/subagents/LLM-span/cache are per-run.
- tool_call_metrics.run_id is **not populated** → tool *usage* is reported as a
  GLOBAL health section, not per-run (flagged in ``attribution_gaps``).
- task_executions / execution_steps are **empty** → no per-node timing is
  persisted; per-node latency lives in Prometheus only (Track B backend).
- tools/subagents **created** carry no run_id → attributed by the run's
  ``created_at`` window joined to ``mutations.created_at`` (generation scope).

Read-only (no writes). Connects via the app's ``DATABASE_URL``.

Usage::

    python scripts/run_metrics.py --suffix 20260707                 # one curve/gen
    python scripts/run_metrics.py --run-id q01-20260707              # one run
    python scripts/run_metrics.py --suffix 20260707 --json out.json  # machine form
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Project root on sys.path so ``src.*`` imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402
from sqlalchemy import select  # noqa: E402

from src.db.models import (  # noqa: E402
    CostLedger,
    EvalResult,
    SubAgentModel,
    SubAgentRunModel,
    ToolCallMetric,
    ToolRegistration,
)
from src.db.session import get_session  # noqa: E402
from src.eval.curve import CapabilityCurve  # noqa: E402

# Selectors are run-id fragments; reject anything that could break a LIKE pattern
# or inject a metacharacter so the ends-with match is safe.
_SELECTOR_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


# ─── pure aggregation core (dict → dataclass; unit-tested with fixtures) ──────


@dataclass(frozen=True, slots=True)
class ModelBreakdown:
    """One provider/model row of the cost/token/call breakdown."""

    provider: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    total_tokens: int
    cost_usd: float
    mean_latency_ms: float | None


@dataclass(frozen=True, slots=True)
class CostSummary:
    total_calls: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    total_tokens: int
    total_cost_usd: float
    by_model: list[ModelBreakdown]
    cache_hit_rate: float  # cached_tokens / input_tokens (0.0 when no input)


@dataclass(frozen=True, slots=True)
class GoalScore:
    goal_id: str
    run_id: str
    score: float
    n_checks: int
    n_rows: int
    verify_passes_estimate: int
    attempt_id: str | None


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    battery_mean: float
    n_goals_ran: int
    per_goal: list[GoalScore]


@dataclass(frozen=True, slots=True)
class SubagentSummary:
    delegated: int
    completed: int
    failed: int
    timeout: int
    total_tokens: int
    total_cost_usd: float


@dataclass(frozen=True, slots=True)
class CreatedSummary:
    """Tools/subagents attributed to the run by ``created_at`` window."""

    tools: int
    subagents: int
    window_start: str | None
    window_end: str | None


@dataclass(frozen=True, slots=True)
class ToolHealthRow:
    """GLOBAL tool success/empty/latency (NOT per-run — tool_call_metrics.run_id
    is unpopulated). Included so the analyst still sees tool reliability."""

    tool_name: str
    calls: int
    success_count: int
    empty_output_count: int
    success_rate: float
    mean_latency_ms: float | None


@dataclass(slots=True)
class RunMetricsReport:
    selector: str
    matched_run_ids: list[str]
    score: ScoreSummary | None
    cost: CostSummary | None
    llm_span_seconds: float | None
    subagents: SubagentSummary | None
    created: CreatedSummary
    global_tool_health: list[ToolHealthRow]
    attribution_gaps: list[str] = field(default_factory=list)


def aggregate_cost(rows: list[dict[str, Any]]) -> CostSummary:
    """GROUP-BY provider,model over cost-ledger rows → totals + per-model rows.

    Pure: takes already-fetched dict rows (``provider``, ``model``, token counts,
    ``cost_usd``, ``latency_ms``) so it is unit-tested with a fixture list.
    """
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    in_tok = out_tok = cached_tok = total_tok = 0
    total_cost = 0.0
    total_calls = 0
    for r in rows:
        key = (str(r.get("provider") or ""), str(r.get("model") or ""))
        b = buckets.setdefault(
            key,
            {
                "calls": 0,
                "input": 0,
                "output": 0,
                "cached": 0,
                "total": 0,
                "cost": 0.0,
                "latencies": [],
            },
        )
        b["calls"] += 1
        b["input"] += int(r.get("input_tokens") or 0)
        b["output"] += int(r.get("output_tokens") or 0)
        b["cached"] += int(r.get("cached_tokens") or 0)
        b["total"] += int(r.get("total_tokens") or 0)
        b["cost"] += float(r.get("cost_usd") or 0.0)
        lat = r.get("latency_ms")
        if lat is not None:
            b["latencies"].append(int(lat))

        in_tok += int(r.get("input_tokens") or 0)
        out_tok += int(r.get("output_tokens") or 0)
        cached_tok += int(r.get("cached_tokens") or 0)
        total_tok += int(r.get("total_tokens") or 0)
        total_cost += float(r.get("cost_usd") or 0.0)
        total_calls += 1

    by_model: list[ModelBreakdown] = []
    for (provider, model), b in sorted(buckets.items()):
        lats = b["latencies"]
        by_model.append(
            ModelBreakdown(
                provider=provider,
                model=model,
                calls=int(b["calls"]),
                input_tokens=int(b["input"]),
                output_tokens=int(b["output"]),
                cached_tokens=int(b["cached"]),
                total_tokens=int(b["total"]),
                cost_usd=round(float(b["cost"]), 6),
                mean_latency_ms=round(sum(lats) / len(lats), 1) if lats else None,
            )
        )

    cache_hit_rate = (cached_tok / in_tok) if in_tok else 0.0
    return CostSummary(
        total_calls=total_calls,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cached_tokens=cached_tok,
        total_tokens=total_tok,
        total_cost_usd=round(total_cost, 6),
        by_model=by_model,
        cache_hit_rate=round(cache_hit_rate, 4),
    )


def _estimate_verify_passes(rows: list[dict[str, Any]]) -> int:
    """Verify passes ≈ total check rows / distinct check names, rounded up.

    Each verify pass writes one row per check; the latest attempt's rows are the
    tail of a re-verify chain. With ``n`` distinct checks and ``r`` total rows,
    the attempt ran ≈ ceil(r/n) verify passes. 0 when no rows. This is an
    ESTIMATE (the exact verify-cycle count is not persisted — execution_steps is
    empty), used only as a convergence/efficiency signal.
    """
    if not rows:
        return 0
    names = {str(r.get("check_name") or "") for r in rows}
    n = len(names) or 1
    r = len(rows)
    return -(-r // n)  # ceil


def score_goals(eval_rows: list[dict[str, Any]]) -> ScoreSummary:
    """Terminal-state score per goal (latest attempt → latest row per check → mean).

    Pure transform of eval rows grouped by ``goal_id``; mirrors
    ``scripts/curve_score.py._goal_score`` exactly via the shared
    ``CapabilityCurve._latest_attempt`` / ``_latest_per_check`` methods so a
    self-correcting run that ends all-pass scores 1.0. The battery mean is the
    mean of the per-goal means of goals that have rows (a missing goal is
    excluded, not counted as 0).
    """
    by_goal: dict[str, list[dict[str, Any]]] = {}
    for r in eval_rows:
        by_goal.setdefault(str(r.get("goal_id") or ""), []).append(r)

    per_goal: list[GoalScore] = []
    ran: list[float] = []
    for goal_id, rows in by_goal.items():
        attempt = CapabilityCurve._latest_attempt(rows)
        attempt_rows = [r for r in rows if r.get("attempt_id") == attempt] or rows
        latest = CapabilityCurve._latest_per_check(attempt_rows)
        score = (
            sum(float(r.get("score") or 0.0) for r in latest) / len(latest)
            if latest
            else 0.0
        )
        if latest:
            ran.append(score)
        run_id = rows[0].get("run_id") if rows else None
        per_goal.append(
            GoalScore(
                goal_id=goal_id,
                run_id=str(run_id) if run_id is not None else "",
                score=round(score, 4),
                n_checks=len(latest),
                n_rows=len(rows),
                verify_passes_estimate=_estimate_verify_passes(rows),
                attempt_id=attempt,
            )
        )

    per_goal.sort(key=lambda g: g.goal_id)
    battery = sum(ran) / len(ran) if ran else 0.0
    return ScoreSummary(
        battery_mean=round(battery, 4), n_goals_ran=len(ran), per_goal=per_goal
    )


def aggregate_subagents(rows: list[dict[str, Any]]) -> SubagentSummary:
    """Sub-agent delegation roll-up from ``sub_agent_runs`` rows (parent_thread_id)."""
    delegated = len(rows)
    completed = sum(1 for r in rows if str(r.get("status") or "") == "completed")
    failed = sum(1 for r in rows if str(r.get("status") or "") == "failed")
    timeout = sum(1 for r in rows if str(r.get("status") or "") == "timeout")
    total_tokens = sum(int(r.get("tokens_used") or 0) for r in rows)
    total_cost = sum(float(r.get("cost_usd") or 0.0) for r in rows)
    return SubagentSummary(
        delegated=delegated,
        completed=completed,
        failed=failed,
        timeout=timeout,
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 6),
    )


def aggregate_tool_health(rows: list[dict[str, Any]]) -> list[ToolHealthRow]:
    """GLOBAL per-tool success/empty/latency (tool_call_metrics is not run-attributed)."""
    buckets: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = str(r.get("tool_name") or "")
        b = buckets.setdefault(
            name, {"calls": 0, "success": 0, "empty": 0, "latencies": []}
        )
        b["calls"] += 1
        if bool(r.get("success")):
            b["success"] += 1
        if bool(r.get("empty_output")):
            b["empty"] += 1
        lat = r.get("latency_ms")
        if lat is not None:
            b["latencies"].append(int(lat))
    out: list[ToolHealthRow] = []
    for name, b in sorted(buckets.items()):
        lats = b["latencies"]
        out.append(
            ToolHealthRow(
                tool_name=name,
                calls=int(b["calls"]),
                success_count=int(b["success"]),
                empty_output_count=int(b["empty"]),
                success_rate=round(int(b["success"]) / int(b["calls"]), 4)
                if b["calls"]
                else 0.0,
                mean_latency_ms=round(sum(lats) / len(lats), 1) if lats else None,
            )
        )
    return out


def count_created(
    tool_rows: list[dict[str, Any]], subagent_rows: list[dict[str, Any]]
) -> CreatedSummary:
    """Count generated tools / subagents (source_mutation_id IS NOT NULL)."""
    tools = sum(1 for r in tool_rows if r.get("source_mutation_id") is not None)
    subagents = sum(1 for r in subagent_rows if r.get("source_mutation_id") is not None)
    starts = [
        str(r["created_at"]) for r in [*tool_rows, *subagent_rows] if r.get("created_at")
    ]
    window_start = min(starts) if starts else None
    window_end = max(starts) if starts else None
    return CreatedSummary(
        tools=tools,
        subagents=subagents,
        window_start=window_start,
        window_end=window_end,
    )


def llm_span(rows: list[dict[str, Any]]) -> float | None:
    """LLM-call span (max(created_at) - min(created_at)) in seconds — the wall-time
    proxy persisted in the DB (task_executions.completed_at is not populated)."""
    times = [t for t in (r.get("_created_dt") for r in rows) if t is not None]
    if len(times) < 2:
        return None
    delta = max(times) - min(times)
    return round(delta.total_seconds(), 1)


# ─── thin DB fetch layer (dict rows → pure core) ──────────────────────────────


def _endswith_clause(col: Any, selector: str) -> Any:
    return col.like(f"%{selector}")


async def fetch_cost_rows(selector: str) -> list[dict[str, Any]]:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(
                    CostLedger.run_id,
                    CostLedger.provider,
                    CostLedger.model,
                    CostLedger.input_tokens,
                    CostLedger.output_tokens,
                    CostLedger.cached_tokens,
                    CostLedger.total_tokens,
                    CostLedger.cost_usd,
                    CostLedger.latency_ms,
                    CostLedger.created_at,
                ).where(_endswith_clause(CostLedger.run_id, selector))
            )
        ).all()
    return [
        {
            "run_id": r.run_id,
            "provider": r.provider,
            "model": r.model,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cached_tokens": r.cached_tokens,
            "total_tokens": r.total_tokens,
            "cost_usd": float(r.cost_usd),
            "latency_ms": r.latency_ms,
            "_created_dt": r.created_at,
        }
        for r in rows
    ]


async def fetch_eval_rows(selector: str) -> list[dict[str, Any]]:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(EvalResult).where(_endswith_clause(EvalResult.run_id, selector))
            )
        ).scalars().all()
    return [
        {
            "goal_id": r.goal_id,
            "run_id": r.run_id,
            "attempt_id": r.attempt_id,
            "check_name": r.check_name,
            "score": float(r.score),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


async def fetch_subagent_rows(selector: str) -> list[dict[str, Any]]:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(
                    SubAgentRunModel.parent_thread_id,
                    SubAgentRunModel.status,
                    SubAgentRunModel.tokens_used,
                    SubAgentRunModel.cost_usd,
                ).where(
                    _endswith_clause(SubAgentRunModel.parent_thread_id, selector)
                )
            )
        ).all()
    return [
        {
            "parent_thread_id": r.parent_thread_id,
            "status": r.status,
            "tokens_used": r.tokens_used,
            "cost_usd": float(r.cost_usd),
        }
        for r in rows
    ]


async def fetch_global_tool_metrics(limit: int = 5000) -> list[dict[str, Any]]:
    """All recent tool_call_metrics rows (GLOBAL — not run-attributed)."""
    async with get_session() as session:
        rows = (
            await session.execute(
                select(
                    ToolCallMetric.tool_name,
                    ToolCallMetric.success,
                    ToolCallMetric.empty_output,
                    ToolCallMetric.latency_ms,
                ).order_by(ToolCallMetric.created_at.desc())
                .limit(limit)
            )
        ).all()
    return [
        {
            "tool_name": r.tool_name,
            "success": r.success,
            "empty_output": r.empty_output,
            "latency_ms": r.latency_ms,
        }
        for r in rows
    ]


async def fetch_created_in_window(
    window_start: Any | None, window_end: Any | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generated tools/subagents whose created_at falls in the run's window."""
    if window_start is None or window_end is None:
        return [], []
    tool_q = select(ToolRegistration.tool_name, ToolRegistration.source_mutation_id).where(
        ToolRegistration.created_at.between(window_start, window_end)
    )
    sub_q = select(SubAgentModel.name, SubAgentModel.source_mutation_id).where(
        SubAgentModel.created_at.between(window_start, window_end)
    )
    async with get_session() as session:
        tools = (await session.execute(tool_q)).all()
        subs = (await session.execute(sub_q)).all()
    return (
        [{"source_mutation_id": t.source_mutation_id} for t in tools],
        [{"source_mutation_id": s.source_mutation_id} for s in subs],
    )


async def build_report(selector: str) -> RunMetricsReport:
    """Compose the full report for one endswith selector."""
    cost_rows, eval_rows, sub_rows, tool_rows = await asyncio.gather(
        fetch_cost_rows(selector),
        fetch_eval_rows(selector),
        fetch_subagent_rows(selector),
        fetch_global_tool_metrics(),
    )

    matched = sorted({str(r["run_id"]) for r in cost_rows if r.get("run_id")})

    cost_summary = aggregate_cost(cost_rows) if cost_rows else None
    score_summary = score_goals(eval_rows) if eval_rows else None
    sub_summary = aggregate_subagents(sub_rows) if sub_rows else None
    span = llm_span(cost_rows)

    # Attribute created tools/subagents to the run's time window.
    starts = [r["_created_dt"] for r in cost_rows if r.get("_created_dt")]
    window_start = min(starts) if starts else None
    window_end = max(starts) if starts else None
    created_tools, created_subs = await fetch_created_in_window(window_start, window_end)
    created = count_created(created_tools, created_subs)

    report = RunMetricsReport(
        selector=selector,
        matched_run_ids=matched,
        score=score_summary,
        cost=cost_summary,
        llm_span_seconds=span,
        subagents=sub_summary,
        created=created,
        global_tool_health=aggregate_tool_health(tool_rows),
        attribution_gaps=_attribution_gaps(),
    )
    return report


def _attribution_gaps() -> list[str]:
    """The known per-run attribution limits (documented, not silently dropped)."""
    return [
        "tool_call_metrics.run_id is unpopulated → tool usage reported globally, "
        "not per-run",
        "task_executions / execution_steps are empty → no per-node timing "
        "persisted (Prometheus-only; Track B backend)",
        "tools/subagents carry no run_id → 'created' attributed by run time-window "
        "(generation scope, fuzzy at boundaries)",
    ]


# ─── rendering ────────────────────────────────────────────────────────────────


def report_to_dict(report: RunMetricsReport) -> dict[str, Any]:
    """JSON-serializable form (dataclasses → dict; _created_dt stripped)."""
    d: dict[str, Any] = {
        "selector": report.selector,
        "matched_run_ids": report.matched_run_ids,
        "llm_span_seconds": report.llm_span_seconds,
        "score": asdict(report.score) if report.score else None,
        "cost": asdict(report.cost) if report.cost else None,
        "subagents": asdict(report.subagents) if report.subagents else None,
        "created": asdict(report.created),
        "global_tool_health": [asdict(r) for r in report.global_tool_health],
        "attribution_gaps": report.attribution_gaps,
    }
    return d


def render_table(report: RunMetricsReport) -> None:
    """Compact human-readable summary to stdout (secrets never appear here)."""
    print(f"\n═ run_metrics · selector={report.selector!r} "
          f"· {len(report.matched_run_ids)} run(s) matched")
    if report.matched_run_ids:
        preview = ", ".join(report.matched_run_ids[:6])
        more = f" (+{len(report.matched_run_ids) - 6} more)" if len(report.matched_run_ids) > 6 else ""
        print(f"  runs: {preview}{more}")

    if report.score:
        print(f"\n  SCORE   battery_mean={report.score.battery_mean:.4f}  "
              f"goals_ran={report.score.n_goals_ran}")
        for g in report.score.per_goal:
            print(f"          {g.goal_id:<28} {g.score:.4f}  "
                  f"checks={g.n_checks}  ~verify_passes={g.verify_passes_estimate}")

    if report.cost:
        c = report.cost
        print(f"\n  COST    ${c.total_cost_usd:.4f}  calls={c.total_calls}  "
              f"span={report.llm_span_seconds}s")
        print(f"  TOKENS  in={c.input_tokens} out={c.output_tokens} "
              f"cached={c.cached_tokens} (hit_rate={c.cache_hit_rate:.2%})")
        for m in c.by_model:
            print(f"          {m.provider}/{m.model:<22} calls={m.calls:<4} "
                  f"${m.cost_usd:.4f} in={m.input_tokens} out={m.output_tokens} "
                  f"cached={m.cached_tokens}")

    if report.subagents:
        s = report.subagents
        print(f"\n  SUBAGNT delegated={s.delegated} completed={s.completed} "
              f"failed={s.failed} timeout={s.timeout} "
              f"tokens={s.total_tokens} ${s.total_cost_usd:.4f}")

    cr = report.created
    if cr.tools or cr.subagents:
        print(f"\n  CREATED tools={cr.tools} subagents={cr.subagents} "
              f"(window {cr.window_start} → {cr.window_end})")

    if report.global_tool_health:
        top = sorted(report.global_tool_health, key=lambda t: -t.calls)[:8]
        print("\n  GLOBAL TOOL HEALTH (not per-run):")
        for t in top:
            print(f"          {t.tool_name:<28} calls={t.calls:<5} "
                  f"success={t.success_rate:.2%} empty={t.empty_output_count} "
                  f"lat={t.mean_latency_ms}")

    print("\n  ATTRIBUTION GAPS:")
    for gap in report.attribution_gaps:
        print(f"          • {gap}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _validate_selector(value: str) -> str:
    if not _SELECTOR_RE.match(value):
        raise argparse.ArgumentTypeError(
            f"selector must match {_SELECTOR_RE.pattern!r} (got {value!r})"
        )
    return value


async def _async_main(args: argparse.Namespace) -> int:
    selector = args.suffix if args.suffix else args.run_id
    if not selector:
        print("error: provide --run-id or --suffix", file=sys.stderr)
        return 2
    report = await build_report(selector)
    render_table(report)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report_to_dict(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("run_metrics JSON written → {}", args.json)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--run-id", type=_validate_selector, help="one run (ends-with match)")
    grp.add_argument(
        "--suffix", type=_validate_selector, help="one curve/generation (ends-with match)"
    )
    parser.add_argument("--json", help="write the full report as JSON to this path")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
