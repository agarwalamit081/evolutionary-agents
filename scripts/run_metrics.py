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

**Attribution (Track-1: all per-run, direct — no fuzzy time-window joins):**
- cost_ledger.run_id, eval_results.run_id, sub_agent_runs.parent_thread_id,
  tool_call_metrics.run_id, execution_steps.run_id, tool_registrations.
  owner_run_id, sub_agent_definitions.owner_run_id are ALL populated → every
  metric here is attributed directly to the run.
- tool *usage* is per-run (``tool_call_metrics.run_id``, populated by the
  execute node's ``_record_tool_metric`` via the active run_id contextvar).
- per-node wall-clock is per-run (``execution_steps.run_id``, written by the
  graph ``_wrap`` node-timer) → reported as a ``node_timing`` breakdown.
- tools/subagents **created** are attributed by ``owner_run_id`` (the run that
  generated them at persist time), not by a created_at window.
- ``llm_span_seconds`` stays the LLM-call wall-clock span (cost-ledger
  created_at spread) — complementary to per-node timing, not a proxy for it.

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
    ExecutionStep,
    SubAgentModel,
    SubAgentRunModel,
    ToolCallMetric,
    ToolRegistration,
)
from src.db.session import get_session  # noqa: E402
from src.eval.curve import (  # noqa: E402
    CapabilityCurve,
    coverage_ratio,
    expected_check_names,
)
# Shared run_id → spec-id stripper so the scorer's goal bucketing can NEVER drift
# from the worker's golden-check resolver (``src.runner._resolve_eval_spec_id``).
# The Track-1 ``-gen{N}-seed{M}-YYYYMMDD`` suffix already broke the resolver by
# leaving ``-seed1`` attached; a second copy here would re-break the scorer the
# same way. Import the single source of truth instead of duplicating the rule.
from src.runner import _strip_date_suffix as _strip_run_suffix  # noqa: E402

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
    # Convergence (experiment-design, Phase 1). A goal is ``converged`` when the
    # verify battery ran to completion on this run — its terminal attempt covered
    # the spec's declared checks AND the run did not end in a non-terminal control
    # status (BUDGET_EXHAUSTED/TIMEOUT/FAILED). ``coverage`` is the fraction of
    # expected checks observed (None when the goal has no spec → unmeasurable).
    # ``non_convergence_reason`` is empty when converged, else names the signal.
    converged: bool = True
    coverage: float | None = None
    non_convergence_reason: str = ""


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    battery_mean: float
    n_goals_ran: int
    per_goal: list[GoalScore]
    # Filtered mean over CONVERGED goals only (the clean experiment-design signal:
    # budget-infeasible / never-finished goals like q06 are excluded, not averaged
    # in as artificial 0.0). ``None`` when no goal has a measurable convergence
    # status (all adhoc/un-spec'd) so a caller can tell "no filtering applied"
    # apart from "filtered == unfiltered". ``excluded_goals`` lists the
    # non-converged goal ids dropped from the filtered mean (transparency, never
    # a silent exclusion).
    battery_mean_converged: float | None = None
    n_goals_converged: int = 0
    excluded_goals: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SubagentSummary:
    delegated: int
    completed: int
    failed: int
    timeout: int
    total_tokens: int
    total_cost_usd: float


@dataclass(frozen=True, slots=True)
class NodeTimingRow:
    """One graph node's per-run wall-clock (from ``execution_steps``)."""

    phase: str
    calls: int
    total_ms: int
    mean_ms: float | None


@dataclass(frozen=True, slots=True)
class NodeTimingSummary:
    """Per-node wall-clock breakdown for a run (``execution_steps``, keyed by run_id).

    Replaces the prior ``execution_steps is empty`` attribution gap: the graph
    ``_wrap`` node-timer now persists one timing row per node invocation.
    """

    by_node: list[NodeTimingRow]
    total_ms: int  # sum of every node duration (excludes inter-node gaps)


@dataclass(frozen=True, slots=True)
class CreatedSummary:
    """Tools/subagents attributed to the run by ``created_at`` window."""

    tools: int
    subagents: int
    window_start: str | None
    window_end: str | None


@dataclass(frozen=True, slots=True)
class ToolHealthRow:
    """Per-run tool success/empty/latency (``tool_call_metrics.run_id`` is now
    populated by the execute node's metric recorder via the active run_id)."""

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
    node_timing: NodeTimingSummary | None = None


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


def _spec_key_from_run_id(run_id: str | None) -> str:
    """Derive a stable goal-bucket key from a run_id (spec id), robust to eval mode.

    The eval_results ``goal_id`` column is only populated with the real spec id
    when the **golden** recomputation checks fire (``verify.py`` writes
    ``goal_id=spec.name``); adhoc-only rows all carry ``goal_id="adhoc-deliverables"``
    regardless of which battery query produced them. Grouping a full battery by
    that column therefore collapses all N queries into a single ``adhoc`` bucket
    (``n_goals_ran=1``) whenever golden was skipped — exactly what happened to the
    Track-1 G0 runs before the resolver was fixed.

    The run_id, by contrast, ALWAYS encodes the spec (``api-battery04_q01-gen0-
    seed1-20260713``), so bucketing by the spec id recovered from it yields the
    correct N-goal matrix whether or not golden fired — and aligns a goal across
    generations in ``generation_compare`` (gen0-seed1's q01 and gen2-seed1's q01
    both bucket under ``battery04_q01``). Stripping reuses the runner's canonical
    ``_strip_date_suffix`` so this can never disagree with the resolver.
    """
    if not run_id:
        return ""
    rid = str(run_id)
    for prefix in ("api-", "cli-", "bench-"):
        if rid.startswith(prefix):
            rid = rid[len(prefix) :]
            break
    return _strip_run_suffix(rid) or rid


# Run-control statuses that mean the run did NOT reach a natural terminal state
# — it was killed by a cap/timeout/cancel before finishing. A goal whose only
# rows came from such a run is non-converged (excluded from the filtered mean)
# even if a partial verify pass happened to write some check rows. Matches the
# resumable-terminal set in ``src/worker/schema.py`` (TIMEOUT/BUDGET_EXHAUSTED)
# plus FAILED/CANCELLED. Redis-only and TTL-bounded, so this is a best-effort
# signal; the durable backbone is the coverage check below.
_NON_CONVERGENT_STATUSES: frozenset[str] = frozenset(
    {"budget_exhausted", "timeout", "failed", "cancelled"}
)

# A goal is "coverage-converged" when its terminal attempt observed at least
# this fraction of the spec's declared checks. 1.0 = the verify battery ran in
# full (skipped checks are still written with their name, so a complete pass
# reaches 1.0). A budget-cut run that never reached verify (no deliverable → no
# checks) lands at 0.0 and is filtered out.
_CONVERGENCE_COVERAGE_THRESHOLD: float = 1.0


def _goal_converged(
    goal_id: str,
    observed_check_names: set[str],
    expected_checks: dict[str, set[str]] | None,
    run_id: str,
    status_by_run: dict[str, str] | None,
) -> tuple[bool, float | None, str]:
    """Decide whether one goal's terminal attempt represents a converged run.

    Returns ``(converged, coverage, reason)``. Two independent signals, EITHER
    can flag non-convergence:
      1. **Coverage** (durable, from eval_results): the terminal attempt's
         observed check-name set vs. the spec's declared set. Below threshold →
         the verify battery did not complete (budget/timeout cut it off, or no
         deliverable was produced for the checks to read).
      2. **Run status** (best-effort, Redis): the run ended in a non-terminal
         control status (BUDGET_EXHAUSTED/TIMEOUT/FAILED/CANCELLED).

    A goal with no registered spec (adhoc/custom) has no measurable coverage →
    ``coverage is None`` and the coverage signal is skipped (do not filter what
    cannot be measured); only a known-bad run status can then flag it. Pure;
    unit-tested with fixtures.
    """
    reason = ""
    coverage: float | None = None

    expected = (expected_checks or {}).get(goal_id)
    if expected:
        coverage = coverage_ratio(observed_check_names, expected)
        if coverage < _CONVERGENCE_COVERAGE_THRESHOLD:
            reason = (
                f"incomplete_check_coverage({coverage:.2f}; "
                f"{len(observed_check_names & expected)}/{len(expected)})"
            )

    status = (status_by_run or {}).get(run_id, "")
    if status and status in _NON_CONVERGENT_STATUSES:
        if reason:
            reason = f"{reason}+run_status={status}"
        else:
            reason = f"run_status={status}"

    return (not reason), coverage, reason


def score_goals(
    eval_rows: list[dict[str, Any]],
    *,
    expected_checks: dict[str, set[str]] | None = None,
    status_by_run: dict[str, str] | None = None,
) -> ScoreSummary:
    """Terminal-state score per goal (latest attempt → latest row per check → mean).

    Pure transform of eval rows grouped by the **spec id recovered from each
    row's run_id** (not the ``goal_id`` column — see ``_spec_key_from_run_id``);
    mirrors ``scripts/curve_score.py._goal_score`` exactly via the shared
    ``CapabilityCurve._latest_attempt`` / ``_latest_per_check`` methods so a
    self-correcting run that ends all-pass scores 1.0. The battery mean is the
    mean of the per-goal means of goals that have rows (a missing goal is
    excluded, not counted as 0).

    Convergence (experiment-design): when ``expected_checks`` and/or
    ``status_by_run`` are supplied, each goal is tagged ``converged`` via
    ``_goal_converged`` and a SECOND battery mean — ``battery_mean_converged``
    — is computed over converged goals only (non-converged goals are listed in
    ``excluded_goals``, never silently dropped). Both means are always reported
    so a reader sees the unfiltered headline AND the clean filtered signal.
    ``battery_mean_converged`` is ``None`` when no goal has a measurable
    convergence status (all goals adhoc/un-spec'd AND no run statuses) → the
    caller can tell "no filtering applied" from "filtered == unfiltered".
    """
    by_goal: dict[str, list[dict[str, Any]]] = {}
    for r in eval_rows:
        by_goal.setdefault(_spec_key_from_run_id(r.get("run_id")), []).append(r)

    per_goal: list[GoalScore] = []
    ran: list[float] = []
    converged_scores: list[float] = []
    excluded: list[str] = []
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
        run_id = str(rows[0].get("run_id")) if rows and rows[0].get("run_id") else ""
        observed_names = {str(r.get("check_name") or "") for r in latest}
        converged, coverage, reason = _goal_converged(
            goal_id, observed_names, expected_checks, run_id, status_by_run
        )
        if latest and converged:
            converged_scores.append(score)
        elif latest and not converged:
            excluded.append(goal_id)
        per_goal.append(
            GoalScore(
                goal_id=goal_id,
                run_id=run_id,
                score=round(score, 4),
                n_checks=len(latest),
                n_rows=len(rows),
                verify_passes_estimate=_estimate_verify_passes(rows),
                attempt_id=attempt,
                converged=converged,
                coverage=round(coverage, 4) if coverage is not None else None,
                non_convergence_reason=reason,
            )
        )

    per_goal.sort(key=lambda g: g.goal_id)
    battery = sum(ran) / len(ran) if ran else 0.0
    # The filtered mean is only meaningful when at least one goal had a
    # measurable convergence signal; otherwise leave it None (distinct from
    # "all goals converged" where it equals the unfiltered mean). When every
    # measured goal was non-converged the filtered set is empty → also None
    # (the ``excluded_goals`` list carries the transparency).
    any_measurable = any(
        g.coverage is not None or bool(status_by_run and status_by_run.get(g.run_id))
        for g in per_goal
    )
    if not any_measurable or not converged_scores:
        battery_converged: float | None = None
    else:
        battery_converged = round(sum(converged_scores) / len(converged_scores), 4)
    n_converged = len(converged_scores)
    return ScoreSummary(
        battery_mean=round(battery, 4),
        n_goals_ran=len(ran),
        per_goal=per_goal,
        battery_mean_converged=battery_converged,
        n_goals_converged=n_converged,
        excluded_goals=sorted(excluded),
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
    """Per-run per-tool success/empty/latency (rows are already run-scoped via
    ``tool_call_metrics.run_id``)."""
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


def aggregate_node_timing(rows: list[dict[str, Any]]) -> NodeTimingSummary | None:
    """Per-node wall-clock roll-up from ``execution_steps`` timing rows.

    Pure: takes already-fetched dict rows (``phase``, ``duration_ms``) so it is
    unit-tested with a fixture list. Groups by node phase → calls, total_ms,
    mean_ms. Returns None when there are no rows (the run produced no timing).
    """
    if not rows:
        return None
    buckets: dict[str, dict[str, Any]] = {}
    total = 0
    for r in rows:
        phase = str(r.get("phase") or "")
        ms = int(r.get("duration_ms") or 0)
        b = buckets.setdefault(phase, {"calls": 0, "total": 0})
        b["calls"] += 1
        b["total"] += ms
        total += ms
    by_node: list[NodeTimingRow] = []
    for phase, b in sorted(buckets.items()):
        calls = int(b["calls"])
        t = int(b["total"])
        by_node.append(
            NodeTimingRow(
                phase=phase,
                calls=calls,
                total_ms=t,
                mean_ms=round(t / calls, 1) if calls else None,
            )
        )
    return NodeTimingSummary(by_node=by_node, total_ms=total)


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


async def fetch_tool_metrics(selector: str) -> list[dict[str, Any]]:
    """Per-run tool_call_metrics rows (run-attributed via ``run_id`` ends-with)."""
    async with get_session() as session:
        rows = (
            await session.execute(
                select(
                    ToolCallMetric.tool_name,
                    ToolCallMetric.success,
                    ToolCallMetric.empty_output,
                    ToolCallMetric.latency_ms,
                ).where(_endswith_clause(ToolCallMetric.run_id, selector))
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


async def fetch_node_timing(selector: str) -> list[dict[str, Any]]:
    """Per-run execution_steps timing rows (phase, duration_ms, status).

    ``execution_steps.run_id`` is written by the graph ``_wrap`` node-timer;
    timing-only rows carry ``task_id IS NULL``. ``status`` is kept so callers can
    see failed-node counts if needed.
    """
    async with get_session() as session:
        rows = (
            await session.execute(
                select(
                    ExecutionStep.phase,
                    ExecutionStep.duration_ms,
                    ExecutionStep.status,
                ).where(_endswith_clause(ExecutionStep.run_id, selector))
            )
        ).all()
    return [
        {"phase": r.phase, "duration_ms": r.duration_ms, "status": r.status}
        for r in rows
    ]


def _expected_checks_for_eval(eval_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Map each spec-key present in ``eval_rows`` → its declared check-name set.

    Keys are the spec ids recovered from each row's run_id (``_spec_key_from_run_id``),
    the SAME bucketing ``score_goals`` uses, so a goal and its expected-check set
    always align. Goals with no registered spec (adhoc) are absent from the map →
    ``_goal_converged`` treats them as unmeasurable. Static per process (the golden
    registry does not mutate at runtime); cheap to recompute, so no caching.
    """
    out: dict[str, set[str]] = {}
    for r in eval_rows:
        key = _spec_key_from_run_id(r.get("run_id"))
        if key and key not in out:
            out[key] = expected_check_names(key)
    return out


async def _try_status_by_run(run_ids: list[str]) -> dict[str, str]:
    """Best-effort lookup of each run's terminal control status from Redis.

    Returns ``{run_id: status_value}`` for runs whose status hash is still live
    (TTL-bounded — historical runs scored days later will be absent, which is
    fine: the durable coverage signal carries convergence detection on its own).
    Fully best-effort: if Redis is unreachable, unconfigured, or the lookup
    raises for any reason, returns ``{}`` so the scorer never breaks on a
    missing status store. Never raises.
    """
    if not run_ids:
        return {}
    try:
        import redis.asyncio as aioredis  # noqa: PLC0415 — optional, guarded

        from src.config.settings import get_settings  # noqa: PLC0415
        from src.worker.status import RunStatusStore  # noqa: PLC0415

        settings = get_settings()
        client = aioredis.from_url(
            settings.redis.redis_url, decode_responses=False
        )
        store = RunStatusStore(client, settings.worker)
        out: dict[str, str] = {}
        try:
            for rid in run_ids:
                rec = await store.get(rid)
                if rec is not None:
                    out[rid] = rec.status.value
        finally:
            await client.aclose()
        return out
    except Exception as exc:  # noqa: BLE001 — best-effort; never break scoring
        logger.debug("status-by-run lookup unavailable: {}", exc)
        return {}


async def fetch_created_by_owner(
    selector: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generated tools/subagents attributed directly to the run via ``owner_run_id``.

    Replaces the prior fuzzy created_at-window join (which smeared attribution
    across concurrent/adjacent runs). ``owner_run_id`` is populated at persist
    time from the active run_id contextvar, so this is exact per-run attribution.
    """
    tool_q = select(
        ToolRegistration.tool_name,
        ToolRegistration.source_mutation_id,
        ToolRegistration.created_at,
    ).where(_endswith_clause(ToolRegistration.owner_run_id, selector))
    sub_q = select(
        SubAgentModel.name,
        SubAgentModel.source_mutation_id,
        SubAgentModel.created_at,
    ).where(_endswith_clause(SubAgentModel.owner_run_id, selector))
    async with get_session() as session:
        tools = (await session.execute(tool_q)).all()
        subs = (await session.execute(sub_q)).all()
    return (
        [
            {
                "source_mutation_id": t.source_mutation_id,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in tools
        ],
        [
            {
                "source_mutation_id": s.source_mutation_id,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in subs
        ],
    )


async def build_report(selector: str) -> RunMetricsReport:
    """Compose the full report for one endswith selector."""
    cost_rows, eval_rows, sub_rows, tool_rows, node_rows = await asyncio.gather(
        fetch_cost_rows(selector),
        fetch_eval_rows(selector),
        fetch_subagent_rows(selector),
        fetch_tool_metrics(selector),
        fetch_node_timing(selector),
    )

    matched = sorted({str(r["run_id"]) for r in cost_rows if r.get("run_id")})

    cost_summary = aggregate_cost(cost_rows) if cost_rows else None
    # Convergence inputs: declared check sets per spec (durable, from eval rows)
    # + best-effort terminal run statuses (Redis, may be empty for old runs).
    expected_checks = _expected_checks_for_eval(eval_rows)
    status_by_run = await _try_status_by_run(matched)
    score_summary = (
        score_goals(
            eval_rows,
            expected_checks=expected_checks or None,
            status_by_run=status_by_run or None,
        )
        if eval_rows
        else None
    )
    sub_summary = aggregate_subagents(sub_rows) if sub_rows else None
    span = llm_span(cost_rows)

    # Attribute created tools/subagents directly via owner_run_id (exact per-run).
    created_tools, created_subs = await fetch_created_by_owner(selector)
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
        node_timing=aggregate_node_timing(node_rows),
    )
    return report


# ─── rendering ────────────────────────────────────────────────────────────────


def report_to_dict(
    report: RunMetricsReport, *, require_converged: bool = False
) -> dict[str, Any]:
    """JSON-serializable form (dataclasses → dict; _created_dt stripped).

    ``require_converged`` adds a top-level ``headline_mean`` / ``headline_basis``
    so a downstream pipeline can read ONE canonical mean: the converged-filtered
    mean when the flag is set (and a filtered mean exists), else the unfiltered
    battery mean. Both means always remain in ``score`` for transparency.
    """
    d: dict[str, Any] = {
        "selector": report.selector,
        "matched_run_ids": report.matched_run_ids,
        "llm_span_seconds": report.llm_span_seconds,
        "score": asdict(report.score) if report.score else None,
        "cost": asdict(report.cost) if report.cost else None,
        "subagents": asdict(report.subagents) if report.subagents else None,
        "created": asdict(report.created),
        "global_tool_health": [asdict(r) for r in report.global_tool_health],
        "node_timing": asdict(report.node_timing) if report.node_timing else None,
    }
    if report.score:
        if require_converged and report.score.battery_mean_converged is not None:
            d["headline_mean"] = report.score.battery_mean_converged
            d["headline_basis"] = "converged"
        else:
            d["headline_mean"] = report.score.battery_mean
            d["headline_basis"] = "all"
    return d


def render_table(
    report: RunMetricsReport, *, require_converged: bool = False
) -> None:
    """Compact human-readable summary to stdout (secrets never appear here)."""
    print(f"\n═ run_metrics · selector={report.selector!r} "
          f"· {len(report.matched_run_ids)} run(s) matched"
          + ("  [HEADLINE=require-converged]" if require_converged else ""))
    if report.matched_run_ids:
        preview = ", ".join(report.matched_run_ids[:6])
        more = f" (+{len(report.matched_run_ids) - 6} more)" if len(report.matched_run_ids) > 6 else ""
        print(f"  runs: {preview}{more}")

    if report.score:
        print(f"\n  SCORE   battery_mean={report.score.battery_mean:.4f}  "
              f"goals_ran={report.score.n_goals_ran}")
        # Converged-filtered mean (experiment-design): excludes budget-exhausted
        # / never-finished goals so they aren't averaged in as artificial 0.0.
        # Reported ALONGSIDE the unfiltered headline — never a silent exclusion.
        if report.score.battery_mean_converged is not None:
            print(f"          battery_mean_converged={report.score.battery_mean_converged:.4f}  "
                  f"converged={report.score.n_goals_converged}/{report.score.n_goals_ran}")
            if report.score.excluded_goals:
                print(f"          excluded(non-converged): "
                      f"{', '.join(report.score.excluded_goals)}")
        for g in report.score.per_goal:
            flag = ""
            if not g.converged:
                flag = f"  [NON-CONVERGED: {g.non_convergence_reason}]"
            elif g.coverage is not None:
                flag = f"  [coverage={g.coverage:.2f}]"
            print(f"          {g.goal_id:<28} {g.score:.4f}  "
                  f"checks={g.n_checks}  ~verify_passes={g.verify_passes_estimate}{flag}")

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
        print("\n  TOOL HEALTH (per-run):")
        for t in top:
            print(f"          {t.tool_name:<28} calls={t.calls:<5} "
                  f"success={t.success_rate:.2%} empty={t.empty_output_count} "
                  f"lat={t.mean_latency_ms}")

    if report.node_timing:
        nt = report.node_timing
        top = sorted(nt.by_node, key=lambda n: -n.total_ms)[:10]
        print(f"\n  NODE TIMING  total={nt.total_ms}ms across {len(nt.by_node)} node(s)")
        for n in top:
            print(f"          {n.phase:<24} calls={n.calls:<4} "
                  f"total={n.total_ms}ms mean={n.mean_ms}ms")
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
    render_table(report, require_converged=args.require_converged)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                report_to_dict(report, require_converged=args.require_converged),
                indent=2,
                ensure_ascii=False,
            ),
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
    parser.add_argument(
        "--require-converged",
        action="store_true",
        help=(
            "Make the converged-filtered mean the HEADLINE (excludes "
            "budget-exhausted / never-finished goals). Both means are always "
            "reported; this only selects which one a pipeline reads as canonical."
        ),
    )
    parser.add_argument("--json", help="write the full report as JSON to this path")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
