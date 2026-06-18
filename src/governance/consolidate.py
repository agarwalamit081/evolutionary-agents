"""Offline capability consolidation (B3): cluster, merge, retire, re-point.

A one-shot governance pass over the capability catalog: clusters active
tools/sub-agents by their capability embedding (greedy pairwise cosine), keeps
the best-scoring member of each redundant cluster, retires the rest
(``is_active=False``), and re-points ``tool_subset`` references that named a
retired tool. Defaults to **dry-run** — a first run only reports the planned
merges without mutating anything.

Reuses M1/M2 primitives: the persisters' capability-row fetchers and
``retire``/``merge_alias``. This rewrites only the capability catalog (which
tools/agents are active) — it does not mutate handler code or promote anything
to production.

Run directly::

    python -m src.governance.consolidate --dry-run --threshold 0.92 --target all
"""

from __future__ import annotations

import argparse
import asyncio
import math
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class MergePlan:
    """One consolidation decision: keep ``target``, retire the rest."""

    target: str
    retired: list[str] = field(default_factory=list)
    similarity: float = 1.0


@dataclass
class ConsolidationReport:
    """Aggregate result of a consolidation pass."""

    tools: list[MergePlan] = field(default_factory=list)
    agents: list[MergePlan] = field(default_factory=list)
    # M4: tools retired for chronic low performance (no surviving twin — these
    # are not redundancy merges, so they are tracked separately from ``tools``).
    performance_retired: list[str] = field(default_factory=list)
    dry_run: bool = True

    @property
    def total_retired(self) -> int:
        return sum(len(p.retired) for p in (*self.tools, *self.agents)) + len(
            self.performance_retired
        )


def _cosine(u: list[float], v: list[float]) -> float:
    """Cosine similarity; 0.0 for zero-norm vectors (768-d embeddings)."""
    dot = nu = nv = 0.0
    for a, b in zip(u, v, strict=True):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def _plan_clusters(
    rows: list[dict[str, Any]],
    threshold: float,
) -> list[MergePlan]:
    """Cluster capability rows by cosine; keep the top scorer per cluster.

    Greedy over score-desc: the highest-scoring row of a redundant group
    becomes its survivor (``target``); every later row within ``threshold``
    cosine of an existing survivor is folded into that cluster (retired). A row
    with no embedding cannot be clustered and survives on its own. Returns one
    :class:`MergePlan` per cluster that retires at least one capability.
    """
    scored = sorted(rows, key=lambda r: r["score"], reverse=True)
    survivors: list[tuple[str, list[float]]] = []  # (name, embedding)
    plans: list[MergePlan] = []
    plan_by_survivor: dict[str, MergePlan] = {}

    for row in scored:
        emb = row.get("embedding")
        if emb is None:
            continue
        match_plan: MergePlan | None = None
        best_sim = threshold
        for surv_name, surv_emb in survivors:
            if surv_emb is None:
                continue
            sim = _cosine(emb, surv_emb)
            if sim >= best_sim:
                best_sim = sim
                match_plan = plan_by_survivor[surv_name]
        if match_plan is not None:
            match_plan.retired.append(row["name"])
            match_plan.similarity = max(match_plan.similarity, best_sim)
        else:
            plan = MergePlan(target=row["name"])
            plans.append(plan)
            plan_by_survivor[row["name"]] = plan
            survivors.append((row["name"], emb))

    return [p for p in plans if p.retired]


async def consolidate_tools(
    threshold: float = 0.92,
    dry_run: bool = True,
    persister: Any | None = None,
) -> ConsolidationReport:
    """Consolidate redundant active tools; re-point ``tool_subset`` refs.

    Args:
        threshold: Minimum cosine similarity to treat two tools as duplicates
            (the stricter consolidation cutoff, distinct from creation-time
            ``capability_dedup_threshold``).
        dry_run: When True (default), only report — no retire/merge_alias calls.
        persister: Optional ``ToolPersister`` (dependency injection for tests).

    Returns:
        A :class:`ConsolidationReport` whose ``tools`` lists every merge.
    """
    from src.tools.dynamic.persister import ToolPersister

    p = persister or ToolPersister()
    try:
        rows = await p._active_tool_capability_rows()  # noqa: SLF001
    except Exception as e:
        logger.error(f"consolidate_tools: capability fetch failed: {e}")
        return ConsolidationReport(dry_run=dry_run)

    plans = _plan_clusters(rows, threshold)
    if not dry_run:
        for plan in plans:
            await p.retire(plan.retired)
            for loser in plan.retired:
                await p.merge_alias(loser, plan.target)

    report = ConsolidationReport(tools=plans, dry_run=dry_run)
    _log_report("tools", report)
    return report


async def consolidate_sub_agents(
    threshold: float = 0.92,
    dry_run: bool = True,
    persister: Any | None = None,
) -> ConsolidationReport:
    """Consolidate redundant active sub-agents (retire lower performers).

    Sub-agent names are not embedded in any persisted JSONB reference (delegation
    resolves them by name from the live registry), so there is no re-point step
    here — only retire the losers. The winner survives and is reloaded next run.

    Args:
        threshold: Minimum cosine similarity to treat two agents as duplicates.
        dry_run: When True (default), only report.
        persister: Optional ``SubAgentPersister`` (dependency injection).

    Returns:
        A :class:`ConsolidationReport` whose ``agents`` lists every merge.
    """
    from src.agents.persister import SubAgentPersister

    p = persister or SubAgentPersister()
    try:
        rows = await p._active_capability_rows()  # noqa: SLF001
    except Exception as e:
        logger.error(f"consolidate_sub_agents: capability fetch failed: {e}")
        return ConsolidationReport(dry_run=dry_run)

    plans = _plan_clusters(rows, threshold)
    if not dry_run:
        for plan in plans:
            await p.retire(plan.retired)

    report = ConsolidationReport(agents=plans, dry_run=dry_run)
    _log_report("sub-agents", report)
    return report


async def consolidate_performance(
    min_runs: int = 20,
    success_floor: float = 0.25,
    dry_run: bool = True,
    persister: Any | None = None,
) -> ConsolidationReport:
    """Retire (or report) chronic low-performing active tools (M4).

    Distinct from :func:`consolidate_tools`: that removes *redundant*
    capabilities (a semantic duplicate of a better twin); this removes tools
    that succeed too rarely after enough calls — there is no survivor to keep,
    so the retiree list is reported via ``performance_retired`` (not a
    :class:`MergePlan`).

    Args:
        min_runs: Minimum ``calls`` before a tool is eligible.
        success_floor: Retire tools with ``success_rate`` below this.
        dry_run: When True (default), only report.
        persister: Optional ``ToolPersister`` (dependency injection for tests).

    Returns:
        A :class:`ConsolidationReport` whose ``performance_retired`` lists names.
    """
    from src.tools.dynamic.persister import ToolPersister

    p = persister or ToolPersister()
    try:
        names = await p.underperforming_tools(min_runs, success_floor)
    except Exception as e:
        logger.error(f"consolidate_performance: underperformer scan failed: {e}")
        return ConsolidationReport(dry_run=dry_run)

    if not dry_run and names:
        await p.retire(names)

    report = ConsolidationReport(performance_retired=list(names), dry_run=dry_run)
    verb = "would retire" if dry_run else "retired"
    for n in names:
        logger.info(
            f"[consolidate/perf] {verb} '{n}' "
            f"(success_rate < {success_floor} over >= {min_runs} calls)"
        )
    if names:
        logger.info(
            f"[consolidate/perf] {len(names)} underperformer(s) "
            f"({'dry-run' if dry_run else 'applied'})"
        )
    return report


async def consolidate_all(
    threshold: float = 0.92,
    dry_run: bool = True,
    min_runs: int = 20,
    success_floor: float = 0.25,
) -> ConsolidationReport:
    """Consolidate redundant tools + sub-agents, then retire underperformers."""
    tools = await consolidate_tools(threshold=threshold, dry_run=dry_run)
    agents = await consolidate_sub_agents(threshold=threshold, dry_run=dry_run)
    perf = await consolidate_performance(
        min_runs=min_runs, success_floor=success_floor, dry_run=dry_run
    )
    return ConsolidationReport(
        tools=tools.tools,
        agents=agents.agents,
        performance_retired=perf.performance_retired,
        dry_run=dry_run,
    )


def _log_report(label: str, report: ConsolidationReport) -> None:
    plans = report.tools if label == "tools" else report.agents
    verb = "would retire" if report.dry_run else "retired"
    for plan in plans:
        logger.info(
            f"[consolidate/{label}] keep '{plan.target}', {verb} "
            f"{plan.retired} (max sim {plan.similarity:.3f})"
        )
    if plans:
        logger.info(
            f"[consolidate/{label}] {len(plans)} merge(s), "
            f"{sum(len(p.retired) for p in plans)} retiree(s) "
            f"({'dry-run' if report.dry_run else 'applied'})"
        )


def _format_report(report: ConsolidationReport) -> str:
    lines: list[str] = []
    mode = "DRY-RUN" if report.dry_run else "APPLIED"
    lines.append(f"Consolidation ({mode}): {report.total_retired} retiree(s)")
    for plan in (*report.tools, *report.agents):
        lines.append(
            f"  keep '{plan.target}' <- retire {plan.retired} "
            f"(sim {plan.similarity:.3f})"
        )
    if report.performance_retired:
        lines.append(
            f"  retire (performance) {report.performance_retired}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``--threshold --dry-run/--no-dry-run --target {tools,agents,all}``."""
    parser = argparse.ArgumentParser(
        description="Consolidate redundant capabilities (B3)."
    )
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report only (default); use --no-dry-run to apply.",
    )
    parser.add_argument(
        "--target",
        choices=["tools", "agents", "all"],
        default="all",
    )
    args = parser.parse_args(argv)

    if args.target == "tools":
        report = asyncio.run(
            consolidate_tools(args.threshold, args.dry_run)
        )
    elif args.target == "agents":
        report = asyncio.run(
            consolidate_sub_agents(args.threshold, args.dry_run)
        )
    else:
        report = asyncio.run(consolidate_all(args.threshold, args.dry_run))

    print(_format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
