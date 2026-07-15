"""Read-only data-access layer for the operator dashboard (Phase 5).

Every helper is best-effort: a DB/Redis hiccup is logged and degrades to an
empty result / zero — it NEVER raises into a request (the dashboard shows "no
data yet" rather than a 500). All access is read-only (SELECT / Redis reads)
over the existing tables and the Redis status hashes — no migrations, no
writes, no new schema.

Run status is Redis-only (``turing:run:{run_id}`` hashes, TTL-bounded — there is
no ``runs`` table), so the run list is a ``SCAN`` over that keyspace. Cost and
eval/mutation history live in Postgres. The two are joined in Python by the
``thread_id`` (``cost_ledger.run_id`` carries the graph ``thread_id`` =
``api-{run_id}`` for API-enqueued runs).

The generation-curve query computes a per-(run, goal) terminal-state mean
(latest row per ``check_name`` per run/goal, then averaged). It is a dashboard
quick-glance view; the CANONICAL terminal-state scorer (latest attempt → latest
row per check) lives in ``src/eval/curve.py`` + ``scripts/run_metrics.py`` and
remains the source of truth for the self-improvement verdict.
"""

from __future__ import annotations

import uuid
from typing import Any, TYPE_CHECKING

import sqlalchemy as sa
from loguru import logger

from src.db.models import (
    ABTestResult,
    AgentConfigVersion,
    CostLedger,
    ExecutionStep,
    Mutation,
    SubAgentModel,
    ToolCallMetric,
    ToolRegistration,
)
from src.graph.enums import MutationType
from src.worker.schema import RunStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.worker.status import RunStatusStore

# The status-hash keyspace (mirrors ``RunStatusStore._KEY_PREFIX``). SCANned to
# list runs — Redis has no list-runs primitive, but the keyspace is small and
# TTL-bounded (``WorkerSettings.status_ttl_s``) so a single SCAN is cheap.
_RUN_KEY_PREFIX = "turing:run:"
# Cap the run list + mutation timeline so a large history never renders a huge
# page (pagination is a fast-follow; the dashboard is a recent-activity view).
_DEFAULT_LIMIT = 100

# The web-search builtin's registered name (``src/tools/builtin/web_search.py``).
# ``tool_call_metrics`` — NOT ``execution_steps`` — is the per-invocation audit
# trail: the execute chokepoint records one row per tool call with ``run_id =
# get_active_run_id()`` (``execution_steps.tool_name`` is NULL — the node-timer
# logs phases only). That ``run_id`` is the SAME bare run_id bound by
# ``runner.run`` and written to ``owner_run_id``, so the attribution join below
# is a direct equality (no prefix stripping).
_WEB_SEARCH_TOOL = "web_search"
# Mutation-type roll-ups for the summary cards. The ``SUB_AGENT_*`` variants
# only *modify* existing agents (a created sub-agent never gets a ``mutations``
# row — see ``sub_agent_timeline``) but still count as prompt/tool mutations.
_PROMPT_TYPES = frozenset({MutationType.PROMPT.value, MutationType.SUB_AGENT_PROMPT.value})
_TOOL_TYPES = frozenset({MutationType.TOOL.value, MutationType.SUB_AGENT_TOOLS.value})


async def list_runs(
    status_store: RunStatusStore,
    *,
    limit: int = _DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Recent runs from Redis, newest-started first (empty on any failure).

    SCANs ``turing:run:*``, parses each hash to a ``RunStatus``, and returns the
    plain dicts the templates render. Sorted by ``started_at`` descending so an
    operator sees the latest activity first; runs lacking ``started_at`` sort to
    the end. Capped at ``limit``.
    """
    redis = status_store._redis  # noqa: SLF001 — read-only SCAN over the public keyspace
    runs: list[RunStatus] = []
    try:
        async for key in redis.scan_iter(match=f"{_RUN_KEY_PREFIX}*", count=200):
            mapping = await redis.hgetall(key)  # type: ignore[func-returns-value]
            if not mapping:
                continue
            try:
                runs.append(RunStatus.from_hash(mapping))  # type: ignore[arg-type]
            except Exception:
                # A malformed/expired hash mid-scan must not abort the listing.
                continue
    except Exception as e:  # noqa: BLE001 — best-effort; degrade to empty
        logger.warning(f"Dashboard: run listing failed: {e}")
        return []

    runs.sort(key=lambda r: r.started_at or "", reverse=True)
    return [_run_to_view(r) for r in runs[:limit]]


def _run_to_view(record: RunStatus) -> dict[str, Any]:
    """Flatten a ``RunStatus`` to the template-friendly dict shape."""
    return {
        "run_id": record.run_id,
        "thread_id": record.thread_id,
        "status": record.status.value,
        "final_output": record.final_output,
        "is_complete": record.is_complete,
        "iteration_count": record.iteration_count,
        "error": record.error,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "results_dir": record.results_dir,
    }


def _run_cost_keys(run_view: dict[str, Any]) -> list[str]:
    """Candidate ``cost_ledger.run_id`` values for a run (thread_id then bare id).

    ``cost_ledger.run_id`` carries the graph ``thread_id`` (``api-{run_id}`` for
    API runs, ``cli-{run_id}`` for CLI runs). The Redis status hash has both the
    bare ``run_id`` and the ``thread_id``, so match either.
    """
    keys: list[str] = []
    tid = run_view.get("thread_id")
    if tid:
        keys.append(str(tid))
    rid = run_view.get("run_id")
    if rid and rid != tid:
        keys.append(str(rid))
    return keys


async def runs_cost_index(session: AsyncSession) -> dict[str, dict[str, Any]]:
    """Aggregate ``cost_ledger`` per ``run_id`` → ``{run_id: {cost, calls, tokens}}``.

    Lets the run list join spend onto each Redis run by ``thread_id``/``run_id``
    without an N+1 (one grouped query, then an in-memory lookup per run). Empty
    dict on any failure (runs render with cost "—").
    """
    try:
        result = await session.execute(
            sa.select(
                CostLedger.run_id,
                sa.func.coalesce(sa.func.sum(CostLedger.cost_usd), 0.0).label("cost_usd"),
                sa.func.count(CostLedger.id).label("calls"),
                sa.func.coalesce(sa.func.sum(CostLedger.total_tokens), 0).label("total_tokens"),
            )
            .where(CostLedger.run_id.is_not(None))
            .group_by(CostLedger.run_id)
        )
        return {
            str(row.run_id): {
                "cost_usd": float(row.cost_usd),
                "calls": int(row.calls),
                "total_tokens": int(row.total_tokens),
            }
            for row in result.all()
        }
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"Dashboard: cost index failed: {e}")
        return {}


def _cost_for_run(
    run_view: dict[str, Any], cost_index: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Look up a run's spend from the precomputed index (zeros when unattributed)."""
    for key in _run_cost_keys(run_view):
        if key in cost_index:
            return cost_index[key]
    return {"cost_usd": 0.0, "calls": 0, "total_tokens": 0}


async def runs_with_cost(
    status_store: RunStatusStore, session: AsyncSession, *, limit: int = _DEFAULT_LIMIT
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recent runs with per-run spend + the index-page summary cards.

    One Redis SCAN + one grouped cost query + the in-memory join — no N+1.
    Returns ``(runs, summary)`` where each run dict carries its ``cost`` block.
    Best-effort throughout: a Redis OR DB failure degrades to an empty list with
    zeroed summary rather than raising.
    """
    runs = await list_runs(status_store, limit=limit)
    cost_index = await runs_cost_index(session)
    for r in runs:
        r["cost"] = _cost_for_run(r, cost_index)
    summary = await summary_cards(runs, cost_index)
    return runs, summary


async def run_cost_breakdown(session: AsyncSession, run_view: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-model spend for one run (its top cost drivers), most expensive first.

    Used in the run-detail view. Empty list on failure / when the run has no
    attributed cost rows.
    """
    keys = _run_cost_keys(run_view)
    if not keys:
        return []
    try:
        result = await session.execute(
            sa.select(
                CostLedger.model,
                sa.func.coalesce(sa.func.sum(CostLedger.cost_usd), 0.0).label("cost_usd"),
                sa.func.count(CostLedger.id).label("calls"),
                sa.func.coalesce(sa.func.sum(CostLedger.total_tokens), 0).label("total_tokens"),
            )
            .where(CostLedger.run_id.in_(keys))
            .group_by(CostLedger.model)
            .order_by(sa.func.sum(CostLedger.cost_usd).desc())
        )
        return [
            {
                "model": row.model,
                "cost_usd": float(row.cost_usd),
                "calls": int(row.calls),
                "total_tokens": int(row.total_tokens),
            }
            for row in result.all()
        ]
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"Dashboard: run cost breakdown failed: {e}")
        return []


async def execution_steps(
    session: AsyncSession, run_view: dict[str, Any], *, limit: int = 200
) -> list[dict[str, Any]]:
    """Per-node execution timeline for one run, chronological (oldest first).

    ``execution_steps`` is a per-node wall-clock log keyed by ``run_id`` (written
    by the graph node-timer in ``task_graph.py`` — one row per node invocation:
    ``phase`` node name, ``status``, ``duration_ms``, ``created_at``). The timer
    hard-codes ``step_number=0`` and leaves ``tool_name``/``tokens_used``/
    ``tool_output`` NULL, so a synthetic 1-based ``seq`` is derived from
    chronological order rather than the column. Empty list on any failure.

    Matches the run by the same ``_run_cost_keys`` candidates (thread_id then
    bare run_id) used by ``run_cost_breakdown`` so both ``api-{id}`` and bare ids
    resolve. Dashboard quick-glance only; the canonical per-node timing lives in
    ``run_metrics.py``.
    """
    keys = _run_cost_keys(run_view)
    if not keys:
        return []
    try:
        result = await session.execute(
            sa.select(
                ExecutionStep.phase,
                ExecutionStep.status,
                ExecutionStep.duration_ms,
                ExecutionStep.created_at,
            )
            .where(ExecutionStep.run_id.in_(keys))
            .order_by(ExecutionStep.created_at.asc(), ExecutionStep.id.asc())
            .limit(limit)
        )
        return [
            {
                "seq": i + 1,
                "phase": r.phase,
                "status": r.status,
                "duration_ms": int(r.duration_ms or 0),
                "duration_s": round(int(r.duration_ms or 0) / 1000, 1),
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for i, r in enumerate(result.all())
        ]
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"Dashboard: execution steps failed: {e}")
        return []


async def run_token_split(
    session: AsyncSession, run_view: dict[str, Any]
) -> dict[str, Any]:
    """Input/output/cached token split for one run (the token-amplification answer).

    An autonomous agent's overhead is overwhelmingly input-side (system prompt +
    tool registry + accumulated context are re-sent each turn), so surfacing the
    input/output/cached split answers "why so many tokens?" with data: the input
    share + what fraction of input was a prompt-cache hit. Matches the run by the
    same ``_run_cost_keys`` candidates as ``run_cost_breakdown``. Zeros on any
    failure / when the run has no attributed rows; ``cache_hit_pct`` is
    ``cached / input``.
    """
    keys = _run_cost_keys(run_view)
    if not keys:
        return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
                "total_tokens": 0, "input_pct": 0.0, "cache_hit_pct": 0.0}
    try:
        r = (await session.execute(
            sa.select(
                sa.func.coalesce(sa.func.sum(CostLedger.input_tokens), 0).label("input_tokens"),
                sa.func.coalesce(sa.func.sum(CostLedger.output_tokens), 0).label("output_tokens"),
                sa.func.coalesce(sa.func.sum(CostLedger.cached_tokens), 0).label("cached_tokens"),
                sa.func.coalesce(sa.func.sum(CostLedger.total_tokens), 0).label("total_tokens"),
            ).where(CostLedger.run_id.in_(keys))
        )).one()
        inp, out, cached, total = int(r.input_tokens), int(r.output_tokens), int(r.cached_tokens), int(r.total_tokens)
        produced = inp + out
        return {
            "input_tokens": inp,
            "output_tokens": out,
            "cached_tokens": cached,
            "total_tokens": total,
            "input_pct": round(100.0 * inp / produced, 1) if produced else 0.0,
            "cache_hit_pct": round(100.0 * cached / inp, 1) if inp else 0.0,
        }
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"Dashboard: run token split failed: {e}")
        return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
                "total_tokens": 0, "input_pct": 0.0, "cache_hit_pct": 0.0}


async def mutation_timeline(
    session: AsyncSession, *, limit: int = _DEFAULT_LIMIT
) -> list[dict[str, Any]]:
    """Recent mutations with their latest A/B stats, newest first.

    One row per mutation (joined 1:1 to its most-recent ``ABTestResult`` so the
    Phase-4 p-value / significance / confidence surface alongside the diff).
    ``has_diff`` flags whether Phase-3 ``diff_content`` was captured. Empty list
    on any failure.
    """
    try:
        # Most-recent ABTestResult per mutation (the engine writes one per A/B).
        latest_ab = (
            sa.select(
                ABTestResult.mutation_id.label("ab_mutation_id"),
                ABTestResult.p_value,
                ABTestResult.is_significant,
                ABTestResult.confidence,
                ABTestResult.sample_size,
                ABTestResult.control_value,
                ABTestResult.treatment_value,
            )
            .order_by(ABTestResult.mutation_id, ABTestResult.created_at.desc())
            .distinct(ABTestResult.mutation_id)
            .subquery()
        )
        stmt = (
            sa.select(
                Mutation.id,
                Mutation.mutation_type,
                Mutation.target_path,
                Mutation.description,
                Mutation.status,
                Mutation.diff_content,
                Mutation.model_used,
                Mutation.created_at,
                latest_ab.c.p_value,
                latest_ab.c.is_significant,
                latest_ab.c.confidence,
                latest_ab.c.sample_size,
                latest_ab.c.control_value,
                latest_ab.c.treatment_value,
            )
            .outerjoin(latest_ab, latest_ab.c.ab_mutation_id == Mutation.id)
            .order_by(Mutation.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows: list[dict[str, Any]] = []
        for r in result.all():
            rows.append(
                {
                    "id": str(r.id),
                    "mutation_type": r.mutation_type,
                    "target_path": r.target_path,
                    "description": (r.description or "")[:200],
                    "status": r.status,
                    "has_diff": bool(r.diff_content),
                    "diff_content": r.diff_content,
                    "model_used": r.model_used,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                    "p_value": float(r.p_value) if r.p_value is not None else None,
                    "is_significant": r.is_significant,
                    "confidence": float(r.confidence) if r.confidence is not None else None,
                    "sample_size": r.sample_size,
                    "control_value": float(r.control_value) if r.control_value is not None else None,
                    "treatment_value": float(r.treatment_value) if r.treatment_value is not None else None,
                }
            )
        return rows
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"Dashboard: mutation timeline failed: {e}")
        return []


async def sub_agent_timeline(
    session: AsyncSession, *, limit: int = _DEFAULT_LIMIT
) -> list[dict[str, Any]]:
    """Recent sub-agent definitions, newest first.

    Sub-agent *creation* writes to ``sub_agent_definitions`` (NOT the ``mutations``
    table — the ``SUB_AGENT_*`` mutation types only *modify* existing agents), so
    the mutation timeline never shows a spawned sub-agent. This surfaces them:
    name, model tier, rolling metrics, the run that spawned it
    (``owner_run_id``), and whether it is live (``is_active``). Empty list on any
    failure.
    """
    try:
        result = await session.execute(
            sa.select(
                SubAgentModel.id,
                SubAgentModel.name,
                SubAgentModel.description,
                SubAgentModel.model_tier,
                SubAgentModel.is_active,
                SubAgentModel.source_mutation_id,
                SubAgentModel.owner_run_id,
                SubAgentModel.total_runs,
                SubAgentModel.success_rate,
                SubAgentModel.quality_score,
                SubAgentModel.created_at,
            )
            .order_by(SubAgentModel.created_at.desc())
            .limit(limit)
        )
        rows: list[dict[str, Any]] = []
        for r in result.all():
            rows.append(
                {
                    "id": str(r.id),
                    "name": r.name,
                    "description": (r.description or "")[:160],
                    "model_tier": r.model_tier,
                    "is_active": bool(r.is_active),
                    "source_mutation_id": str(r.source_mutation_id) if r.source_mutation_id else None,
                    "owner_run_id": r.owner_run_id,
                    "total_runs": int(r.total_runs or 0),
                    "success_rate": round(float(r.success_rate or 0.0), 3),
                    "quality_score": round(float(r.quality_score or 0.0), 3),
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                }
            )
        return rows
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"Dashboard: sub-agent timeline failed: {e}")
        return []


async def mutation_counts(session: AsyncSession) -> dict[str, Any]:
    """Mutation counts by type + how many capabilities are LIVE.

    ``by_type`` maps each raw ``mutation_type`` value (prompt/tool/code/…) to its
    row count; ``prompts_mutated``/``tools_mutated`` roll up the ``SUB_AGENT_*``
    variants for clean summary cards. ``deployed_tools``/``active_subagents``/
    ``active_configs`` count what actually reached production — channel-A tools
    active in the registry, active sub-agents, and the one active config version.
    Each piece is independently best-effort (a single failed count does not zero
    the rest). Zeros on any failure.
    """
    by_type: dict[str, int] = {}
    try:
        rows = (await session.execute(
            sa.select(
                Mutation.mutation_type.label("mtype"),
                sa.func.count(Mutation.id).label("n"),
            ).group_by(Mutation.mutation_type)
        )).all()
        by_type = {str(r.mtype): int(r.n) for r in rows}
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"Dashboard: mutation counts failed: {e}")

    deployed_tools = await _scalar_count(
        session, sa.select(sa.func.count(ToolRegistration.id)).where(ToolRegistration.is_active.is_(True)),
        "deployed-tool count",
    )
    active_subagents = await _scalar_count(
        session, sa.select(sa.func.count(SubAgentModel.id)).where(SubAgentModel.is_active.is_(True)),
        "active-subagent count",
    )
    active_configs = await _scalar_count(
        session, sa.select(sa.func.count(AgentConfigVersion.id)).where(AgentConfigVersion.is_active.is_(True)),
        "active-config count",
    )
    return {
        "by_type": by_type,
        "prompts_mutated": sum(by_type.get(t, 0) for t in _PROMPT_TYPES),
        "tools_mutated": sum(by_type.get(t, 0) for t in _TOOL_TYPES),
        "total_mutations": sum(by_type.values()),
        "deployed_tools": deployed_tools,
        "active_subagents": active_subagents,
        "active_configs": active_configs,
    }


async def _scalar_count(session: AsyncSession, stmt: Any, label: str) -> int:
    """Run a count SELECT, returning 0 + a warning on any failure."""
    try:
        return int((await session.execute(stmt)).scalar() or 0)
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"Dashboard: {label} failed: {e}")
        return 0


async def evolution_summary(
    session: AsyncSession, counts: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Did the application actually evolve? The "is the app different now?" answer.

    Combines the two evolution channels: **channel-B** live PROMPT promotions
    (``PromotionGate.active_promotions`` reads ``current.json`` from the
    ``evolved_handlers_dir`` — the same ``turing-workspace`` volume the api mounts,
    so the dashboard sees what the worker promoted) and **channel-A** live counts
    (deployed tools / active sub-agents). ``any_evolved`` is the single yes/no.
    Best-effort: a promotion-read failure degrades to the channel-A counts alone,
    NEVER raises (observability-only, mirrors the cost-ledger-resilience pattern).

    Pass the already-computed ``mutation_counts`` dict to avoid re-querying it.
    """
    live: list[dict[str, Any]] = []
    try:
        from src.evolution.promote import PromotionGate

        live = list(PromotionGate().active_promotions())
    except Exception as e:  # noqa: BLE001 — observability-only
        logger.warning(f"Dashboard: live promotions read failed: {e}")
        live = []
    base = counts if counts is not None else await mutation_counts(session)
    total_live = len(live)
    return {
        "live_prompt_promotions": live,
        "total_live_promotions": total_live,
        "deployed_tools": int(base.get("deployed_tools", 0)),
        "active_subagents": int(base.get("active_subagents", 0)),
        "active_configs": int(base.get("active_configs", 0)),
        "any_evolved": bool(total_live or base.get("deployed_tools") or base.get("active_subagents")),
    }


async def web_search_summary(session: AsyncSession) -> dict[str, Any]:
    """Page-level web-search usage: total ``web_search`` calls + distinct runs.

    ``tool_call_metrics`` (not ``execution_steps`` — its ``tool_name`` is NULL) is
    the per-invocation audit trail. Zeros on any failure or when metrics
    recording is off (``TOOL_METRICS_ENABLED``, default on).
    """
    try:
        r = (await session.execute(
            sa.select(
                sa.func.count(ToolCallMetric.id).label("calls"),
                sa.func.count(ToolCallMetric.run_id.distinct()).label("runs"),
            ).where(ToolCallMetric.tool_name == _WEB_SEARCH_TOOL)
        )).one()
        return {"total_calls": int(r.calls or 0), "runs_using_search": int(r.runs or 0)}
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"Dashboard: web-search summary failed: {e}")
        return {"total_calls": 0, "runs_using_search": 0}


async def web_search_runs(session: AsyncSession, run_ids: list[str | None]) -> set[str]:
    """Which of the given owner-run-ids invoked ``web_search`` (one grouped query).

    Shared by ``mutations_web_search`` and the sub-agent badge: the run that
    spawned a tool/sub-agent (``owner_run_id``) is the SAME bare run_id recorded
    on ``tool_call_metrics.run_id`` (both come from ``get_active_run_id``), so the
    match is a direct equality. Empty set on any failure.
    """
    norm = [str(r) for r in run_ids if r]
    if not norm:
        return set()
    try:
        rows = (await session.execute(
            sa.select(ToolCallMetric.run_id)
            .where(
                ToolCallMetric.tool_name == _WEB_SEARCH_TOOL,
                ToolCallMetric.run_id.in_(norm),
            )
            .distinct()
        )).scalars().all()
        return {str(r) for r in rows if r is not None}
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"Dashboard: web-search run lookup failed: {e}")
        return set()


async def mutations_web_search(
    session: AsyncSession, mutation_ids: list[str]
) -> dict[str, bool]:
    """Per-mutation web-search badge: did the run that produced this mutation
    invoke ``web_search``?

    Only TOOL/SUB_AGENT_* mutations carry an ``owner_run_id`` (resolved via
    ``tool_registrations``/``sub_agent_definitions`` ``source_mutation_id``);
    PROMPT/CODE mutations have no run link and are absent from the result (the
    template renders "—" for them — no false signal). Three queries total
    (tool→run, sub-agent→run, then the web-search set) — constant, never N+1
    regardless of row count.
    """
    norm = [str(m) for m in mutation_ids if m]
    if not norm:
        return {}
    try:
        uuids = [uuid.UUID(m) for m in norm]  # ids are str(UUID); safe
    except ValueError as e:  # a non-UUID id should not occur, but never 500
        logger.warning(f"Dashboard: non-UUID mutation id in web-search attribution: {e}")
        return {}
    mid_to_run: dict[str, str] = {}
    for model in (ToolRegistration, SubAgentModel):
        try:
            rows = (await session.execute(
                sa.select(model.source_mutation_id, model.owner_run_id).where(
                    model.source_mutation_id.in_(uuids),
                    model.owner_run_id.is_not(None),
                )
            )).all()
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning(f"Dashboard: mutation→run resolution failed: {e}")
            continue
        for smid, orid in rows:
            if smid is not None and orid is not None:
                mid_to_run[str(smid)] = orid
    if not mid_to_run:
        return {}
    ws_runs = await web_search_runs(session, list(set(mid_to_run.values())))
    return {mid: (run in ws_runs) for mid, run in mid_to_run.items()}


# Per-(run, goal) terminal-state mean. A SQL constant (no user input) using
# PostgreSQL ``DISTINCT ON`` to keep the latest row per check, then averaging.
# Read-only; the suffix filter is applied in Python on the result set, so no
# caller value is ever interpolated into the SQL.
_CURVE_SQL = sa.text(
    """
    SELECT run_id, goal_id, AVG(score) AS mean_score, COUNT(*) AS n_checks
    FROM (
        SELECT DISTINCT ON (run_id, goal_id, check_name)
               run_id, goal_id, check_name, score
        FROM eval_results
        ORDER BY run_id, goal_id, check_name, created_at DESC
    ) latest_per_check
    GROUP BY run_id, goal_id
    ORDER BY run_id, goal_id
    """
)


async def generation_curve(
    session: AsyncSession,
    *,
    suffix: str | None = None,
    limit_runs: int = 20,
) -> dict[str, Any]:
    """Per-run × per-goal terminal-state mean matrix for the curve view.

    Returns ``{"runs": [run_id...], "goals": [goal_id...], "matrix": {run_id:
    {goal_id: {"mean": float, "n": int}}}, "run_means": {run_id: float},
    "goal_means": {goal_id: float}}``. ``suffix`` (e.g. a run_id substring like
    a date ``20260713`` or a gen tag) filters which runs appear as columns;
    ``None`` shows the most recent ``limit_runs`` runs by max(created_at).

    Dashboard quick-glance only — the canonical scorer is ``run_metrics.py``.
    """
    try:
        result = await session.execute(_CURVE_SQL)
        all_rows = result.all()
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(f"Dashboard: generation curve failed: {e}")
        return {"runs": [], "goals": [], "matrix": {}, "run_means": {}, "goal_means": {}}

    # Apply the suffix filter (Python-side; the SQL carries no caller input).
    filtered = [
        r for r in all_rows if (suffix is None or suffix in (r.run_id or ""))
    ]
    if not filtered:
        return {"runs": [], "goals": [], "matrix": {}, "run_means": {}, "goal_means": {}}

    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    goal_set: set[str] = set()
    for r in filtered:
        run_id = str(r.run_id)
        goal_id = str(r.goal_id)
        mean = float(r.mean_score) if r.mean_score is not None else 0.0
        matrix.setdefault(run_id, {})[goal_id] = {"mean": mean, "n": int(r.n_checks or 0)}
        goal_set.add(goal_id)

    # Column order: most runs first is arbitrary in SQL — order by run_id desc
    # so the latest generations land leftmost after we cap to limit_runs.
    run_ids = sorted(matrix.keys(), reverse=True)[:limit_runs]
    goals = sorted(goal_set)

    run_means: dict[str, float] = {}
    for rid in run_ids:
        scores = [cell["mean"] for cell in matrix[rid].values()]
        run_means[rid] = round(sum(scores) / len(scores), 4) if scores else 0.0
    goal_means: dict[str, float] = {}
    for g in goals:
        scores = [matrix[rid][g]["mean"] for rid in run_ids if g in matrix[rid]]
        goal_means[g] = round(sum(scores) / len(scores), 4) if scores else 0.0

    return {
        "runs": run_ids,
        "goals": goals,
        "matrix": matrix,
        "run_means": run_means,
        "goal_means": goal_means,
    }


async def summary_cards(
    runs: list[dict[str, Any]],
    cost_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Top-of-page counts for the index view (no DB hit — pure Python over the
    already-fetched run list + cost index)."""
    in_flight = sum(
        1 for r in runs if r.get("status") in ("queued", "running")
    )
    completed = sum(1 for r in runs if r.get("is_complete"))
    # Total attributed spend across every known run.
    total_cost = round(sum(c["cost_usd"] for c in cost_index.values()), 4)
    return {
        "runs_total": len(runs),
        "runs_in_flight": in_flight,
        "runs_completed": completed,
        "total_cost_usd": total_cost,
    }
