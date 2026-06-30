"""Aggregate READ path over the ``cost_ledger`` (``CostTracker`` is write-only).

Reports per-run / per-model spend breakdowns. The script ``scripts/cost_query.py``
encoded this aggregate-read logic — promoted here so it ships with the app and is
reachable via ``python main.py --cost`` (the API has no auth, so the spend report
lives behind a CLI flag, never an unauthenticated route).

Layering:
  - ``build_cost_filter`` — pure: builds ORM predicates + scope labels (no SQL
    string interpolation; every value is a bound parameter via the ORM).
  - ``cost_breakdown`` — executes the ORM aggregates against an injected session
    (DI seam, like ``src/db/backfills.run_backfill``) and returns a structured
    ``CostBreakdown`` (NOT a print) so callers can render or assert on it.
  - ``format_cost_breakdown`` — pure: renders the breakdown to a report string.

No connection string or password is ever printed, hardcoded, or read from
``os.environ`` here — the session comes from ``src.db.session.get_session``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import CostLedger


@dataclass(frozen=True)
class SpendRow:
    """One aggregate row.

    ``primary`` is the run_id (or ``(none)``) or the model name; ``secondary``
    is the model for a run×model row, else ``""`` for a single-dimension row or
    a run subtotal.
    """

    primary: str
    secondary: str
    calls: int
    spend: float
    tokens: int


@dataclass(frozen=True)
class CostBreakdown:
    scope: str
    matched: bool
    total_calls: int
    total_spend: float
    total_tokens: int
    is_by_model: bool
    detail: tuple[SpendRow, ...] = field(default_factory=tuple)
    by_run: tuple[SpendRow, ...] = field(default_factory=tuple)


def _utc_today_window() -> tuple[dt.datetime, dt.datetime]:
    """Half-open UTC day window [today 00:00, tomorrow 00:00)."""
    today = dt.datetime.now(dt.timezone.utc).date()
    start = dt.datetime.combine(today, dt.time.min, tzinfo=dt.timezone.utc)
    return start, start + dt.timedelta(days=1)


def _parse_since(value: str) -> dt.datetime:
    """Parse a ``YYYY-MM-DD`` (or full ISO) into a UTC-aware datetime.

    A naive result (date-only input) is pinned to UTC midnight so the filter is
    timezone-consistent with ``created_at`` (``DateTime(timezone=True)``).
    """
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def build_cost_filter(
    *,
    run_id: str = "",
    model: str = "",
    today: bool = False,
    since: str = "",
) -> tuple[list[sa.ColumnElement[bool]], list[str]]:
    """Build the ORM WHERE predicates (AND) + human-readable scope labels.

    Every filter value flows through the ORM as a bound parameter — none are
    interpolated into SQL, so there is no injection surface. ``--today`` and
    ``--since`` are mutually exclusive at the CLI layer; both set here would
    AND together (today AND since-date), which the handler forbids upstream.
    """
    preds: list[sa.ColumnElement[bool]] = []
    labels: list[str] = []
    if run_id:
        preds.append(CostLedger.run_id == run_id)
        labels.append(f"run_id={run_id}")
    if model:
        preds.append(CostLedger.model == model)
        labels.append(f"model={model}")
    if today:
        start, end = _utc_today_window()
        preds.append(CostLedger.created_at >= start)
        preds.append(CostLedger.created_at < end)
        labels.append("today")
    if since:
        preds.append(CostLedger.created_at >= _parse_since(since))
        labels.append(f"since={since}")
    return preds, labels


def _scope(labels: list[str]) -> str:
    return ("[" + ", ".join(labels) + "]") if labels else "[all-time]"


def _total_select(preds: list[sa.ColumnElement[bool]]) -> sa.Select:
    sel = sa.select(
        sa.func.count().label("calls"),
        sa.func.coalesce(sa.func.sum(CostLedger.cost_usd), 0.0).label("spend"),
        sa.func.coalesce(sa.func.sum(CostLedger.total_tokens), 0).label("tok"),
    )
    if preds:
        sel = sel.where(sa.and_(*preds))
    return sel


async def cost_breakdown(
    *,
    run_id: str = "",
    model: str = "",
    by_model: bool = False,
    today: bool = False,
    since: str = "",
    session: AsyncSession,
) -> CostBreakdown:
    """Run the aggregate spend queries against ``session`` and return a breakdown.

    ``by_model=True`` collapses across runs (one row per model); otherwise the
    detail is per run×model plus run subtotals (sorted by spend desc). A filter
    matching zero rows returns ``matched=False`` (the caller renders a notice).
    """
    preds, labels = build_cost_filter(
        run_id=run_id, model=model, today=today, since=since
    )
    scope = _scope(labels)

    tot = (await session.execute(_total_select(preds))).one()
    total_calls = int(tot.calls)
    total_spend = float(tot.spend)
    total_tokens = int(tot.tok)
    if total_calls == 0:
        return CostBreakdown(
            scope=scope,
            matched=False,
            total_calls=0,
            total_spend=0.0,
            total_tokens=0,
            is_by_model=by_model,
        )

    if by_model:
        sel = (
            sa.select(
                CostLedger.model,
                sa.func.count(),
                sa.func.coalesce(sa.func.sum(CostLedger.cost_usd), 0.0),
                sa.func.coalesce(sa.func.sum(CostLedger.total_tokens), 0),
            )
            .group_by(CostLedger.model)
            .order_by(sa.func.sum(CostLedger.cost_usd).desc())
        )
        if preds:
            sel = sel.where(sa.and_(*preds))
        rows = (await session.execute(sel)).all()
        detail = tuple(
            SpendRow(
                primary=str(r[0]),
                secondary="",
                calls=int(r[1]),
                spend=float(r[2]),
                tokens=int(r[3]),
            )
            for r in rows
        )
        return CostBreakdown(
            scope=scope,
            matched=True,
            total_calls=total_calls,
            total_spend=total_spend,
            total_tokens=total_tokens,
            is_by_model=True,
            detail=detail,
        )

    # by run × model (+ run subtotals)
    sel = (
        sa.select(
            sa.func.coalesce(sa.func.nullif(CostLedger.run_id, ""), "(none)"),
            CostLedger.model,
            sa.func.count(),
            sa.func.coalesce(sa.func.sum(CostLedger.cost_usd), 0.0),
            sa.func.coalesce(sa.func.sum(CostLedger.total_tokens), 0),
        )
        .group_by(CostLedger.run_id, CostLedger.model)
        .order_by(CostLedger.run_id, sa.func.sum(CostLedger.cost_usd).desc())
    )
    if preds:
        sel = sel.where(sa.and_(*preds))
    rows = (await session.execute(sel)).all()
    detail = tuple(
        SpendRow(
            primary=str(r[0]),
            secondary=str(r[1]),
            calls=int(r[2]),
            spend=float(r[3]),
            tokens=int(r[4]),
        )
        for r in rows
    )

    subtotals: dict[str, list[float]] = {}
    for r in rows:
        bucket = subtotals.setdefault(str(r[0]), [0.0, 0.0, 0.0])
        bucket[0] += int(r[2])
        bucket[1] += float(r[3])
        bucket[2] += int(r[4])
    by_run = tuple(
        SpendRow(
            primary=run,
            secondary="",
            calls=int(b[0]),
            spend=b[1],
            tokens=int(b[2]),
        )
        for run, b in sorted(subtotals.items(), key=lambda kv: kv[1][1], reverse=True)
    )
    return CostBreakdown(
        scope=scope,
        matched=True,
        total_calls=total_calls,
        total_spend=total_spend,
        total_tokens=total_tokens,
        is_by_model=False,
        detail=detail,
        by_run=by_run,
    )


def format_cost_breakdown(bd: CostBreakdown) -> str:
    """Render a ``CostBreakdown`` to the report string (pure, no I/O)."""
    if not bd.matched:
        return f"cost_ledger{bd.scope}: no rows matched."

    lines: list[str] = [f"cost_ledger{bd.scope}"]
    lines.append(
        f"  TOTAL: ${bd.total_spend:.4f} | {bd.total_calls} calls | "
        f"{bd.total_tokens:,} tokens\n"
    )

    if bd.is_by_model:
        lines.append("  by model:")
        for row in bd.detail:
            lines.append(
                f"    {row.primary:28s} ${row.spend:8.4f} | "
                f"{row.calls:5d} calls | {row.tokens:>12,} tokens"
            )
        return "\n".join(lines)

    lines.append("  by run:")
    for row in bd.by_run:
        lines.append(
            f"    {row.primary:24s} ${row.spend:8.4f} | "
            f"{row.calls:5d} calls | {row.tokens:>12,} tokens"
        )

    lines.append("\n  by run × model:")
    for row in bd.detail:
        lines.append(
            f"    {row.primary:18s} {row.secondary:22s} ${row.spend:8.4f} | "
            f"{row.calls:4d} calls | {row.tokens:>12,} tokens"
        )
    return "\n".join(lines)
