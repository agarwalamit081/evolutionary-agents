"""SI-5 — ``--cost`` spend-breakdown (``src.llm.cost_queries``).

Promoted from ``scripts/cost_query.py``: the aggregate READ path over
``cost_ledger`` (``CostTracker`` is write-only). Rewritten as ORM aggregates
(``sa.select(sa.func.coalesce(sa.func.sum(...)))``) to match the codebase
pattern (not the script's raw ``text()``).

Layered for testability: the pure helpers (filter builder + formatter) are
unit-tested directly; ``cost_breakdown`` is exercised through a mock async
session (no DB) — the DI seam (``session`` injected) makes that possible.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from src.llm.cost_queries import (
    CostBreakdown,
    SpendRow,
    _parse_since,
    build_cost_filter,
    cost_breakdown,
    format_cost_breakdown,
)


class _MockSession:
    """Fake AsyncSession: 1st execute() → total (.one()), 2nd → detail (.all())."""

    def __init__(self, total: SimpleNamespace, detail: list[tuple]) -> None:
        self._total = total
        self._detail = detail
        self._n = 0

    async def execute(self, stmt: object) -> SimpleNamespace:
        del stmt  # the exact SQL is not asserted here (that is the ORM's job)
        self._n += 1
        if self._n == 1:
            return SimpleNamespace(one=lambda: self._total)
        return SimpleNamespace(all=lambda: self._detail)


class TestParseSince:
    def test_date_only_is_utc_midnight(self) -> None:
        parsed = _parse_since("2026-06-01")
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == dt.timedelta(0)
        assert (parsed.hour, parsed.day) == (0, 1)

    def test_naive_datetime_pinned_to_utc(self) -> None:
        parsed = _parse_since("2026-06-01T12:00:00")
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == dt.timedelta(0)

    def test_aware_datetime_offset_preserved(self) -> None:
        parsed = _parse_since("2026-06-01T12:00:00+02:00")
        assert parsed.utcoffset() == dt.timedelta(hours=2)


class TestBuildCostFilter:
    def test_no_filters_is_all_time(self) -> None:
        preds, labels = build_cost_filter()
        assert preds == []
        assert labels == []

    def test_run_id_filter(self) -> None:
        preds, labels = build_cost_filter(run_id="cli-q07")
        assert len(preds) == 1
        assert labels == ["run_id=cli-q07"]

    def test_model_filter(self) -> None:
        preds, labels = build_cost_filter(model="glm-4.7")
        assert len(preds) == 1
        assert labels == ["model=glm-4.7"]

    def test_today_adds_half_open_window(self) -> None:
        preds, labels = build_cost_filter(today=True)
        assert len(preds) == 2  # created_at >= start AND created_at < end
        assert labels == ["today"]

    def test_since_filter(self) -> None:
        preds, labels = build_cost_filter(since="2026-06-01")
        assert len(preds) == 1
        assert labels == ["since=2026-06-01"]

    def test_combined_filters_preserve_label_order(self) -> None:
        preds, labels = build_cost_filter(
            run_id="cli-q07", model="glm-4.7", today=True
        )
        assert len(preds) == 4  # run_id + model + 2 today-window predicates
        assert labels == ["run_id=cli-q07", "model=glm-4.7", "today"]


class TestFormatCostBreakdown:
    def test_no_match_notice(self) -> None:
        bd = CostBreakdown(
            scope="[today]",
            matched=False,
            total_calls=0,
            total_spend=0.0,
            total_tokens=0,
            is_by_model=False,
        )
        assert format_cost_breakdown(bd) == "cost_ledger[today]: no rows matched."

    def test_by_model_render(self) -> None:
        bd = CostBreakdown(
            scope="[all-time]",
            matched=True,
            total_calls=3,
            total_spend=0.012,
            total_tokens=1500,
            is_by_model=True,
            detail=(
                SpendRow("glm-4.7", "", 2, 0.008, 1000),
                SpendRow("qwen3.5-flash", "", 1, 0.004, 500),
            ),
        )
        out = format_cost_breakdown(bd)
        assert "TOTAL: $0.0120" in out
        assert "3 calls" in out and "1,500 tokens" in out
        assert "by model:" in out
        assert "glm-4.7" in out and "0.0080" in out
        assert "by run" not in out  # by-model mode has no run section

    def test_by_run_render_has_subtotals_and_detail(self) -> None:
        bd = CostBreakdown(
            scope="[all-time]",
            matched=True,
            total_calls=2,
            total_spend=0.012,
            total_tokens=1500,
            is_by_model=False,
            detail=(
                SpendRow("cli-q07", "glm-4.7", 1, 0.008, 1000),
                SpendRow("cli-q07", "qwen3.5-flash", 1, 0.004, 500),
            ),
            by_run=(SpendRow("cli-q07", "", 2, 0.012, 1500),),
        )
        out = format_cost_breakdown(bd)
        assert "by run:" in out
        assert "by run × model:" in out
        assert "TOTAL: $0.0120" in out
        assert "glm-4.7" in out and "qwen3.5-flash" in out


class TestCostBreakdownQuery:
    async def test_empty_filter_returns_not_matched(self) -> None:
        session = _MockSession(SimpleNamespace(calls=0, spend=0, tok=0), [])
        bd = await cost_breakdown(session=session)
        assert bd.matched is False
        assert bd.total_calls == 0
        assert bd.scope == "[all-time]"
        assert bd.detail == () and bd.by_run == ()

    async def test_by_model_aggregates(self) -> None:
        total = SimpleNamespace(calls=2, spend=0.0012, tok=1500)
        detail = [("glm-4.7", 1, 0.0006, 800), ("qwen3.5-flash", 1, 0.0006, 700)]
        session = _MockSession(total, detail)
        bd = await cost_breakdown(by_model=True, session=session)
        assert bd.matched is True
        assert bd.is_by_model is True
        assert bd.total_calls == 2
        assert bd.total_spend == pytest.approx(0.0012)
        assert bd.total_tokens == 1500
        assert len(bd.detail) == 2
        assert bd.detail[0].primary == "glm-4.7"
        assert bd.detail[0].calls == 1 and bd.detail[0].tokens == 800
        assert bd.by_run == ()  # by-model mode computes no run subtotals

    async def test_by_run_aggregates_with_subtotals(self) -> None:
        total = SimpleNamespace(calls=2, spend=0.0012, tok=1500)
        detail = [
            ("cli-q07", "glm-4.7", 1, 0.0006, 800),
            ("cli-q07", "qwen3.5-flash", 1, 0.0006, 700),
        ]
        session = _MockSession(total, detail)
        bd = await cost_breakdown(session=session)
        assert bd.matched is True
        assert bd.is_by_model is False
        assert len(bd.detail) == 2
        assert bd.detail[0].secondary == "glm-4.7"
        assert len(bd.by_run) == 1
        assert bd.by_run[0].primary == "cli-q07"
        assert bd.by_run[0].calls == 2  # 1 + 1 summed across models
        assert bd.by_run[0].tokens == 1500  # 800 + 700
        assert bd.by_run[0].spend == pytest.approx(0.0012)

    async def test_by_run_subtotals_sorted_by_spend_desc(self) -> None:
        total = SimpleNamespace(calls=3, spend=0.003, tok=3000)
        # cli-q07 spends more than cli-q08 overall, but appears 2nd in detail;
        # the subtotal sort must still rank cli-q07 first.
        detail = [
            ("cli-q08", "glm-4.7", 1, 0.0006, 1000),
            ("cli-q07", "glm-4.7", 1, 0.0024, 2000),
        ]
        session = _MockSession(total, detail)
        bd = await cost_breakdown(session=session)
        assert [r.primary for r in bd.by_run] == ["cli-q07", "cli-q08"]

    async def test_scope_label_reflects_filters(self) -> None:
        total = SimpleNamespace(calls=1, spend=0.0006, tok=800)
        detail = [("cli-q07", "glm-4.7", 1, 0.0006, 800)]
        session = _MockSession(total, detail)
        bd = await cost_breakdown(run_id="cli-q07", model="glm-4.7", session=session)
        assert bd.scope == "[run_id=cli-q07, model=glm-4.7]"


class TestCostCliExitContract:
    """``--cost-today`` and ``--cost-since`` are mutually exclusive (exit 2)."""

    def test_today_and_since_mutually_exclusive(self) -> None:
        import click.testing

        import main as main_mod

        result = click.testing.CliRunner().invoke(
            main_mod.main, ["--cost", "--cost-today", "--cost-since", "2026-06-01"]
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output
