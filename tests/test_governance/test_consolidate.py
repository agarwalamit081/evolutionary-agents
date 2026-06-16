"""Tests for src.governance.consolidate (B3 offline consolidation).

``_plan_clusters`` is pure (no DB) and is exercised directly. ``consolidate_tools``
/ ``consolidate_sub_agents`` take an injected persister (a fake whose row-fetcher
returns controlled rows) so the dry-run/applied wiring is verified without a
database. ``ToolPersister.merge_alias`` (the tool_subset re-point) is tested with
a mocked session.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.governance.consolidate import (
    ConsolidationReport,
    MergePlan,
    _plan_clusters,
    consolidate_sub_agents,
    consolidate_tools,
)


def _row(name: str, emb: list[float] | None, score: float) -> dict[str, Any]:
    return {"name": name, "embedding": emb, "score": score}


_E0 = [1.0] + [0.0] * 767  # identical direction
_E1 = [0.0, 1.0] + [0.0] * 766  # orthogonal to _E0


class TestPlanClusters:
    def test_merges_redundant_keeps_higher_score(self) -> None:
        plans = _plan_clusters([_row("v1", _E0, 1.0), _row("v3", _E0, 3.0)], 0.92)
        assert len(plans) == 1
        assert plans[0].target == "v3"  # higher score survives
        assert plans[0].retired == ["v1"]

    def test_below_threshold_untouched(self) -> None:
        """Orthogonal embeddings never merge."""
        plans = _plan_clusters([_row("a", _E0, 1.0), _row("b", _E1, 1.0)], 0.92)
        assert plans == []

    def test_three_way_cluster_one_survivor(self) -> None:
        plans = _plan_clusters(
            [_row("low", _E0, 1.0), _row("mid", _E0, 2.0), _row("high", _E0, 3.0)],
            0.92,
        )
        assert len(plans) == 1
        assert plans[0].target == "high"
        assert sorted(plans[0].retired) == ["low", "mid"]

    def test_none_embedding_not_clustered(self) -> None:
        """A row with no embedding survives alone — nothing merged into/with it."""
        plans = _plan_clusters(
            [_row("noemb", None, 9.0), _row("dup", _E0, 1.0)], 0.92
        )
        assert plans == []

    def test_two_distinct_clusters(self) -> None:
        """Two separate redundant pairs → two plans."""
        e2 = [0.0, 0.0, 1.0] + [0.0] * 765
        plans = _plan_clusters(
            [
                _row("a1", _E0, 1.0),
                _row("a2", _E0, 2.0),
                _row("b1", e2, 1.0),
                _row("b2", e2, 2.0),
            ],
            0.92,
        )
        assert len(plans) == 2
        targets = {p.target for p in plans}
        assert targets == {"a2", "b2"}


def _bind_session(session: MagicMock) -> Any:
    @asynccontextmanager
    async def _get_session() -> AsyncGenerator[MagicMock, None]:
        yield session

    return _get_session


class _FakeToolPersister:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._active_tool_capability_rows = AsyncMock(return_value=rows)
        self.retire = AsyncMock(return_value=0)
        self.merge_alias = AsyncMock(return_value=0)


class _FakeAgentPersister:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._active_capability_rows = AsyncMock(return_value=rows)
        self.retire = AsyncMock(return_value=0)


class TestConsolidateTools:
    @pytest.mark.asyncio
    async def test_dry_run_reports_without_mutating(self) -> None:
        rows = [_row("v1", _E0, 1.0), _row("v3", _E0, 3.0)]
        fp = _FakeToolPersister(rows)
        report = await consolidate_tools(threshold=0.92, dry_run=True, persister=fp)
        assert report.dry_run is True
        assert len(report.tools) == 1
        assert report.tools[0].target == "v3"
        fp.retire.assert_not_awaited()
        fp.merge_alias.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_applied_retires_and_repoints(self) -> None:
        rows = [_row("v1", _E0, 1.0), _row("v3", _E0, 3.0)]
        fp = _FakeToolPersister(rows)
        report = await consolidate_tools(threshold=0.92, dry_run=False, persister=fp)
        assert report.dry_run is False
        fp.retire.assert_awaited_once_with(["v1"])
        fp.merge_alias.assert_awaited_once_with("v1", "v3")

    @pytest.mark.asyncio
    async def test_no_duplicates_no_merges(self) -> None:
        fp = _FakeToolPersister([_row("a", _E0, 1.0), _row("b", _E1, 1.0)])
        report = await consolidate_tools(persister=fp)
        assert report.tools == []
        fp.retire.assert_not_awaited()


class TestConsolidateSubAgents:
    @pytest.mark.asyncio
    async def test_applied_retires_without_repoint(self) -> None:
        rows = [_row("weak", _E0, 1.0), _row("strong", _E0, 3.0)]
        fp = _FakeAgentPersister(rows)
        report = await consolidate_sub_agents(
            threshold=0.92, dry_run=False, persister=fp
        )
        assert len(report.agents) == 1
        assert report.agents[0].target == "strong"
        fp.retire.assert_awaited_once_with(["weak"])
        # Agents have no persisted name-references → no merge_alias.
        assert not hasattr(fp, "merge_alias")


class TestMergeAlias:
    @pytest.mark.asyncio
    async def test_repoints_tool_subset_references(self) -> None:
        from src.db.models import SubAgentModel
        from src.tools.dynamic.persister import ToolPersister

        # A sub-agent scoped to the soon-to-be-retired "old_tool".
        model = SubAgentModel(
            id=uuid.uuid4(),
            name="scoped_agent",
            description="d",
            tool_subset=["keep_tool", "old_tool"],
            is_active=True,
        )
        select_result = MagicMock()
        select_result.scalars.return_value.all.return_value = [model]

        # Capture every statement execute() sees; the 1st is the SELECT, the
        # 2nd the UPDATE whose bind params carry the re-pointed tool_subset.
        captured: list[Any] = []

        async def _exec(stmt: Any) -> Any:
            captured.append(stmt)
            if len(captured) == 1:
                return select_result  # SELECT SubAgentModel
            return MagicMock()  # UPDATE

        session = MagicMock()
        session.execute = AsyncMock(side_effect=_exec)
        persister = ToolPersister()
        with patch("src.db.session.get_session", new=_bind_session(session)):
            count = await persister.merge_alias("old_tool", "new_tool")

        assert count == 1
        # JSONB has no literal_binds renderer; read the raw bind params instead.
        update_stmt = captured[1]
        params = update_stmt.compile().params
        subset_values = [v for v in params.values() if isinstance(v, list)]
        assert subset_values == [["keep_tool", "new_tool"]]  # re-pointed + deduped

    @pytest.mark.asyncio
    async def test_same_source_target_noop(self) -> None:
        from src.tools.dynamic.persister import ToolPersister

        assert await ToolPersister().merge_alias("x", "x") == 0

    @pytest.mark.asyncio
    async def test_no_references_returns_zero(self) -> None:
        from src.tools.dynamic.persister import ToolPersister

        select_result = MagicMock()
        select_result.scalars.return_value.all.return_value = []  # no refs
        session = MagicMock()
        session.execute = AsyncMock(return_value=select_result)
        persister = ToolPersister()
        with patch("src.db.session.get_session", new=_bind_session(session)):
            count = await persister.merge_alias("old", "new")
        assert count == 0


class TestReportShape:
    def test_total_retired_sums_plans(self) -> None:
        report = ConsolidationReport(
            tools=[MergePlan(target="t1", retired=["a", "b"])],
            agents=[MergePlan(target="a1", retired=["c"])],
            dry_run=True,
        )
        assert report.total_retired == 3
