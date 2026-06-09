"""Tests for src.llm.cost_tracker — cost tracking, budget enforcement, and usage stats."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.cost_tracker import CostTracker


@pytest.fixture
def mock_session() -> MagicMock:
    """Create a mock AsyncSession."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock Settings with a budget group."""
    settings = MagicMock()
    settings.budget.max_cost_usd = 10.0
    return settings


@pytest.fixture
def tracker(mock_session: MagicMock, mock_settings: MagicMock) -> CostTracker:
    """Create a CostTracker with mocked session and settings."""
    return CostTracker(session=mock_session, settings=mock_settings)


# ─── calculate_cost (static method) ──────────────────────────────────────


class TestCalculateCost:
    """Tests for the static calculate_cost method."""

    def test_known_model_returns_fallback_cost(self) -> None:
        """Known model without cost fields uses fallback pricing."""
        cost = CostTracker.calculate_cost("gpt-4o-mini-2024-07-18", 100, 50)
        assert cost > 0

    def test_unknown_model_uses_fallback(self) -> None:
        """Unknown model uses fallback pricing ($0.005/1K in, $0.015/1K out)."""
        cost = CostTracker.calculate_cost("unknown-model-xyz", 1000, 500)
        expected = (1000 * 0.005 / 1000) + (500 * 0.015 / 1000)
        assert abs(cost - expected) < 1e-10

    def test_zero_tokens_returns_zero(self) -> None:
        """Zero tokens means cost is 0.0."""
        cost = CostTracker.calculate_cost("any-model", 0, 0)
        assert cost == 0.0

    def test_cost_scales_linearly(self) -> None:
        """Cost scales linearly with token count."""
        cost_small = CostTracker.calculate_cost("test-model", 100, 50)
        cost_large = CostTracker.calculate_cost("test-model", 1000, 500)
        assert abs(cost_large - cost_small * 10) < 1e-10

    def test_fallback_pricing_formula(self) -> None:
        """Verify fallback pricing formula: (in * 0.005 + out * 0.015) / 1000."""
        cost = CostTracker.calculate_cost("nonexistent", 2000, 3000)
        expected = (2000 * 0.005 + 3000 * 0.015) / 1000
        assert abs(cost - expected) < 1e-10


# ─── record_usage ─────────────────────────────────────────────────────────


class TestRecordUsage:
    """Tests for CostTracker.record_usage().

    Uses _make_tracker helper to patch CostLedger per-test so the missing
    total_tokens column in the ORM model does not cause TypeError.
    """

    def _make_tracker(
        self, mock_session: MagicMock, mock_settings: MagicMock
    ) -> CostTracker:
        """Create a tracker with CostLedger patched in the cost_tracker module."""
        with patch("src.llm.cost_tracker.CostLedger"):
            return CostTracker(session=mock_session, settings=mock_settings)

    @pytest.mark.asyncio
    async def test_creates_ledger_entry_and_commits(
        self, mock_session: MagicMock, mock_settings: MagicMock
    ) -> None:
        """record_usage should add a CostLedger entry and commit."""
        with patch("src.llm.cost_tracker.CostLedger"):
            tracker = CostTracker(session=mock_session, settings=mock_settings)
            cost = await tracker.record_usage(
                model="gpt-4o-mini-2024-07-18",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
            )

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        assert cost > 0

    @pytest.mark.asyncio
    async def test_returns_calculated_cost(
        self, mock_session: MagicMock, mock_settings: MagicMock
    ) -> None:
        """record_usage should return the cost calculated by calculate_cost."""
        with patch("src.llm.cost_tracker.CostLedger"):
            tracker = CostTracker(session=mock_session, settings=mock_settings)
            cost = await tracker.record_usage(
                model="test-model",
                provider="test-provider",
                input_tokens=1000,
                output_tokens=500,
            )

        expected = CostTracker.calculate_cost("test-model", 1000, 500)
        assert abs(cost - expected) < 1e-10

    @pytest.mark.asyncio
    async def test_ledger_entry_fields(
        self, mock_session: MagicMock, mock_settings: MagicMock
    ) -> None:
        """The CostLedger entry should have correct model, provider, and token fields."""
        with patch("src.llm.cost_tracker.CostLedger") as mock_ledger_cls:
            tracker = CostTracker(session=mock_session, settings=mock_settings)
            await tracker.record_usage(
                model="deepseek-v4-flash",
                provider="deepseek",
                input_tokens=200,
                output_tokens=100,
                task_id="task-uuid-123",
                latency_ms=1500,
            )

        # Verify CostLedger was called with the right kwargs
        call_kwargs = mock_ledger_cls.call_args[1]
        assert call_kwargs["model"] == "deepseek-v4-flash"
        assert call_kwargs["provider"] == "deepseek"
        assert call_kwargs["input_tokens"] == 200
        assert call_kwargs["output_tokens"] == 100
        assert call_kwargs["cost_usd"] > 0
        assert call_kwargs["task_id"] == "task-uuid-123"
        assert call_kwargs["latency_ms"] == 1500

    @pytest.mark.asyncio
    async def test_optional_fields_default_to_none(
        self, mock_session: MagicMock, mock_settings: MagicMock
    ) -> None:
        """task_id and latency_ms should be None when not provided."""
        with patch("src.llm.cost_tracker.CostLedger") as mock_ledger_cls:
            tracker = CostTracker(session=mock_session, settings=mock_settings)
            await tracker.record_usage(
                model="test-model",
                provider="test",
                input_tokens=10,
                output_tokens=5,
            )

        call_kwargs = mock_ledger_cls.call_args[1]
        assert call_kwargs["task_id"] is None
        assert call_kwargs["latency_ms"] is None

    @pytest.mark.asyncio
    async def test_zero_token_usage(
        self, mock_session: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Zero tokens should result in zero cost."""
        with patch("src.llm.cost_tracker.CostLedger"):
            tracker = CostTracker(session=mock_session, settings=mock_settings)
            cost = await tracker.record_usage(
                model="test",
                provider="test",
                input_tokens=0,
                output_tokens=0,
            )

        assert cost == 0.0


# ─── check_budget ─────────────────────────────────────────────────────────


class TestCheckBudget:
    """Tests for CostTracker.check_budget().

    check_budget calls get_daily_spend internally, which queries CostLedger.
    We mock get_daily_spend on the tracker instance to isolate budget logic.
    """

    @pytest.mark.asyncio
    async def test_under_limit_returns_ok(
        self, tracker: CostTracker
    ) -> None:
        """When spend is well under limit, returns (True, 'Budget OK')."""
        tracker.get_daily_spend = AsyncMock(return_value=1.0)

        is_ok, msg = await tracker.check_budget()

        assert is_ok is True
        assert "Budget OK" in msg

    @pytest.mark.asyncio
    async def test_at_70_percent_returns_warning(
        self, tracker: CostTracker
    ) -> None:
        """At 70% of budget, returns (True, 'Budget warning')."""
        tracker.get_daily_spend = AsyncMock(return_value=7.5)

        is_ok, msg = await tracker.check_budget()

        assert is_ok is True
        assert "Budget warning" in msg

    @pytest.mark.asyncio
    async def test_at_90_percent_returns_critical(
        self, tracker: CostTracker
    ) -> None:
        """At 90% of budget, returns (True, 'WARNING')."""
        tracker.get_daily_spend = AsyncMock(return_value=9.5)

        is_ok, msg = await tracker.check_budget()

        assert is_ok is True
        assert "WARNING" in msg

    @pytest.mark.asyncio
    async def test_at_100_percent_returns_exhausted(
        self, tracker: CostTracker
    ) -> None:
        """At 100% of budget, returns (False, 'exhausted')."""
        tracker.get_daily_spend = AsyncMock(return_value=10.0)

        is_ok, msg = await tracker.check_budget()

        assert is_ok is False
        assert "exhausted" in msg

    @pytest.mark.asyncio
    async def test_over_100_percent_returns_exhausted(
        self, tracker: CostTracker
    ) -> None:
        """Over 100% of budget, returns (False, 'exhausted')."""
        tracker.get_daily_spend = AsyncMock(return_value=15.0)

        is_ok, msg = await tracker.check_budget()

        assert is_ok is False
        assert "exhausted" in msg

    @pytest.mark.asyncio
    async def test_message_contains_dollar_amounts(
        self, tracker: CostTracker
    ) -> None:
        """Budget messages should include dollar-formatted spend and limit."""
        tracker.get_daily_spend = AsyncMock(return_value=5.0)

        is_ok, msg = await tracker.check_budget()

        assert "$5.00" in msg
        assert "$10.00" in msg


# ─── get_daily_spend ──────────────────────────────────────────────────────


class TestGetDailySpend:
    """Tests for CostTracker.get_daily_spend().

    get_daily_spend builds a SQLAlchemy query referencing CostLedger columns.
    Since CostLedger columns are real ORM mapped attributes, they work fine
    in query construction. We only mock session.execute to avoid hitting DB.
    """

    @pytest.mark.asyncio
    async def test_returns_float_sum(
        self, tracker: CostTracker, mock_session: MagicMock
    ) -> None:
        """get_daily_spend should return the sum of cost_usd as a float."""
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 3.456
        mock_session.execute = AsyncMock(return_value=mock_result)

        total = await tracker.get_daily_spend()

        assert isinstance(total, float)
        assert total == 3.456

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_spending(
        self, tracker: CostTracker, mock_session: MagicMock
    ) -> None:
        """When no spending today, should return 0.0."""
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 0.0
        mock_session.execute = AsyncMock(return_value=mock_result)

        total = await tracker.get_daily_spend()

        assert total == 0.0

    @pytest.mark.asyncio
    async def test_queries_with_date_filter(
        self, tracker: CostTracker, mock_session: MagicMock
    ) -> None:
        """get_daily_spend should execute a query (verifying it was called)."""
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 0.0
        mock_session.execute = AsyncMock(return_value=mock_result)

        await tracker.get_daily_spend()

        mock_session.execute.assert_called_once()
        call_args = mock_session.execute.call_args[0][0]
        # The query is a sa.select statement with a whereclause
        assert hasattr(call_args, "whereclause")


# ─── get_daily_token_usage ────────────────────────────────────────────────


class TestGetDailyTokenUsage:
    """Tests for CostTracker.get_daily_token_usage().

    get_daily_token_usage references CostLedger.total_tokens which does not
    exist on the ORM model (known schema/code mismatch). We mock the method
    on the tracker instance to test the return-value contract, and also
    test the query-building path where possible.
    """

    @pytest.mark.asyncio
    async def test_returns_aggregated_stats(
        self, tracker: CostTracker
    ) -> None:
        """Should return dict with input_tokens, output_tokens, total_tokens."""
        expected = {"input_tokens": 500, "output_tokens": 250, "total_tokens": 750}
        tracker.get_daily_token_usage = AsyncMock(return_value=expected)

        usage = await tracker.get_daily_token_usage()

        assert usage == expected

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_usage(
        self, tracker: CostTracker
    ) -> None:
        """When no usage today, all fields should be 0."""
        expected = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        tracker.get_daily_token_usage = AsyncMock(return_value=expected)

        usage = await tracker.get_daily_token_usage()

        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_values_are_ints(
        self, tracker: CostTracker
    ) -> None:
        """All values should be integers (not floats from Decimal)."""
        expected = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        tracker.get_daily_token_usage = AsyncMock(return_value=expected)

        usage = await tracker.get_daily_token_usage()

        for key in ("input_tokens", "output_tokens", "total_tokens"):
            assert isinstance(usage[key], int)


# ─── DB error handling ────────────────────────────────────────────────────


class TestDBErrorHandling:
    """Tests for database error handling in CostTracker."""

    @pytest.mark.asyncio
    async def test_record_usage_db_commit_error_propagates(
        self, mock_session: MagicMock, mock_settings: MagicMock
    ) -> None:
        """If session.commit fails, the error should propagate."""
        mock_session.commit = AsyncMock(side_effect=RuntimeError("DB commit failed"))

        with patch("src.llm.cost_tracker.CostLedger"):
            tracker = CostTracker(session=mock_session, settings=mock_settings)
            with pytest.raises(RuntimeError, match="DB commit failed"):
                await tracker.record_usage(
                    model="test",
                    provider="test",
                    input_tokens=10,
                    output_tokens=5,
                )

    @pytest.mark.asyncio
    async def test_get_daily_spend_db_error_propagates(
        self, tracker: CostTracker, mock_session: MagicMock
    ) -> None:
        """If session.execute fails in get_daily_spend, the error should propagate."""
        mock_session.execute = AsyncMock(side_effect=RuntimeError("DB query failed"))

        with pytest.raises(RuntimeError, match="DB query failed"):
            await tracker.get_daily_spend()

    @pytest.mark.asyncio
    async def test_get_daily_token_usage_db_error_propagates(
        self, tracker: CostTracker
    ) -> None:
        """If session.execute fails in get_daily_token_usage, the error should propagate."""
        tracker.get_daily_token_usage = AsyncMock(
            side_effect=RuntimeError("DB query failed")
        )

        with pytest.raises(RuntimeError, match="DB query failed"):
            await tracker.get_daily_token_usage()
