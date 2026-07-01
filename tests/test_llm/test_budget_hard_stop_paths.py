"""Budget-enforcement hard-stop paths — the 70/90/100% ladder + resume re-trip.

Companion to ``tests/test_llm/test_gateway.py::TestBudgetEnforcement`` (which
pins the two terminal branches: hard-stop raises, no-cheaper raises). This file
covers the DISTINCT angles called out by the run-control hardening design:

* the 90% ``critical`` threshold signals a downgrade (not a hard raise) and the
  call still proceeds;
* ``budget_hard_stop=False`` degrades (downgrade) instead of raising even when a
  cheaper model exists;
* ``check_budget``'s per-run token cap carries cumulative spend so a resumed
  run re-trips the cap on the next attempt (the documented deferred caveat);
* ``CostTracker.check_budget`` computes ``spent = cumulative - baseline`` so a
  resumed run with a fresh baseline does NOT pre-trip;
* the warn/critical messages carry the threshold-scaled spend.

These exercise the gateway's budget branch + ``CostTracker.check_budget`` as
PURE LOGIC against a fake cost accumulator / in-memory ledger — no live DB.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.llm.exceptions import BudgetExhaustedError
from src.llm.gateway import LLMGateway
from src.llm.models import LLMResponse


def _make_settings() -> Settings:
    """Default settings; tests flip ``budget.budget_hard_stop`` explicitly."""
    return Settings()


def _make_gateway(settings: Settings) -> LLMGateway:
    return LLMGateway(settings)


def _make_litellm_response(content: str = "ok") -> MagicMock:
    """A MagicMock shaped like a litellm ModelResponse (post-_build_response)."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.model = "gpt-4o-mini-2024-07-18"
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return resp


@pytest.fixture
def settings() -> Settings:
    return _make_settings()


@pytest.fixture
def gateway(settings: Settings) -> LLMGateway:
    return _make_gateway(settings)


@pytest.fixture
def simple_messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "Hello world"}]


# ─── Gateway hard-stop vs downgrade at the cap (100%) ────────────────


class TestHardCapBranchSelection:
    """The per-run cap (100%) branch: raise-on hard-stop vs degrade-by-default."""

    @pytest.mark.asyncio
    async def test_hard_stop_true_raises_not_silent_degrade(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """Crossing the cap with budget_hard_stop=True raises BudgetExhaustedError
        (the typed terminal signal), NOT a silent model-tier downgrade."""
        tracker = MagicMock()
        tracker.check_budget = AsyncMock(return_value=(False, "Per-run token cap reached"))
        gateway.set_cost_tracker(tracker)
        gateway.set_run_id("cli-resume-1")
        gateway._settings.budget.budget_hard_stop = True

        with patch.object(gateway, "_get_cheaper_fallback") as fb:
            with pytest.raises(BudgetExhaustedError, match="Per-run token cap reached"):
                await gateway.acompletion(messages=simple_messages, model="claude-sonnet-4-6")
            # Hard-stop must NOT consult the downgrade path.
            assert not fb.called

    @pytest.mark.asyncio
    async def test_hard_stop_false_degrades_to_cheaper_fallback(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """With budget_hard_stop=False (default), crossing the cap DOWNGRADES to a
        cheaper model and the call proceeds — no raise."""
        tracker = MagicMock()
        tracker.check_budget = AsyncMock(return_value=(False, "Per-run token cap reached"))
        tracker.record_usage = AsyncMock(return_value=0.001)
        gateway.set_cost_tracker(tracker)
        gateway.set_run_id("cli-resume-2")
        gateway._settings.budget.budget_hard_stop = False

        mock_resp = _make_litellm_response("downgraded answer")
        with patch.object(gateway, "_get_cheaper_fallback", return_value="gpt-4o-mini-2024-07-18"):
            with patch("src.llm.gateway.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
                mock_litellm.Usage = MagicMock
                for exc in (
                    "RateLimitError",
                    "Timeout",
                    "ServiceUnavailableError",
                    "APIConnectionError",
                    "AuthenticationError",
                    "BadRequestError",
                ):
                    setattr(mock_litellm, exc, Exception)

                result = await gateway.acompletion(
                    messages=simple_messages, model="claude-sonnet-4-6"
                )

        assert isinstance(result, LLMResponse)
        # The cheaper fallback model was the one actually called.
        assert mock_litellm.acompletion.await_args.kwargs.get("model") == "gpt-4o-mini-2024-07-18"


# ─── The 90% critical threshold: downgrade signal, call proceeds ─────


class TestCriticalThresholdDowngrade:
    """At the 90% critical threshold ``check_budget`` returns within-budget=True
    with a WARNING, so the gateway does NOT raise and does NOT downgrade — the
    call proceeds normally (the downgrade only fires when within_budget=False)."""

    @pytest.mark.asyncio
    async def test_critical_warning_proceeds_without_raise(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        tracker = MagicMock()
        tracker.check_budget = AsyncMock(
            return_value=(True, "WARNING: Daily budget at 90%: $9.00 / $10.00")
        )
        tracker.record_usage = AsyncMock(return_value=0.001)
        gateway.set_cost_tracker(tracker)
        gateway._settings.budget.budget_hard_stop = True  # even with hard-stop on

        mock_resp = _make_litellm_response("normal answer")
        with patch.object(gateway, "_get_cheaper_fallback") as fb:
            with patch("src.llm.gateway.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
                mock_litellm.Usage = MagicMock
                for exc in (
                    "RateLimitError",
                    "Timeout",
                    "ServiceUnavailableError",
                    "APIConnectionError",
                    "AuthenticationError",
                    "BadRequestError",
                ):
                    setattr(mock_litellm, exc, Exception)

                result = await gateway.acompletion(
                    messages=simple_messages, model="claude-sonnet-4-6"
                )

        assert isinstance(result, LLMResponse)
        # within-budget=True → no downgrade consultation.
        assert not fb.called


# ─── Cumulative resume re-trip (the documented deferred caveat) ──────


def _make_cost_tracker_with_ledger(
    settings: Settings,
    *,
    run_usage: int,
    daily_spend: float = 0.0,
    run_spend: float = 0.0,
) -> Any:
    """A real ``CostTracker`` whose DB reads are faked by overriding the three
    query methods — so ``check_budget`` runs its PURE threshold logic against a
    planted cumulative run-usage + daily spend + cumulative run $ spend without a
    live Postgres."""
    from src.llm.cost_tracker import CostTracker

    tracker = CostTracker(session=MagicMock(), settings=settings)
    tracker.get_run_token_usage = AsyncMock(return_value=run_usage)  # type: ignore[method-assign]
    tracker.get_daily_spend = AsyncMock(return_value=daily_spend)  # type: ignore[method-assign]
    tracker.get_run_spend = AsyncMock(return_value=run_spend)  # type: ignore[method-assign]
    return tracker


class TestResumeReTripsCap:
    """``get_run_token_usage`` is cumulative, so a resumed run re-trips the cap."""

    @pytest.mark.asyncio
    async def test_resumed_run_retrips_cap_immediately(self, settings: Settings) -> None:
        """A run that already spent >= per_task_token_limit re-trips the cap on the
        very next attempt — the documented deferred caveat. With budget_hard_stop
        the next ``check_budget`` returns False (cap reached)."""
        settings.budget.per_task_token_limit = 200_000
        settings.budget.budget_hard_stop = True
        # Prior attempt already logged 407K tokens attributed to this run_id.
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=407_000, daily_spend=0.0
        )

        within, msg = await tracker.check_budget("cli-q09")

        assert within is False
        assert "Per-run token cap reached" in msg

    @pytest.mark.asyncio
    async def test_fresh_baseline_prevents_pretrip_on_resume(self, settings: Settings) -> None:
        """``set_run_baseline`` shifts the window so a resumed run measures only
        THIS attempt's spend — the fix for the q09 re-enqueue pre-trip. With the
        baseline set to the prior cumulative, the cap is measured from zero."""
        settings.budget.per_task_token_limit = 200_000
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=407_000, daily_spend=0.0
        )
        # The runner captured 407K as the baseline at attempt start.
        tracker.set_run_baseline(407_000)

        within, msg = await tracker.check_budget("cli-q09")

        # spent = max(0, 407_000 - 407_000) = 0 < 200_000 → within budget.
        assert within is True
        assert "cap reached" not in msg

    @pytest.mark.asyncio
    async def test_baseline_clamped_nonnegative(self, settings: Settings) -> None:
        """``set_run_baseline`` clamps negatives so a coerce-failure can never
        grant a larger budget than intended."""
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=100_000, daily_spend=0.0
        )
        tracker.set_run_baseline(-50_000)

        within, _ = await tracker.check_budget("cli-q09")

        # spent = max(0, 100_000 - 0) = 100_000 (baseline clamped to 0).
        assert within is True


# ─── Two-tier per-run COST caps (attempt-relative + cumulative-absolute) ──


class TestPerRunCostCaps:
    """The two per-run COST tiers: ``per_run_cost_limit`` (attempt-relative,
    baseline-subtracted — the q09 resume-safe fix) and
    ``per_run_cost_limit_absolute`` (cumulative across all attempts, NO baseline
    — the q06 redelivery-forever backstop, Fix B)."""

    @pytest.mark.asyncio
    async def test_per_attempt_cost_cap_trips(self, settings: Settings) -> None:
        """Tier 3 (attempt-relative): cumulative $1.5 with a $0.5 baseline → this
        attempt spent $1.0 ≥ the $1.0 per-attempt cap → trips. Locks the cost tier
        that was previously untested and that the two-tier refactor restructured."""
        settings.budget.per_run_cost_limit = 1.0
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=0, daily_spend=0.0, run_spend=1.5
        )
        tracker.set_run_cost_baseline(0.5)

        within, msg = await tracker.check_budget("cli-q06")

        assert within is False
        assert "Per-run cost cap reached" in msg

    @pytest.mark.asyncio
    async def test_per_attempt_cost_baseline_prevents_pretrip(
        self, settings: Settings
    ) -> None:
        """Tier 3 is resume-safe: a fresh baseline (set to the prior cumulative)
        means THIS attempt's spend is ~$0 even though the run has spent $1.5 total
        → does not trip the $1.0 per-attempt cap. (The q09 resume property,
        expressed for the cost tier.)"""
        settings.budget.per_run_cost_limit = 1.0
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=0, daily_spend=0.0, run_spend=1.5
        )
        tracker.set_run_cost_baseline(1.5)

        within, msg = await tracker.check_budget("cli-q06")

        assert within is True
        assert "cap reached" not in msg

    @pytest.mark.asyncio
    async def test_cumulative_absolute_trips_despite_baseline_reset(
        self, settings: Settings
    ) -> None:
        """Tier 4 — the q06 redelivery-forever fix. A run that redelivered N× has
        spent $3.5 TOTAL. The per-attempt baseline was reset (to $2.6) so tier 3
        sees only $0.9 this attempt (< $1.0) and does NOT trip — exactly the hole
        that let q06 run unbounded. Tier 4 ignores the baseline and trips on the
        cumulative $3.5 ≥ $3.0 absolute cap."""
        settings.budget.per_run_cost_limit = 1.0
        settings.budget.per_run_cost_limit_absolute = 3.0
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=0, daily_spend=0.0, run_spend=3.5
        )
        tracker.set_run_cost_baseline(2.6)

        within, msg = await tracker.check_budget("cli-q06")

        assert within is False
        assert "Cumulative run cost cap reached" in msg
        # Sanity: tier 3 alone would have allowed this (spent $0.9 < $1.0).
        assert "Per-run cost cap reached" not in msg

    @pytest.mark.asyncio
    async def test_cumulative_absolute_disabled_by_default(
        self, settings: Settings
    ) -> None:
        """Tier 4 is opt-in: with ``per_run_cost_limit_absolute = 0`` (default) an
        arbitrarily large cumulative spend does not trip tier 4 — only the active
        per-attempt tier (here also 0/disabled) governs, so the run stays within."""
        settings.budget.per_run_cost_limit = 0.0
        settings.budget.per_run_cost_limit_absolute = 0.0
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=0, daily_spend=0.0, run_spend=999.0
        )

        within, msg = await tracker.check_budget("cli-q06")

        assert within is True
        assert "cap reached" not in msg

    @pytest.mark.asyncio
    async def test_two_tiers_normal_resume_not_tripped(
        self, settings: Settings
    ) -> None:
        """Both tiers active but a NORMAL resume (cumulative $1.2, this attempt
        $0) stays within: tier 3 spent $0 < $1.0 and tier 4 cumulative $1.2 <
        $3.0. The backstop must not falsely kill a healthy resumed run."""
        settings.budget.per_run_cost_limit = 1.0
        settings.budget.per_run_cost_limit_absolute = 3.0
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=0, daily_spend=0.0, run_spend=1.2
        )
        tracker.set_run_cost_baseline(1.2)

        within, msg = await tracker.check_budget("cli-q06")

        assert within is True
        assert "cap reached" not in msg



# ─── The 70% warn / 90% critical messages (pure check_budget logic) ──


class TestWarnCriticalMessages:
    """``check_budget`` scales the warn/critical messages against the thresholds."""

    @pytest.mark.asyncio
    async def test_warn_threshold_message(self, settings: Settings) -> None:
        settings.budget.max_cost_usd = 10.0
        settings.budget.budget_warn_threshold = 0.70
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=0, daily_spend=7.50
        )

        within, msg = await tracker.check_budget(None)

        assert within is True
        assert "Budget warning" in msg

    @pytest.mark.asyncio
    async def test_critical_threshold_message(self, settings: Settings) -> None:
        settings.budget.max_cost_usd = 10.0
        settings.budget.budget_critical_threshold = 0.90
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=0, daily_spend=9.10
        )

        within, msg = await tracker.check_budget(None)

        assert within is True
        assert "WARNING" in msg and "90%" in msg

    @pytest.mark.asyncio
    async def test_daily_cap_exhausted_blocks(self, settings: Settings) -> None:
        settings.budget.max_cost_usd = 10.0
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=0, daily_spend=11.00
        )

        within, msg = await tracker.check_budget("cli-x")

        assert within is False
        assert "Daily budget exhausted" in msg

    @pytest.mark.asyncio
    async def test_ok_when_under_all_thresholds(self, settings: Settings) -> None:
        settings.budget.max_cost_usd = 10.0
        tracker = _make_cost_tracker_with_ledger(
            settings, run_usage=1_000, daily_spend=1.00
        )

        within, msg = await tracker.check_budget("cli-ok")

        assert within is True
        assert "Budget OK" in msg


# ─── _get_cheaper_fallback tier ordering (drives the degrade branch) ─


class TestCheaperFallbackTierOrder:
    """The degrade branch is driven by ``_get_cheaper_fallback``. A moderate-tier
    model degrades to a cheaper one; a very-cheap (tier-0) model has no fallback."""

    def test_moderate_model_has_cheaper_fallback(self, gateway: LLMGateway) -> None:
        """A moderate-tier model resolves to SOME cheaper fallback."""
        fb = gateway._get_cheaper_fallback("claude-sonnet-4-6")
        assert fb is not None
        assert fb != "claude-sonnet-4-6"

    def test_cheapest_model_has_no_fallback(self, gateway: LLMGateway) -> None:
        """A tier-0 (very-cheap) model has no cheaper fallback → degrade branch
        would instead raise BudgetExhaustedError (covered by test_gateway)."""
        fb = gateway._get_cheaper_fallback("gpt-4o-mini-2024-07-18")
        assert fb is None

    def test_unknown_model_has_no_fallback(self, gateway: LLMGateway) -> None:
        fb = gateway._get_cheaper_fallback("does-not-exist-model")
        assert fb is None
