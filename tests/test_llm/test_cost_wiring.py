"""Integration tests: a real CostTracker wired into the gateway records usage.

Validates the §10.1 wiring — that a completion through the gateway lands a
``cost_ledger`` row and consults the budget gate — without a live database
(the session is mocked, litellm is patched).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.db.models import CostLedger
from src.llm.cost_tracker import CostTracker
from src.llm.gateway import LLMGateway


def _make_settings() -> Settings:
    return Settings()


def _make_gateway(settings: Settings) -> LLMGateway:
    with patch.object(LLMGateway, "_configure_litellm"):
        gw = LLMGateway(settings)
    gw._rate_limiter = MagicMock()
    gw._rate_limiter.acquire = AsyncMock(return_value=None)
    return gw


def _mock_session(daily_spend: float = 0.0) -> MagicMock:
    """AsyncSession mock whose ``execute().scalar_one()`` yields the daily spend."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    result = MagicMock()
    result.scalar_one.return_value = daily_spend
    session.execute = AsyncMock(return_value=result)
    return session


def _make_litellm_response(
    *, content: str = "ok", input_tokens: int = 12, output_tokens: int = 8
) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    usage.total_tokens = input_tokens + output_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


_LITELLM_ATTRS = (
    "Usage",
    "RateLimitError",
    "Timeout",
    "ServiceUnavailableError",
    "APIConnectionError",
    "AuthenticationError",
    "BadRequestError",
)


def _patch_litellm(mock_litellm: Any, response: MagicMock) -> None:
    mock_litellm.acompletion = AsyncMock(return_value=response)
    for attr in _LITELLM_ATTRS:
        setattr(mock_litellm, attr, MagicMock if attr == "Usage" else Exception)


@pytest.mark.asyncio
async def test_wired_tracker_records_ledger_row_and_checks_budget() -> None:
    """A completion through a gateway with a real CostTracker lands a row and
    consults the budget gate (the §10.1 wiring regression)."""
    settings = _make_settings()
    gateway = _make_gateway(settings)
    session = _mock_session(daily_spend=0.0)
    response = _make_litellm_response()

    with patch("src.llm.gateway.litellm") as mock_litellm:
        _patch_litellm(mock_litellm, response)
        gateway.set_cost_tracker(CostTracker(session, settings))
        await gateway.acompletion(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini-2024-07-18",
        )

    # Budget gate consulted (pre-call): check_budget -> get_daily_spend.
    session.execute.assert_awaited()
    # Usage recorded (post-call): a CostLedger row is added and committed.
    session.add.assert_called_once()
    session.commit.assert_awaited_once()
    ledger = session.add.call_args.args[0]
    assert isinstance(ledger, CostLedger)
    # The recorded row reflects the real call.
    assert ledger.model == "gpt-4o-mini-2024-07-18"
    assert ledger.provider == "openai"
    assert ledger.input_tokens == 12
    assert ledger.output_tokens == 8
    assert ledger.total_tokens == 20
    assert ledger.cost_usd > 0


@pytest.mark.asyncio
async def test_real_tracker_budget_exhausted_triggers_model_downgrade() -> None:
    """When the real tracker reports the budget exhausted, the gateway
    downgrades to a cheaper model — proving check_budget drives the gate, not
    just logs."""
    settings = _make_settings()
    gateway = _make_gateway(settings)
    # Pin the DEFAULT downgrade path (budget_hard_stop=False) so an ambient
    # BUDGET_HARD_STOP=true in .env cannot flip this to the terminal-raise path
    # (the opt-in hard-stop is covered by the sibling test in test_gateway.py).
    gateway._settings.budget.budget_hard_stop = False
    # Daily spend just over the budget -> check_budget returns (False, ...).
    session = _mock_session(daily_spend=settings.budget.max_cost_usd + 1.0)
    response = _make_litellm_response(content="cheap answer")

    with (
        patch("src.llm.gateway.litellm") as mock_litellm,
        patch.object(
            gateway, "_get_cheaper_fallback", return_value="gpt-4o-mini-2024-07-18"
        ),
    ):
        _patch_litellm(mock_litellm, response)
        gateway.set_cost_tracker(CostTracker(session, settings))
        result = await gateway.acompletion(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-6",  # moderate tier — has a cheaper fallback
        )

    assert result.model == "gpt-4o-mini-2024-07-18"
