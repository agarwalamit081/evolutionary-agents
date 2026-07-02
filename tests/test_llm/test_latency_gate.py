"""Tests for src.llm.latency_gate and gateway latency-gate integration.

Mirrors the circuit-breaker test structure: a pure-gate section (state machine
without the gateway/litellm) and a gateway-integration section (the fallback
loop skips a provider the gate has demoted).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import LatencyGateSettings, Settings
from src.llm.gateway import LLMGateway
from src.llm.latency_gate import LatencyGate, LatencyGateOpenError


# ─── Pure gate behavior ───────────────────────────────────────────────


def _gate(**overrides: Any) -> LatencyGate:
    defaults: dict[str, Any] = {
        "threshold_ms": 100.0,
        "min_samples": 3,
        "cooldown_s": 120.0,
        "alpha": 0.5,
        "enabled": True,
    }
    defaults.update(overrides)
    return LatencyGate(**defaults)


class TestLatencyGateCore:
    """EWMA demotion behavior without the gateway or litellm."""

    @pytest.mark.asyncio
    async def test_disabled_gate_is_noop(self) -> None:
        gate = _gate(enabled=False)
        # A disabled gate must not demote even under sustained slow calls...
        for _ in range(10):
            await gate.record_call("zai", 10_000.0)
        assert gate.get_demoted_until("zai") == 0.0
        # ...and before_call must never raise.
        await gate.before_call("zai")

    @pytest.mark.asyncio
    async def test_under_threshold_never_demotes(self) -> None:
        gate = _gate(threshold_ms=100.0)
        for _ in range(5):
            await gate.record_call("openai", 50.0)
        assert gate.get_demoted_until("openai") == 0.0
        await gate.before_call("openai")

    @pytest.mark.asyncio
    async def test_slow_success_demotes_after_min_samples(self) -> None:
        gate = _gate(threshold_ms=100.0, min_samples=3)
        # Two slow calls: under min_samples → not yet demoted.
        await gate.record_call("deepseek", 500.0)
        await gate.record_call("deepseek", 500.0)
        assert gate.get_demoted_until("deepseek") == 0.0
        # Third slow call: EWMA 500 > 100, samples >= 3 → demoted.
        await gate.record_call("deepseek", 500.0)
        assert gate.get_demoted_until("deepseek") > 0.0
        with pytest.raises(LatencyGateOpenError):
            await gate.before_call("deepseek")

    @pytest.mark.asyncio
    async def test_does_not_demote_before_min_samples(self) -> None:
        """A single egregious outlier must not demote (min_samples buffers)."""
        gate = _gate(threshold_ms=100.0, min_samples=3)
        await gate.record_call("groq", 10_000.0)
        await gate.record_call("groq", 10_000.0)
        assert gate.get_demoted_until("groq") == 0.0
        await gate.before_call("groq")  # not demoted

    @pytest.mark.asyncio
    async def test_fast_call_lowers_ewma_below_threshold(self) -> None:
        """A fast recovery drags the EWMA under the threshold before the
        min_samples demotion gate opens, so no demotion fires."""
        # min_samples=4 lets the EWMA decay below the threshold (sample 4) after
        # the slow outlier (sample 1). With min_samples=3 it would demote at the
        # 3rd call (EWMA 132.5 > 100) before recovery could pull it down.
        gate = _gate(threshold_ms=100.0, min_samples=4, alpha=0.5)
        await gate.record_call("zai", 500.0)  # ewma = 500
        await gate.record_call("zai", 10.0)  # ewma = 255
        await gate.record_call("zai", 10.0)  # ewma = 132.5 (> 100, but samples 3 < 4)
        await gate.record_call("zai", 10.0)  # ewma = 71.25 (< 100) → no demotion
        assert gate.get_demoted_until("zai") == 0.0

    @pytest.mark.asyncio
    async def test_demotion_blocks_then_readmits_after_cooldown(self) -> None:
        """A demoted provider is skipped until cooldown elapses, then re-admitted
        for a probe (self-healing)."""
        gate = _gate(threshold_ms=100.0, min_samples=2, cooldown_s=60.0)
        clock = {"t": 1000.0}
        gate._clock = lambda: clock["t"]  # type: ignore[assignment]
        await gate.record_call("mistral", 500.0)
        await gate.record_call("mistral", 500.0)  # demoted_until = 1060.0
        assert gate.get_demoted_until("mistral") == pytest.approx(1060.0)

        # Before cooldown: blocked.
        clock["t"] = 1000.1
        with pytest.raises(LatencyGateOpenError):
            await gate.before_call("mistral")

        # After cooldown: re-admitted (demotion cleared).
        clock["t"] = 1061.0
        await gate.before_call("mistral")  # must not raise
        assert gate.get_demoted_until("mistral") == 0.0

    @pytest.mark.asyncio
    async def test_per_provider_isolation(self) -> None:
        gate = _gate(threshold_ms=100.0, min_samples=2)
        for _ in range(3):
            await gate.record_call("zai", 500.0)
        # zai demoted...
        assert gate.get_demoted_until("zai") > 0.0
        # ...openai unaffected.
        assert gate.get_demoted_until("openai") == 0.0
        await gate.before_call("openai")  # must not raise

    @pytest.mark.asyncio
    async def test_non_numeric_or_negative_latency_ignored(self) -> None:
        gate = _gate(threshold_ms=100.0, min_samples=1)
        await gate.record_call("deepseek", None)  # type: ignore[arg-type]
        await gate.record_call("deepseek", "slow")  # type: ignore[arg-type]
        await gate.record_call("deepseek", float("nan"))
        await gate.record_call("deepseek", -5.0)
        # None of these corrupted the EWMA or demoted.
        assert gate.get_demoted_until("deepseek") == 0.0
        await gate.before_call("deepseek")

    @pytest.mark.asyncio
    async def test_demotion_increments_metric(self) -> None:
        fake_counter = MagicMock()
        with patch("src.llm.latency_gate.LATENCY_GATE_DEMOTIONS", fake_counter):
            gate = _gate(threshold_ms=100.0, min_samples=2)
            await gate.record_call("zai", 500.0)
            await gate.record_call("zai", 500.0)
        fake_counter.labels.assert_called_once_with(provider="zai")
        fake_counter.labels.return_value.inc.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_metric_failure_does_not_break_demotion(self) -> None:
        """A broken/missing Prometheus registry must never break the gate."""
        broken_counter = MagicMock()
        broken_counter.labels.side_effect = RuntimeError("registry gone")
        with patch("src.llm.latency_gate.LATENCY_GATE_DEMOTIONS", broken_counter):
            gate = _gate(threshold_ms=100.0, min_samples=2)
            await gate.record_call("zai", 500.0)
            # Must not raise despite the metric blowing up.
            await gate.record_call("zai", 500.0)
        # Demotion still applied.
        assert gate.get_demoted_until("zai") > 0.0


class TestLatencyGateDefaults:
    """The production default is OFF + a threshold above the moderate primaries."""

    def test_default_settings_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "LATENCY_GATE_ENABLED",
            "LATENCY_GATE_THRESHOLD_MS",
            "LATENCY_GATE_MIN_SAMPLES",
        ):
            monkeypatch.delenv(var, raising=False)
        s = LatencyGateSettings(_env_file=None)
        assert s.latency_gate_enabled is False
        assert s.latency_gate_threshold_ms == 150_000.0
        assert s.latency_gate_min_samples == 3


# ─── Gateway integration ──────────────────────────────────────────────


def _make_settings(**overrides: Any) -> Settings:
    return Settings(**overrides)


def _make_litellm_response(content: str = "ok") -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 3
    usage.total_tokens = 8
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_gateway() -> LLMGateway:
    with patch.object(LLMGateway, "_configure_litellm"):
        gw = LLMGateway(_make_settings())
    gw._rate_limiter = MagicMock()
    gw._rate_limiter.acquire = AsyncMock(return_value=None)
    # Replace the (default-off) gate with an enabled one for the test.
    gw._latency_gate = _gate(threshold_ms=100.0, min_samples=2)
    return gw


class TestGatewayLatencyGateIntegration:
    """The fallback loop skips a provider the latency gate has demoted."""

    @pytest.mark.asyncio
    async def test_demoted_provider_skipped_to_fallback(self) -> None:
        """Pre-demote the primary provider's latency gate; assert the response is
        served by a fallback-chain provider and the primary is skipped."""
        gw = _make_gateway()

        # claude-sonnet-4-6 (anthropic) is the primary; pre-demote anthropic.
        await gw._latency_gate.record_call("anthropic", 500.0)
        await gw._latency_gate.record_call("anthropic", 500.0)
        assert gw._latency_gate.get_demoted_until("anthropic") > 0.0

        mock_resp = _make_litellm_response("fallback served")
        call_models: list[str] = []

        async def _side_effect(**kwargs: Any) -> MagicMock:
            call_models.append(str(kwargs.get("model", "")))
            return mock_resp

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_side_effect)
            mock_litellm.Usage = MagicMock
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            result = await gw._execute_with_fallback(
                messages=[{"role": "user", "content": "hi"}],
                model="claude-sonnet-4-6",
            )

        assert result.content == "fallback served"
        # The anthropic (primary) model must NOT have been attempted.
        assert not any("claude-sonnet-4-6" in m for m in call_models), call_models
        assert result.provider != "anthropic"

