"""Per-tier reasoning/thinking control (2C).

Locks ``thinking_params_for`` (provider mappings, tier gating, the Anthropic
temperature invariant) and the gateway merge that makes a caller's explicit
``thinking``/``reasoning_effort`` always win. The feature is OFF by default →
no thinking params are emitted when disabled.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import ReasoningControlSettings, Settings
from src.graph.enums import TaskComplexity
from src.llm.gateway import LLMGateway
from src.llm.thinking_control import thinking_params_for


def _settings(
    *,
    on: bool = True,
    complex_thinking: str = "medium",
    simple_thinking: str = "none",
    budget_complex: int = 8000,
    budget_medium: int = 4000,
) -> SimpleNamespace:
    """A lightweight stand-in for ReasoningControlSettings (pure unit tests)."""
    return SimpleNamespace(
        enabled=on,
        complex_thinking=complex_thinking,
        simple_thinking=simple_thinking,
        anthropic_budget_tokens_complex=budget_complex,
        anthropic_budget_tokens_medium=budget_medium,
    )


def _gw_settings(*, on: bool = True) -> Settings:
    return Settings(reasoning=ReasoningControlSettings(enabled=on))


def _make_gateway(settings: Settings) -> LLMGateway:
    with patch.object(LLMGateway, "_configure_litellm"):
        gw = LLMGateway(settings)
    gw._rate_limiter = MagicMock()
    gw._rate_limiter.acquire = AsyncMock(return_value=None)
    return gw


def _mock_resp() -> MagicMock:
    msg = MagicMock()
    msg.content = "ok"
    msg.tool_calls = None
    msg.reasoning_content = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 1
    usage.completion_tokens = 1
    usage.total_tokens = 2
    usage._cache_read_input_tokens = None
    usage._cache_creation_input_tokens = None
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ─── thinking_params_for unit tests ─────────────────────────────────


class TestThinkingParamsFor:
    def test_disabled_returns_empty(self) -> None:
        s = _settings(on=False)
        assert thinking_params_for(TaskComplexity.COMPLEX, "anthropic", "claude-haiku-4-5-20251001", s) == {}

    def test_none_complexity_returns_empty(self) -> None:
        s = _settings()
        assert thinking_params_for(None, "anthropic", "claude-haiku-4-5-20251001", s) == {}

    def test_complex_anthropic_enables_with_budget_and_temp(self) -> None:
        s = _settings()
        out = thinking_params_for(TaskComplexity.COMPLEX, "anthropic", "claude-haiku-4-5-20251001", s)
        assert out["thinking"] == {"type": "enabled", "budget_tokens": 8000}
        assert out["temperature"] == 1.0

    def test_critical_anthropic_uses_complex_budget(self) -> None:
        s = _settings(budget_complex=1234)
        out = thinking_params_for(TaskComplexity.CRITICAL, "anthropic", "claude-sonnet-4-6", s)
        assert out["thinking"]["budget_tokens"] == 1234

    def test_simple_anthropic_no_param(self) -> None:
        # Anthropic thinking is off by default → no disable emitted.
        s = _settings()
        assert thinking_params_for(TaskComplexity.SIMPLE, "anthropic", "claude-haiku-4-5-20251001", s) == {}

    def test_complex_deepseek_enables_via_extra_body(self) -> None:
        s = _settings()
        out = thinking_params_for(TaskComplexity.COMPLEX, "deepseek", "deepseek-v4-flash", s)
        assert out == {"extra_body": {"thinking": {"type": "enabled"}}}

    def test_simple_deepseek_disabled_via_extra_body(self) -> None:
        # DeepSeek thinking is ON by default → explicit disable on trivial tasks.
        s = _settings()
        out = thinking_params_for(TaskComplexity.SIMPLE, "deepseek", "deepseek-v4-flash", s)
        assert out == {"extra_body": {"thinking": {"type": "disabled"}}}

    def test_complex_zai_enables(self) -> None:
        s = _settings()
        out = thinking_params_for(TaskComplexity.COMPLEX, "zai", "glm-4.7", s)
        assert out == {"extra_body": {"thinking": {"type": "enabled"}}}

    def test_simple_zai_no_param(self) -> None:
        # GLM off by default → no disable for trivial tasks.
        s = _settings()
        assert thinking_params_for(TaskComplexity.SIMPLE, "zai", "glm-4.7", s) == {}

    def test_complex_openai_oseries_reasoning_effort(self) -> None:
        s = _settings()
        out = thinking_params_for(TaskComplexity.COMPLEX, "openai", "o3-mini", s)
        assert out == {"reasoning_effort": "medium"}

    def test_unsupported_provider_returns_empty(self) -> None:
        s = _settings()
        assert thinking_params_for(TaskComplexity.COMPLEX, "groq", "llama-3.3-70b-versatile", s) == {}

    def test_complex_thinking_none_disables_everywhere(self) -> None:
        # If the operator sets complex_thinking="none", even complex tasks get
        # no thinking (deepseek still gets its default-disable).
        s = _settings(complex_thinking="none")
        assert thinking_params_for(TaskComplexity.COMPLEX, "anthropic", "claude-haiku-4-5-20251001", s) == {}
        assert thinking_params_for(TaskComplexity.COMPLEX, "deepseek", "deepseek-v4-flash", s) == {
            "extra_body": {"thinking": {"type": "disabled"}}
        }


# ─── gateway merge + caller-override ────────────────────────────────


class TestGatewayThinkingMerge:
    @pytest.mark.asyncio
    async def test_tier_thinking_injected_when_caller_omits(self) -> None:
        gw = _make_gateway(_gw_settings(on=True))
        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_resp())
            mock_litellm.Usage = MagicMock
            for attr in (
                "RateLimitError", "Timeout", "ServiceUnavailableError",
                "APIConnectionError", "AuthenticationError", "BadRequestError",
            ):
                setattr(mock_litellm, attr, Exception)
            await gw.acompletion(
                messages=[{"role": "user", "content": "x"}],
                model="claude-haiku-4-5-20251001",
                complexity=TaskComplexity.COMPLEX,
            )
            kw = mock_litellm.acompletion.call_args.kwargs
        assert kw["thinking"] == {"type": "enabled", "budget_tokens": 8000}
        assert kw["temperature"] == 1.0

    @pytest.mark.asyncio
    async def test_caller_thinking_override_wins(self) -> None:
        """A caller's explicit ``thinking`` is not clobbered by the tier."""
        gw = _make_gateway(_gw_settings(on=True))
        explicit = {"type": "enabled", "budget_tokens": 5000}
        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_resp())
            mock_litellm.Usage = MagicMock
            for attr in (
                "RateLimitError", "Timeout", "ServiceUnavailableError",
                "APIConnectionError", "AuthenticationError", "BadRequestError",
            ):
                setattr(mock_litellm, attr, Exception)
            await gw.acompletion(
                messages=[{"role": "user", "content": "x"}],
                model="claude-haiku-4-5-20251001",
                complexity=TaskComplexity.COMPLEX,
                thinking=explicit,
            )
            kw = mock_litellm.acompletion.call_args.kwargs
        assert kw["thinking"] == explicit  # caller wins, not the 8000-budget tier

    @pytest.mark.asyncio
    async def test_disabled_no_thinking_kwarg(self) -> None:
        gw = _make_gateway(_gw_settings(on=False))
        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_resp())
            mock_litellm.Usage = MagicMock
            for attr in (
                "RateLimitError", "Timeout", "ServiceUnavailableError",
                "APIConnectionError", "AuthenticationError", "BadRequestError",
            ):
                setattr(mock_litellm, attr, Exception)
            await gw.acompletion(
                messages=[{"role": "user", "content": "x"}],
                model="claude-haiku-4-5-20251001",
                complexity=TaskComplexity.COMPLEX,
            )
            kw = mock_litellm.acompletion.call_args.kwargs
        assert "thinking" not in kw
