"""Native JSON-schema structured outputs (2D).

Locks ``build_native_response_format`` (provider mappings, json_object mode,
disabled-passthrough), the gateway ``response_schema``→native wiring, the
plain ``response_format`` forwarding, and the pre-emptive Anthropic
tool_choice-conflict guard. The feature is OFF by default → ``response_schema``
is ignored (back-compat).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import NativeStructuredSettings, Settings
from src.llm.gateway import LLMGateway
from src.llm.structured_output import (
    build_native_response_format,
    is_anthropic_4x_or_newer,
)

_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"x": {"type": "integer"}}}


def _settings(on: bool = True) -> SimpleNamespace:
    return SimpleNamespace(enabled=on)


def _gw_settings(on: bool = False) -> Settings:
    return Settings(native_structured=NativeStructuredSettings(enabled=on))


def _make_gateway(settings: Settings) -> LLMGateway:
    with patch.object(LLMGateway, "_configure_litellm"):
        gw = LLMGateway(settings)
    gw._rate_limiter = MagicMock()
    gw._rate_limiter.acquire = AsyncMock(return_value=None)
    return gw


def _mock_resp() -> MagicMock:
    msg = MagicMock()
    msg.content = '{"x": 1}'
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


# ─── is_anthropic_4x_or_newer ───────────────────────────────────────


class TestAnthropicVersionDetect:
    def test_4x_models(self) -> None:
        assert is_anthropic_4x_or_newer("claude-haiku-4-5-20251001")
        assert is_anthropic_4x_or_newer("claude-sonnet-4-6")
        assert is_anthropic_4x_or_newer("claude-opus-4-8")

    def test_pre_4x_models(self) -> None:
        assert not is_anthropic_4x_or_newer("claude-haiku-2-5")
        assert not is_anthropic_4x_or_newer("claude-sonnet-3-5")

    def test_non_anthropic(self) -> None:
        assert not is_anthropic_4x_or_newer("gpt-4o-mini-2024-07-18")


# ─── build_native_response_format ───────────────────────────────────


class TestBuildNativeResponseFormat:
    def test_disabled_returns_none(self) -> None:
        assert build_native_response_format(_SCHEMA, "openai", _settings(on=False)) is None

    def test_openai_json_schema_strict(self) -> None:
        out = build_native_response_format(_SCHEMA, "openai", _settings())
        assert out == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": _SCHEMA, "strict": True},
        }

    def test_deepseek_json_schema_strict(self) -> None:
        out = build_native_response_format(_SCHEMA, "deepseek", _settings())
        assert out is not None
        assert out["type"] == "json_schema"
        assert out["json_schema"]["strict"] is True

    def test_anthropic_json_schema_no_strict(self) -> None:
        out = build_native_response_format(_SCHEMA, "anthropic", _settings())
        assert out is not None
        assert out["type"] == "json_schema"
        assert "strict" not in out["json_schema"]

    def test_gemini_json_schema_no_strict(self) -> None:
        out = build_native_response_format(_SCHEMA, "google", _settings())
        assert out is not None
        assert out["type"] == "json_schema"
        assert "strict" not in out["json_schema"]

    def test_unknown_provider_falls_back_to_json_object(self) -> None:
        out = build_native_response_format(_SCHEMA, "groq", _settings())
        assert out == {"type": "json_object"}

    def test_no_schema_returns_json_object_mode(self) -> None:
        out = build_native_response_format(None, "openai", _settings())
        assert out == {"type": "json_object"}

    def test_custom_schema_name(self) -> None:
        out = build_native_response_format(
            _SCHEMA, "openai", _settings(), schema_name="plan"
        )
        assert out is not None
        assert out["json_schema"]["name"] == "plan"


# ─── gateway wiring + guard ─────────────────────────────────────────


class TestGatewayNativeStructured:
    @pytest.mark.asyncio
    async def test_response_schema_forwarded_to_kwargs_openai(self) -> None:
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
                model="gpt-4o-mini-2024-07-18",
                response_schema=_SCHEMA,
            )
            kw = mock_litellm.acompletion.call_args.kwargs
        assert kw["response_format"]["type"] == "json_schema"
        assert kw["response_format"]["json_schema"]["strict"] is True

    @pytest.mark.asyncio
    async def test_response_schema_ignored_when_disabled(self) -> None:
        """With the feature OFF, response_schema is ignored (no response_format)."""
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
                model="gpt-4o-mini-2024-07-18",
                response_schema=_SCHEMA,
            )
            kw = mock_litellm.acompletion.call_args.kwargs
        assert "response_format" not in kw

    @pytest.mark.asyncio
    async def test_raw_response_format_forwarded(self) -> None:
        """A plain response_format (not via the feature) is forwarded unchanged."""
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
                model="gpt-4o-mini-2024-07-18",
                response_format={"type": "json_object"},
            )
            kw = mock_litellm.acompletion.call_args.kwargs
        assert kw["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_pre4x_anthropic_tool_choice_conflict_dropped(self) -> None:
        """json_schema + explicit tool_choice on pre-4.x Anthropic → tool_choice
        dropped + warning (tool-conversion forces tool_choice → 400 otherwise)."""
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
                model="claude-haiku-2-5",  # pre-4.x anthropic
                response_schema=_SCHEMA,
                tool_choice={"type": "function", "function": {"name": "f"}},
            )
            kw = mock_litellm.acompletion.call_args.kwargs
        assert kw["response_format"]["type"] == "json_schema"
        assert "tool_choice" not in kw  # dropped by the guard

    @pytest.mark.asyncio
    async def test_4x_anthropic_keeps_tool_choice(self) -> None:
        """4.x Anthropic uses output_format (no forced tool_choice) → no guard."""
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
                model="claude-haiku-4-5-20251001",  # 4.x anthropic
                response_schema=_SCHEMA,
                tool_choice={"type": "function", "function": {"name": "f"}},
            )
            kw = mock_litellm.acompletion.call_args.kwargs
        assert "tool_choice" in kw  # NOT dropped on 4.x
