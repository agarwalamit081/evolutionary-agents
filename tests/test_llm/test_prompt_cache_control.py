"""Native prompt caching (2A): Anthropic ``cache_control`` breakpoints.

Locks the opt-in, Anthropic-only message transformation and the cache-token
read-back. The feature is OFF by default → zero behavior change; these tests
assert present-when-enabled / absent-when-disabled across the helper, the
``_parse_response`` Usage read-back, and the live gateway wiring.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import PromptCacheControlSettings, Settings
from src.llm.gateway import LLMGateway
from src.llm.prompt_cache_control import inject_cache_breakpoints


# ─── helpers ─────────────────────────────────────────────────────────


def _make_settings(*, cache_on: bool = False, min_tokens: int = 1024) -> Settings:
    return Settings(
        prompt_cache=PromptCacheControlSettings(
            enabled=cache_on, min_system_tokens=min_tokens
        )
    )


def _make_gateway(settings: Settings) -> LLMGateway:
    with patch.object(LLMGateway, "_configure_litellm"):
        gw = LLMGateway(settings)
    gw._rate_limiter = MagicMock()
    gw._rate_limiter.acquire = AsyncMock(return_value=None)
    return gw


def _long_system(n_chars: int = 8192) -> str:
    """A system prompt comfortably above the default 1024-token floor."""
    return "You are a meticulous analyst. " * (n_chars // 28)


def _mock_resp(content: str = "ok") -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.reasoning_content = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    # A real litellm.Usage leaves the cache fields None when the provider did
    # not report caching (MagicMock would otherwise auto-vivify them truthy).
    usage._cache_read_input_tokens = None
    usage._cache_creation_input_tokens = None
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ─── inject_cache_breakpoints unit tests ─────────────────────────────


class TestInjectCacheBreakpoints:
    def test_disabled_passthrough_same_object(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": _long_system()},
            {"role": "user", "content": "hi"},
        ]
        out = inject_cache_breakpoints(msgs, "anthropic", enabled=False)
        assert out is msgs  # untouched when off

    def test_non_anthropic_passthrough_same_object(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": _long_system()},
        ]
        out = inject_cache_breakpoints(msgs, "openai", enabled=True)
        assert out is msgs  # cache_control is Anthropic-only

    def test_anthropic_long_system_gets_breakpoint(self) -> None:
        system_text = _long_system()
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": "hi"},
        ]
        out = inject_cache_breakpoints(msgs, "anthropic", enabled=True)
        assert out is not msgs
        assert out[0]["role"] == "system"
        content = out[0]["content"]
        assert isinstance(content, list) and len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == system_text
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        # The user message is untouched.
        assert out[1] == {"role": "user", "content": "hi"}

    def test_anthropic_short_system_unchanged(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": "be brief"},  # ~2 tokens
            {"role": "user", "content": "hi"},
        ]
        out = inject_cache_breakpoints(msgs, "anthropic", enabled=True)
        # Below the floor → no breakpoint; same object returned.
        assert out is msgs

    def test_anthropic_structured_content_tags_trailing_block(self) -> None:
        msgs: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": _long_system()},
                ],
            },
        ]
        out = inject_cache_breakpoints(msgs, "anthropic", enabled=True)
        assert out is not msgs
        assert out[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        # Original input block is NOT mutated.
        assert "cache_control" not in msgs[0]["content"][-1]

    def test_input_not_mutated(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": _long_system()},
        ]
        snapshot = dict(msgs[0])
        inject_cache_breakpoints(msgs, "anthropic", enabled=True)
        assert msgs[0] == snapshot  # caller's dicts stay reusable

    def test_only_first_qualifying_system_marked(self) -> None:
        """Anthropic caps breakpoints; exactly one is injected."""
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": _long_system()},
            {"role": "system", "content": _long_system(6000)},
        ]
        out = inject_cache_breakpoints(msgs, "anthropic", enabled=True)
        marked = sum(
            1
            for m in out
            if m.get("role") == "system"
            and isinstance(m.get("content"), list)
            and m["content"][-1].get("cache_control")
        )
        assert marked == 1


# ─── _parse_response cache-token read-back ───────────────────────────


class TestParseResponseCacheTokens:
    def test_reads_cache_tokens_when_present(self) -> None:
        gw = _make_gateway(_make_settings(cache_on=True))
        resp = _mock_resp()
        resp.usage._cache_read_input_tokens = 4096  # type: ignore[attr-defined]
        resp.usage._cache_creation_input_tokens = 1024  # type: ignore[attr-defined]

        out = gw._parse_response(resp, "claude-haiku-4-5-20251001", "anthropic")
        assert out.cache_read_tokens == 4096
        assert out.cache_creation_tokens == 1024

    def test_zero_cache_tokens_when_absent(self) -> None:
        gw = _make_gateway(_make_settings())
        resp = _mock_resp()
        # No _cache_* attributes on usage → getattr defaults → 0.
        out = gw._parse_response(resp, "gpt-4o-mini-2024-07-18", "openai")
        assert out.cache_read_tokens == 0
        assert out.cache_creation_tokens == 0


# ─── gateway wiring: messages reach litellm with the breakpoint ──────


class TestGatewayWiring:
    @pytest.mark.asyncio
    async def test_enabled_anthropic_messages_carry_breakpoint(self) -> None:
        """End-to-end: with caching ON, the messages handed to litellm carry an
        Anthropic cache_control block on the system message."""
        gw = _make_gateway(_make_settings(cache_on=True))
        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_resp())
            mock_litellm.Usage = MagicMock
            for attr in (
                "RateLimitError", "Timeout", "ServiceUnavailableError",
                "APIConnectionError", "AuthenticationError", "BadRequestError",
            ):
                setattr(mock_litellm, attr, Exception)

            await gw.acompletion(
                messages=[
                    {"role": "system", "content": _long_system()},
                    {"role": "user", "content": "summarize"},
                ],
                model="claude-haiku-4-5-20251001",
            )
            captured = list(
                mock_litellm.acompletion.call_args.kwargs["messages"]
            )

        sys_msg = next(m for m in captured if m.get("role") == "system")
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_disabled_messages_have_no_breakpoint(self) -> None:
        """With caching OFF, the messages handed to litellm carry no marker."""
        gw = _make_gateway(_make_settings(cache_on=False))
        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_resp())
            mock_litellm.Usage = MagicMock
            for attr in (
                "RateLimitError", "Timeout", "ServiceUnavailableError",
                "APIConnectionError", "AuthenticationError", "BadRequestError",
            ):
                setattr(mock_litellm, attr, Exception)

            await gw.acompletion(
                messages=[
                    {"role": "system", "content": _long_system()},
                    {"role": "user", "content": "summarize"},
                ],
                model="claude-haiku-4-5-20251001",
            )
            captured = list(
                mock_litellm.acompletion.call_args.kwargs["messages"]
            )

        sys_msg = next(m for m in captured if m.get("role") == "system")
        # No cache_control injected when disabled.
        assert isinstance(sys_msg["content"], str)
        assert "cache_control" not in sys_msg
