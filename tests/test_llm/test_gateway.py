"""Tests for src.llm.gateway — LLMGateway with resilience features."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.graph.enums import TaskComplexity
from src.llm.exceptions import BudgetExhaustedError
from src.llm.gateway import LLMGateway
from src.llm.models import LLMResponse, ToolCallResponse


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_settings(**overrides: Any) -> Settings:
    """Create a Settings instance suitable for testing (no .env required)."""
    return Settings(**overrides)


def _make_litellm_response(
    content: str | None = "Hello!",
    model: str = "gpt-4o-mini-2024-07-18",  # pyright: ignore[reportUnusedParameter]
    input_tokens: int = 10,
    output_tokens: int = 5,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Build a mock that mimics litellm.ModelResponse structure."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    usage.total_tokens = input_tokens + output_tokens

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_tool_call(tc_id: str = "tc_1", name: str = "get_weather", arguments: str = '{"city":"SF"}') -> MagicMock:
    """Build a mock tool_call object."""
    tc = MagicMock()
    tc.id = tc_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_gateway(settings: Settings | None = None) -> LLMGateway:
    """Create an LLMGateway with mocked internal components."""
    if settings is None:
        settings = _make_settings()
    with patch.object(LLMGateway, "_configure_litellm"):
        gw = LLMGateway(settings)
    # Replace rate limiter with a no-op so tests never block
    gw._rate_limiter = MagicMock()
    gw._rate_limiter.acquire = AsyncMock(return_value=None)
    return gw


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> Settings:
    return _make_settings()


@pytest.fixture
def gateway(settings: Settings) -> LLMGateway:
    return _make_gateway(settings)


@pytest.fixture
def simple_messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "Hello world"}]


# ─── Test __init__ ───────────────────────────────────────────────────


class TestLLMGatewayInit:
    """Tests for LLMGateway.__init__ attribute setup."""

    def test_init_stores_settings(self, settings: Settings) -> None:
        """Settings reference is stored on the gateway."""
        gw = _make_gateway(settings)
        assert gw._settings is settings

    def test_init_creates_model_router(self, settings: Settings) -> None:
        """ModelRouter is initialized during construction."""
        gw = _make_gateway(settings)
        assert gw._model_router is not None

    def test_init_cost_tracker_default_none(self, settings: Settings) -> None:
        """Cost tracker starts as None (lazy injection)."""
        gw = _make_gateway(settings)
        assert gw._cost_tracker is None

    def test_init_cache_default_none(self, settings: Settings) -> None:
        """Prompt cache starts as None (lazy injection)."""
        gw = _make_gateway(settings)
        assert gw._cache is None

    def test_set_cost_tracker_injects(self, gateway: LLMGateway) -> None:
        """set_cost_tracker injects the tracker."""
        tracker = MagicMock()
        gateway.set_cost_tracker(tracker)
        assert gateway._cost_tracker is tracker

    def test_set_cache_injects(self, gateway: LLMGateway) -> None:
        """set_cache injects the cache."""
        cache = MagicMock()
        gateway.set_cache(cache)
        assert gateway._cache is cache

    def test_set_rate_limiter_redis_forwards(self, gateway: LLMGateway) -> None:
        """set_rate_limiter_redis forwards the client to the limiter."""
        gateway._rate_limiter = MagicMock()
        client = MagicMock()

        gateway.set_rate_limiter_redis(client)

        gateway._rate_limiter.attach_redis.assert_called_once_with(client)

    def test_init_run_id_default_none(self, settings: Settings) -> None:
        """run_id starts as None (bound later from the run's thread_id)."""
        gw = _make_gateway(settings)
        assert gw._run_id is None

    def test_set_run_id_injects(self, gateway: LLMGateway) -> None:
        """set_run_id binds the per-run cost attribution key."""
        gateway.set_run_id("cli-q05")
        assert gateway._run_id == "cli-q05"


# ─── Test _extract_provider ──────────────────────────────────────────


class TestExtractProvider:
    """Tests for LLMGateway._extract_provider static method."""

    def test_openai_gpt_prefix(self) -> None:
        assert LLMGateway._extract_provider("gpt-4o-mini-2024-07-18") == "openai"

    def test_openai_embedding_prefix(self) -> None:
        assert LLMGateway._extract_provider("text-embedding-3-small") == "openai"

    def test_anthropic_claude_prefix(self) -> None:
        assert LLMGateway._extract_provider("claude-sonnet-4-6") == "anthropic"

    def test_deepseek_prefix(self) -> None:
        assert LLMGateway._extract_provider("deepseek-v4-flash") == "deepseek"

    def test_google_gemini_prefix(self) -> None:
        assert LLMGateway._extract_provider("gemini-2.5-flash") == "google"

    def test_zai_glm_prefix(self) -> None:
        assert LLMGateway._extract_provider("glm-4.7") == "zai"

    def test_groq_llama_prefix(self) -> None:
        assert LLMGateway._extract_provider("llama-3.1-8b-instant") == "groq"

    def test_provider_slash_prefix(self) -> None:
        """Models with provider/ prefix extract the provider part."""
        assert LLMGateway._extract_provider("groq/llama-3.3-70b-versatile") == "groq"

    def test_unknown_model_returns_unknown(self) -> None:
        assert LLMGateway._extract_provider("totally-unknown-model") == "unknown"

    def test_alibaba_qwen_prefix(self) -> None:
        """Qwen is an Alibaba model series served via the DashScope API."""
        assert LLMGateway._extract_provider("qwen3.5-flash") == "alibaba"
        assert LLMGateway._extract_provider("qwen3.7-plus") == "alibaba"


# ─── Test _estimate_tokens ──────────────────────────────────────────


class TestEstimateTokens:
    """Tests for LLMGateway._estimate_tokens static method."""

    def test_single_message(self) -> None:
        messages = [{"role": "user", "content": "a" * 40}]
        tokens = LLMGateway._estimate_tokens(messages)
        assert tokens == 10  # 40 chars / 4

    def test_multiple_messages(self) -> None:
        messages = [
            {"role": "system", "content": "a" * 20},
            {"role": "user", "content": "b" * 20},
        ]
        tokens = LLMGateway._estimate_tokens(messages)
        assert tokens == 10  # 40 total chars / 4

    def test_empty_messages_returns_minimum_one(self) -> None:
        """Empty messages list returns at least 1 token."""
        tokens = LLMGateway._estimate_tokens([])
        assert tokens == 1

    def test_string_input(self) -> None:
        tokens = LLMGateway._estimate_tokens("Hello world!")
        assert tokens == max(1, len("Hello world!") // 4)

    def test_minimum_one_token(self) -> None:
        """Even 0-length content returns at least 1."""
        messages = [{"role": "user", "content": ""}]
        tokens = LLMGateway._estimate_tokens(messages)
        assert tokens >= 1


# ─── Test _build_kwargs ──────────────────────────────────────────────


class TestBuildKwargs:
    """Tests for LLMGateway._build_kwargs."""

    def test_basic_kwargs(self, gateway: LLMGateway) -> None:
        kwargs = gateway._build_kwargs("gpt-4o-mini-2024-07-18", 0.7, 1024, None)
        assert kwargs["model"] == "gpt-4o-mini-2024-07-18"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1024

    def test_max_tokens_from_registry_when_none(self, gateway: LLMGateway) -> None:
        """When max_tokens is None, uses registry default for the model."""
        kwargs = gateway._build_kwargs("gpt-4o-mini-2024-07-18", 0.5, None, None)
        # gpt-4o-mini-2024-07-18 has max_output=16_000 in MODEL_REGISTRY
        assert kwargs["max_tokens"] == 16_000

    def test_max_tokens_fallback_4096_for_unknown_model(self, gateway: LLMGateway) -> None:
        """Unknown model with no registry entry falls back to 4096."""
        kwargs = gateway._build_kwargs("nonexistent-model-xyz", 0.5, None, None)
        assert kwargs["max_tokens"] == 4096

    def test_metadata_included_when_provided(self, gateway: LLMGateway) -> None:
        meta = {"task_id": "abc"}
        kwargs = gateway._build_kwargs("gpt-4o-mini-2024-07-18", 0.5, 100, meta)
        assert kwargs["metadata"] == meta

    def test_no_metadata_key_when_none(self, gateway: LLMGateway) -> None:
        kwargs = gateway._build_kwargs("gpt-4o-mini-2024-07-18", 0.5, 100, None)
        assert "metadata" not in kwargs

    def test_nvidia_shims_to_openai_with_nim_base(self, gateway: LLMGateway) -> None:
        """litellm in this build rejects the bare ``nvidia/`` provider prefix
        ("LLM Provider NOT provided"). The NVIDIA NIM endpoint is
        OpenAI-compatible, so a registered nvidia model_id is rewritten to the
        ``openai/`` shim against the pinned NIM api_base (verified live for all
        16 registered NVIDIA models)."""
        kwargs = gateway._build_kwargs("nvidia-llama-3.3-70b", 0.5, 100, None)
        assert kwargs["model"] == "openai/meta/llama-3.3-70b-instruct"
        assert kwargs["api_base"] == "https://integrate.api.nvidia.com/v1"

    def test_nvidia_nested_model_path_is_shimmed(self, gateway: LLMGateway) -> None:
        """The shim handles nested ``nvidia/<org>/<model>`` ids, not just
        ``nvidia/<model>`` — the org segment is preserved under openai/."""
        kwargs = gateway._build_kwargs("nvidia-qwen3-next-80b", 0.5, 100, None)
        assert kwargs["model"] == "openai/qwen/qwen3-next-80b-a3b-instruct"
        assert kwargs["api_base"] == "https://integrate.api.nvidia.com/v1"

    def test_non_nvidia_model_is_not_shimmed(self, gateway: LLMGateway) -> None:
        """A plain OpenAI model is left untouched — no openai/ prefix injected
        and no NIM base pinned."""
        kwargs = gateway._build_kwargs("gpt-4o-mini-2024-07-18", 0.5, 100, None)
        assert kwargs["model"] == "gpt-4o-mini-2024-07-18"
        assert kwargs.get("api_base") != "https://integrate.api.nvidia.com/v1"

    def test_alibaba_qwen_pins_dashscope_api_base(self) -> None:
        """Qwen (alibaba provider) calls pin api_base to the DashScope endpoint.

        Qwen models are registered with an ``openai/`` model_id prefix; without
        the api_base pin litellm would route them to OpenAI's endpoint using the
        DASHSCOPE_API_KEY and the call would fail. The key must also reach the
        call via the alibaba provider lookup. The default endpoint is the
        INTERNATIONAL (Bailian) one — DashScope keys are region-bound and the
        China endpoint rejects an intl key.
        """
        settings = _make_settings()
        settings.llm.dashscope_api_key = "sk-dashscope-test"
        gw = _make_gateway(settings)
        kwargs = gw._build_kwargs("qwen3.5-flash", 0.5, 100, None)
        assert kwargs["model"] == "openai/qwen3.5-flash"
        assert kwargs["api_base"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        assert kwargs["api_key"] == "sk-dashscope-test"

    def test_alibaba_api_base_override(self) -> None:
        """ALIBABA_API_BASE override is honored over the default endpoint."""
        settings = _make_settings()
        settings.llm.dashscope_api_key = "sk-dashscope-test"
        settings.llm.alibaba_api_base = "https://custom-dashscope.example.com/v1"
        gw = _make_gateway(settings)
        kwargs = gw._build_kwargs("qwen3.7-plus", 0.5, 100, None)
        assert kwargs["api_base"] == "https://custom-dashscope.example.com/v1"

    def test_non_alibaba_provider_has_no_api_base(self, gateway: LLMGateway) -> None:
        """Only nvidia/anthropic/alibaba providers receive an api_base pin."""
        kwargs = gateway._build_kwargs("gpt-4o-mini-2024-07-18", 0.5, 100, None)
        assert "api_base" not in kwargs

    def test_request_timeout_always_set(self, gateway: LLMGateway) -> None:
        """A hard timeout is passed to litellm so a dead provider can't stall.

        Regression: without an explicit ``timeout`` kwarg litellm falls back to
        its ~600s default, and one unresponsive provider hangs the whole run.
        """
        kwargs = gateway._build_kwargs("gpt-4o-mini-2024-07-18", 0.5, 100, None)
        assert "timeout" in kwargs, "timeout must be set on every litellm call"
        assert kwargs["timeout"] == gateway._settings.llm.request_timeout

    def test_request_timeout_override_honored(self) -> None:
        """REQUEST_TIMEOUT override flows through to the litellm kwargs."""
        settings = _make_settings()
        settings.llm.request_timeout = 15.0
        gw = _make_gateway(settings)
        kwargs = gw._build_kwargs("gpt-4o-mini-2024-07-18", 0.5, 100, None)
        assert kwargs["timeout"] == 15.0

    def test_per_call_timeout_override_wins_over_default(self) -> None:
        """A per-call ``timeout`` (e.g. codegen) overrides REQUEST_TIMEOUT.

        Code-gen calls (tool_create + evolution) legitimately exceed the reasoning
        default (observed live: a 58s deepseek-v4-pro codegen cut at 60s). They
        pass a longer timeout that must reach litellm unchanged.
        """
        settings = _make_settings()
        settings.llm.request_timeout = 45.0
        gw = _make_gateway(settings)
        kwargs = gw._build_kwargs(
            "gpt-4o-mini-2024-07-18", 0.5, 100, None, timeout=200.0
        )
        assert kwargs["timeout"] == 200.0

    def test_timeout_defaults_declared(self) -> None:
        """request_timeout defaults to 90s; codegen_timeout to 180s.

        Asserted against the declared field defaults (not a constructed instance)
        so the test is immune to ambient REQUEST_TIMEOUT/CODEGEN_TIMEOUT env.
        """
        from src.config.settings import LLMProviderSettings

        fields = LLMProviderSettings.model_fields
        assert fields["request_timeout"].default == 90.0
        assert fields["codegen_timeout"].default == 180.0

    def test_qwen_default_max_tokens_within_api_cap(self, gateway: LLMGateway) -> None:
        """qwen3.5-flash's registry max_output must stay within the DashScope
        max_tokens API cap.

        Regression: the gateway sends spec.max_output as max_tokens on calls
        that don't override it. qwen3.5-flash was registered with max_output=
        66_000, which DashScope hard-rejects with
        ``Range of max_tokens should be [1, 65536]`` — so every cheap-classify
        call 400'd and fell through the fallback chain. The registry value is
        the source of truth, so guard the cap here.
        """
        kwargs = gateway._build_kwargs("qwen3.5-flash", 0.5, None, None)
        assert kwargs["max_tokens"] <= 65_536


# ─── Test _get_cheaper_fallback ──────────────────────────────────────


class TestGetCheaperFallback:
    """Tests for LLMGateway._get_cheaper_fallback.

    ModelTier is a str enum (values: "very_cheap", "cheap", "moderate").
    The method uses _TIER_ORDER to compare tiers numerically and walks
    the model's fallback chain first, then scans the registry.
    """

    def test_returns_none_for_unknown_model(self, gateway: LLMGateway) -> None:
        result = gateway._get_cheaper_fallback("nonexistent-model")
        assert result is None

    def test_moderate_tier_returns_cheaper_model(self, gateway: LLMGateway) -> None:
        """Moderate-tier models return a cheaper fallback from their chain."""
        result = gateway._get_cheaper_fallback("claude-sonnet-4-6")
        assert result is not None
        # Should return a model in a cheaper tier (CHEAP or VERY_CHEAP)
        from src.config.model_registry import MODEL_REGISTRY, ModelTier

        fb_spec = MODEL_REGISTRY.get(result)
        assert fb_spec is not None
        assert fb_spec.tier in {ModelTier.CHEAP, ModelTier.VERY_CHEAP}

    def test_cheap_tier_returns_very_cheap_model(self, gateway: LLMGateway) -> None:
        """Cheap-tier models return a very-cheap fallback."""
        result = gateway._get_cheaper_fallback("claude-haiku-4-5-20251001")
        assert result is not None
        from src.config.model_registry import MODEL_REGISTRY, ModelTier

        fb_spec = MODEL_REGISTRY.get(result)
        assert fb_spec is not None
        assert fb_spec.tier == ModelTier.VERY_CHEAP

    def test_very_cheap_tier_returns_none(self, gateway: LLMGateway) -> None:
        """Very-cheap-tier models have no cheaper fallback — returns None."""
        result = gateway._get_cheaper_fallback("gpt-4o-mini-2024-07-18")
        assert result is None

    def test_prefers_fallback_chain_over_registry_scan(self, gateway: LLMGateway) -> None:
        """Models with defined fallback chains prefer chain models over registry scan."""
        result = gateway._get_cheaper_fallback("claude-sonnet-4-6")
        assert result is not None
        from src.config.model_registry import FALLBACK_CHAINS

        # Result should be from claude-sonnet-4-6's fallback chain
        chain = FALLBACK_CHAINS.get("claude-sonnet-4-6", [])
        if chain:
            # The result should be from the chain (cheaper tier)
            from src.config.model_registry import MODEL_REGISTRY

            fb_spec = MODEL_REGISTRY.get(result)
            assert fb_spec is not None
            assert fb_spec.tier != "moderate"

    def test_fallback_skips_provider_marked_disabled(
        self, gateway: LLMGateway, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the budget-fallback must skip every model whose provider
        is in _TEMPORARY_DISABLED_PROVIDERS. Without this guard, exhausting the
        daily budget would degrade a run onto a provider that returns 400/quota
        on every call, burning the whole fallback chain. (Monkeypatches the set
        so the test does not depend on which provider is temporarily disabled.)"""
        import src.llm.model_router as mr
        from src.config.model_registry import MODEL_REGISTRY, ModelTier

        # claude-sonnet-4-6 is MODERATE → it has cheaper fallbacks across
        # providers. Disable 'openai' and assert the result is still a cheaper
        # model, just never an openai one.
        monkeypatch.setattr(mr, "_TEMPORARY_DISABLED_PROVIDERS", frozenset({"openai"}))
        result = gateway._get_cheaper_fallback("claude-sonnet-4-6")
        assert result is not None
        fb_spec = MODEL_REGISTRY.get(result)
        assert fb_spec is not None
        assert fb_spec.tier in {ModelTier.CHEAP, ModelTier.VERY_CHEAP}
        assert fb_spec.provider != "openai", (
            f"budget fallback returned disabled provider 'openai' ({result})"
        )

    def test_is_provider_disabled_reads_temporary_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_provider_disabled returns True exactly for providers in the
        temporary-disabled set (whatever it currently holds)."""
        import src.llm.model_router as mr
        from src.llm.model_router import ModelRouter

        monkeypatch.setattr(mr, "_TEMPORARY_DISABLED_PROVIDERS", frozenset({"groq"}))
        assert ModelRouter.is_provider_disabled("groq") is True
        assert ModelRouter.is_provider_disabled("openai") is False


# ─── Test _parse_response ────────────────────────────────────────────


class TestParseResponse:
    """Tests for LLMGateway._parse_response."""

    def test_basic_text_response(self, gateway: LLMGateway) -> None:
        mock_resp = _make_litellm_response(
            content="Paris is the capital of France.",
            model="gpt-4o-mini-2024-07-18",
            input_tokens=20,
            output_tokens=8,
        )
        parsed = gateway._parse_response(mock_resp, "gpt-4o-mini-2024-07-18", "openai")

        assert isinstance(parsed, LLMResponse)
        assert parsed.content == "Paris is the capital of France."
        assert parsed.model == "gpt-4o-mini-2024-07-18"
        assert parsed.provider == "openai"
        assert parsed.input_tokens == 20
        assert parsed.output_tokens == 8
        assert parsed.total_tokens == 28
        assert parsed.finish_reason == "stop"
        assert parsed.tool_calls is None

    def test_response_with_tool_calls(self, gateway: LLMGateway) -> None:
        tc = _make_tool_call()
        mock_resp = _make_litellm_response(
            content="",
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        parsed = gateway._parse_response(mock_resp, "gpt-4o-mini-2024-07-18", "openai")

        assert parsed.tool_calls is not None
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["id"] == "tc_1"
        assert parsed.tool_calls[0]["function"]["name"] == "get_weather"
        assert parsed.tool_calls[0]["function"]["arguments"] == '{"city":"SF"}'

    def test_empty_content_becomes_empty_string(self, gateway: LLMGateway) -> None:
        mock_resp = _make_litellm_response(content=None)
        parsed = gateway._parse_response(mock_resp, "gpt-4o-mini-2024-07-18", "openai")
        assert parsed.content == ""

    def test_missing_usage_defaults_to_zero(self, gateway: LLMGateway) -> None:
        mock_resp = _make_litellm_response()
        mock_resp.usage = None  # Simulate missing usage
        parsed = gateway._parse_response(mock_resp, "test-model", "test")
        assert parsed.input_tokens == 0
        assert parsed.output_tokens == 0

    def test_cost_calculated(self, gateway: LLMGateway) -> None:
        mock_resp = _make_litellm_response(input_tokens=1000, output_tokens=500)
        parsed = gateway._parse_response(mock_resp, "test-model-xyz", "test")
        # Uses fallback pricing since test-model-xyz is not in registry
        assert parsed.cost_usd > 0


# ─── Test acompletion ────────────────────────────────────────────────


class TestAcompletion:
    """Tests for LLMGateway.acompletion end-to-end flow."""

    @pytest.mark.asyncio
    async def test_basic_completion_returns_llm_response(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        mock_resp = _make_litellm_response(content="Hi there!", input_tokens=5, output_tokens=3)

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.Usage = MagicMock
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            result = await gateway.acompletion(
                messages=simple_messages,
                model="gpt-4o-mini-2024-07-18",
            )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hi there!"
        assert result.model == "gpt-4o-mini-2024-07-18"
        assert result.provider == "openai"

    @pytest.mark.asyncio
    async def test_run_id_threads_into_record_usage(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """A bound run_id reaches CostTracker.record_usage so the cost row is
        attributable to the run. Regression for the always-NULL task_id gap:
        without this threading, per-run cost attribution is impossible."""
        mock_resp = _make_litellm_response(content="ok", input_tokens=4, output_tokens=2)
        tracker = MagicMock()
        tracker.record_usage = AsyncMock(return_value=0.001)
        tracker.check_budget = AsyncMock(return_value=(True, "ok"))
        gateway.set_cost_tracker(tracker)
        gateway.set_run_id("cli-q05")

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.Usage = MagicMock
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            await gateway.acompletion(
                messages=simple_messages,
                model="gpt-4o-mini-2024-07-18",
            )

        tracker.record_usage.assert_awaited_once()
        assert tracker.record_usage.call_args.kwargs["run_id"] == "cli-q05"

    @pytest.mark.asyncio
    async def test_per_call_timeout_threads_to_litellm(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """acompletion(timeout=...) threads through to the litellm call kwargs.

        End-to-end regression: the codegen per-call override (e.g. 180s) must reach
        the actual ``litellm.acompletion`` invocation, not just ``_build_kwargs``.
        """
        mock_resp = _make_litellm_response(content="ok", input_tokens=2, output_tokens=1)

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.Usage = MagicMock
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            await gateway.acompletion(
                messages=simple_messages,
                model="gpt-4o-mini-2024-07-18",
                timeout=200.0,
            )

        mock_litellm.acompletion.assert_awaited_once()
        assert mock_litellm.acompletion.call_args.kwargs["timeout"] == 200.0

    @pytest.mark.asyncio
    async def test_tool_choice_conflict_deepseek_retries_with_thinking_disabled(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """deepseek's thinking mode rejects a forced tool_choice (400 "Thinking
        mode does not support this tool_choice"). The gateway retries the SAME
        model with thinking DISABLED via extra_body and tool_choice KEPT — not
        dropped, and not bounced down the fallback chain.

        Verified live: extra_body={"thinking":{"type":"disabled"}} makes
        deepseek-v4-flash honor the forced tool_choice (litellm 1.83.14's
        native ``thinking=`` param only accepts "enabled" and drops "disabled",
        so extra_body is required). Dropping tool_choice instead made deepseek
        narrate rather than call the file tool (9 wasted write-nudges on q4).
        """

        class _FakeBadRequest(Exception):
            pass

        mock_resp = _make_litellm_response(content="ok")
        seen_tc: list[Any] = []
        seen_eb: list[Any] = []

        async def fake_acompletion(
            messages: list[dict[str, Any]],  # pyright: ignore[reportUnusedParameter]
            **kwargs: Any,
        ) -> Any:
            seen_tc.append(kwargs.get("tool_choice"))
            seen_eb.append(kwargs.get("extra_body"))
            if len(seen_tc) == 1:
                raise _FakeBadRequest(
                    'DeepseekException - {"error":{"message":'
                    '"Thinking mode does not support this tool_choice"}}'
                )
            return mock_resp

        # Ensure the deepseek primary is attempted first (not pre-filtered for
        # a missing test-env API key).
        gateway._model_router._has_provider_key = MagicMock(return_value=True)

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = fake_acompletion
            mock_litellm.Usage = MagicMock
            # Under the patch the except clause resolves these from the mock.
            mock_litellm.AuthenticationError = _FakeBadRequest
            mock_litellm.BadRequestError = _FakeBadRequest

            result = await gateway.acompletion(
                messages=simple_messages,
                model="deepseek-v4-flash",
                tool_choice={"type": "function", "function": {"name": "file_writer"}},
            )

        # Two calls on the same model: the retry keeps tool_choice AND disables
        # thinking via extra_body.
        assert result.content == "ok"
        assert len(seen_tc) == 2
        assert seen_tc[0] is not None
        assert seen_tc[1] is not None  # kept, not dropped
        assert seen_eb[1] == {"thinking": {"type": "disabled"}}

    @pytest.mark.asyncio
    async def test_tool_choice_conflict_non_deepseek_drops_tool_choice(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """For a non-deepseek model that rejects a forced tool_choice, the
        thinking-disable stage does not apply (DeepSeek-specific API); the
        gateway falls back to retrying with tool_choice dropped.
        """

        class _FakeBadRequest(Exception):
            pass

        mock_resp = _make_litellm_response(content="ok")
        seen_tc: list[Any] = []

        async def fake_acompletion(
            messages: list[dict[str, Any]],  # pyright: ignore[reportUnusedParameter]
            **kwargs: Any,
        ) -> Any:
            seen_tc.append(kwargs.get("tool_choice"))
            if len(seen_tc) == 1:
                raise _FakeBadRequest("invalid tool_choice")
            return mock_resp

        gateway._model_router._has_provider_key = MagicMock(return_value=True)

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = fake_acompletion
            mock_litellm.Usage = MagicMock
            mock_litellm.AuthenticationError = _FakeBadRequest
            mock_litellm.BadRequestError = _FakeBadRequest

            result = await gateway.acompletion(
                messages=simple_messages,
                model="gpt-4.1-mini-2025-04-14",  # provider openai, not deepseek
                tool_choice={"type": "function", "function": {"name": "file_writer"}},
            )

        assert result.content == "ok"
        assert len(seen_tc) == 2
        assert seen_tc[0] is not None
        assert seen_tc[1] is None  # dropped (fallback)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_key",
        [
            "deepseek-v4-flash",
            "alibaba-deepseek-v4-flash",
            "nvidia-deepseek-v4-flash",
        ],
    )
    async def test_proactive_thinking_disable_for_deepseek_variants(
        self,
        gateway: LLMGateway,
        simple_messages: list[dict[str, Any]],
        model_key: str,
    ) -> None:
        """P0 — deepseek-v4-flash ships with thinking ON; a FORCED tool_choice
        makes the provider 400 "Thinking mode does not support this
        tool_choice". The gateway must DISABLE thinking proactively on the
        FIRST call (via extra_body) so no thinking tokens are burned on a
        doomed reject→retry. Covers all three hostings (native,
        alibaba/DashScope, nvidia free-tier) — they share the
        ``deepseek-v4-flash`` registry substring. ``litellm.drop_params`` makes
        the extra_body a harmless no-op for any hosting that ignores it.
        """
        mock_resp = _make_litellm_response(content="ok")
        seen_eb: list[Any] = []
        seen_calls = 0

        async def fake_acompletion(
            messages: list[dict[str, Any]],  # pyright: ignore[reportUnusedParameter]
            **kwargs: Any,
        ) -> Any:
            nonlocal seen_calls
            seen_calls += 1
            seen_eb.append(kwargs.get("extra_body"))
            return mock_resp

        # Ensure the chosen primary is attempted (not pre-filtered for a missing
        # test-env API key).
        gateway._model_router._has_provider_key = MagicMock(return_value=True)

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = fake_acompletion
            mock_litellm.Usage = MagicMock
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            result = await gateway.acompletion(
                messages=simple_messages,
                model=model_key,
                tool_choice={"type": "function", "function": {"name": "file_writer"}},
            )

        # Single call — proactive disable means no reject→retry burn.
        assert seen_calls == 1
        assert result.content == "ok"
        # The very first call already disabled thinking.
        assert seen_eb[0] == {"thinking": {"type": "disabled"}}

    @pytest.mark.asyncio
    async def test_no_proactive_disable_without_forced_tool_choice(
        self,
        gateway: LLMGateway,
        simple_messages: list[dict[str, Any]],
    ) -> None:
        """With NO tool_choice set (the common discovery path — the execute
        node only forces a tool_choice on write-nudges), the proactive
        thinking-disable must NOT fire — deepseek may think freely when no
        tool is being forced.
        """
        mock_resp = _make_litellm_response(content="ok")
        seen_eb: list[Any] = []

        async def fake_acompletion(
            messages: list[dict[str, Any]],  # pyright: ignore[reportUnusedParameter]
            **kwargs: Any,
        ) -> Any:
            seen_eb.append(kwargs.get("extra_body"))
            return mock_resp

        gateway._model_router._has_provider_key = MagicMock(return_value=True)

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = fake_acompletion
            mock_litellm.Usage = MagicMock
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            await gateway.acompletion(
                messages=simple_messages,
                model="deepseek-v4-flash",
            )

        # No thinking-disable injected when no tool_choice is forced.
        assert seen_eb[0] is None

    @pytest.mark.asyncio
    async def test_no_proactive_disable_for_non_deepseek(
        self,
        gateway: LLMGateway,
        simple_messages: list[dict[str, Any]],
    ) -> None:
        """A forced tool_choice on a NON-deepseek model must NOT get the
        deepseek-specific thinking-disable injected.
        """
        mock_resp = _make_litellm_response(content="ok")
        seen_eb: list[Any] = []

        async def fake_acompletion(
            messages: list[dict[str, Any]],  # pyright: ignore[reportUnusedParameter]
            **kwargs: Any,
        ) -> Any:
            seen_eb.append(kwargs.get("extra_body"))
            return mock_resp

        gateway._model_router._has_provider_key = MagicMock(return_value=True)

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = fake_acompletion
            mock_litellm.Usage = MagicMock
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            await gateway.acompletion(
                messages=simple_messages,
                model="gpt-4.1-mini-2025-04-14",
                tool_choice={"type": "function", "function": {"name": "file_writer"}},
            )

        assert seen_eb[0] is None

    @pytest.mark.asyncio
    async def test_uses_complexity_routing_when_no_model(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """When model is None but complexity is given, model router is consulted."""
        mock_resp = _make_litellm_response()

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.Usage = MagicMock
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            result = await gateway.acompletion(
                messages=simple_messages,
                complexity=TaskComplexity.SIMPLE,
            )

        assert isinstance(result, LLMResponse)
        # The model router should have selected some model
        assert len(result.model) > 0

    @pytest.mark.asyncio
    async def test_returns_cached_response_on_hit(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """When cache returns a hit, litellm is never called."""
        cached_response = LLMResponse(
            content="cached answer",
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=5,
            output_tokens=3,
            total_tokens=8,
            cost_usd=0.0001,
            cached=True,
        )
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=cached_response)
        gateway.set_cache(mock_cache)

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock()

            result = await gateway.acompletion(
                messages=simple_messages,
                model="gpt-4o-mini-2024-07-18",
            )

        assert result.cached is True
        assert result.content == "cached answer"
        # litellm.acompletion should NOT have been called
        mock_litellm.acompletion.assert_not_awaited()


# ─── Test acompletion_with_tools ─────────────────────────────────────


class TestAcompletionWithTools:
    """Tests for LLMGateway.acpletion_with_tools."""

    @pytest.mark.asyncio
    async def test_returns_tool_call_response(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        tc = _make_tool_call()
        mock_resp = _make_litellm_response(
            content="",
            tool_calls=[tc],
            finish_reason="tool_calls",
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.Usage = MagicMock
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            result = await gateway.acompletion_with_tools(
                messages=simple_messages,
                tools=tools,
                model="gpt-4o-mini-2024-07-18",
            )

        assert isinstance(result, ToolCallResponse)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "get_weather"


# ─── Test Budget Enforcement ─────────────────────────────────────────


class TestBudgetEnforcement:
    """Tests for budget check during acompletion.

    _get_cheaper_fallback uses _TIER_ORDER for tier comparison.
    When budget is exhausted, the gateway tries to find a cheaper fallback
    (the default, downgrade path). If no cheaper model exists — OR the opt-in
    ``budget_hard_stop`` is set — it raises ``BudgetExhaustedError`` (caught by
    the worker as the terminal, resumable BUDGET_EXHAUSTED status).
    """

    @pytest.mark.asyncio
    async def test_budget_exhausted_with_no_cheaper_fallback_raises_budget_exhausted(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """When budget is exhausted and model is already cheapest,
        BudgetExhaustedError is raised (typed signal, not a bare RuntimeError)."""
        mock_tracker = MagicMock()
        mock_tracker.check_budget = AsyncMock(return_value=(False, "Daily budget exhausted"))
        gateway.set_cost_tracker(mock_tracker)
        # Pin the DEFAULT path (budget_hard_stop=False) so an ambient
        # BUDGET_HARD_STOP=true in .env cannot pre-empt the typed-raise branch
        # below. The opt-in hard-stop alternative is covered by the sibling test
        # ``test_budget_hard_stop_raises_even_when_cheaper_fallback_available``.
        gateway._settings.budget.budget_hard_stop = False

        with pytest.raises(BudgetExhaustedError, match="Budget exhausted"):
            await gateway.acompletion(
                messages=simple_messages,
                model="gpt-4o-mini-2024-07-18",  # VERY_CHEAP — no cheaper option
            )

    @pytest.mark.asyncio
    async def test_budget_hard_stop_raises_even_when_cheaper_fallback_available(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """Opt-in hard-stop (D): with ``budget_hard_stop`` set, the gateway raises
        BudgetExhaustedError immediately — it does NOT downgrade, even though a
        cheaper fallback exists. Default-off behavior (downgrade) is covered by
        the sibling test below; this pins the opt-in HARD-stop alternative that
        prevents a degraded run from fabricating (battery-04 q09)."""
        mock_tracker = MagicMock()
        mock_tracker.check_budget = AsyncMock(return_value=(False, "Daily budget exhausted"))
        gateway.set_cost_tracker(mock_tracker)
        # Flip the opt-in flag on for this test only.
        gateway._settings.budget.budget_hard_stop = True

        # A cheaper fallback IS available — hard-stop must ignore it and raise.
        with patch.object(
            gateway, "_get_cheaper_fallback", return_value="gpt-4o-mini-2024-07-18"
        ) as mock_fallback:
            with pytest.raises(BudgetExhaustedError, match="Daily budget exhausted"):
                await gateway.acompletion(
                    messages=simple_messages,
                    model="claude-sonnet-4-6",
                )
        # The fallback was never consulted (raise happens before the downgrade).
        assert not mock_fallback.called

    @pytest.mark.asyncio
    async def test_budget_exhausted_falls_back_when_cheaper_available(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """When budget is exhausted but _get_cheaper_fallback returns a model, it is used."""
        mock_tracker = MagicMock()
        mock_tracker.check_budget = AsyncMock(return_value=(False, "Daily budget exhausted"))
        mock_tracker.record_usage = AsyncMock(return_value=0.001)
        gateway.set_cost_tracker(mock_tracker)
        # Pin the DEFAULT downgrade path (budget_hard_stop=False) so an ambient
        # BUDGET_HARD_STOP=true in .env cannot flip this to a terminal raise
        # (the opt-in hard-stop is covered by the sibling test).
        gateway._settings.budget.budget_hard_stop = False

        mock_resp = _make_litellm_response(content="cheap answer")

        with patch.object(gateway, "_get_cheaper_fallback", return_value="gpt-4o-mini-2024-07-18"):
            with patch("src.llm.gateway.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
                mock_litellm.Usage = MagicMock
                mock_litellm.RateLimitError = Exception
                mock_litellm.Timeout = Exception
                mock_litellm.ServiceUnavailableError = Exception
                mock_litellm.APIConnectionError = Exception
                mock_litellm.AuthenticationError = Exception
                mock_litellm.BadRequestError = Exception

                result = await gateway.acompletion(
                    messages=simple_messages,
                    model="claude-sonnet-4-6",
                )

        assert isinstance(result, LLMResponse)
        # Should have been routed to the cheaper model
        assert result.model == "gpt-4o-mini-2024-07-18"

    @pytest.mark.asyncio
    async def test_cost_recorded_after_success(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """Cost tracker records usage after a successful completion."""
        mock_tracker = MagicMock()
        mock_tracker.check_budget = AsyncMock(return_value=(True, "Budget OK"))
        mock_tracker.record_usage = AsyncMock(return_value=0.0001)
        gateway.set_cost_tracker(mock_tracker)

        mock_resp = _make_litellm_response(input_tokens=10, output_tokens=5)

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.Usage = MagicMock
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            await gateway.acompletion(
                messages=simple_messages,
                model="gpt-4o-mini-2024-07-18",
            )

        mock_tracker.record_usage.assert_awaited_once()
        call_kwargs = mock_tracker.record_usage.call_args
        assert call_kwargs.kwargs["input_tokens"] == 10
        assert call_kwargs.kwargs["output_tokens"] == 5


# ─── Test Fallback Chain ─────────────────────────────────────────────


class TestFallbackChain:
    """Tests for _execute_with_fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_on_transient_error(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """When the first model fails with a transient error, fallback is tried."""
        error = Exception("rate limited")
        # We need to use actual litellm exception types, but they may not be
        # easily constructable. Patch the transient errors tuple to include our error.
        mock_resp = _make_litellm_response(content="fallback success")

        call_count = 0

        async def _side_effect(**_kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise error
            return mock_resp

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_side_effect)
            mock_litellm.RateLimitError = type(error)
            mock_litellm.Timeout = type(error)
            mock_litellm.ServiceUnavailableError = type(error)
            mock_litellm.APIConnectionError = type(error)
            mock_litellm.AuthenticationError = type("AuthErr", (Exception,), {})
            mock_litellm.BadRequestError = type("BadReqErr", (Exception,), {})
            mock_litellm.Usage = MagicMock

            # Patch the module-level _TRANSIENT_ERRORS to include our error type
            with patch("src.llm.gateway._TRANSIENT_ERRORS", (type(error),)):
                result = await gateway._execute_with_fallback(
                    messages=simple_messages,
                    model="claude-sonnet-4-6",
                )

        assert result.content == "fallback success"
        assert call_count >= 2  # First call failed, second succeeded

    @pytest.mark.asyncio
    async def test_all_fallbacks_exhausted_raises(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """When all fallback models fail, RuntimeError is raised."""

        async def _always_fail(**_kwargs: Any) -> None:
            raise Exception("unavailable")

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_always_fail)
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception
            mock_litellm.Usage = MagicMock

            with patch("src.llm.gateway._TRANSIENT_ERRORS", (Exception,)):
                with pytest.raises(RuntimeError, match="All fallbacks exhausted"):
                    await gateway._execute_with_fallback(
                        messages=simple_messages,
                        model="claude-sonnet-4-6",
                    )


# ─── Test _configure_litellm ─────────────────────────────────────────


class TestConfigureLitellm:
    """Tests for litellm global configuration during init."""

    def test_configure_litellm_sets_flags(self) -> None:
        with patch("src.llm.gateway.litellm") as mock_litellm, \
             patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"}):
            gw = LLMGateway.__new__(LLMGateway)
            gw._settings = _make_settings()
            gw._configure_litellm()

        mock_litellm.set_verbose = False
        mock_litellm.drop_params = True
        assert mock_litellm.success_callback == []
        assert mock_litellm.failure_callback == []

    def test_configure_litellm_with_langsmith_tracing(self) -> None:
        """litellm callbacks include langsmith when tracing is enabled."""
        with patch("src.llm.gateway.litellm") as mock_litellm, \
             patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
            gw = LLMGateway.__new__(LLMGateway)
            gw._settings = _make_settings()
            gw._configure_litellm()

        assert mock_litellm.success_callback == ["langsmith"]
        assert mock_litellm.failure_callback == ["langsmith"]


# ─── Test D2: multimodal/vision ──────────────────────────────────────


class TestGatewayVision:
    """Tests for the opt-in gateway vision path (D2).

    Default-off ⇒ every call is byte-identical to text-only. With
    ``AgentSettings.vision_enabled`` on, an ``images=`` payload is folded into
    the last user message as OpenAI content blocks and the fallback chain is
    restricted to image-capable models (``ModelSpec.supports_images``).
    """

    def test_build_content_blocks_returns_str_when_no_images(self) -> None:
        """No usable images ⇒ the plain text is returned unchanged (identity)."""
        from src.llm.gateway import build_content_blocks

        assert build_content_blocks("Hello", None) == "Hello"
        assert build_content_blocks("Hello", []) == "Hello"

    def test_build_content_blocks_returns_blocks_with_images(self) -> None:
        """Images ⇒ a text block + one image_url block per usable image; falsy
        / non-str entries are dropped."""
        from src.llm.gateway import build_content_blocks

        images: list[Any] = ["https://x/a.png", "", 5, "data:image/png;base64,AAA"]
        out = build_content_blocks("Describe this", images)
        assert isinstance(out, list)
        assert out[0] == {"type": "text", "text": "Describe this"}
        image_blocks = [b for b in out if b.get("type") == "image_url"]
        assert len(image_blocks) == 2  # the empty string and the int were dropped
        assert image_blocks[0] == {"type": "image_url", "image_url": {"url": "https://x/a.png"}}

    def test_content_char_len_str(self) -> None:
        """A plain string content contributes its length."""
        from src.llm.gateway import _content_char_len

        assert _content_char_len("abcd") == 4
        assert _content_char_len("") == 0

    def test_content_char_len_block_list(self) -> None:
        """A multimodal block list: text blocks sum their text, each image_url
        block contributes the flat per-image constant (not base64 length)."""
        from src.llm.gateway import _IMAGE_FLAT_CHARS, _content_char_len

        content = [
            {"type": "text", "text": "ab"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 10_000}},
            {"type": "image_url", "image_url": {"url": "https://x/b.png"}},
        ]
        # 2 text chars + 2 × flat per-image allowance (the huge data-URI is NOT
        # counted by its encoded length — the bug this fixes).
        assert _content_char_len(content) == 2 + 2 * _IMAGE_FLAT_CHARS

    def test_content_char_len_non_str_non_list_is_zero(self) -> None:
        """Non-str / non-list content (e.g. an int or None) contributes 0 and
        never raises — the legacy text path stays safe."""
        from src.llm.gateway import _content_char_len

        assert _content_char_len(12345) == 0
        assert _content_char_len(None) == 0

    def test_estimate_tokens_tolerates_list_content(self) -> None:
        """Regression: ``_estimate_tokens`` must not raise (or undercount) when a
        message's ``content`` is a multimodal block list — the shape a vision
        payload produces. Pre-fix, ``len(list)`` counted BLOCKS not chars."""
        from src.llm.gateway import _IMAGE_FLAT_CHARS

        text_len = 400
        list_msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a" * text_len},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            }
        ]
        # Post-fix: text counted by char length (400) + the flat per-image
        # allowance → (400 + 256)//4 = 164. Pre-fix ``len(list)`` counted BLOCKS
        # (2) → 2//4 = 0 → max(1,0) = 1, severely undercounting a vision payload.
        assert LLMGateway._estimate_tokens(list_msgs) == (text_len + _IMAGE_FLAT_CHARS) // 4
        assert LLMGateway._estimate_tokens(list_msgs) > 1

    def test_attach_images_to_last_user_attaches_and_does_not_mutate_original(self) -> None:
        """Images fold into the LAST user message only; the caller's list and the
        prior assistant turn are untouched."""
        from src.llm.gateway import _attach_images_to_last_user

        original = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "describe this image"},
        ]
        result = _attach_images_to_last_user(original, ["https://x/a.png"])
        # Original untouched (copy semantics).
        assert original[-1]["content"] == "describe this image"
        assert original is not result
        # Only the last user message became multimodal.
        assert isinstance(result[-1]["content"], list)
        assert result[-1]["content"][0] == {"type": "text", "text": "describe this image"}
        assert {"type": "image_url", "image_url": {"url": "https://x/a.png"}} in result[-1]["content"]
        # Prior turns preserved verbatim.
        assert result[0]["content"] == "first question"
        assert result[1]["content"] == "answer"

    def test_attach_images_with_no_user_message_drops_images(self) -> None:
        """No user turn to attach to ⇒ fail safe (return a copy unchanged) rather
        than fabricate a prompt."""
        from src.llm.gateway import _attach_images_to_last_user

        original = [{"role": "system", "content": "be helpful"}]
        result = _attach_images_to_last_user(original, ["https://x/a.png"])
        assert result == original
        assert result is not original

    @pytest.mark.asyncio
    async def test_acompletion_vision_off_drops_images(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """Default-off (vision_enabled=False): an ``images=`` payload is ignored
        and plain text content reaches the provider — byte-identical to a
        text-only call."""
        mock_resp = _make_litellm_response(content="ok", input_tokens=2, output_tokens=1)
        # Pin vision OFF explicitly (mirrors the _on test below) so this stays
        # hermetic to the ambient .env value of VISION_ENABLED.
        gateway._settings.agent.vision_enabled = False
        assert gateway._settings.agent.vision_enabled is False

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.Usage = MagicMock
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            await gateway.acompletion(
                messages=simple_messages,
                model="gpt-4o-mini-2024-07-18",
                images=["https://x/a.png"],
            )

        sent_content = mock_litellm.acompletion.call_args.kwargs["messages"][-1]["content"]
        assert sent_content == "Hello world"  # images dropped; no block list

    @pytest.mark.asyncio
    async def test_acompletion_vision_on_builds_multimodal(
        self, gateway: LLMGateway, simple_messages: list[dict[str, Any]]
    ) -> None:
        """Opt-in (vision_enabled=True): the image is folded into the last user
        message as an image_url content block that reaches the provider."""
        mock_resp = _make_litellm_response(content="a chart", input_tokens=3, output_tokens=2)
        gateway._settings.agent.vision_enabled = True

        with patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.Usage = MagicMock
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            result = await gateway.acompletion(
                messages=simple_messages,
                model="gpt-4o-mini-2024-07-18",
                images=["https://x/a.png"],
            )

        sent_content = mock_litellm.acompletion.call_args.kwargs["messages"][-1]["content"]
        assert isinstance(sent_content, list)
        assert {"type": "text", "text": "Hello world"} in sent_content
        assert {"type": "image_url", "image_url": {"url": "https://x/a.png"}} in sent_content
        assert isinstance(result, LLMResponse)

    @pytest.mark.asyncio
    async def test_fallback_chain_unfiltered_without_require_vision(self, gateway: LLMGateway) -> None:
        """Regression: without require_vision the chain is NOT pruned for vision
        capability — a non-vision primary is still attempted first."""
        mock_resp = _make_litellm_response(content="ok", input_tokens=2, output_tokens=1)
        # Force the key-check to True so it cannot mask the vision filter.
        gateway._model_router._has_provider_key = lambda _p: True  # type: ignore[method-assign]

        with patch.dict(
            "src.llm.gateway.FALLBACK_CHAINS",
            {"text-only-fake-model": ["gpt-4o-mini-2024-07-18"]},
        ), patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.Usage = MagicMock
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            await gateway._execute_with_fallback(
                messages=[{"role": "user", "content": "hi"}],
                model="text-only-fake-model",
                require_vision=False,
            )

        # Non-vision primary attempted first (filter did not remove it).
        assert mock_litellm.acompletion.call_args.kwargs["model"] == "text-only-fake-model"

    @pytest.mark.asyncio
    async def test_fallback_chain_filtered_to_vision_when_require_vision(self, gateway: LLMGateway) -> None:
        """With require_vision the non-vision primary is skipped in favor of the
        first vision-capable entry in the fallback chain
        (gpt-4o-mini-2024-07-18 supports images in the registry)."""
        mock_resp = _make_litellm_response(content="ok", input_tokens=2, output_tokens=1)
        gateway._model_router._has_provider_key = lambda _p: True  # type: ignore[method-assign]

        with patch.dict(
            "src.llm.gateway.FALLBACK_CHAINS",
            {"text-only-fake-model": ["gpt-4o-mini-2024-07-18"]},
        ), patch("src.llm.gateway.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.Usage = MagicMock
            mock_litellm.RateLimitError = Exception
            mock_litellm.Timeout = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.APIConnectionError = Exception
            mock_litellm.AuthenticationError = Exception
            mock_litellm.BadRequestError = Exception

            await gateway._execute_with_fallback(
                messages=[{"role": "user", "content": "hi"}],
                model="text-only-fake-model",
                require_vision=True,
            )

        # Non-vision primary skipped; the vision-capable fallback was attempted.
        assert mock_litellm.acompletion.call_args.kwargs["model"] == "gpt-4o-mini-2024-07-18"
