"""Unified LLM Gateway wrapping litellm with resilience features.

All LLM calls flow through this gateway, which provides:
- Rate limiting per provider
- Redis prompt caching
- Automatic fallback chains on failure
- Cost tracking in PostgreSQL
- Structured output extraction
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import litellm
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config.model_registry import FALLBACK_CHAINS, MODEL_REGISTRY, ModelTier
from src.config.settings import Settings
from src.llm.cache import PromptCache
from src.llm.cost_tracker import CostTracker
from src.llm.model_router import ModelRouter
from src.graph.enums import TaskComplexity
from src.llm.models import LLMResponse, ToolCallResponse
from src.llm.rate_limiter import RateLimiterRegistry
from src.llm.structured_output import StructuredOutputManager


# Transient errors that warrant retry (litellm exposes these at runtime)
_TRANSIENT_ERRORS = (
    litellm.RateLimitError,  # type: ignore[attr-defined]
    litellm.Timeout,  # type: ignore[attr-defined]
    litellm.ServiceUnavailableError,  # type: ignore[attr-defined]
    litellm.APIConnectionError,  # type: ignore[attr-defined]
)

# Max retries for transient errors
_MAX_RETRIES = 3

# Tier ordering for cheaper fallback selection
_TIER_ORDER: dict[ModelTier, int] = {
    ModelTier.VERY_CHEAP: 0,
    ModelTier.CHEAP: 1,
    ModelTier.MODERATE: 2,
}


class LLMGateway:
    """Unified LLM gateway with rate limiting, caching, fallbacks, and cost tracking.

    Usage:
        gateway = LLMGateway(settings)
        response = await gateway.acompletion(
            messages=[{"role": "user", "content": "Hello"}],
            complexity=TaskComplexity.SIMPLE,
        )
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._rate_limiter = RateLimiterRegistry(settings)
        self._model_router = ModelRouter(settings)
        self._structured_output = StructuredOutputManager()
        self._cost_tracker: CostTracker | None = None
        self._cache: PromptCache | None = None

        # Configure litellm
        self._configure_litellm()

    def set_cost_tracker(self, tracker: CostTracker) -> None:
        """Inject a cost tracker (requires async DB session)."""
        self._cost_tracker = tracker

    def set_cache(self, cache: PromptCache) -> None:
        """Inject a Redis prompt cache."""
        self._cache = cache

    async def acompletion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        complexity: TaskComplexity | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.5,
        max_tokens: int | None = None,
        cache_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a completion request with full resilience pipeline.

        Pipeline: rate-limit → cache lookup → litellm call (retry) → cost record → cache store.

        Args:
            messages: Chat messages in OpenAI format.
            model: Explicit model to use (overrides complexity routing).
            complexity: Task complexity for automatic model routing.
            tools: Optional tool definitions for function calling.
            response_format: Optional response format specification.
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens in the response.
            cache_key: Optional cache key override.
            metadata: Optional metadata for logging.

        Returns:
            LLMResponse with content, usage stats, and cost.
        """
        # Determine model
        if model is None and complexity is not None:
            model = self._model_router.route(complexity)
        elif model is None:
            model = self._model_router.route(TaskComplexity.SIMPLE)

        assert model is not None  # guaranteed by routing logic

        provider = self._extract_provider(model)

        # Rate limiting
        await self._rate_limiter.acquire(provider, self._estimate_tokens(messages))

        # Cache lookup
        if self._cache and cache_key is None:
            cached = await self._cache.get(messages, model, temperature, max_tokens)
            if cached is not None:
                logger.debug(f"Cache hit for {model}")
                return cached

        # Budget check
        if self._cost_tracker:
            within_budget, budget_msg = await self._cost_tracker.check_budget()
            if not within_budget:
                logger.error(f"Budget exhausted: {budget_msg}")
                # Try fallback to cheaper model
                fallback = self._get_cheaper_fallback(model)
                if fallback:
                    logger.warning(f"Falling back to cheaper model: {fallback}")
                    model = fallback
                    provider = self._extract_provider(model)
                else:
                    raise RuntimeError(f"Budget exhausted and no cheaper fallback: {budget_msg}")

        # Execute with retry and fallback
        start_time = time.monotonic()
        response = await self._execute_with_fallback(
            messages=messages,
            model=model,
            provider=provider,
            tools=tools,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )
        latency_ms = int((time.monotonic() - start_time) * 1000)

        # Cost tracking
        if self._cost_tracker:
            await self._cost_tracker.record_usage(
                model=response.model,
                provider=response.provider,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=latency_ms,
            )

        # Cache store
        if self._cache:
            await self._cache.set(response, messages, model, temperature, max_tokens)

        return response

    async def astream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        complexity: TaskComplexity | None = None,
        temperature: float = 0.5,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion response token by token.

        Args:
            messages: Chat messages in OpenAI format.
            model: Explicit model to use.
            complexity: Task complexity for automatic routing.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            metadata: Optional metadata for logging.

        Yields:
            Token strings as they arrive.
        """
        if model is None and complexity is not None:
            model = self._model_router.route(complexity)
        elif model is None:
            model = self._model_router.route(TaskComplexity.SIMPLE)

        assert model is not None  # guaranteed by routing logic

        provider = self._extract_provider(model)
        await self._rate_limiter.acquire(provider, self._estimate_tokens(messages))

        kwargs = self._build_kwargs(model, temperature, max_tokens, metadata)

        try:
            response = await litellm.acompletion(
                messages=messages,
                stream=True,
                **kwargs,
            )
            async for chunk in response:  # type: ignore[union-attr]
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except _TRANSIENT_ERRORS as exc:
            logger.warning(f"Streaming failed for {model}: {exc}")
            # Fallback: try next in chain
            fallbacks = FALLBACK_CHAINS.get(model, [])
            for fallback_model in fallbacks:
                try:
                    fb_kwargs = self._build_kwargs(fallback_model, temperature, max_tokens, metadata)
                    response = await litellm.acompletion(
                        messages=messages,
                        stream=True,
                        **fb_kwargs,
                    )
                    async for chunk in response:  # type: ignore[union-attr]
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta
                    return
                except Exception:
                    continue
            raise RuntimeError(f"All streaming fallbacks exhausted for {model}")

    async def acompletion_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        complexity: TaskComplexity | None = None,
        temperature: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> ToolCallResponse:
        """Send a completion request with tool definitions.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Tool definitions for function calling.
            model: Explicit model to use.
            complexity: Task complexity for routing.
            temperature: Sampling temperature.
            metadata: Optional metadata for logging.

        Returns:
            ToolCallResponse with tool calls and optional text content.
        """
        response = await self.acompletion(
            messages=messages,
            model=model,
            complexity=complexity,
            tools=tools,
            temperature=temperature,
            metadata=metadata,
        )

        return ToolCallResponse(
            content=response.content if response.content else None,
            tool_calls=response.tool_calls or [],
            model=response.model,
            provider=response.provider,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            cost_usd=response.cost_usd,
        )

    # ─── Internal Methods ────────────────────────────────────────────────

    async def _execute_with_fallback(
        self,
        messages: list[dict[str, Any]],
        model: str,
        provider: str,  # noqa: ARG002 — kept for caller API compat
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.5,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Execute an LLM call with automatic fallback on failure."""
        fallback_chain = [model] + FALLBACK_CHAINS.get(model, [])

        # Pre-filter: skip providers without API keys to reduce log noise.
        # If ALL providers lack keys (e.g., test env), fall back to trying all.
        available_chain: list[str] = []
        for m in fallback_chain:
            p = self._extract_provider(m)
            if self._model_router._has_provider_key(p):
                available_chain.append(m)
            else:
                logger.debug(f"Skipping fallback {m}: no API key for {p}")

        if available_chain:
            fallback_chain = available_chain

        last_error: Exception | None = None
        for attempt_model in fallback_chain:
            attempt_provider = self._extract_provider(attempt_model)
            try:
                kwargs = self._build_kwargs(attempt_model, temperature, max_tokens, metadata)
                if tools:
                    kwargs["tools"] = tools
                if response_format:
                    kwargs["response_format"] = response_format

                response = await self._retry_call(messages, **kwargs)
                return self._parse_response(response, attempt_model, attempt_provider)

            except _TRANSIENT_ERRORS as exc:
                last_error = exc
                logger.warning(
                    f"LLM call failed for {attempt_model}: {exc.__class__.__name__}: {exc}"
                )
                continue
            except (litellm.AuthenticationError, litellm.BadRequestError) as exc:  # type: ignore[attr-defined]
                logger.error(f"Non-retryable error for {attempt_model}: {exc}")
                continue
            except Exception as exc:
                last_error = exc
                logger.error(f"Unexpected error for {attempt_model}: {exc}")
                continue

        raise RuntimeError(
            f"All fallbacks exhausted for {model}. Last error: {last_error}"
        )

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
        before_sleep=lambda state: logger.warning(
            f"Retrying LLM call (attempt {state.attempt_number}): {state.outcome.exception()}"  # type: ignore[union-attr]
        ),
    )
    async def _retry_call(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Execute a litellm call with tenacity retry."""
        return await litellm.acompletion(messages=messages, **kwargs)

    def _parse_response(self, response: Any, model: str, provider: str) -> LLMResponse:
        """Parse a litellm response into our standard LLMResponse."""
        choice = response.choices[0]
        content = choice.message.content or ""

        # Extract tool calls if present
        tool_calls = None
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        usage = response.usage or litellm.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        input_tokens = usage.prompt_tokens or 0
        output_tokens = usage.completion_tokens or 0

        cost = CostTracker.calculate_cost(model, input_tokens, output_tokens)

        return LLMResponse(
            content=content,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage.total_tokens or (input_tokens + output_tokens),
            cost_usd=cost,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
        )

    def _build_kwargs(
        self,
        model: str,
        temperature: float,
        max_tokens: int | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for litellm call."""
        kwargs: dict[str, Any] = {"model": model, "temperature": temperature}

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        else:
            # Use model-specific defaults
            spec = MODEL_REGISTRY.get(model)
            if spec and spec.max_output:
                kwargs["max_tokens"] = spec.max_output
            else:
                kwargs["max_tokens"] = 4096

        if metadata:
            kwargs["metadata"] = metadata

        # Set provider API key
        provider = self._extract_provider(model)
        api_key = self._get_api_key(provider)
        if api_key:
            kwargs["api_key"] = api_key

        return kwargs

    def _get_api_key(self, provider: str) -> str | None:
        """Get the API key for a specific provider."""
        try:
            return self._settings.llm.get_provider_key(provider)
        except Exception:
            return None

    def _get_cheaper_fallback(self, model: str) -> str | None:
        """Find a cheaper fallback model for budget conservation."""
        spec = MODEL_REGISTRY.get(model)
        if not spec:
            return None

        current_order = _TIER_ORDER.get(spec.tier, 0)
        if current_order == 0:
            return None  # Already cheapest tier

        # Walk fallback chain first — prefer provider diversity
        for fb in FALLBACK_CHAINS.get(model, []):
            fb_spec = MODEL_REGISTRY.get(fb)
            if fb_spec and _TIER_ORDER.get(fb_spec.tier, 0) < current_order:
                return fb

        # Scan registry for any cheaper model
        for mid, mspec in MODEL_REGISTRY.items():
            if mid != model and _TIER_ORDER.get(mspec.tier, 0) < current_order:
                return mid

        return None

    @staticmethod
    def _extract_provider(model: str) -> str:
        """Extract provider from litellm model identifier."""
        return ModelRouter._extract_provider(model)

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, Any]] | str) -> int:
        """Rough token estimation: ~4 chars per token."""
        if isinstance(messages, str):
            return max(1, len(messages) // 4)
        total_chars = sum(len(m.get("content", "")) for m in messages if isinstance(m, dict))
        return max(1, total_chars // 4)

    def _configure_litellm(self) -> None:
        """Configure litellm global settings."""
        litellm.set_verbose = False  # type: ignore[attr-defined]
        litellm.drop_params = True  # Drop unsupported params instead of erroring
        litellm.success_callback = []
        litellm.failure_callback = []
        logger.debug("litellm configured: drop_params=True, verbose=False")
