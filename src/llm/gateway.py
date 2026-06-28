"""Unified LLM Gateway wrapping litellm with resilience features.

All LLM calls flow through this gateway, which provides:
- Rate limiting per provider
- Redis prompt caching
- Automatic fallback chains on failure
- Cost tracking in PostgreSQL
- Structured output extraction
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import litellm
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config.model_registry import FALLBACK_CHAINS, MODEL_REGISTRY, ModelTier
from src.config.settings import Settings
from src.graph.enums import TaskComplexity
from src.graph.models import CostRecord
from src.llm.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from src.llm.cache import PromptCache
from src.llm.cost_tracker import CostTracker
from src.llm.exceptions import BudgetExhaustedError
from src.llm.nvidia_shim import nvidia_shim_model_id
from src.llm.prompt_cache_control import inject_cache_breakpoints
from src.llm.thinking_control import thinking_params_for
from src.llm.model_router import ModelRouter
from src.llm.models import BatchRequest, BatchResponse, LLMResponse, ToolCallResponse
from src.llm.rate_limiter import RateLimiterRegistry
from src.llm.structured_output import (
    StructuredOutputManager,
    build_native_response_format,
    is_anthropic_4x_or_newer,
)


# Transient errors that warrant retry (litellm exposes these at runtime)
_TRANSIENT_ERRORS = (
    litellm.RateLimitError,  # type: ignore[attr-defined]
    litellm.Timeout,  # type: ignore[attr-defined]
    litellm.ServiceUnavailableError,  # type: ignore[attr-defined]
    litellm.APIConnectionError,  # type: ignore[attr-defined]
    # Hard asyncio.wait_for backstop raised when the per-call timeout is not
    # enforced by the underlying client (see ``_retry_call``). Treated as
    # transient so the tenacity layer retries — and the fallback chain moves
    # on to the next provider once the budget exhausts.
    asyncio.TimeoutError,
)

# Tier ordering for cheaper fallback selection
_TIER_ORDER: dict[ModelTier, int] = {
    ModelTier.VERY_CHEAP: 0,
    ModelTier.CHEAP: 1,
    ModelTier.MODERATE: 2,
}

# Multimodal (D2): flat per-image character allowance used when estimating
# tokens for a vision payload. Image token cost depends on tiling/detail, NOT on
# the (huge) base64 length of a data-URI, so a literal char count would blow up
# the estimate and over-reserve the rate-limiter budget. 256 chars ≈ a small
# reserve that keeps ``_estimate_tokens`` sane for image_url blocks.
_IMAGE_FLAT_CHARS = 256


def build_content_blocks(
    text: str, images: list[str] | None
) -> str | list[dict[str, Any]]:
    """Build OpenAI-format message ``content`` for an optional image payload.

    With no usable images this returns the plain ``text`` unchanged
    (byte-identical to the text-only path), so non-vision calls are untouched.
    With one or more images it returns the multimodal block list OpenAI/litellm
    expect: a leading text block followed by an ``image_url`` block per image.

    Args:
        text: The textual prompt for the message.
        images: Optional image references — ``https://``/``http://`` URLs or
            ``data:image/...;base64,...`` data-URIs. Falsy or non-str entries
            are dropped.

    Returns:
        The original ``str`` when there are no usable images, else a content
        block list (text block + one ``image_url`` block per image).
    """
    usable = [img for img in (images or []) if isinstance(img, str) and img]
    if not usable:
        return text
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for img in usable:
        blocks.append({"type": "image_url", "image_url": {"url": img}})
    return blocks


def _content_char_len(content: Any) -> int:
    """Approximate character length of a message ``content`` field.

    Tolerates the multimodal block-list shape (``[{"type":"text",...},
    {"type":"image_url",...}]``) that text-only code assumed was always a
    string. For a block list, text blocks contribute their ``text`` length and
    each ``image_url`` block contributes a flat constant (see
    ``_IMAGE_FLAT_CHARS``). Non-str / non-list content contributes 0, so this
    never raises on the legacy text path.
    """
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                total += len(block.get("text", "") or "")
            elif btype == "image_url":
                total += _IMAGE_FLAT_CHARS
        return total
    return 0


def _attach_images_to_last_user(
    messages: list[dict[str, Any]], images: list[str]
) -> list[dict[str, Any]]:
    """Return a new message list with ``images`` folded into the last user turn.

    Only the caller's list is copied — the original is never mutated. The last
    ``role == "user"`` message (searched from the end) gets its ``content``
    rewritten to the multimodal block form via ``build_content_blocks``; any
    existing list content is collapsed to text first so the image blocks append
    cleanly. If no user message exists the images are dropped (a vision request
    with no prompt is malformed — fail safe by ignoring rather than fabricating
    a prompt).
    """
    result = list(messages)
    for idx in range(len(result) - 1, -1, -1):
        msg = result[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            text: Any = msg.get("content", "")
            if isinstance(text, list):
                text = "\n".join(
                    str(b.get("text", ""))
                    for b in text
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            elif not isinstance(text, str):
                text = str(text)
            new_msg = dict(msg)
            new_msg["content"] = build_content_blocks(text, images)
            result[idx] = new_msg
            return result
    return result


class LLMGateway:
    """Unified LLM gateway with rate limiting, caching, fallbacks, and cost tracking.

    Usage:
        gateway = LLMGateway(settings)
        response = await gateway.acompletion(
            messages=[{"role": "user", "content": "Hello"}],
            complexity=TaskComplexity.SIMPLE,
        )
    """

    # Grace (seconds) added to the configured per-call timeout when wrapping
    # the litellm call in ``asyncio.wait_for`` (Bug T1 hard backstop). Lets the
    # cleaner in-library timeout fire first when the client honors it; the
    # wait_for is the guaranteed cap when it does not.
    _WAIT_FOR_GRACE: float = 5.0

    def __init__(self, settings: Settings, *, pinned_model: str | None = None) -> None:
        self._settings = settings
        self._rate_limiter = RateLimiterRegistry(settings)
        self._model_router = ModelRouter(settings)
        # Per-provider circuit breaker (sits ABOVE per-call tenacity retry):
        # decides whether a provider should be attempted at all. When open for
        # a provider, the fallback loop skips to the next provider in the chain.
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=settings.circuit_breaker.cb_failure_threshold,
            recovery_timeout=settings.circuit_breaker.cb_recovery_timeout,
            half_open_max_calls=settings.circuit_breaker.cb_half_open_max_calls,
        )
        self._structured_output = StructuredOutputManager()
        self._cost_tracker: CostTracker | None = None
        self._cache: PromptCache | None = None
        # When set (e.g. via the CLI ``--model`` flag), overrides complexity
        # routing for every call so a single run can be pinned to one model.
        # Provider/key/endpoint resolution still flows through ``_build_kwargs``
        # exactly as for a routed model, so a pinned model with no provider key
        # still falls back through its chain on failure.
        self._pinned_model = pinned_model
        # Per-run correlation key for cost-ledger attribution. Set via
        # ``set_run_id`` from the run's graph ``thread_id`` (main.py) so every
        # real LLM call's cost row is attributable to the run that issued it.
        # ``None`` until bound — rows then record a NULL run_id (unattributed),
        # matching the pre-attribution behavior.
        self._run_id: str | None = None
        # In-memory accumulation of every real LLM call's cost/tokens for this
        # gateway instance. Flushed into graph state (cost_records /
        # total_tokens_used) by the store_memory node. A fresh gateway is built
        # per run, so this is naturally scoped to one graph execution.
        self._cost_records: list[CostRecord] = []

        # History compression (runs every 5th call to reduce overhead)
        from src.llm.history_compressor import HistoryCompressor
        self._history_compressor = HistoryCompressor()

        # Configure litellm
        self._configure_litellm()

    def set_cost_tracker(self, tracker: CostTracker) -> None:
        """Inject a cost tracker (requires async DB session)."""
        self._cost_tracker = tracker

    def set_cache(self, cache: PromptCache) -> None:
        """Inject a Redis prompt cache."""
        self._cache = cache

    def set_rate_limiter_redis(self, client: Any) -> None:
        """Attach a shared Redis client for cross-process rate coordination.

        Forwards to the rate limiter so the worker fleet (multiple gateway
        instances across processes) shares ONE provider RPM/TPM budget instead
        of each process slamming its own 60-RPM bucket. Best-effort + opt-in
        (``RATE_LIMIT_CROSS_PROCESS_ENABLED``); degrades to in-memory-only when
        the client is absent or registration fails.
        """
        self._rate_limiter.attach_redis(client)

    def set_run_id(self, run_id: str | None) -> None:
        """Bind the per-run correlation key for cost-ledger attribution.

        Set from the run's graph ``thread_id`` (always defined in main.py:
        ``cli-{run_id}`` for a ``--run-id`` run, else ``cli-{pid}-{obj}``). The
        value flows into every ``CostTracker.record_usage`` call so cost rows
        carry the run identifier, enabling per-run spend attribution.
        """
        self._run_id = run_id

    def get_cost_records(self) -> list[CostRecord]:
        """Return the accumulated cost records for this gateway instance.

        Each real (non-cached) LLM call appends one CostRecord. Used by the
        store_memory node and the e2e/main runners to populate graph state.
        """
        return list(self._cost_records)

    def reset_cost_records(self) -> None:
        """Clear the accumulated cost records."""
        self._cost_records.clear()

    async def cache_stats(self) -> dict[str, Any]:
        """Return prompt-cache hit/miss stats, or zeros when caching is disabled.

        Delegates to the injected :class:`PromptCache` (see :meth:`set_cache`).
        When no cache is wired up the returned counters are all zero rather than
        raising, so callers (metrics sinks, health endpoints) can treat the
        absence of a cache as a degenerate-but-valid state.
        """
        if self._cache is None:
            return {"hits": 0, "misses": 0, "hit_rate": 0.0, "size_est": 0}
        return await self._cache.stats()

    async def acompletion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        complexity: TaskComplexity | None = None,
        node: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        response_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
        images: list[str] | None = None,
    ) -> LLMResponse:
        """Send a completion request with full resilience pipeline.

        Pipeline: rate-limit → cache lookup → litellm call (retry) → cost record → cache store.

        Args:
            messages: Chat messages in OpenAI format.
            model: Explicit model to use (overrides complexity routing).
            complexity: Task complexity for automatic model routing.
            node: Optional graph-node name (e.g. ``"plan"``, ``"execute"``,
                ``"verify"``). Threads into ``ModelRouter.route`` so per-node
                tier overrides (NODE_TIER_MAP) and reasoning-model selection
                (verify/reflect on complex/critical) apply. Ignored when
                ``model`` is given explicitly.
            tools: Optional tool definitions for function calling.
            response_format: Optional response format specification.
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens in the response.
            cache_key: Optional cache key override.
            metadata: Optional metadata for logging.
            timeout: Optional hard per-call timeout (seconds) overriding the
                ``request_timeout`` default; used by long code-generation calls.

        Returns:
            LLMResponse with content, usage stats, and cost.
        """
        # Determine model
        if model is None:
            if self._pinned_model is not None:
                model = self._pinned_model
            elif complexity is not None:
                model = self._model_router.route(complexity, node=node)
            else:
                model = self._model_router.route(TaskComplexity.SIMPLE, node=node)

        assert model is not None  # guaranteed by routing logic

        provider = self._extract_provider(model)

        temperature = self._resolve_temperature(temperature)

        # Compress older messages to reduce token consumption
        messages = self._history_compressor.compress(messages)

        # Multimodal (D2): when vision is enabled and the caller supplied images,
        # fold them into the LAST user message as OpenAI-format content blocks
        # (text + image_url) BEFORE the cache lookup so the image payload
        # participates in the cache key, and flag the call so the fallback loop
        # restricts the chain to image-capable models. Off (default) or no
        # images ⇒ the messages list is untouched (byte-identical to the
        # text-only path) and ``require_vision`` stays False.
        require_vision = False
        if images and self._settings.agent.vision_enabled:
            messages = _attach_images_to_last_user(messages, images)
            require_vision = True

        # Rate limiting moved into _retry_call (retry-aware, keyed on the ACTUAL
        # attempt provider) so a fallback onto a different provider is paced
        # against THAT provider's budget — the old single acquire here paced only
        # the original model's provider and left every fallback unpaced (a
        # glm-4.7→deepseek fallback hit deepseek with zero rate limiting).

        # Cache lookup
        if self._cache and cache_key is None:
            cached = await self._cache.get(messages, model, temperature, max_tokens)
            if cached is not None:
                logger.debug(f"Cache hit for {model}")
                return cached

        # Budget check
        if self._cost_tracker:
            within_budget, budget_msg = await self._cost_tracker.check_budget(self._run_id)
            if not within_budget:
                logger.error(f"Budget exhausted: {budget_msg}")
                # Opt-in budget hard-stop (D): instead of silently downgrading to a
                # cheaper fallback — which under downgrade can fabricate (battery-04
                # q09 degraded onto a free-tier provider and never completed) — stop
                # cleanly here. The worker catches BudgetExhaustedError → terminal
                # BUDGET_EXHAUSTED (resumable via checkpoint). Default OFF, so the
                # downgrade path below is unchanged unless budget_hard_stop is set.
                if self._settings.budget.budget_hard_stop:
                    raise BudgetExhaustedError(budget_msg)
                # Default: downgrade to the cheapest available fallback model.
                fallback = self._get_cheaper_fallback(model)
                if fallback:
                    logger.warning(f"Falling back to cheaper model: {fallback}")
                    model = fallback
                    provider = self._extract_provider(model)
                else:
                    # Already on the cheapest tier — no downgrade possible. Raise
                    # the typed signal (not a bare RuntimeError) so the worker marks
                    # a terminal BUDGET_EXHAUSTED instead of redelivering a doomed
                    # retry (a budget-exhausted run won't recover on redelivery).
                    raise BudgetExhaustedError(
                        f"Budget exhausted and no cheaper fallback: {budget_msg}"
                    )

        # Native JSON-schema structured outputs (opt-in). Convert a caller-
        # supplied schema to a provider-native response_format using the FINAL
        # (post-fallback) model/provider. Ignored when the feature is disabled
        # (build returns None → caller's raw response_format, if any, stands).
        # See structured_output.build_native_response_format.
        if response_schema is not None:
            native = build_native_response_format(
                response_schema, provider, self._settings.native_structured
            )
            if native is not None:
                response_format = native

        # Execute with retry and fallback
        start_time = time.monotonic()
        response = await self._execute_with_fallback(
            messages=messages,
            model=model,
            complexity=complexity,
            tools=tools,
            response_format=response_format,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            require_vision=require_vision,
        )
        latency_ms = int((time.monotonic() - start_time) * 1000)

        # Accumulate this real (non-cached) call's cost in memory so it can be
        # flushed into graph state (cost_records / total_tokens_used) by the
        # store_memory node. Cache hits skip this path (they incur no spend).
        self._cost_records.append(
            CostRecord(
                provider=response.provider,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cached_tokens=0,
                cost_usd=response.cost_usd,
                latency_ms=latency_ms,
            )
        )

        # Cost tracking
        if self._cost_tracker:
            await self._cost_tracker.record_usage(
                model=response.model,
                provider=response.provider,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=latency_ms,
                run_id=self._run_id,
            )

        # Cache store
        if self._cache:
            await self._cache.set(response, messages, model, temperature, max_tokens)

        return response

    async def abatch(
        self,
        requests: list[BatchRequest],
        *,
        complexity: TaskComplexity | None = None,
    ) -> list[BatchResponse]:
        """Run many independent completions concurrently.

        Concurrency is bounded by ``BatchingSettings.max_concurrency`` (an
        ``asyncio.Semaphore``); the existing per-provider rate limiter still
        gates RPM/TPM underneath each individual ``acompletion``. There is no
        async litellm batch primitive in this build (``litellm.batch_completion``
        is a sync ThreadPool wrapper), so gather+semaphore is the supported
        approach.

        Partial failures are isolated: a request whose ``acompletion`` raises
        yields a ``BatchResponse`` with empty content and ``metadata["error"]``
        set, so one bad request never aborts the batch. ``asyncio.CancelledError``
        is deliberately NOT caught — it propagates so the caller can clean up.

        When batching is disabled (``LLM_BATCH_ENABLED=false``) the semaphore
        collapses to size 1, i.e. the requests run sequentially while keeping
        the same isolation contract.

        Args:
            requests: Independent completions. Each carries its own
                messages/model/temperature/max_tokens; ``model=""`` falls back
                to ``complexity`` routing like a normal call.
            complexity: Optional complexity for routing requests whose model is
                empty.

        Returns:
            One ``BatchResponse`` per request, in input order.
        """
        if not requests:
            return []

        batch_cfg = self._settings.batching
        max_concurrency = max(1, batch_cfg.max_concurrency) if batch_cfg.enabled else 1
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _one(req: BatchRequest) -> BatchResponse:
            async with semaphore:
                try:
                    resp = await self.acompletion(
                        messages=req.messages,
                        model=req.model or None,
                        complexity=complexity,
                        temperature=req.temperature,
                        max_tokens=req.max_tokens,
                    )
                    return BatchResponse(
                        content=resp.content,
                        model=resp.model,
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                        cost_usd=resp.cost_usd,
                        metadata=req.metadata,
                    )
                except Exception as exc:
                    # Isolate the failure: record it on the result, keep going.
                    logger.warning(
                        f"Batch request failed: {exc.__class__.__name__}: {exc}"
                    )
                    return BatchResponse(
                        content="",
                        model=req.model,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        metadata={**(req.metadata or {}), "error": str(exc)},
                    )

        return await asyncio.gather(*[_one(r) for r in requests])

    async def astream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        complexity: TaskComplexity | None = None,
        node: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion response token by token.

        Args:
            messages: Chat messages in OpenAI format.
            model: Explicit model to use.
            complexity: Task complexity for automatic routing.
            node: Optional graph-node name (see ``acompletion``); threads into
                ``ModelRouter.route`` for per-node tier overrides.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            metadata: Optional metadata for logging.

        Yields:
            Token strings as they arrive.
        """
        if model is None:
            if self._pinned_model is not None:
                model = self._pinned_model
            elif complexity is not None:
                model = self._model_router.route(complexity, node=node)
            else:
                model = self._model_router.route(TaskComplexity.SIMPLE, node=node)

        assert model is not None  # guaranteed by routing logic

        provider = self._extract_provider(model)
        temperature = self._resolve_temperature(temperature)
        kwargs = self._build_kwargs(model, temperature, max_tokens, metadata)
        # Retry-aware pacing (mirrors _retry_call): reserve input + the capped
        # max_tokens against the ACTUAL provider's budget. Keyed on the model's
        # provider; the fallback loop below re-acquires per fallback model.
        await self._rate_limiter.acquire(
            provider, self._estimate_tokens(messages) + int(kwargs.get("max_tokens") or 0)
        )

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
                    fb_provider = self._extract_provider(fallback_model)
                    await self._rate_limiter.acquire(
                        fb_provider,
                        self._estimate_tokens(messages)
                        + int(fb_kwargs.get("max_tokens") or 0),
                    )
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
        node: str | None = None,
        tool_choice: dict[str, Any] | None = None,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> ToolCallResponse:
        """Send a completion request with tool definitions.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Tool definitions for function calling.
            model: Explicit model to use.
            complexity: Task complexity for routing.
            node: Optional graph-node name (see ``acompletion``); threads into
                ``ModelRouter.route`` for per-node tier overrides.
            temperature: Sampling temperature.
            metadata: Optional metadata for logging.

        Returns:
            ToolCallResponse with tool calls and optional text content.
        """
        response = await self.acompletion(
            messages=messages,
            model=model,
            complexity=complexity,
            node=node,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            metadata=metadata,
            timeout=timeout,
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

    def _resolve_temperature(self, temperature: float | None) -> float:
        """Resolve a sentinel ``None`` temperature to the configured default.

        Centralizes the sampling-temperature default so callers can override it
        via ``LLM_DEFAULT_TEMPERATURE`` (ResilienceSettings) without touching
        every call site. Explicit values pass through unchanged.
        """
        if temperature is None:
            return self._settings.resilience.llm_default_temperature
        return temperature

    async def _execute_with_fallback(
        self,
        messages: list[dict[str, Any]],
        model: str,
        complexity: TaskComplexity | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        require_vision: bool = False,
    ) -> LLMResponse:
        """Execute an LLM call with automatic fallback on failure."""
        # Resolve the sentinel default once into a fresh local. A plain
        # reassignment of the `temperature` parameter would be widened back to
        # float | None at the fallback loop's join point; a non-parameter local
        # keeps the inferred float type throughout the function.
        resolved_temperature: float = self._resolve_temperature(temperature)
        fallback_chain = [model] + FALLBACK_CHAINS.get(model, [])

        # Multimodal (D2): when a vision payload is attached, restrict the chain
        # to image-capable models (``ModelSpec.supports_images``) so a text-only
        # fallback never silently drops the images. Fail-safe: if the filter
        # would empty the chain (every entry non-vision or unknown), keep the
        # original chain unchanged — a degraded text attempt is preferable to
        # raising immediately (the provider may still tolerate the block).
        if require_vision:
            vision_chain = [
                m
                for m in fallback_chain
                if MODEL_REGISTRY.get(m) is not None
                and MODEL_REGISTRY[m].supports_images
            ]
            if vision_chain:
                fallback_chain = vision_chain
            else:
                logger.warning(
                    f"No vision-capable model in fallback chain for {model}; "
                    f"attempting original chain (provider may drop the images)"
                )

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
                # Circuit breaker: skip providers whose breaker is open, falling
                # through to the next provider in the chain (outage protection).
                await self._circuit_breaker.before_call(attempt_provider)
            except CircuitBreakerOpenError:
                logger.info(
                    f"Circuit open for {attempt_provider}, skipping to next fallback"
                )
                continue
            try:
                kwargs = self._build_kwargs(
                    attempt_model, resolved_temperature, max_tokens, metadata,
                    thinking=thinking, reasoning_effort=reasoning_effort,
                    timeout=timeout,
                )
                if tools:
                    kwargs["tools"] = tools
                if response_format:
                    kwargs["response_format"] = response_format
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice

                # Pre-emptive guard: on pre-4.x Anthropic, a json_schema
                # response_format is served via tool-conversion which FORCES
                # tool_choice — competing with an explicit tool_choice and
                # causing a 400. Drop tool_choice + warn; the existing
                # :484-525 recovery catches any residual conflict. 4.x+
                # Anthropic uses output_format (no forced tool_choice) so it is
                # exempt. Native structured outputs (2D).
                if (
                    response_format
                    and response_format.get("type") == "json_schema"
                    and tool_choice
                    and attempt_provider == "anthropic"
                    and not is_anthropic_4x_or_newer(attempt_model)
                ):
                    logger.warning(
                        f"Dropping tool_choice for {attempt_model}: json_schema "
                        f"on pre-4.x Anthropic forces tool_choice via "
                        f"tool-conversion (conflict)."
                    )
                    kwargs.pop("tool_choice", None)

                # Per-tier reasoning/thinking control (opt-in). Fills in
                # provider-native thinking params the caller did NOT set, so a
                # caller's explicit thinking/reasoning_effort always wins
                # (setdefault). ``temperature`` is force-applied only when an
                # invariant demands it (Anthropic needs 1.0 with thinking).
                tier_thinking = thinking_params_for(
                    complexity, attempt_provider, attempt_model,
                    self._settings.reasoning,
                )
                for tk, tv in tier_thinking.items():
                    if tk in ("thinking", "reasoning_effort", "extra_body"):
                        kwargs.setdefault(tk, tv)
                    else:
                        kwargs[tk] = tv

                # P0 — proactive DeepSeek thinking-disable. deepseek-v4-flash
                # ships with thinking ON; a FORCED tool_choice (the write-nudge
                # forcing a specific tool) makes the provider 400 with "Thinking
                # mode does not support this tool_choice". The reactive handler
                # below recovers, but only AFTER the first attempt already
                # burned thinking tokens. Disable thinking pre-flight for any
                # deepseek-v4-flash variant (native / alibaba-DashScope /
                # nvidia) whenever a FORCED tool_choice is set and the caller
                # did not explicitly request thinking. In this codebase the
                # only tool_choice a caller ever sets is a forced dict
                # (execute.py write-nudge), so "tool_choice present" ⟺ "forced".
                # ``litellm.drop_params`` makes the extra_body a harmless no-op
                # for any provider that ignores it, so this is safe across all
                # three hostings.
                if (
                    kwargs.get("tool_choice")
                    and "deepseek-v4-flash" in attempt_model
                    and not kwargs.get("thinking")
                    and "thinking" not in (kwargs.get("extra_body") or {})
                ):
                    kwargs["extra_body"] = {
                        **(kwargs.get("extra_body") or {}),
                        "thinking": {"type": "disabled"},
                    }
                    logger.debug(
                        f"Proactively disabled thinking for {attempt_model} "
                        f"(forced tool_choice pre-flight)"
                    )

                # Provider-native prompt caching (Anthropic cache_control).
                # Opt-in; disabled → passthrough (same list object). Computed
                # per-attempt because fallback may cross providers and
                # cache_control is Anthropic-only. See prompt_cache_control.py.
                cache_cfg = self._settings.prompt_cache
                call_messages = inject_cache_breakpoints(
                    messages,
                    attempt_provider,
                    enabled=cache_cfg.enabled,
                    min_system_tokens=cache_cfg.min_system_tokens,
                )

                response = await self._retry_call(
                    call_messages, provider=attempt_provider, **kwargs
                )
                await self._circuit_breaker.record_success(attempt_provider)
                return self._parse_response(response, attempt_model, attempt_provider)

            except _TRANSIENT_ERRORS as exc:
                last_error = exc
                await self._circuit_breaker.record_failure(
                    attempt_provider, transient=True
                )
                logger.warning(
                    f"LLM call failed for {attempt_model}: {exc.__class__.__name__}: {exc}"
                )
                continue
            except (litellm.AuthenticationError, litellm.BadRequestError) as exc:  # type: ignore[attr-defined]
                # Auth/validation errors must NOT trip the breaker (one bad key
                # is not a provider outage); record with transient=False.
                await self._circuit_breaker.record_failure(
                    attempt_provider, transient=False
                )
                # Recoverable tool_choice / thinking-mode conflict. Thinking-mode
                # models (e.g. deepseek-v4-flash, thinking ON by default) reject
                # a *forced* tool_choice with a 400 "Thinking mode does not
                # support this tool_choice". That is not a provider outage.
                #
                # Stage 1 — DeepSeek (all 3 hostings: native deepseek-v4-flash,
                # alibaba-deepseek-v4-flash / DashScope, nvidia-deepseek-v4-flash):
                # the API supports disabling thinking via
                # extra_body={"thinking":{"type":"disabled"}}. (litellm 1.83.14's
                # native ``thinking=`` param only accepts "enabled" and silently
                # drops "disabled", so extra_body is required.) With thinking
                # disabled the model honors the forced tool_choice — strictly
                # better than dropping tool_choice, which made deepseek narrate
                # instead of calling the file tool (9 wasted write-nudges / q4).
                # Verified empirically: extra_body path → tool_called=True. The
                # proactive pre-flight above usually prevents this 400 entirely;
                # this reactive stage remains as the belt-and-suspenders path
                # (e.g. caller explicitly enabled thinking despite a forced
                # tool_choice). Matched on the model substring so all three
                # hostings share one path.
                #
                # Stage 2 — fallback for any other model/cause: drop tool_choice
                # and rely on the execute-node write-nudge's system-prompt hint.
                if tool_choice and "tool_choice" in str(exc).lower():
                    if "deepseek-v4-flash" in attempt_model:
                        logger.warning(
                            f"{attempt_model} rejects forced tool_choice "
                            f"(thinking-mode); retrying same model with "
                            f"thinking disabled"
                        )
                        retry_kwargs = dict(kwargs)
                        retry_kwargs["extra_body"] = {
                            **(kwargs.get("extra_body") or {}),
                            "thinking": {"type": "disabled"},
                        }
                        try:
                            response = await self._retry_call(
                                call_messages, provider=attempt_provider, **retry_kwargs
                            )
                            await self._circuit_breaker.record_success(attempt_provider)
                            return self._parse_response(
                                response, attempt_model, attempt_provider
                            )
                        except Exception as retry_exc:
                            logger.warning(
                                f"thinking-disabled retry for {attempt_model} "
                                f"also failed: {retry_exc.__class__.__name__}: "
                                f"{retry_exc}"
                            )
                            last_error = retry_exc

                    # Fallback (non-deepseek, or thinking-disable didn't work):
                    # drop tool_choice and rely on the write-nudge system prompt.
                    logger.warning(
                        f"{attempt_model} tool_choice conflict unresolved; "
                        f"retrying same model with tool_choice dropped"
                    )
                    kwargs.pop("tool_choice", None)
                    try:
                        response = await self._retry_call(
                            call_messages, provider=attempt_provider, **kwargs
                        )
                        await self._circuit_breaker.record_success(attempt_provider)
                        return self._parse_response(
                            response, attempt_model, attempt_provider
                        )
                    except Exception as retry_exc:
                        logger.warning(
                            f"tool_choice-less retry for {attempt_model} "
                            f"also failed: {retry_exc.__class__.__name__}: "
                            f"{retry_exc}"
                        )
                        last_error = retry_exc
                else:
                    logger.error(f"Non-retryable error for {attempt_model}: {exc}")
                continue
            except Exception as exc:
                last_error = exc
                await self._circuit_breaker.record_failure(
                    attempt_provider, transient=True
                )
                logger.error(f"Unexpected error for {attempt_model}: {exc}")
                continue

        raise RuntimeError(
            f"All fallbacks exhausted for {model}. Last error: {last_error}"
        )

    async def _retry_call(
        self,
        messages: list[dict[str, Any]],
        *,
        provider: str,
        **kwargs: Any,
    ) -> Any:
        """Execute a litellm call with tenacity retry + retry-aware rate pacing.

        Retry params (attempts, backoff) are read from ``ResilienceSettings`` at
        call time so they're tunable via ``.env`` (LLM_MAX_RETRIES,
        LLM_RETRY_INITIAL_DELAY/MAX_DELAY/JITTER) without redecorating at import.

        Rate pacing is RETRY-AWARE: the per-provider budget is acquired on EVERY
        tenacity attempt (not once per logical call), so a retry fan-out is
        charged against the budget exactly as the provider charges it — every
        HTTP hit counts, including retries. Keyed on ``provider`` (the ACTUAL
        attempt provider passed by the fallback loop), so a fallback onto a
        different provider is paced against THAT provider's budget; the old
        single top-level acquire paced only the original model's provider and
        left every fallback unpaced. The estimated budget reserves input tokens
        PLUS the (capped) max_tokens so the TPM window reflects the reserved
        output, not just the prompt. A limiter hiccup is best-effort (logged at
        debug, never blocks the call) — rate limiting is observability-only.
        """
        resilience = self._settings.resilience
        estimated_tokens = self._estimate_tokens(messages) + int(
            kwargs.get("max_tokens") or 0
        )
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_TRANSIENT_ERRORS),
            stop=stop_after_attempt(resilience.llm_max_retries),
            wait=wait_exponential_jitter(
                initial=resilience.llm_retry_initial_delay,
                max=resilience.llm_retry_max_delay,
                jitter=resilience.llm_retry_jitter,
            ),
            before_sleep=lambda state: logger.warning(
                f"Retrying LLM call (attempt {state.attempt_number}): "
                f"{state.outcome.exception()}"  # type: ignore[union-attr]
            ),
            reraise=True,
        ):
            with attempt:
                # Retry-aware pacing: reserve the per-provider budget on every
                # attempt. Best-effort — never let a limiter error abort a call.
                try:
                    await self._rate_limiter.acquire(provider, estimated_tokens)
                except Exception:
                    logger.debug(
                        f"rate limiter acquire failed for {provider}; "
                        f"proceeding best-effort"
                    )
                # Hard per-call backstop (Bug T1): litellm/httpx occasionally
                # IGNORE the ``timeout`` kwarg on a hanging provider (observed
                # live: 90s timeout → 264s wall on a degraded Z.AI glm-4.7
                # call). That lets one call burn the entire tenacity budget
                # (~13 min before fallback) and stalls the run.
                # ``asyncio.wait_for`` enforces the cap regardless of the
                # underlying client: it cancels the litellm task at the cap and
                # raises ``asyncio.TimeoutError`` (a member of
                # ``_TRANSIENT_ERRORS``), so tenacity retries and the fallback
                # chain advances promptly. A small grace (``_WAIT_FOR_GRACE``)
                # over the configured timeout lets the cleaner in-library
                # timeout fire first when it works.
                call_timeout = float(
                    kwargs.get("timeout") or self._settings.llm.request_timeout
                )
                try:
                    return await asyncio.wait_for(
                        litellm.acompletion(messages=messages, **kwargs),
                        timeout=call_timeout + self._WAIT_FOR_GRACE,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Hard timeout backstop fired for {provider} after "
                        f"{call_timeout + self._WAIT_FOR_GRACE:.0f}s "
                        f"(litellm timeout={call_timeout:.0f}s was not "
                        f"enforced); retrying/falling back"
                    )
                    raise
        raise RuntimeError("unreachable")  # pragma: no cover

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

        # Extract reasoning content (DeepSeek thinking mode, etc.)
        reasoning_content = None
        if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            reasoning_content = str(choice.message.reasoning_content)

        usage = response.usage or litellm.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        input_tokens = usage.prompt_tokens or 0
        output_tokens = usage.completion_tokens or 0

        # Provider-native prompt-cache accounting. litellm surfaces Anthropic's
        # cache hits on the Usage object as _cache_read_input_tokens /
        # _cache_creation_input_tokens (absent/0 when caching is off or the
        # provider does not report them). getattr keeps this resilient across
        # litellm versions and mocked responses.
        cache_read_tokens = int(getattr(usage, "_cache_read_input_tokens", 0) or 0)
        cache_creation_tokens = int(getattr(usage, "_cache_creation_input_tokens", 0) or 0)
        if cache_read_tokens or cache_creation_tokens:
            logger.debug(
                f"Prompt cache ({model}): read={cache_read_tokens} "
                f"created={cache_creation_tokens} tokens"
            )

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
            reasoning_content=reasoning_content,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )

    def _build_kwargs(
        self,
        model: str,
        temperature: float,
        max_tokens: int | None,
        metadata: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Build keyword arguments for litellm call.

        Resolves registry keys (e.g., 'deepseek-v4-flash') to litellm
        model_ids (e.g., 'deepseek/deepseek-v4-flash') so litellm can
        identify the correct provider.
        """
        # Resolve registry key → litellm model_id
        spec = MODEL_REGISTRY.get(model)
        litellm_model = spec.model_id if spec else model

        kwargs: dict[str, Any] = {"model": litellm_model, "temperature": temperature}

        # Hard per-call timeout so an unresponsive provider cannot stall the run
        # on litellm's ~600s default. The tenacity retry layer still handles
        # litellm.Timeout as a transient error. A caller may pass a longer
        # ``timeout`` (e.g. codegen) to override the ``request_timeout`` default.
        kwargs["timeout"] = (
            timeout if timeout is not None else self._settings.llm.request_timeout
        )

        # litellm's OWN internal retries default to 3, which layered UNDER our
        # tenacity layer silently multiplies every transient-error hit (one
        # logical call → up to 9 provider hits = 3 litellm × 3 tenacity). A
        # single 429 then fans into ~9 immediate re-hits the provider correctly
        # rate-limits, which our retries hit even harder — the recurring "Z.AI
        # degradation". Pin to the configured value (default 0) so tenacity is
        # the SINGLE retry authority (it already backs off with jitter). Applies
        # to every provider. Env: LLM_LITELLM_NUM_RETRIES.
        kwargs["num_retries"] = self._settings.resilience.llm_litellm_num_retries

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        else:
            # Use model-specific defaults, hard-capped so a model that
            # over-declares its max_output (e.g. glm-4.7=128_000,
            # deepseek-v4-flash=~384_000) cannot reserve a runaway output
            # budget in the rate limiter's TPM accounting and inflate provider
            # rate-limit pressure on every classify/plan/verify/codegen call.
            # No single node response approaches 16K. Explicit caller max_tokens
            # (the optimizer passes a specific value) is NOT capped — a caller
            # asking for a concrete size knows its need. Env: LLM_MAX_OUTPUT_CAP.
            cap = self._settings.resilience.llm_max_output_cap
            if spec and spec.max_output:
                kwargs["max_tokens"] = min(spec.max_output, cap)
            else:
                kwargs["max_tokens"] = min(
                    self._settings.resilience.llm_default_max_tokens, cap
                )

        if metadata:
            kwargs["metadata"] = metadata

        # Set provider API key
        provider = self._extract_provider(model)
        api_key = self._get_api_key(provider)
        if api_key:
            kwargs["api_key"] = api_key

        # NVIDIA NIM OpenAI-compatible shim: litellm in this build rejects the
        # bare ``nvidia/`` prefix, so a registered nvidia model_id is rewritten
        # to ``openai/<id>`` against the pinned NIM api_base. Pure + unit-tested
        # (src/llm/nvidia_shim.py); verified live for all 16 registered models.
        litellm_model, nvidia_kwargs = nvidia_shim_model_id(provider, litellm_model)
        kwargs.update(nvidia_kwargs)

        # Pin the Anthropic endpoint so an ambient ANTHROPIC_BASE_URL inherited
        # from the environment (e.g. a Claude-Code→Z.AI gateway) cannot misroute
        # the app's own ANTHROPIC_API_KEY to a relay that rejects it. The app
        # authenticates with the key from settings; it must hit the endpoint that
        # key is valid for. Override via ANTHROPIC_API_BASE only if intentional.
        if provider == "anthropic":
            kwargs["api_base"] = self._settings.llm.anthropic_api_base or "https://api.anthropic.com"

        # Pin the Alibaba (Qwen / DashScope service) endpoint. Qwen models are
        # registered with an ``openai/`` model_id prefix; without an api_base
        # pin litellm would route them to OpenAI's endpoint using the
        # DASHSCOPE_API_KEY (same defect class as the resolved Anthropic
        # misroute). The DashScope key is only valid against the OpenAI-
        # compatible-mode endpoint below. Default to the INTERNATIONAL (Bailian)
        # endpoint — DashScope keys are region-bound and the China endpoint
        # (dashscope.aliyuncs.com) rejects an international key. Override via
        # ALIBABA_API_BASE to use the China endpoint instead.
        if provider == "alibaba":
            kwargs["api_base"] = (
                self._settings.llm.alibaba_api_base
                or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )

        # DeepSeek thinking mode — pass through thinking/reasoning_effort
        # LiteLLM standardizes these for DeepSeek models.
        if thinking is not None:
            kwargs["thinking"] = thinking
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

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

        # Walk fallback chain first — prefer provider diversity. Skip any
        # candidate whose provider is temporarily disabled (e.g. anthropic under
        # a quota cap) — selecting it would burn the fallback chain on a 400.
        for fb in FALLBACK_CHAINS.get(model, []):
            fb_spec = MODEL_REGISTRY.get(fb)
            if (
                fb_spec
                and _TIER_ORDER.get(fb_spec.tier, 0) < current_order
                and not ModelRouter.is_provider_disabled(fb_spec.provider)
            ):
                return fb

        # Scan registry for any cheaper model (also skipping disabled providers)
        for mid, mspec in MODEL_REGISTRY.items():
            if (
                mid != model
                and _TIER_ORDER.get(mspec.tier, 0) < current_order
                and not ModelRouter.is_provider_disabled(mspec.provider)
            ):
                return mid

        return None

    @staticmethod
    def _extract_provider(model: str) -> str:
        """Extract provider from litellm model identifier."""
        return ModelRouter._extract_provider(model)

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, Any]] | str) -> int:
        """Rough token estimation: ~4 chars per token.

        Content-tolerant: uses ``_content_char_len`` so a multimodal
        (block-list) ``content`` field — the shape a vision payload produces —
        is counted instead of raising on ``len(list)``.
        """
        if isinstance(messages, str):
            return max(1, len(messages) // 4)
        total_chars = sum(
            _content_char_len(m.get("content", "")) for m in messages if isinstance(m, dict)
        )
        return max(1, total_chars // 4)

    def _configure_litellm(self) -> None:
        """Configure litellm global settings."""
        litellm.set_verbose = False  # type: ignore[attr-defined]
        litellm.drop_params = True  # Drop unsupported params instead of erroring

        # Silence litellm's internal loggers (DEBUG spam like
        # "NO SHARED SESSION", "Creating new ClientSession",
        # "model isn't mapped yet"). These propagate via the stdlib logging
        # module into loguru; raising them to WARNING removes the noise without
        # changing behavior.
        for _litellm_logger in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
            logging.getLogger(_litellm_logger).setLevel(logging.WARNING)

        # Enable LangSmith callbacks for litellm when tracing is active
        if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
            litellm.success_callback = ["langsmith"]
            litellm.failure_callback = ["langsmith"]
            logger.debug("litellm configured with LangSmith callbacks")
        else:
            litellm.success_callback = []
            litellm.failure_callback = []

        logger.debug("litellm configured: drop_params=True, verbose=False")
