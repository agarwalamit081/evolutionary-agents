"""
pydantic-settings configuration for the Turing Agent.

All configuration loaded from .env via nested BaseSettings classes.
Application code accesses config through get_settings() singleton.

Provider-specific API keys (no generic LLM_API_KEY):
  OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY, etc.
"""

from __future__ import annotations

import functools
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── LLM Provider Settings ─────────────────────────────────────────


class LLMProviderSettings(BaseSettings):
    """API keys and configuration for all supported LLM providers."""

    # Provider API Keys
    anthropic_api_key: Optional[str] = None
    # Explicit Anthropic endpoint. When set, the gateway pins Anthropic calls
    # here instead of inheriting an ambient ANTHROPIC_BASE_URL (e.g. a
    # Claude-Code→Z.AI gateway) that would misroute the app's own key. Defaults
    # to the standard public Anthropic API in the gateway.
    anthropic_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_org_id: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    zai_api_key: Optional[str] = None
    minimax_api_key: Optional[str] = None
    minimax_group_id: Optional[str] = None
    mistral_api_key: Optional[str] = None
    mistral_org_id: Optional[str] = None
    moonshot_api_key: Optional[str] = None
    dashscope_api_key: Optional[str] = None
    # Explicit Alibaba (Qwen / DashScope service) OpenAI-compatible endpoint.
    # Qwen models are registered with an ``openai/`` model_id prefix, so without
    # an api_base pin litellm routes them to OpenAI's endpoint using the
    # DASHSCOPE_API_KEY and the call fails. The gateway defaults to the
    # INTERNATIONAL (Bailian) endpoint (dashscope-intl.aliyuncs.com); set this
    # to the China endpoint (dashscope.aliyuncs.com) only for a China-region
    # key. The provider is "alibaba"; only the API key field keeps the dashscope
    # name (env DASHSCOPE_API_KEY).
    alibaba_api_base: Optional[str] = None
    alibaba_workspace_id: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None

    # Default model selection
    default_llm_provider: str = "deepseek"
    default_llm_model: str = "deepseek-v4-flash"

    # Fast model for classification, routing, simple tasks
    fast_llm_provider: str = "openai"
    fast_llm_model: str = "gpt-4o-mini-2024-07-18"

    # Heavy model for reasoning, planning, complex analysis
    reasoning_llm_provider: str = "deepseek"
    reasoning_llm_model: str = "deepseek-v4-pro"

    # Embedding generation (§10.2) — litellm embedding model + output dimension.
    # ``embedding_dim`` MUST match the pgvector ``Vector(N)`` columns:
    # ``cold_memories.embedding`` and ``memory_embeddings.embedding`` are both
    # ``Vector(768)`` in src/db/models.py. text-embedding-3-small supports a
    # ``dimensions`` param so its output is reduced to 768 to fit the columns.
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 768

    # Hard timeout (seconds) for each LLM completion call, passed to litellm as
    # ``timeout``. Without it litellm falls back to its long default (~600s), so
    # a single unresponsive provider stalls the entire agent run. Bounded by the
    # tenacity retry layer (_MAX_RETRIES=3): worst case ≈ timeout × retries.
    # Default 90s (REQUEST_TIMEOUT) — reasoning/embedding calls fail fast to the
    # fallback chain while leaving margin for a slow first token.
    request_timeout: float = 90.0

    # Longer timeout for code-generation calls (tool_create + evolution CODE
    # mutations) that route to a code-strong model and emit a full handler in one
    # shot — these legitimately exceed the reasoning default (observed live: a
    # 58.4s deepseek-v4-pro codegen call was cut at the old 60s default). Passed
    # via a per-call ``timeout`` override on the gateway so reasoning/embedding
    # calls keep the shorter ``request_timeout`` and fail fast to the fallback.
    codegen_timeout: float = 180.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator(
        "default_llm_provider",
        "fast_llm_provider",
        "reasoning_llm_provider",
    )
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Ensure provider is in the allowed list."""
        allowed = {
            "anthropic",
            "openai",
            "deepseek",
            "zai",
            "minimax",
            "mistral",
            "moonshot",
            "alibaba",
            "groq",
            "google",
            "openrouter",
            "nvidia",
            "ollama",
        }
        if v not in allowed:
            raise ValueError(
                f"Invalid LLM provider: {v}. Must be one of {sorted(allowed)}"
            )
        return v

    def get_provider_key(self, provider: str) -> Optional[str]:
        """Get the API key for a given provider name.

        Returns None if the provider key is not configured.
        """
        key_map: dict[str, Optional[str]] = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "deepseek": self.deepseek_api_key,
            "zai": self.zai_api_key,
            "minimax": self.minimax_api_key,
            "mistral": self.mistral_api_key,
            "moonshot": self.moonshot_api_key,
            "alibaba": self.dashscope_api_key,
            "groq": self.groq_api_key,
            "google": self.google_api_key,
            "openrouter": self.openrouter_api_key,
            "nvidia": self.nvidia_api_key,
            "ollama": None,  # Local, no key needed
        }
        return key_map.get(provider)

    def has_provider_key(self, provider: str) -> bool:
        """Check if a provider has a non-empty API key configured."""
        key = self.get_provider_key(provider)
        return key is not None and len(key.strip()) > 0


# ─── LLM Resilience Settings ────────────────────────────────────────


class ResilienceSettings(BaseSettings):
    """LLM call resilience: retry, sampling defaults, and output caps.

    Centralizes the magic numbers previously hardcoded in
    ``src/llm/gateway.py`` (``_MAX_RETRIES``, the retry backoff, the four
    ``temperature=0.5`` defaults, and the ``max_tokens=4096`` fallback) so an
    operator can tune the whole LLM I/O envelope from ``.env`` without code
    changes. Distinct from ``LLMProviderSettings.request_timeout`` (the hard
    per-call timeout), which stays where it is to avoid churning callers.
    """

    # Max retry attempts on transient LLM errors (429/5xx/timeout/conn).
    llm_max_retries: int = 3  # Env: LLM_MAX_RETRIES
    # tenacity wait_exponential_jitter params (seconds).
    llm_retry_initial_delay: float = 1.0  # Env: LLM_RETRY_INITIAL_DELAY
    llm_retry_max_delay: float = 30.0  # Env: LLM_RETRY_MAX_DELAY
    llm_retry_jitter: float = 2.0  # Env: LLM_RETRY_JITTER
    # Default sampling temperature when a caller omits it (gateway acompletion/
    # astream/acompletion_with_tools all defaulted to 0.5).
    llm_default_temperature: float = 0.5  # Env: LLM_DEFAULT_TEMPERATURE
    # Default output cap when no model spec supplies max_tokens.
    llm_default_max_tokens: int = 4096  # Env: LLM_DEFAULT_MAX_TOKENS
    # litellm's OWN internal retries (litellm/main.py defaults num_retries=3
    # when it is not passed). Layered UNDER this tenacity layer, that silently
    # multiplies every transient-error HTTP hit (one logical call → up to 9
    # provider hits = 3 litellm × 3 tenacity) — the root cause of the recurring
    # "Z.AI degradation": a single 429 fans into ~9 immediate re-hits, which
    # Z.AI correctly rate-limits, which our retries hit even harder. Set 0 so
    # tenacity is the SINGLE retry authority (it already backs off with jitter).
    # Applies to EVERY provider — GLM is hosted on zai/nvidia/etc. alike.
    llm_litellm_num_retries: int = 0  # Env: LLM_LITELLM_NUM_RETRIES
    # Hard cap on the default max_tokens when a caller omits it. Without this,
    # _build_kwargs falls back to spec.max_output (128_000 for glm-4.7) on EVERY
    # classify/plan/verify/codegen call. A 128K max_tokens reserves that whole
    # budget in the rate limiter's TPM accounting and signals an enormous
    # response to the provider, which inflates rate-limit pressure for no
    # benefit (no single classify/plan response approaches 16K). 16K covers any
    # realistic single response; callers genuinely needing more pass max_tokens
    # explicitly. Sits ABOVE spec.max_output in min() so it also bounds models
    # that over-declare (e.g. deepseek-v4-flash ~384K).
    llm_max_output_cap: int = 16384  # Env: LLM_MAX_OUTPUT_CAP

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("llm_default_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is a sane sampling range (0–2)."""
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"Temperature must be between 0 and 2. Got: {v}")
        return v


# ─── Circuit Breaker Settings ───────────────────────────────────────


class CircuitBreakerSettings(BaseSettings):
    """Per-provider circuit breaker thresholds.

    Exposed (hoisted from the ``CircuitBreaker`` constructor) so provider
    reliability tuning doesn't require editing ``src/llm/circuit_breaker.py``.

    ``cb_failure_threshold`` defaults to 3 (tuned down from the original 5): a
    multi-provider rate-limit storm spreads transient failures across providers
    and rarely accumulates 5 *consecutive* failures on any single one, so the
    breaker's OPEN path never engaged and the fallback chain kept hammering a
    rate-limited provider. 3 opens a clearly-down provider one failure sooner.
    """

    cb_failure_threshold: int = 3  # Env: CB_FAILURE_THRESHOLD
    cb_recovery_timeout: float = 60.0  # Env: CB_RECOVERY_TIMEOUT
    cb_half_open_max_calls: int = 1  # Env: CB_HALF_OPEN_MAX_CALLS

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("cb_failure_threshold", "cb_half_open_max_calls")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive integers."""
        if v < 1:
            raise ValueError(f"Must be a positive integer. Got: {v}")
        return v


# ─── Rate Limiter Settings ──────────────────────────────────────────


class RateLimiterSettings(BaseSettings):
    """Per-provider rate-limit fallbacks.

    ``src/llm/rate_limiter.py`` keeps an explicit PROVIDER_LIMITS table for
    known providers; these are the RPM/TPM used when a provider is absent from
    that table.
    """

    rate_limit_default_rpm: int = 60  # Env: RATE_LIMIT_DEFAULT_RPM
    rate_limit_default_tpm: int = 100_000  # Env: RATE_LIMIT_DEFAULT_TPM
    # Per-provider RPM/TPM OVERRIDES layered on top of the curated
    # PROVIDER_LIMITS table in src/llm/rate_limiter.py (Phase 4 G). A JSON map
    # of provider → [rpm, tpm]; empty (default) → the curated table is used
    # unchanged. Lets a deployment tune a provider's limits without a code
    # change. Env: RATE_LIMIT_PROVIDER_OVERRIDES.
    rate_limit_provider_overrides: dict[str, list[int]] = Field(default_factory=dict)
    # When True AND a Redis client is attached to the limiter, the per-provider
    # rate budget is enforced CROSS-PROCESS via an atomic Redis token-bucket so
    # concurrent workers/sub-agents SHARE one RPM/TPM budget per provider. The
    # in-memory aiolimiter is per-LLMGateway-instance (built per-run), so two
    # workers each silently got their own 60-RPM zai bucket = 120 RPM against a
    # 60-RPM provider — the real trigger of the recurring "Z.AI degradation".
    # Keyed off provider name, so it applies to GLM on zai/nvidia/etc. alike.
    rate_limit_cross_process_enabled: bool = True  # Env: RATE_LIMIT_CROSS_PROCESS_ENABLED
    # Max blocking acquire attempts (with backoff) in the cross-process path
    # before proceeding best-effort. Rate limiting is observability-only — it
    # must NEVER hard-fail a run, so after bounded waiting the call proceeds.
    rate_limit_max_wait_attempts: int = 5  # Env: RATE_LIMIT_MAX_WAIT_ATTEMPTS

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("rate_limit_default_rpm", "rate_limit_default_tpm")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive integers."""
        if v < 1:
            raise ValueError(f"Must be a positive integer. Got: {v}")
        return v


# ─── Routing Settings (findings F2 — operator-tunable tier maps) ────


class RoutingSettings(BaseSettings):
    """Operator overrides for the curated complexity/node model tier maps.

    ``model_router.py`` ships curated ``COMPLEXITY_TIER_MAP`` /
    ``NODE_TIER_MAP`` as the production defaults. These JSON knobs let a
    deployment retune a single (complexity, node) or (complexity) routing
    decision without a code change — read at ``route()`` call-time by
    ``ModelRouter._apply_routing_overrides``.

    Keys:
      • node-tier: ``"<COMPLEXITY>:<node>"`` (e.g. ``"COMPLEX:execute"``)
      • complexity-tier: a bare ``"<COMPLEXITY>"`` (e.g. ``"COMPLEX"``)
    Values: a litellm model_id present in MODEL_REGISTRY. Empty (default) →
    the curated tables are used unchanged; invalid JSON / unknown model_id is
    ignored so a bad env can never break routing. A node-tier override wins
    over a complexity-tier override for the same decision.

    Env: ``ROUTING_NODE_TIER_OVERRIDES_JSON`` / ``ROUTING_COMPLEXITY_TIER_OVERRIDES_JSON``.
    """

    routing_node_tier_overrides_json: str = "{}"
    routing_complexity_tier_overrides_json: str = "{}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# ─── Prompt Cache Control Settings (provider-native caching) ────────


class PromptCacheControlSettings(BaseSettings):
    """Anthropic-native prompt caching via ``cache_control`` breakpoints.

    When enabled, a long system message sent to an Anthropic model is tagged
    with an ephemeral ``cache_control`` marker so the reused system prompt is
    served from the provider's cache on subsequent calls. litellm forwards the
    marker to the Anthropic API unmodified. Off by default — when disabled the
    messages list is passed through unchanged (zero behavior change).
    Implemented in ``src/llm/prompt_cache_control.py``.
    """

    # Master switch. Env: PROMPT_CACHE_CONTROL_ENABLED
    enabled: bool = False
    # Minimum estimated system-prompt size (tokens) before a breakpoint is worth
    # the cache write cost. Env: PROMPT_CACHE_CONTROL_MIN_SYSTEM_TOKENS
    min_system_tokens: int = 1024

    model_config = SettingsConfigDict(
        env_prefix="prompt_cache_control_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("min_system_tokens")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure a positive minimum size."""
        if v < 1:
            raise ValueError(f"Must be a positive integer. Got: {v}")
        return v


# ─── Batching Settings (concurrent request batching) ────────────────


class BatchingSettings(BaseSettings):
    """Concurrent request batching for ``LLMGateway.abatch``.

    ``abatch`` runs many independent completions via ``asyncio.gather`` bounded
    by ``max_concurrency`` (an ``asyncio.Semaphore``); the per-provider rate
    limiter still gates RPM/TPM under each call. No async litellm batch
    primitive exists in this build (``litellm.batch_completion`` is a sync
    ThreadPool wrapper), so gather+semaphore is the supported approach. Off by
    default; when disabled, ``abatch`` runs sequentially (concurrency 1) while
    preserving its isolation contract.
    """

    # Master switch. Env: LLM_BATCH_ENABLED
    enabled: bool = False
    # Max concurrent in-flight completions. Env: LLM_BATCH_MAX_CONCURRENCY
    max_concurrency: int = 5

    model_config = SettingsConfigDict(
        env_prefix="llm_batch_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("max_concurrency")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure a positive concurrency level."""
        if v < 1:
            raise ValueError(f"Must be a positive integer. Got: {v}")
        return v


# ─── Reasoning Control Settings (per-tier thinking) ─────────────────


class ReasoningControlSettings(BaseSettings):
    """Per-tier extended thinking/reasoning control.

    When enabled, complex/critical tasks get provider-native extended thinking
    (Anthropic ``thinking``, DeepSeek/Z.AI ``extra_body``, OpenAI o-series
    ``reasoning_effort``) and trivial/simple tasks get it off (DeepSeek gets an
    explicit disable since its thinking is ON by default). Implemented in
    ``src/llm/thinking_control.py``. Off by default → no thinking params are
    emitted when disabled (zero behavior change).
    """

    # Master switch. Env: REASONING_CONTROL_ENABLED
    enabled: bool = False
    # Effort for complex/critical tasks: "none"/"low"/"medium"/"high".
    # Env: REASONING_CONTROL_COMPLEX_THINKING
    complex_thinking: str = "medium"
    # Effort for trivial/simple tasks: "none" disables thinking.
    # Env: REASONING_CONTROL_SIMPLE_THINKING
    simple_thinking: str = "none"
    # Anthropic thinking budget (tokens) for complex/critical tasks.
    # Env: REASONING_CONTROL_ANTHROPIC_BUDGET_TOKENS_COMPLEX
    anthropic_budget_tokens_complex: int = 8000
    # Anthropic thinking budget (tokens) for moderate/other enabled tiers.
    # Env: REASONING_CONTROL_ANTHROPIC_BUDGET_TOKENS_MEDIUM
    anthropic_budget_tokens_medium: int = 4000

    model_config = SettingsConfigDict(
        env_prefix="reasoning_control_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator(
        "anthropic_budget_tokens_complex", "anthropic_budget_tokens_medium"
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive token budgets."""
        if v < 1:
            raise ValueError(f"Must be a positive integer. Got: {v}")
        return v


# ─── Native Structured Output Settings (JSON-schema) ────────────────


class NativeStructuredSettings(BaseSettings):
    """Provider-native JSON-schema structured outputs.

    When enabled, a caller-supplied JSON schema (``acompletion(response_schema=...)``)
    is converted to a provider-native ``response_format`` so the model emits
    schema-conformant JSON natively (OpenAI/DeepSeek strict ``json_schema``,
    Anthropic ``output_format``, Gemini ``response_schema``) rather than via
    prompt instructions + json_repair. Implemented in ``src/llm/structured_output.py``.
    Off by default → ``response_schema`` args are ignored (back-compat).
    """

    # Master switch. Env: NATIVE_STRUCTURED_OUTPUT_ENABLED
    enabled: bool = False

    model_config = SettingsConfigDict(
        env_prefix="native_structured_output_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# ─── Database Settings ──────────────────────────────────────────────


class DatabaseSettings(BaseSettings):
    """PostgreSQL database connection configuration."""

    database_url: str = "postgresql+asyncpg://postgres:changeme@localhost:5433/turing_agent"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # .env keys are UPPERCASE (DATABASE_URL, DATABASE_POOL_SIZE, ...). pydantic
        # derives the lookup key from the lowercase field name, so case_sensitive=True
        # silently ignores every override and falls to the code default — the app would
        # connect to localhost:5432 (default) instead of the configured 5433. ALL
        # settings groups here read case-insensitively; see LangSmithSettings for the
        # same documented rationale.
        case_sensitive=False,
    )

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure DATABASE_URL uses asyncpg driver."""
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use asyncpg driver: "
                "postgresql+asyncpg://user:pass@host:port/dbname"
            )
        return v


# ─── Redis Settings ─────────────────────────────────────────────────


class RedisSettings(BaseSettings):
    """Redis cache and session storage configuration."""

    redis_url: str = "redis://localhost:6380/0"
    redis_ttl_hot_memory: int = 86400  # 24 hours
    redis_ttl_session: int = 3600  # 1 hour
    redis_ttl_rate_limit: int = 60  # 1 minute
    cache_ttl_seconds: int = 3600  # LLM prompt cache TTL

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Ensure REDIS_URL starts with redis://."""
        if not v.startswith("redis://"):
            raise ValueError(f"REDIS_URL must start with redis://. Got: {v}")
        return v


# ─── Tool Cache Settings ────────────────────────────────────────────


class ToolCacheSettings(BaseSettings):
    """Redis-backed result cache for idempotent, read-only tools.

    Only opt-in read-only tools are cached (web_search, file_reader); mutating
    tools (file_writer) are never cached. Any Redis failure degrades to a
    transparent cache miss and never breaks a tool call.
    """

    tool_cache_enabled: bool = True
    tool_cache_ttl_seconds: int = 3600  # 1 hour
    # Shorter TTL for recency-sensitive queries (Gap 5 dynamic TTL). A query
    # carrying a time-window param (time_range/timelimit/tbs/days/…) or a
    # lexical recency cue ("latest"/"news"/"today"/"2025" in the text) is
    # cached for this long instead of the full ``tool_cache_ttl_seconds``, so a
    # moved time window isn't served stale. Evergreen queries keep the base TTL.
    tool_cache_recency_ttl_seconds: int = 300  # 5 minutes

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# ─── Tool Limits Settings ───────────────────────────────────────────


class ToolLimitsSettings(BaseSettings):
    """Per-tool timeouts, size caps, and retry params for the built-in tools.

    Centralizes the module-level constants previously hardcoded in
    ``src/tools/builtin/`` (terminal_command, http_request, web_scraper,
    code_executor, web_search) so an operator can bound tool I/O from ``.env``.
    Tools read these at call-time via ``get_settings().tools`` (not at import)
    so changes take effect without a process restart in long-running hosts.
    """

    terminal_command_timeout: float = 30.0  # Env: TERMINAL_COMMAND_TIMEOUT
    terminal_max_output_bytes: int = 16_000  # Env: TERMINAL_MAX_OUTPUT_BYTES
    http_request_timeout: float = 15.0  # Env: HTTP_REQUEST_TIMEOUT
    http_max_response_chars: int = 8000  # Env: HTTP_MAX_RESPONSE_CHARS
    http_max_body_bytes: int = 1_000_000  # Env: HTTP_MAX_BODY_BYTES
    web_scraper_timeout: float = 20.0  # Env: WEB_SCRAPER_TIMEOUT
    web_scraper_max_bytes: int = 5 * 1024 * 1024  # Env: WEB_SCRAPER_MAX_BYTES
    web_scraper_max_chars: int = 8000  # Env: WEB_SCRAPER_MAX_CHARS
    # curl-cffi TLS-impersonation anti-bot tier (Gap 7). On an anti-bot signal
    # (403/429) the scraper retries once with a Chrome-impersonated TLS session,
    # which bypasses Cloudflare/bot-WAF JA3 blocking that httpx cannot. Opt-in
    # shape (default on) but degrades to httpx-only if curl_cffi isn't importable.
    web_scraper_curl_cffi_enabled: bool = True  # Env: WEB_SCRAPER_CURL_CFFI_ENABLED
    web_scraper_curl_cffi_impersonate: str = "chrome"  # Env: WEB_SCRAPER_CURL_CFFI_IMPERSONATE
    code_executor_timeout: int = 30  # Env: CODE_EXECUTOR_TIMEOUT
    # glm-ocr OCR tool (Z.AI layout_parsing): the API is a single synchronous
    # POST that can take several seconds on a multi-page PDF; bound it like the
    # other fetch tools. Env: OCR_PARSER_TIMEOUT / OCR_PARSER_MAX_CHARS.
    ocr_parser_timeout: float = 30.0  # Env: OCR_PARSER_TIMEOUT
    ocr_parser_max_chars: int = 8000  # Env: OCR_PARSER_MAX_CHARS
    # image_generator tool (litellm aimage_generation). The model is a separate
    # API surface from chat completions; the default ``gpt-image-1`` is a real
    # litellm image-gen model (mode=image_generation, ~$0.011/image at low/1024²).
    # Swap for e.g. ``dashscope/qwen-image-2.0`` to reuse the DASHSCOPE key.
    # Env: IMAGE_GEN_* (IMAGE_GEN_MODEL / IMAGE_GEN_DEFAULT_SIZE /
    # IMAGE_GEN_DEFAULT_QUALITY / IMAGE_GEN_TIMEOUT / IMAGE_GEN_API_BASE).
    image_gen_model: str = "gpt-image-1"  # Env: IMAGE_GEN_MODEL
    image_gen_default_size: str = "1024x1024"  # Env: IMAGE_GEN_DEFAULT_SIZE
    image_gen_default_quality: str = "low"  # Env: IMAGE_GEN_DEFAULT_QUALITY
    image_gen_timeout: float = 60.0  # Env: IMAGE_GEN_TIMEOUT
    image_gen_api_base: str = ""  # Env: IMAGE_GEN_API_BASE (empty = provider default)
    web_search_max_attempts: int = 3  # Env: WEB_SEARCH_MAX_ATTEMPTS
    web_search_delay_min: float = 0.2  # Env: WEB_SEARCH_DELAY_MIN
    web_search_delay_max: float = 0.6  # Env: WEB_SEARCH_DELAY_MAX

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def validate_delay_range(self) -> "ToolLimitsSettings":
        """Ensure web_search_delay_min <= web_search_delay_max."""
        if self.web_search_delay_min > self.web_search_delay_max:
            raise ValueError(
                f"WEB_SEARCH_DELAY_MIN ({self.web_search_delay_min}) must be <= "
                f"WEB_SEARCH_DELAY_MAX ({self.web_search_delay_max})"
            )
        return self


class ToolSandboxSettings(BaseSettings):
    """Runtime ``code_executor`` isolation (Phase 2c — opt-in, default off).

    The ``code_executor`` builtin normally runs untrusted one-off LLM code in a
    HOST subprocess (network on, no mem cap, full host FS) — the T2-high
    sandbox-bypass gap from findings-03. When ``code_executor_mode='docker'``,
    the tool instead routes the code through ``SandboxExecutor`` docker mode:
    network disabled, container rootfs read-only, a memory cap, and the agent's
    results dir mounted read-write so ``results/<file>`` deliverables still
    persist. If Docker is unavailable it falls back to the host subprocess (with
    a WARNING) so a run never hard-fails on a missing daemon.

    Distinct from ``EvolutionSettings``: that sandbox vets ALREADY-statically-
    validated handler code and runs host BY DESIGN (``execute_code_subprocess``);
    this gates UNVALIDATED runtime one-off code, so it defaults OFF until an
    operator opts in. The ``self-evolving-agent-toolbox`` image mirrors
    ``src/tools/dynamic/allowlist.py`` so an import of an allowlisted third-party
    dep resolves in the container too.
    """

    code_executor_mode: Literal["subprocess", "docker", "runner"] = "subprocess"
    code_executor_sandbox_image: str = "self-evolving-agent-toolbox:latest"
    code_executor_sandbox_memory_mb: int = 512
    # Timeout for the docker code run. Defaults higher than the host
    # ``code_executor_timeout`` because a cold container start adds latency.
    code_executor_sandbox_timeout: int = 60
    # Host path mounted read-write at ``code_executor_sandbox_workdir_dest``
    # inside the container so a script's ``results/<file>`` writes persist. Empty
    # → resolves to ``results_root()`` at call time (the agent's RESULTS_ROOT).
    code_executor_results_mount: str = ""
    # Container path where the results mount lands. Paired with
    # ``working_dir=/workspace`` so a relative ``results/foo.md`` resolves to the
    # mounted host results dir — the same contract the host subprocess honors.
    code_executor_sandbox_workdir_dest: str = "/workspace/results"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# ─── Runner Settings ───────────────────────────────────────────────


class RunnerSettings(BaseSettings):
    """Remote no-DinD code-execution runner (Phase 3b/c — opt-in via mode).

    The runner is a dedicated container that is the agent's SINGLE sink for
    executing generated code — both the ``code_executor`` builtin (unvalidated
    one-off LLM scripts) and the evolution sandbox (already-SafetyPipeline-
    validated handler code). It runs a tiny HTTP server that executes submitted
    Python as a constrained subprocess IN ITS OWN container: no Docker socket
    (so no Docker-in-Docker), no DATABASE/REDIS/search credentials, and (in
    compose) no internet egress (it sits on an ``internal: true`` network).
    The disposable container itself is the isolation boundary — per-invocation
    gVisor/Kata hardening is a Phase-5 doc item, not this phase.

    The worker reads ``runner_url`` to reach it; when the configured sandbox
    mode (``EVOLUTION_SANDBOX_MODE`` / ``CODE_EXECUTOR_MODE``) is ``runner``,
    ``SandboxExecutor`` and ``code_executor`` POST code here instead of calling
    ``docker.from_env()``, so the worker needs NO Docker access at all and the
    worker compose service drops its ``/var/run/docker.sock`` mount.

    See :mod:`src.sandbox.runner_client` (this side) and
    :mod:`src.sandbox.runner_server` (the runner container).
    """

    # In-compose default; the runner service listens on 8090 internally. Host /
    # CLI runs that opt into runner mode set RUNNER_URL explicitly.
    runner_url: str = "http://runner:8090"
    # Bound on the TCP handshake only — a down runner fails fast so callers
    # fall back without paying the full request timeout.
    runner_connect_timeout_s: float = 5.0
    # Hard cap on a single requested execution timeout so a runaway caller
    # can't ask the runner to hold a subprocess open for an hour.
    runner_max_timeout_s: int = 300

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# ─── Budget Settings ────────────────────────────────────────────────


class BudgetSettings(BaseSettings):
    """Token budget and cost control configuration."""

    daily_token_budget: int = 500000
    # Per-run token cap (cumulative across the run_id — ``CostTracker`` sums
    # ``cost_ledger.total_tokens``). Raised 200K→500K: a complex multi-deliverable
    # run with recompute/verification probes (battery q07/q08/q09 / complex-arxiv
    # style) routinely spends 250-300K of real work before converging. At the old
    # 200-300K cap, such a run hit BudgetExhaustedError JUST after writing its
    # deliverables (complex-arxiv-stats-4 died at 313K, ~22s after the last
    # write); the gateway died mid-pass, and (pre-A2) the node-level ``except
    # Exception`` degraded to heuristics that could never complete because the
    # never-cleared ``errors`` accumulator (operator.add) kept the heuristic gate
    # blocked — looping to the iteration cap. Now A2 makes a hard-stop terminate
    # cleanly at the worker (BUDGET_EXHAUSTED, resumable), and 500K gives the
    # LLM-path enough room to converge on a fresh attempt before the cap fires.
    # The downgrade path (``gateway.py``) still engages when budget_hard_stop is
    # OFF. Env: PER_TASK_TOKEN_LIMIT.
    per_task_token_limit: int = 500000
    max_cost_usd: float = 10.0
    budget_warn_threshold: float = 0.70
    budget_critical_threshold: float = 0.90
    # Opt-in budget HARD-stop (battery-04 q09 fix D). Default OFF: when the
    # per-run token cap is reached the gateway downgrades to a cheaper fallback
    # (current behavior — a cheaper model almost always exists). When ON, the
    # gateway instead RAISES ``BudgetExhaustedError`` so the run stops cleanly:
    # the worker marks ``BUDGET_EXHAUSTED`` (resumable — the checkpoint
    # persists, so ``main.py --resume <run_id>`` picks up later). This prevents
    # the failure mode where downgrade cascaded onto a free-tier provider
    # (429s) and the degraded run FABRICATED a deliverable. KNOWN LIMITATION:
    # ``get_run_token_usage`` is cumulative, so a resumed run re-trips the cap
    # immediately — a resume-window delta (baseline the spent count at resume)
    # is a deferred follow-up. Env: BUDGET_HARD_STOP.
    budget_hard_stop: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("budget_warn_threshold", "budget_critical_threshold")
    @classmethod
    def validate_thresholds(cls, v: float) -> float:
        """Ensure thresholds are between 0 and 1."""
        if not 0.0 < v <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1. Got: {v}")
        return v

    @model_validator(mode="after")
    def validate_critical_gt_warn(self) -> "BudgetSettings":
        """Ensure critical threshold is greater than warn threshold."""
        if self.budget_critical_threshold <= self.budget_warn_threshold:
            raise ValueError(
                f"Critical threshold ({self.budget_critical_threshold}) must be "
                f"greater than warn threshold ({self.budget_warn_threshold})"
            )
        return self


# ─── Evolution Settings ─────────────────────────────────────────────


class EvolutionSettings(BaseSettings):
    """Self-evolution system configuration."""

    evolution_enabled: bool = True  # aligns with .env.example (Env: EVOLUTION_ENABLED)
    evolution_interval: int = 10  # Every N tasks
    evolution_max_mutations: int = 5
    evolution_sandbox_timeout: int = 30  # Seconds
    evolution_require_human_approval: bool = True
    evolution_sandbox_memory_mb: int = 256
    evolution_sandbox_image: str = "python:3.12-slim"
    evolution_sandbox_mode: Literal["docker", "subprocess", "runner"] = "docker"
    evolution_shadow_repo_path: str = ".turing/evolution-repo"
    evolution_source_dir: str = "src"
    # Max regeneration attempts after a validation failure (0 = single attempt,
    # no retry). NOT routed through validate_positive_int so 0 stays legal.
    max_evolution_retries: int = 3
    # Code-emitting mutation LLM params (previously hardcoded in
    # src/evolution/templates.py): sampling temperature and the fraction of the
    # model's max output tokens the regeneration loop may consume.
    evolution_temperature: float = 0.4  # Env: EVOLUTION_TEMPERATURE
    evolution_max_tokens_factor: float = 0.9  # Env: EVOLUTION_MAX_TOKENS_FACTOR
    # Sandbox subprocess timeouts (previously hardcoded in
    # src/sandbox/executor.py asyncio.wait_for): venv creation + package install.
    sandbox_venv_create_timeout: int = 60  # Env: SANDBOX_VENV_CREATE_TIMEOUT
    sandbox_package_install_timeout: int = 120  # Env: SANDBOX_PACKAGE_INSTALL_TIMEOUT
    # Phase 8 — evolution→live promotion gate (opt-in). When true, a deployed
    # PROMPT mutation that passes the eval canary (score >= eval_canary_min_score)
    # is written as a versioned artifact under ``evolved_handlers_dir`` and a
    # ``current`` pointer is updated so the live agent loads it via the prompt
    # builder (tagged [evolved]). CODE/TOOL mutations already reach live via the
    # DB tool registry; this gate is scoped to PROMPT mutations. Off by default
    # so nothing is promoted until the operator opts in. Env: EVOLUTION_PROMOTE_TO_LIVE.
    evolution_promote_to_live: bool = False
    # Wall-clock budget (seconds) for the INLINE promotion canary
    # (``GoldenCanary.score`` → ``BenchmarkHarness.run_benchmark``), per battery
    # goal. When promotion is on, the engine scores a deployed PROMPT mutation by
    # running a full golden goal (default ``battery04_q01``) SYNCHRONOUSLY inside
    # the live run's evolve node, ON THE LIVE GATEWAY. A non-converging goal
    # (e.g. q01's tz-shift) otherwise blocks ``run_cycle`` → ``evolve`` → the
    # live run until the worker wall-clock kills it (observed live: a 30-min
    # held-hostage run, adhoc-eval-proof-1, 2026-06-27), AND bills its LLM cost
    # to the live run_id. The budget bounds that: a goal that cannot score
    # within it is abandoned (no score recorded → no promotion) and the live run
    # proceeds. The canary's gateway run_id is also scoped to its own ``bench-``
    # thread_id (harness.run_benchmark) so its cost never attributes to the live
    # run. ``<= 0`` disables the bound — an escape hatch for controlled OFFLINE
    # benchmark contexts only, NOT for live worker runs. Env:
    # PROMOTION_CANARY_TIMEOUT_S.
    promotion_canary_timeout_s: float = 180.0
    # Directory holding promoted, versioned handler artifacts (prompts first).
    # Layout: ``<dir>/prompts/<node>.<sha>.json`` (immutable versions) +
    # ``<dir>/prompts/current.json`` (the live pointer manifest the builder reads).
    # Default lives under .turing/ (gitignored scratch), NOT in core src/. Env:
    # EVOLVED_HANDLERS_DIR.
    evolved_handlers_dir: str = ".turing/evolved"
    # Phase 5 G2 — VCS-tracked promotion. The runtime ``evolved_handlers_dir``
    # above is gitignored scratch (pointer + immutable versions live there for
    # the builder to read live). This SECOND dir mirrors only the IMMUTABLE
    # per-version artifact (``<dir>/<node>.<sha>.json``) into the VCS-tracked
    # tree (repo-root ``prompts/evolved/``) so a promotion lands in git — the
    # runtime ``current.json`` pointer is intentionally NOT committed (mutable
    # runtime state → noisy churn). Env: EVOLUTION_TRACKED_PROMPTS_DIR.
    evolution_tracked_prompts_dir: str = "prompts/evolved"
    # Phase 5 G2 — auto-commit the tracked artifact to the main project repo on
    # promotion (explicit-path ``git add``, never ``-A``; local-only, NEVER push).
    # Autonomous git-history writes are sensitive, so this is a SEPARATE opt-in
    # on top of EVOLUTION_PROMOTE_TO_LIVE: when off the tracked FILE is still
    # written (operator commits it on review), when on the gate commits it.
    # Best-effort + non-fatal: a failed commit (e.g. the worker container has no
    # ``.git``) never blocks promotion — the live pointer is the source of truth.
    # Env: EVOLUTION_PROMOTE_TO_VCS.
    evolution_promote_to_vcs: bool = False
    # Phase 2 C1 — curve regression-guard on PROMPT promotion (opt-in, default
    # off). When true, the engine's Phase-8 promotion path reads the nightly
    # capability-curve verdict (``CapabilityCurve.detect_regression``) BEFORE
    # calling ``promotion_gate.promote`` and SKIPS the promotion when the battery
    # curve is regressed — so a mutation does not go live during a known capability
    # regression window. Only ``regressed`` blocks: ``inconclusive`` (too few
    # nightly battery points — the cold-start case) is allowed through, since the
    # single-goal ``GoldenCanary`` still gates each mutation and starving a fresh
    # deploy of all promotion until the battery has run enough nights would be
    # counter-productive. Fail-open: a curve-read error never blocks promotion
    # (the curve is observability-only; the canary is authoritative). Off by
    # default ⇒ byte-identical behavior until toggled. Env:
    # EVOLUTION_REQUIRE_CURVE_CLEAR.
    evolution_require_curve_clear: bool = False
    # Phase 4 E — evolve→execute edge for deployed TOOL mutations (opt-in,
    # default off). When true, after a TOOL mutation deploys (it has already
    # passed the engine's safety + sandbox + post-deploy smoke gates), the
    # evolve node live-registers its handler in the ToolRegistry and signals
    # ``route_after_evolve`` to run ONE execute pass so the new tool is
    # reachable in-run (closes the q3 reliability gap where a deployed TOOL
    # mutation landed in the shadow repo but the run ended before it executed).
    # Fail-closed: any materialization/safety hiccup skips re-execution. Bounded
    # to a single pass per run by ``AgentState.evolve_reexecute_done``. The gate
    # is TOOL-specific: PROMPT/CODE mutations and config-JSON TOOL templates
    # (target_path not ending in ``.py``) never re-execute. Env:
    # EVOLUTION_REEXECUTE_TOOL.
    evolution_reexecute_tool: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator(
        "evolution_interval",
        "evolution_max_mutations",
        "evolution_sandbox_timeout",
        "evolution_sandbox_memory_mb",
        "sandbox_venv_create_timeout",
        "sandbox_package_install_timeout",
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive integers."""
        if v < 1:
            raise ValueError(f"Must be a positive integer. Got: {v}")
        return v

    @field_validator("evolution_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure evolution sampling temperature is a sane range (0–2)."""
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"Temperature must be between 0 and 2. Got: {v}")
        return v

    @field_validator("evolution_max_tokens_factor")
    @classmethod
    def validate_factor(cls, v: float) -> float:
        """Ensure max-tokens factor is a fraction of the model cap (0–1]."""
        if not 0.0 < v <= 1.0:
            raise ValueError(f"Factor must be between 0 and 1. Got: {v}")
        return v


# ─── Agent Settings ─────────────────────────────────────────────────


class AgentSettings(BaseSettings):
    """Core agent execution limits and safety controls."""

    max_iterations: int = 60
    # Complexity-aware runtime iteration cap (B1). The routers and error handler
    # terminate a run at ``effective_max_iterations(state)`` (src/graph/
    # iteration_cap.py), which falls back to the tier below when no explicit cap
    # is pinned. A TRIVIAL/SIMPLE goal stops loop-hunting early; COMPLEX/CRITICAL
    # keep full headroom. An explicit per-run cap (CLI ``--max-iterations``, an
    # eval spec, or the worker schema → ``state["max_iterations"]``) ALWAYS wins.
    # NOTE: ``max_iterations`` above stays the recursion_limit BASIS (runner.py
    # computes ``recursion_limit = max(max_iterations*8, 100)`` at graph-build
    # time, before complexity is classified, so it cannot be complexity-aware).
    # The @model_validator below guards the invariant
    # ``max_iterations >= max(max_iterations_complex, max_iterations_critical)``.
    # Env: MAX_ITERATIONS_TRIVIAL/SIMPLE/COMPLEX/CRITICAL.
    max_iterations_trivial: int = 12
    max_iterations_simple: int = 15
    max_iterations_complex: int = 60
    max_iterations_critical: int = 60
    # Convergence early-exit (B3). When ``verify`` emits an identical output
    # fingerprint across this many consecutive passes AND the plan is
    # exhausted, ``route_after_verify`` accepts the partial result via
    # ``store_memory`` instead of looping to the iteration hard-cap. The run
    # is "stably stuck" — repeating the same result with no forward progress.
    # Default 3: a transient repeat on one verify is common (re-execution of
    # an unchanged step); 3 consecutive unchanged passes is a real plateau.
    # Env: CONVERGENCE_STABLE_THRESHOLD.
    convergence_stable_threshold: int = 3
    # Capability-cap gap-loop break (battery-04 q09 fix B). When the active
    # tool/sub-agent population is at its cumulative cap (max_active_tools /
    # max_active_sub_agents), ``agent_spawn`` converts agent-gaps→tool-gaps and
    # ``tool_create`` skips registration — so a run whose plan still calls for
    # missing capabilities ping-pongs spawn↔create with NO forward progress
    # until the iteration hard-cap (the q09 non-terminating loop). The routers
    # count CONSECUTIVE cap-blocks (a spawn/create round that produced no new
    # capability); once this many accumulate, they stop re-routing into
    # spawn/create and route to plan/verify instead, forcing convergence. Reset
    # to 0 on any real progress (a capability IS created). Default 2 = ONE
    # fully-saturated cycle: the counter climbs +1 per cap-blocked spawn/create
    # node, so a single spawn-blocked-then-create-blocked round reaches 2 with
    # zero progress (real progress resets it). 3 needs 1.5 cycles, which a slow
    # model (deepseek-v4-flash, ~2 min/generation) never reaches inside a
    # worker wall-clock timeout — live-validated on q09 (counter climbed 0→1→2
    # and stalled before a 600s timeout fired). ON by default — correctness fix.
    # Env: CAP_LOOP_BREAK_THRESHOLD.
    cap_loop_break_threshold: int = 2
    # Run caps — single source of truth for tool/sub-agent creation limits.
    # Enforcement sites (tool generator, agent_spawn, structure_analysis) read
    # these fields directly; there are NO module-level MAX_*_PER_RUN constants
    # (they were dead/stale and have been removed).
    # Env: MAX_TOOLS_PER_RUN, MAX_SUB_AGENTS_PER_RUN.
    max_tools_per_run: int = 12
    max_sub_agents_per_run: int = 5
    # Tool-handler code generation: route to a code-strong model instead of the
    # CHEAP tier (complexity=SIMPLE → Haiku), which truncates non-trivial
    # handlers so they fail the AST gate and never persist (battery-02 N5's
    # duplicate_finder). deepseek-v4-pro is a strong, low-cost coder with a full
    # FALLBACK_CHAINS entry; alt gpt-4.1-mini-2025-04-14. Empty string → fall
    # back to the legacy complexity=SIMPLE path. Env: TOOL_GENERATION_MODEL.
    tool_generation_model: str = "deepseek-v4-pro"
    # Semantic capability dedup (B3): before creating a tool/sub-agent the agent
    # embeds the capability gap and reuses an existing capability whose
    # cosine similarity to the gap is >= this threshold, instead of spawning a
    # duplicate. Conservative (0.85) to avoid false reuse; distinct from the
    # consolidation redundancy cutoff (capability_redundancy_threshold, M3).
    # Env: CAPABILITY_DEDUP_THRESHOLD.
    capability_dedup_threshold: float = 0.85
    # Cumulative capability caps + retirement (B3 de-bloat). Distinct from the
    # per-run caps above (max_tools_per_run / max_sub_agents_per_run), which
    # bound creation within ONE run. These bound the TOTAL active population and
    # trigger retirement when a capability is a chronic low performer
    # (success_rate < retire_success_floor OR empty_output_rate >=
    # retire_empty_output_floor, over >= retire_min_runs), stale (unused for
    # retire_recency_days), or redundant (cosine >=
    # capability_redundancy_threshold with a better-scoring twin). Enforced on
    # load by SubAgentPersister.load_active_agents / ToolPersister.load_active_tools
    # when settings is passed. Env: MAX_ACTIVE_TOOLS, MAX_ACTIVE_SUB_AGENTS,
    # CAPABILITY_REDUNDANCY_THRESHOLD, RETIRE_MIN_RUNS, RETIRE_SUCCESS_FLOOR,
    # RETIRE_EMPTY_OUTPUT_FLOOR, RETIRE_RECENCY_DAYS.
    max_active_tools: int = 25  # every active tool loads into the registry/selection prompt → keep tight
    # Sub-agents are delegated-to selectively (not all injected into every prompt),
    # so a higher stored cap is safe and preserves a richer specialist ecosystem.
    max_active_sub_agents: int = 60
    capability_redundancy_threshold: float = 0.92
    retire_min_runs: int = 20
    retire_success_floor: float = 0.5
    # Phase 4 G — retire tools that chronically return BLANK output even when
    # they "succeed" (success_rate can look healthy while the tool is useless).
    # A generated tool is retired when calls >= retire_min_runs AND
    # (success_rate < retire_success_floor OR empty_output_rate >=
    # retire_empty_output_floor). Env: RETIRE_SUCCESS_FLOOR,
    # RETIRE_EMPTY_OUTPUT_FLOOR.
    retire_empty_output_floor: float = 0.8
    retire_recency_days: int = 30
    # Phase-4 gap fix — retire objective dead weight (0-call tools). Distinct from
    # the performance path above: retire_underperforming INTENTIONALLY spares
    # untried tools (a tool is never retired for performance before a fair
    # chance), so a generated tool with calls == 0 that is neither redundant, nor
    # over the cumulative cap, nor has enough calls to underperform would survive
    # FOREVER — the nightly GovernancePruner was a no-op on exactly this cruft
    # (battery-04 q07: 8 of 25 slots were never-invoked dead weight). This knob
    # retires active generated tools with calls == 0 whose created_at age exceeds
    # ``retire_unused_days`` (the age gate is the safety — a freshly-spawned tool
    # no run has picked YET is NOT retired; "never used" must be durable). The
    # periodic GovernancePruner runs it; ``<= 0`` disables (no retirement). Env:
    # RETIRE_UNUSED_DAYS. Default 30 mirrors retire_recency_days (the existing
    # "stale" bar) so it is effective whenever GOVERNANCE_PRUNE_ENABLED is on.
    retire_unused_days: int = 30
    # Per-tool success-metrics recording (M4). When true, the execute chokepoint
    # records each tool invocation (success/empty/latency) to tool_call_metrics
    # and updates the running aggregates on tool_registrations, which the
    # performance-retirement path above (retire_min_runs/retire_success_floor)
    # scores. Gated so a DB hiccup in the recorder never breaks a run.
    # Env: TOOL_METRICS_ENABLED.
    tool_metrics_enabled: bool = True
    # Tool retrieval-before-selection (findings-05; default OFF). When true,
    # execute/plan no longer inject EVERY active tool into the prompt — instead
    # they keep the built-in tools (always) plus the top-k dynamically-created
    # tools whose capability embeddings are semantically nearest the current
    # goal/step. Reuses the existing capability_embedding index for RECALL (not
    # just dedup): the MAX_ACTIVE_TOOLS=25 cap is a symptom of injecting all;
    # retrieval bounds the prompt to what's relevant. Falls back to the full
    # set when disabled, when the embedding provider is unavailable (hash
    # fallback is not semantically meaningful), or on any retrieval error — so
    # behavior is unchanged until toggled on and never starves the run.
    # Built-ins are always included because they are not in the embedding index
    # (only tool_create/agent_spawn persist embeddings). Env:
    # TOOL_RETRIEVAL_ENABLED, TOOL_RETRIEVAL_TOP_K.
    tool_retrieval_enabled: bool = False
    tool_retrieval_top_k: int = 8
    # E2 — success-metric score-blend. When on, tool retrieval widens to a pool
    # (top_k × ``tool_retrieval_blend_pool``) by cosine, then RE-RANKS by
    # ``cosine · (1 + blend_weight · success_rate · (1 − empty_output_rate))``
    # before taking the top_k — so a reliable, slightly-less-similar tool can
    # outrank a flaky near-match. Defaults reproduce pure-cosine ranking until
    # toggled on; untested tools (no metrics) default to success_rate=1.0 so a
    # cold-start tool is never starved. Env: TOOL_RETRIEVAL_BLEND_*.
    tool_retrieval_blend_success: bool = False
    tool_retrieval_blend_pool: int = 2
    tool_retrieval_blend_weight: float = 0.5
    # F1 — semantic sub-agent selection before the delegation fan-out. When on,
    # the SPAWNED agents are ranked by their stored capability embedding against
    # the subtask and only the top ``agent_selection_top_k`` actually run — the
    # rest are deselected (membership decided by ranking; survivors keep spawn
    # order so tier-grouping / provider-spread is preserved). ``agent_spawn``
    # already decided membership; this prunes the fan-out. Reuses the
    # sub_agent_definitions.capability_embedding index (RECALL — before this,
    # ``find_similar`` was its only consumer and it gated dedup, never ranked
    # recall). Defaults preserve the all-spawn fan-out until toggled on. Env:
    # AGENT_SELECTION_*.
    agent_selection_enabled: bool = False
    agent_selection_top_k: int = 3
    # F3 — destructive-tool human-in-the-loop gate. When on, the execute node
    # routes any tool flagged ``destructiveHint=True`` (terminal_command,
    # http_request, index_corpus) through a LangGraph ``interrupt()`` approval
    # checkpoint before invoking it; approval resumes, rejection/no-human-blocks
    # returns a blocked ToolResult (safe default — the tool does NOT run). Tools
    # whose blast radius is already bounded (file_writer path-confined,
    # code_executor sandboxed in the no-DinD runner) are intentionally NOT
    # flagged destructive, so they never gate. Default off ⇒ no behavior change
    # for any tool. Env: DESTRUCTIVE_TOOL_HITL_ENABLED.
    destructive_tool_hitl_enabled: bool = False
    # D2: opt-in gateway multimodal/vision. When on, a caller may pass ``images``
    # to ``LLMGateway.acompletion``; the gateway folds them into the last user
    # message as OpenAI-format content blocks (text + image_url) and restricts
    # the fallback chain to image-capable models (ModelSpec.supports_images).
    # Default off ⇒ behavior is byte-identical to text-only (no message
    # mutation, no chain filtering). Env: VISION_ENABLED.
    vision_enabled: bool = False
    context_window_reserve: float = 0.15  # 15% margin
    hitl_enabled: bool = True
    workspace_root: str = ".turing/workspace"
    results_root: str = "results"
    # Phase 7: per-run results subfolders. When true AND a run_id is active
    # (set by main.py via _paths.set_active_run_id), deliverables written
    # through the shared resolver land under ``results_root / <run_id> / ...``
    # so each run is isolated on disk. Reads (verify/file_reader/eval checks)
    # fall back to the flat root when the run subdir has nothing — so recall
    # of older flat deliverables (battery-03) still works. No-op when no
    # run_id is set (non-run-id runs behave exactly as before). Env: RESULTS_PER_RUN_SUBDIR.
    results_per_run_subdir: bool = True

    # D8: host-subprocess confinement for code_executor (subprocess mode only).
    # When the host path guard is ON, the bootstrap-injected ``open`` wrapper
    # rejects any path resolving OUTSIDE the results/workspace tree — checked
    # AFTER relocation, so a script reads/writes its own deliverables + fixtures
    # but cannot reach repo-root secrets (``open(".env")``) or system files
    # (``open("/etc/hosts")``). Default OFF (behavior unchanged): docker/runner
    # modes are already confined; this hardens the host fallback only. Env:
    # CODE_EXECUTOR_HOST_PATH_GUARD.
    code_executor_host_path_guard: bool = False
    # D8: the host-subprocess working directory. ``project_root`` (default) is
    # the cwd every other file-touching tool shares, so ``glob('results/*.md')``
    # resolves uniformly — KEEP this default (``results_subdir`` double-nests a
    # ``results/<file>`` write and breaks that contract). ``results_subdir``
    # opts into cwd = the per-run results subfolder (``results_root`` fallback)
    # for tighter disk isolation where a run's scripts use bare names. Env:
    # CODE_EXECUTOR_HOST_CWD.
    code_executor_host_cwd: Literal["project_root", "results_subdir"] = "project_root"

    # Memory folding (autonomous context compression)
    memory_folding_enabled: bool = True
    # Cooldown between folds. Tuned to the default max_iterations (~18-25): an
    # interval of 10 couldn't fit a 2nd fold before the cap (needs iter 20).
    # At 6, folds are feasible at ~iter 6/12/18 (capped at max_folds=3).
    memory_folding_interval: int = 6
    memory_folding_token_threshold: int = 50_000
    memory_folding_max_folds: int = 3
    # Primary trigger: fold once the conversation reaches this many messages.
    memory_folding_message_floor: int = 10
    memory_folding_message_threshold: int = 14
    # Tertiary context-size trigger (chars // 4 estimate).
    memory_folding_message_token_estimate: int = 8_000

    # Hot-memory recall: how many recent observations retrieve_context surfaces
    # from the recency-ranked ZSET (newest-first). The legacy hot.search() path
    # had no ordering guarantee and hard-capped at 2; this knob makes the count
    # explicit and deterministic. Env: MEMORY_HOT_RECALL_SIZE
    memory_hot_recall_size: int = 3

    # Planning: cap generated plan length to the iteration budget so large
    # multi-unit goals decompose within max_iterations instead of blowing the
    # run budget (the binding constraint — money budget is secondary). 30 lets
    # complex multi-unit goals (e.g. "document 12 patterns") decompose fully;
    # plan_node still clamps to the remaining iteration budget via
    # min(planning_max_steps, remaining), so this never overshoots max_iterations.
    planning_max_steps: int = 30
    # Proactive structure analysis: detect tool-creation / parallel sub-agent
    # intent from the goal before the execute loop and seed the spawn nodes.
    structure_analysis_enabled: bool = True
    # E3: opt-in LLM-assist refinement. When on AND the goal is COMPLEX/CRITICAL
    # AND the deterministic regex pass found nothing, a one-shot gateway call
    # (glm-4.7 under the configured routing) infers capability gaps the static
    # patterns miss. Fail-safe: any LLM/parse error leaves the regex result
    # unchanged. Default off → behavior is byte-identical to regex-only.
    structure_analysis_llm_assist_enabled: bool = False  # Env: STRUCTURE_ANALYSIS_LLM_ASSIST_ENABLED

    # ── Ambiguity-resolution cascade (Feature B; all default-off) ─────────
    # Master switch: when off, route_after_classify always returns "plan" and
    # the topology is byte-identical to today. When on, an ambiguous goal
    # (severity >= threshold) routes classify -> disambiguate -> plan, running
    # an LLM-self-resolve -> web-grounding -> re-score cascade that carries the
    # resolution forward as ADVISORY planner context (literal goal unchanged).
    clarifying_gate_enabled: bool = False  # Env: CLARIFYING_GATE_ENABLED
    clarifying_severity_threshold: float = 0.5  # Env: CLARIFYING_SEVERITY_THRESHOLD
    # Web grounding runs within the gate when a query is emitted (default on).
    clarifying_web_grounding_enabled: bool = True  # Env: CLARIFYING_WEB_GROUNDING_ENABLED
    clarifying_max_queries: int = 3  # Env: CLARIFYING_MAX_QUERIES  (cap on grounding queries)
    # HITL is the last resort of the cascade. NOTE: there is no Command(resume=)
    # resume surface in the worker/CLI today, so when enabled the HITL step
    # degrades to advisory (carries notes forward) rather than stalling the run.
    clarifying_hitl_threshold: float = 0.7  # Env: CLARIFYING_HITL_THRESHOLD
    clarifying_hitl_enabled: bool = False  # Env: CLARIFYING_HITL_ENABLED
    # Feature C: per-step atomicity. plan_quality is ALWAYS computed + attached
    # as advisory telemetry. When this gate is on, a too_coarse step (>=2
    # conjunctions) triggers ONE bounded heuristic conjunction-split (guarded
    # by atomicity_replan_done so a reflect->plan loop can't re-split forever).
    # Pure heuristic — zero LLM cost. Default off: plans pass through unchanged.
    plan_atomicity_enforce: bool = False  # Env: PLAN_ATOMICITY_ENFORCE

    # Feature E: persist the classify node's refined_intent (Feature A) as a
    # durable semantic fact (memory_type="fact") so later runs recall the real
    # desired outcome behind a recurring goal. Best-effort + non-fatal (a store
    # hiccup never aborts the terminal sink). Default off.
    persist_intent_facts: bool = False  # Env: PERSIST_INTENT_FACTS

    # Concurrency + loop bounds (previously module constants in
    # src/graph/nodes/execute.py and src/graph/nodes/tool_create.py, and the
    # verify data-tool cap in src/graph/nodes/verify.py).
    # Max tools executed in parallel within a single execute step.
    max_concurrent_tools: int = 5  # Env: MAX_CONCURRENT_TOOLS
    # Extra LLM turns spent nudging the model to actually write deliverables.
    max_write_nudges: int = 2  # Env: MAX_WRITE_NUDGES
    # Max tool-handler regeneration attempts after a validation failure.
    tool_gen_max_attempts: int = 3  # Env: TOOL_GEN_MAX_ATTEMPTS
    # Cap on data-bearing tools inspected by the verify node.
    verify_max_data_tools: int = 8  # Env: VERIFY_MAX_DATA_TOOLS
    # Memory-folding LLM params (previously hardcoded in src/memory/folding.py).
    memory_folding_temperature: float = 0.1  # Env: MEMORY_FOLDING_TEMPERATURE
    memory_folding_max_tokens: int = 2048  # Env: MEMORY_FOLDING_MAX_TOKENS
    # Phase 5: semantic/fact memory tier. When a memory fold is persisted, the
    # episode summary is mined for durable facts (entity-ish knowledge) via the
    # gateway and stored as warm memory_type="fact" — de-conflicted from the
    # episodic cold tier and recalled alongside skills/folded memory. Extraction
    # is best-effort (a gateway failure yields no facts, never an error).
    memory_fact_extraction_enabled: bool = True  # Env: MEMORY_FACT_EXTRACTION_ENABLED
    memory_fact_max_per_fold: int = 5  # Env: MEMORY_FACT_MAX_PER_FOLD
    # F2: bounded retry count for framework-mandated tool DB-persistence. A
    # validated tool must reach the DB (cross-run recall) even if the first
    # write hits a transient connection error — each attempt opens a fresh
    # session, so a poisoned session recovers on the next call (CostTracker-
    # resilience pattern). Read at call-time in _persist_tool.
    tool_persist_max_attempts: int = 2  # Env: TOOL_PERSIST_MAX_ATTEMPTS

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator(
        "max_iterations",
        "max_tools_per_run",
        "max_sub_agents_per_run",
        "planning_max_steps",
        "max_concurrent_tools",
        "max_write_nudges",
        "tool_gen_max_attempts",
        "verify_max_data_tools",
        "memory_folding_max_tokens",
        "tool_persist_max_attempts",
        "clarifying_max_queries",
        "memory_hot_recall_size",
        "max_iterations_trivial",
        "max_iterations_simple",
        "max_iterations_complex",
        "max_iterations_critical",
        "convergence_stable_threshold",
        "cap_loop_break_threshold",
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive integers."""
        if v < 1:
            raise ValueError(f"Must be a positive integer. Got: {v}")
        return v

    @model_validator(mode="after")
    def validate_iteration_cap_basis(self) -> AgentSettings:
        """The recursion_limit basis must cover every tier cap (B1 invariant).

        ``runner.py`` computes ``recursion_limit = max(max_iterations*8, 100)``
        at graph-build time using ``max_iterations``. If a complexity tier cap
        exceeded it, a run on that tier would hit ``GraphRecursionError`` before
        its tier cap — silently trapping COMPLEX/CRITICAL runs. Require the
        basis to be >= the largest tier cap.
        """
        tier_max = max(
            self.max_iterations_complex,
            self.max_iterations_critical,
        )
        if self.max_iterations < tier_max:
            raise ValueError(
                f"max_iterations ({self.max_iterations}) must be >= "
                f"max(max_iterations_complex, max_iterations_critical) "
                f"({tier_max}) — it is the recursion_limit basis."
            )
        return self

    @field_validator("clarifying_severity_threshold", "clarifying_hitl_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure an ambiguity/HITL threshold is a probability in [0.0, 1.0]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0. Got: {v}")
        return v

    @field_validator("memory_folding_temperature")
    @classmethod
    def validate_folding_temperature(cls, v: float) -> float:
        """Ensure folding temperature is a sane sampling range (0–2)."""
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"Folding temperature must be between 0 and 2. Got: {v}")
        return v

    @field_validator("context_window_reserve")
    @classmethod
    def validate_reserve(cls, v: float) -> float:
        """Ensure reserve is between 0 and 1."""
        if not 0.0 <= v < 1.0:
            raise ValueError(f"Context reserve must be between 0 and 1. Got: {v}")
        return v


# ─── Logging Settings ──────────────────────────────────────────────


class LoggingSettings(BaseSettings):
    """Structured logging configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["structured", "text"] = "structured"
    log_dir: str = "./logs"
    log_rotation: str = "00:00"  # Daily at midnight
    log_retention: str = "30 days"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# ─── Observability Settings ────────────────────────────────────────


class ObservabilitySettings(BaseSettings):
    """OpenTelemetry tracing and Prometheus metrics."""

    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4318"
    otel_service_name: str = "turing-agent"
    otel_sampling_rate: float = 0.10

    prometheus_enabled: bool = False
    prometheus_port: int = 9090

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("otel_sampling_rate")
    @classmethod
    def validate_sampling_rate(cls, v: float) -> float:
        """Ensure sampling rate is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Sampling rate must be between 0 and 1. Got: {v}")
        return v


# ─── LangSmith Settings ─────────────────────────────────────────────


class LangSmithSettings(BaseSettings):
    """LangSmith tracing configuration for LangGraph and litellm."""

    langchain_tracing_v2: bool = False
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "turing-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Conventional env vars are uppercase (LANGCHAIN_TRACING_V2, LANGSMITH_API_KEY).
        # case_sensitive=True would silently fail to populate the lowercase fields, so
        # main.py would force LANGCHAIN_TRACING_V2=false and tracing would never engage.
        case_sensitive=False,
    )

    @property
    def is_configured(self) -> bool:
        """Check if LangSmith tracing is fully configured."""
        return self.langchain_tracing_v2 and bool(
            self.langsmith_api_key and self.langsmith_api_key.strip()
        )


class EvalSettings(BaseSettings):
    """Evaluation harness configuration (Phase 3 correctness layer).

    Eval is opt-in: ``eval_enabled`` gates the verify-node correctness checks.
    A run with no registered battery GoalSpec still gets a generic ad-hoc
    deliverable eval row (parsed + non-empty per on-disk deliverable) when
    ``eval_adhoc_deliverables`` is on — pure observability, never enforced.
    ``eval_enforce`` (default False) makes a failing correctness check downgrade
    a "complete" verdict to incomplete so the agent retries; when False the
    score is recorded and a grounding warning emitted without changing the
    verdict. The LLM-judge and the persistent eval store have their own toggles.
    """

    eval_enabled: bool = True  # aligns with .env.example (Env: EVAL_ENABLED)
    eval_enforce: bool = True  # aligns with .env.example (Env: EVAL_ENFORCE — completion gating)
    eval_llm_judge_enabled: bool = True  # Env: EVAL_LLM_JUDGE_ENABLED
    eval_canary_min_score: float = 0.8  # Env: EVAL_CANARY_MIN_SCORE
    eval_store_enabled: bool = True  # Env: EVAL_STORE_ENABLED
    # F-e: run correctness checks the first time deliverables appear on disk on
    # an INCOMPLETE verify (not just at is_complete), folding the failing checks'
    # reasons into the re-plan as advisory feedback. Bounded to one rescue per
    # run (state.eval_rescue_attempted) so the LLM-judge fires at most ~once
    # extra. Never forces completion — the iteration hard-cap still self-completes.
    eval_rescue_incomplete: bool = True  # Env: EVAL_RESCUE_INCOMPLETE
    # Ad-hoc deliverable eval: a run with no battery GoalSpec still records a
    # generic structural eval_results row (parse + non-empty per on-disk
    # deliverable) so fresh/unspecced queries are evaluated too. Observability-
    # only — never enforces (a parse hiccup must not loop a real run; verify's
    # own completion gate already enforces well-formedness). Default on: zero
    # LLM cost, pure observability; disable to skip eval entirely for ad-hoc runs.
    eval_adhoc_deliverables: bool = True  # Env: EVAL_ADHOC_DELIVERABLES

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# ─── Search Settings (Phase 1: SearXNG + Meilisearch corpus stack) ────


class SearchSettings(BaseSettings):
    """Web-search + corpus-search configuration (Phase 1 overhaul).

    The search stack runs SearXNG as the primary keyless live-search service and
    Meilisearch as the corpus keyword/hybrid index. Paid providers are an
    automatic fallback behind SearXNG; heavy providers (Firecrawl/Apify) are
    provisioned-only (explicit ``deep_crawl`` opt-in). Each service listens on
    its canonical port INSIDE the container (SearXNG 8080, Meilisearch 7700);
    ``docker-compose.yml`` maps them to non-default HOST ports (8081 / 7701) so
    this stack never clashes with SearXNG/Meilisearch containers other projects
    may run on the defaults. These host-run defaults therefore point at
    ``localhost:8081`` / ``localhost:7701``; the compose ``agent`` service
    overrides them to the in-container hostnames (``http://searxng:8080`` /
    ``http://meilisearch:7700``).
    """

    # Primary live-search service (SearXNG, keyless/self-hosted).
    search_primary: str = "searxng"  # Env: SEARCH_PRIMARY
    searxng_url: str = "http://localhost:8081"  # Env: SEARXNG_URL
    searxng_timeout: float = 10.0  # Env: SEARXNG_TIMEOUT
    searxng_max_results_per_query: int = 10  # Env: SEARXNG_MAX_RESULTS_PER_QUERY

    # Corpus index service (Meilisearch, BM25 + hybrid).
    meilisearch_url: str = "http://localhost:7701"  # Env: MEILISEARCH_URL
    meilisearch_key: str = ""  # Env: MEILISEARCH_KEY (master key; empty disables auth in dev)
    meilisearch_index: str = "turing_corpus"  # Env: MEILISEARCH_INDEX
    meilisearch_timeout: float = 10.0  # Env: MEILISEARCH_TIMEOUT
    # Indexing is async in Meilisearch: POST /documents returns a taskUid whose
    # task runs in the background. _meili_add_documents polls GET /tasks/{uid}
    # until terminal (succeeded/failed) so a search in the same coroutine sees
    # the docs — otherwise it races the still-enqueued task. Bounded so a stuck
    # task can't hang the index leg; on exhaustion it proceeds (search may lag).
    meilisearch_task_poll_interval: float = 0.1  # Env: MEILISEARCH_TASK_POLL_INTERVAL (s)
    meilisearch_task_max_polls: int = 30  # Env: MEILISEARCH_TASK_MAX_POLLS (~3s cap @0.1s)

    # Lightweight paid providers tried in order when SearXNG fails/throttles.
    # Each is only attempted if its API key is set. Heavy providers are excluded
    # from this chain (provisioned-only, see deep_crawl_enabled).
    search_fallback_providers: str = "tavily,serper,brave,serpapi,serpstack,llmlayer"  # Env: SEARCH_FALLBACK_PROVIDERS

    # Tavily adapter tuning (S12). Tavily returns a relevance ``score`` per hit
    # (0..1); these shape the request and the result filter. Defaults keep the
    # cheap/broad path: search_depth=basic, topic=general, no domain filter,
    # no score floor. search_depth=advanced pulls higher-quality answers but
    # costs more credits; topic=news activates recency (then ``days`` applies).
    # include/exclude_domains are comma-separated hostnames (e.g. "arxiv.org").
    tavily_search_depth: str = "basic"  # Env: TAVILY_SEARCH_DEPTH (basic|advanced)
    tavily_topic: str = "general"  # Env: TAVILY_TOPIC (general|news)
    tavily_days: int = 3  # Env: TAVILY_DAYS (days back; only used when topic=news)
    tavily_include_domains: str = ""  # Env: TAVILY_INCLUDE_DOMAINS (comma-sep hostnames)
    tavily_exclude_domains: str = ""  # Env: TAVILY_EXCLUDE_DOMAINS (comma-sep hostnames)
    tavily_min_score: float = 0.0  # Env: TAVILY_MIN_SCORE (drop hits with score below this)

    # Batch/parallel fan-out control (web_search(queries) + corpus_search(queries)).
    search_batch_concurrency: int = 5  # Env: SEARCH_BATCH_CONCURRENCY

    # AI-format extraction / chunking (web_scraper -> corpus index).
    chunk_size: int = 1200  # Env: CHUNK_SIZE (target chars per chunk)
    chunk_overlap: int = 150  # Env: CHUNK_OVERLAP

    # Heavy providers (Firecrawl/Apify): provisioned in .env but OFF by default.
    # Set True to allow the deep_crawl tool flag to engage them.
    deep_crawl_enabled: bool = False  # Env: DEEP_CRAWL_ENABLED

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def fallback_providers(self) -> list[str]:
        """Ordered fallback provider list (whitespace-trimmed, lowercased)."""
        return [p.strip().lower() for p in self.search_fallback_providers.split(",") if p.strip()]


class WorkerSettings(BaseSettings):
    """Redis-Streams queue seam configuration (Phase 2b overhaul).

    The API role enqueues run requests to a Redis Stream; worker processes
    consume them via a consumer group. This decouples request ingestion
    (stateless, scales horizontally) from run execution (heavy, one at a time
    per worker). At-least-once delivery: a worker XACKs only after the run's
    checkpoint is durable, so a mid-run crash leaves the message in the group's
    pending-entries list for another worker to reclaim (``reclaim_min_idle_ms``
    gates how long a stuck entry must idle before ``XAUTOCLAIM`` reassigns it).
    """

    # Stream + consumer-group names. The stream is created on first XADD (or via
    # MKSTREAM on group creation). Env: WORKER_RUNS_STREAM / WORKER_GROUP.
    runs_stream: str = "turing:runs"  # Env: WORKER_RUNS_STREAM
    group: str = "turing-workers"  # Env: WORKER_GROUP

    # This consumer's name within the group (for PEL ownership / XAUTOCLAIM).
    # EMPTY by default → src/worker/__main__._resolve_consumer_name derives a
    # per-process-unique ``worker-{hostname}-{pid}`` so replica workers never
    # share a pending-entries list (XAUTOCLAIM stays sound). Set an explicit
    # value ONLY for a single, fixed-name worker; a fixed default here would make
    # EVERY replica collide on it (the bug this default shipped for years).
    # Env: WORKER_CONSUMER_NAME.
    consumer_name: str = ""  # Env: WORKER_CONSUMER_NAME

    # How many messages to pull per XREADGROUP sweep.
    read_batch_size: int = 5  # Env: WORKER_READ_BATCH_SIZE

    # XREADGROUP block timeout (ms): how long to wait for new messages before
    # looping back to reclaim-stale. 0 = non-blocking. Env: WORKER_BLOCK_MS.
    block_ms: int = 5000  # Env: WORKER_BLOCK_MS

    # Min idle time (ms) before a pending entry is eligible for XAUTOCLAIM by
    # another consumer (crash recovery). Env: WORKER_RECLAIM_MIN_IDLE_MS.
    reclaim_min_idle_ms: int = 30000  # Env: WORKER_RECLAIM_MIN_IDLE_MS

    # TTL (s) on per-run status hashes (``turing:run:{run_id}``) so the status
    # store self-cleans instead of growing unbounded. Env: WORKER_STATUS_TTL_S.
    status_ttl_s: int = 86400  # Env: WORKER_STATUS_TTL_S

    # Dead-letter cap: after this many failed delivery attempts a run is XACKed +
    # marked FAILED permanently (NOT redelivered). Guards against infinite
    # poison-message redelivery — a DETERMINISTIC executor failure (e.g. a missing
    # dep crashing graph build) would otherwise be re-handed to a worker every
    # ``reclaim_min_idle_ms`` forever (Bug B). Transient failures still retry up to
    # this many times before dead-lettering. Env: WORKER_DEAD_LETTER_MAX_ATTEMPTS.
    dead_letter_max_attempts: int = 3  # Env: WORKER_DEAD_LETTER_MAX_ATTEMPTS

    # TTL (s) on the per-run lease lock (Bug C — double-claim).
    # ``reclaim_min_idle_ms`` (XAUTOCLAIM) is shorter than a normal run, so a peer
    # worker's reclaim would otherwise steal a still-healthy in-flight entry and
    # process the SAME goal a second time concurrently. The lease lock is SET-NX
    # per run_id before any work; the holder renews it every ttl/3 while the run
    # is live and releases it on completion. A crash lets it expire, after which
    # ``reclaim_stale`` soundly hands the run to a peer (no double-processing).
    # Must exceed the longest expected run; renewal keeps it alive. A value of 0
    # disables the lease (legacy behavior) — do NOT do that in a multi-worker pool.
    # Env: WORKER_LOCK_TTL_S.
    lock_ttl_s: int = 120  # Env: WORKER_LOCK_TTL_S

    # Run-level wall-clock timeout (battery-04 q09 fix A). Default 0 = OFF
    # (current behavior — a run loops until its iteration cap or convergence).
    # When > 0, the worker wraps ``execute_run`` in ``asyncio.timeout(...)``:
    # on expiry the run is marked ``TIMEOUT`` + XACKed (terminal — NOT
    # redelivered, since re-running would just hit the same wall) and the
    # checkpoint persists so it is resumable via ``main.py --resume <run_id>``.
    # Bounds the non-terminating failure mode where saturated capability caps
    # (or a stuck tool) looped past the iteration cap's protection. A per-run
    # override (``RunRequest.run_timeout_s`` → ``RunJob``) wins over this
    # default; 0/None at both levels = no timeout. Env: WORKER_RUN_TIMEOUT_S.
    run_timeout_s: float = 0.0  # Env: WORKER_RUN_TIMEOUT_S

    model_config = SettingsConfigDict(
        # env_prefix makes the documented WORKER_* vars (see .env.example) map to
        # these fields: WORKER_CONSUMER_NAME → consumer_name, etc. Without it, the
        # field names map to bare CONSUMER_NAME/RUNS_STREAM/… and every WORKER_*
        # var in .env.example is silently ignored (Phase-3 fix of a Phase-2b bug).
        env_prefix="worker_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class SchedulerSettings(BaseSettings):
    """Nightly capability-curve battery scheduler (#197 Phase 5).

    There is no time-triggered run path in the deployed stack: runs enter the
    ``turing:runs`` stream only via the API HTTP endpoint or the host CLI
    ``--eval``. The capability curve (correctness score over time) therefore only
    updates when a human runs ``--eval``. This scheduler is the missing feeder:
    a sidecar (``python -m src.scheduler``) that, on a cron schedule, enqueues
    every ``BATTERY04_GOALS`` spec as a ``RunJob`` into ``turing:runs`` via the
    SAME ``RunsQueue.enqueue`` seam the API uses — so the worker (not the host
    CLI) runs them through the real deployed stack and the eval layer populates
    ``eval_results`` autonomously. Each run gets a date-suffixed ``run_id``
    (``battery04_q01-20260622``) so nightly runs are isolated under
    ``results/<spec>-<date>/`` while the resolver
    (``runner._resolve_eval_spec_id``) strips the suffix to score against the
    spec — keeping the curve per-attempt via ``eval_attempt_id``.

    Opt-in: ``enabled`` defaults False so a host-run ``python -m src.scheduler``
    with no env is a clean no-op (exit 0). The compose ``scheduler`` service is
    profile-gated AND forces ``SCHEDULER_ENABLED=true`` (mirroring the anchor's
    forced-topology pattern), so bringing the profile up is the explicit opt-in.
    """

    enabled: bool = False  # Env: SCHEDULER_ENABLED — opt-in; compose forces true.
    # 5-field crontab consumed by APScheduler ``CronTrigger.from_crontab``.
    # Default: nightly 02:00 UTC. Env: SCHEDULER_CRON.
    cron: str = "0 2 * * *"  # Env: SCHEDULER_CRON
    # IANA zone for the cron schedule. Env: SCHEDULER_TIMEZONE.
    timezone: str = "UTC"  # Env: SCHEDULER_TIMEZONE
    # Optional pinned model for every battery run (registry key or litellm id).
    # Empty → each run uses its complexity-tiered default. Env: SCHEDULER_MODEL.
    model: str = ""  # Env: SCHEDULER_MODEL
    # Battery runs SHOULD exercise evolution (the richest learning signal);
    # default False (do NOT skip). Env: SCHEDULER_NO_EVOLUTION.
    no_evolution: bool = False  # Env: SCHEDULER_NO_EVOLUTION
    # ``strftime`` format appended to each spec id as ``-<suffix>`` for per-night
    # results isolation. The default ``%Y%m%d`` → ``-20260622``; the resolver
    # strips a trailing ``-YYYYMMDD`` (8 digits). Changing this to a format with
    # embedded hyphens (e.g. ``%Y-%m-%d``) would break that strip — keep it
    # compact. Env: SCHEDULER_DATE_SUFFIX_FORMAT.
    date_suffix_format: str = "%Y%m%d"  # Env: SCHEDULER_DATE_SUFFIX_FORMAT
    # Smoke / partial-curve cap: 0 = enqueue EVERY spec (the production nightly —
    # the full capability curve); >0 = enqueue only the first N (e.g. 1 for a
    # cheap one-spec plumbing smoke that does not run the full $1.50 battery).
    # Env: SCHEDULER_SPEC_LIMIT.
    spec_limit: int = 0  # Env: SCHEDULER_SPEC_LIMIT

    model_config = SettingsConfigDict(
        # env_prefix maps the documented SCHEDULER_* vars to these fields:
        # SCHEDULER_ENABLED → enabled, SCHEDULER_CRON → cron, etc. Mirrors the
        # WorkerSettings pattern (a Phase-2b bug shipped for years because the
        # prefix was missing) — declared explicitly so these vars are honored.
        env_prefix="scheduler_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class CapabilityCurveSettings(BaseSettings):
    """Battery capability-curve trend + regression→rollback gate (Phase 2 C1).

    The agent's central claim is that it self-improves. The nightly battery
    (``src/scheduler``) already *collects* per-check correctness scores over
    time in ``eval_results``; the promotion gate already rolls back a PROMPT
    mutation whose *single-goal canary* regressed at promotion time. What was
    missing: a *temporal* regression gate that watches the battery **trend**
    across nights and reverts a promotion whose benefit did not hold. This
    settings group drives that analytics + gate layer
    (``src/eval/curve.py`` + ``src/evolution/curve_gate.py``).

    Regression definition (the conjunction prevents noise from ever firing a
    rollback): ``current < score_floor`` AND ``(best_prior - current) >=
    regression_delta`` AND >= ``min_points`` nights observed. Detection is
    always-on and read-only (cheap); **auto-rollback is opt-in** (default
    False) matching the codebase's promotion/capabilities default-off
    convention. When ``auto_rollback`` is off the gate still sets the
    ``capability_curve_score`` gauge, increments ``capability_curve_regressions_total``,
    records telemetry, and logs a WARNING — the safety evidence without the
    risk. ``curve_cron`` defaults to 05:00 UTC (after the 02:00 battery) so the
    gate reads the just-written night.
    """

    # Register the nightly curve-gate job at all. Opt-in like the scheduler:
    # default False so a host run with no env does nothing.
    gate_enabled: bool = False  # Env: CAPABILITY_CURVE_GATE_ENABLED
    # 5-field crontab for the gate. Default 05:00 UTC (after the 02:00 battery).
    curve_cron: str = "0 5 * * *"  # Env: CAPABILITY_CURVE_CURVE_CRON
    # IANA zone for the gate cron. Env: CAPABILITY_CURVE_TIMEZONE.
    timezone: str = "UTC"  # Env: CAPABILITY_CURVE_TIMEZONE
    # Minimum drop (best_prior - current) to call a regression. Env:
    # CAPABILITY_CURVE_REGRESSION_DELTA. With score_floor this is the AND of
    # the floor+delta conjunction — a delta-only dip below a held floor is NOT
    # a regression (the noise guard).
    regression_delta: float = 0.1  # Env: CAPABILITY_CURVE_REGRESSION_DELTA
    # Absolute score floor: current must be BELOW this to be a regression at
    # all (a curve holding above 0.5 is healthy even if it wiggles). Env:
    # CAPABILITY_CURVE_SCORE_FLOOR.
    score_floor: float = 0.5  # Env: CAPABILITY_CURVE_SCORE_FLOOR
    # Only promotions made within this many days are suspect rollback targets
    # (a regression is most plausibly the most-recent active promotion). Env:
    # CAPABILITY_CURVE_LOOKBACK_DAYS.
    lookback_days: int = 30  # Env: CAPABILITY_CURVE_LOOKBACK_DAYS
    # Minimum nights of battery history before a regression can be declared
    # (too few points is inconclusive, never a rollback). Env:
    # CAPABILITY_CURVE_MIN_POINTS.
    min_points: int = 2  # Env: CAPABILITY_CURVE_MIN_POINTS
    # When True, a confirmed regression reverts the suspect recent PROMPT
    # promotion via ``PromotionGate.rollback``. Default False (detect + log +
    # telemetry only) — the human opts into automatic reversion. Env:
    # CAPABILITY_CURVE_AUTO_ROLLBACK.
    auto_rollback: bool = False  # Env: CAPABILITY_CURVE_AUTO_ROLLBACK

    model_config = SettingsConfigDict(
        # env_prefix maps CAPABILITY_CURVE_* vars to these fields, mirroring the
        # Scheduler/Worker pattern (a missing prefix silently ignores vars).
        env_prefix="capability_curve_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class GovernancePruneSettings(BaseSettings):
    """Opt-in periodic capability-governance prune (battery-04 q09 fix C).

    Governance (``src/governance/consolidate.py``) already runs at LOAD time
    when a worker boots (semantic dedup, cumulative-cap retirement, redundancy
    + performance retirement) — but a long-lived worker accumulates the active
    tool/sub-agent population across runs and never prunes until restart. q09
    saturated its caps (25/25 tools, 60/60 sub-agents) mid-life and then could
    never create a needed capability, looping spawn↔create with no progress.

    This registers a periodic job that re-runs the existing retire/redundancy
    passes on a cron, lowering active counts to free cap headroom WITHOUT
    raising the caps themselves. It reuses ``AgentSettings`` retire knobs
    (``retire_min_runs`` / ``retire_success_floor`` / ``retire_recency_days`` /
    ``retire_empty_output_floor`` / ``capability_redundancy_threshold``) plus the
    Phase-4 ``retire_unused_days`` dead-weight pass (0-call tools aged past the
    gate) — no other retirement thresholds here.
    Opt-in (default False) like the scheduler/curve-gate, so a host run with no
    env does nothing. ``cron`` defaults to 04:00 UTC (before the 02:00 battery
    shifts if the operator wants a clean slate overnight).
    """

    # Register the periodic prune job at all. Default False.
    enabled: bool = False  # Env: GOVERNANCE_PRUNE_ENABLED
    # 5-field crontab for the prune. Default 04:00 UTC.
    cron: str = "0 4 * * *"  # Env: GOVERNANCE_PRUNE_CRON
    # IANA zone for the prune cron. Env: GOVERNANCE_PRUNE_TIMEZONE.
    timezone: str = "UTC"  # Env: GOVERNANCE_PRUNE_TIMEZONE

    model_config = SettingsConfigDict(
        # env_prefix maps GOVERNANCE_PRUNE_* vars to these fields, mirroring the
        # Scheduler/Worker/CapabilityCurve pattern (a missing prefix silently
        # ignores vars — bug #223 class).
        env_prefix="governance_prune_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class OptimizerSettings(BaseSettings):
    """Metric-driven prompt-optimization sidecar (Phase 2 C2: DSPy + GEPA).

    Turns the golden-eval correctness harness (``GoldenCanary``) into an
    AUTOMATIC prompt-improvement loop beyond the engine's one-shot LLM PROMPT
    mutation (``engine.py`` ``_llm_generate``). Runs in its OWN container
    (``src/optimizer``) so the heavy ML deps stay out of the slim api/worker
    images: DSPy + GEPA only, **no torch** (DSPy declares no torch dependency;
    ``import dspy`` is torch-free — verified against dspy==3.2.1). The sidecar
    imports ``src/``, runs its own ``LLMGateway`` against the SHARED
    ``cost_ledger`` DB (so cost/budget/circuit-breaker work natively), and is
    driven by a nightly scheduler job (NOT per-run) to bound spend.

    Architecture (forced by GEPA's real API, verified against dspy==3.2.1):
    GEPA's ``metric`` is called as ``metric(example, pred, trace, pred_name,
    pred_trace) -> float | {score, feedback}``; GEPA applies each candidate
    instruction to the DSPy *student module's predictor* and runs the student
    itself (one LLM call) — it never hands the candidate instruction back to
    caller code. So the eval canary (which scores the FULL agent graph) cannot
    be GEPA's in-loop metric. Instead GEPA searches candidate instructions
    against a CHEAP PROXY metric over a DSPy module (bounds cost; supplies
    GEPA's reflective-feedback loop); the optimized instruction is then
    VALIDATED against the real ``GoldenCanary`` correctness score (full agent
    runs) before promotion through the EXISTING ``PromotionGate`` (canary-gated,
    auto-rollback). The eval metric stays the promotion gate — so the DoD
    ("prompts improve against the eval metric, automatically") is satisfied;
    GEPA supplies the search. The proxy→canary transfer risk is backstopped by
    C1 (``capability_curve``): a promoted prompt that later regresses the
    battery trend is detected + (optionally) rolled back.

    Precondition: only safe once C1's regression gate is proven
    (``require_curve_clear``). Every knob defaults off; the optimizer is
    observability/evolution-only — it never raises into a user run.
    """

    # Register the optimizer at all. Default False — a host run with no env
    # does nothing; the compose ``optimizer`` service (profile-gated) forces it.
    enabled: bool = False  # Env: OPTIMIZER_ENABLED
    # Search backend. ``dspy-gepa`` (default, reflective) / ``dspy-mipro``
    # (few-shot bootstrap) / ``dspy-copro`` (coordinate). All three are real
    # DSPy teleprompters sharing the same student+trainset+metric; ``textgrad``
    # (torch) is deferred — requested via the /optimize body, not here. Env:
    # OPTIMIZER_BACKEND.
    backend: Literal["dspy-gepa", "dspy-mipro", "dspy-copro"] = "dspy-gepa"  # noqa: E501
    # The graph node whose system prompt is optimized. v1 ships a ``classify``
    # profile (cleanest DSPy signature — goal→complexity, exact-match metric);
    # ``execute``/``verify`` profiles are pluggable but un-shipped in v1 (a
    # clear ConfigurationError, NOT a stub). Env: OPTIMIZER_TARGET_NODE.
    target_node: str = "classify"  # Env: OPTIMIZER_TARGET_NODE
    # How many golden specs the FINAL canary validates against (cheapest = 1).
    # GEPA's proxy loop is bounded separately by max_candidates/max_trials.
    # Env: OPTIMIZER_EVAL_SPEC_LIMIT.
    eval_spec_limit: int = 2  # Env: OPTIMIZER_EVAL_SPEC_LIMIT
    # MIPROv2/COPRO candidate-search breadth (num_candidates); unused by GEPA,
    # whose budget is max_trials. Env: OPTIMIZER_MAX_CANDIDATES.
    max_candidates: int = 8  # Env: OPTIMIZER_MAX_CANDIDATES
    # GEPA full-eval rounds → ``max_full_evals`` (×(trainset+valset) metric
    # calls). 0 lets GEPA pick via ``auto="light"``. Env: OPTIMIZER_MAX_TRIALS.
    max_trials: int = 1  # Env: OPTIMIZER_MAX_TRIALS
    # LM call params for the DSPy student module AND the reflection/proposal LM.
    max_tokens: int = 1024  # Env: OPTIMIZER_MAX_TOKENS
    temperature: float = 0.7  # Env: OPTIMIZER_TEMPERATURE
    # Reflection/proposal model for the teleprompter search — GEPA's
    # ``reflection_lm`` (REQUIRED — its probe raises "requires a reflection
    # language model" without one); MIPROv2's & COPRO's ``prompt_model``. All
    # three benefit from a STRONGER model than the cheap student. Empty → route a
    # COMPLEX-tier model via ModelRouter (genuinely stronger than the SIMPLE
    # student, e.g. glm-4.7; NOT anthropic-blocked). A literal model id pins one.
    # Env: OPTIMIZER_REFLECTION_MODEL.
    reflection_model: str = ""  # Env: OPTIMIZER_REFLECTION_MODEL
    # Nightly trigger. Default 03:30 UTC — between the 02:00 battery and the
    # 05:00 curve-gate, so the optimizer runs on a fresh night then the gate
    # re-reads the trend next morning. Env: OPTIMIZER_CRON.
    cron: str = "30 3 * * *"  # Env: OPTIMIZER_CRON
    # IANA zone for the cron. Env: OPTIMIZER_TIMEZONE.
    timezone: str = "UTC"  # Env: OPTIMIZER_TIMEZONE
    # The SIDECAR's own aiohttp bind (server side). Internal-only on turing-net
    # (no host port is published). The optimizer service reads these; the
    # scheduler does NOT — it connects via ``optimizer_url`` below.
    # Env: OPTIMIZER_HOST/PORT.
    host: str = "0.0.0.0"  # Env: OPTIMIZER_HOST
    port: int = 8095  # Env: OPTIMIZER_PORT
    # Client-side connect URL the SCHEDULER POSTs /optimize to (mirrors
    # ``RunnerSettings.runner_url``): the compose service DNS name ``optimizer``
    # + the bind port. The scheduler service runs on a minimal env (NOT
    # *agent-common), so it sets OPTIMIZER_URL explicitly; host/CLI runs that
    # drive the sidecar another way override it. The node/backend/eval knobs are
    # the SIDECAR's concern — the scheduler POSTs an empty body and those
    # defaults apply. Env: OPTIMIZER_URL.
    optimizer_url: str = "http://optimizer:8095"  # Env: OPTIMIZER_URL
    # C1 precondition: refuse to optimize while the capability curve shows a
    # regression OR is inconclusive (too few nights to judge). False lets an
    # operator override on a cold curve (accepting the proxy-transfer risk).
    # Env: OPTIMIZER_REQUIRE_CURVE_CLEAR.
    require_curve_clear: bool = True  # Env: OPTIMIZER_REQUIRE_CURVE_CLEAR
    # Min canary-score MARGIN (candidate − baseline) required to promote. None
    # → reuse ``EvalSettings.eval_canary_min_score`` as an absolute floor.
    # Env: OPTIMIZER_CANARY_MIN_SCORE.
    canary_min_score: Optional[float] = None  # Env: OPTIMIZER_CANARY_MIN_SCORE
    # Hard spend cap (US$) for one optimization run. Queried BEFORE compile();
    # over-cap → skip with no spend. Spend lands in cost_ledger under run_id
    # ``optimizer-<node>-<ts>`` (its own cap, not a user run's). Env:
    # OPTIMIZER_MAX_COST_USD.
    max_cost_usd: float = 0.50  # Env: OPTIMIZER_MAX_COST_USD

    model_config = SettingsConfigDict(
        # env_prefix maps OPTIMIZER_* vars to these fields, mirroring the
        # CapabilityCurve/GovernancePrune pattern (a missing prefix silently
        # ignores vars — bug #223 class).
        env_prefix="optimizer_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("eval_spec_limit", "max_candidates", "max_tokens")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive integers (a 0-budget optimizer is a no-op misconfig)."""
        if v < 1:
            raise ValueError(f"Must be a positive integer. Got: {v}")
        return v

    @field_validator("max_cost_usd")
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """Ensure a non-zero spend cap (uncapped optimization is forbidden)."""
        if v <= 0:
            raise ValueError(f"Must be positive. Got: {v}")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure LM sampling temperature is a sane range (0–2)."""
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"Temperature must be between 0 and 2. Got: {v}")
        return v

    @field_validator("canary_min_score", mode="before")
    @classmethod
    def _coerce_optional_margin(cls, v: object) -> float | None:
        """Tolerate a blank or comment-placeholder value as ``None``.

        ``.env`` templates leave an optional knob as
        ``OPTIMIZER_CANARY_MIN_SCORE=   # comment``; pydantic-settings reads the
        trailing ``# comment`` as the value for an otherwise-empty field, which
        then fails ``float`` coercion at ``Settings`` construction (crashing
        every importer of ``get_settings()``). ``mode="before"`` lets us map any
        unparseable value to ``None`` — the documented default meaning "no
        explicit margin -> reuse ``EVAL_CANARY_MIN_SCORE`` as the floor". A
        genuinely numeric value still validates normally.
        """
        if v is None:
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if isinstance(v, str):
            s = v.strip().lstrip("#").strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                return None
        return None


# ─── Root Settings ──────────────────────────────────────────────────


class AgentCronSettings(BaseSettings):
    """Agent-settable durable cron (Phase 5 I1).

    Lets the agent schedule its OWN future work via the
    ``create_scheduled_task`` builtin: a durable ``scheduled_tasks`` row the
    scheduler consumer (``src.scheduler.cron_consumer``) fires into the
    ``turing:runs`` stream on a cron. Default-off — the tool rejects writes and
    the consumer registers nothing until ``AGENT_CRON_ENABLED`` is set, so a
    host run is byte-identical to pre-I1 behavior (mirrors the battery /
    curve-gate / optimizer opt-in convention).

    The cap is the cron-abuse guard: an agent that loops creating tasks can
    never exceed ``max_tasks`` enabled rows. The goal-size bound keeps the
    stored row + the enqueued ``RunJob.goal`` bounded.
    """

    # Master opt-in. Default False so the tool no-ops and the daemon registers
    # nothing on a clean host run. Env: AGENT_CRON_ENABLED.
    enabled: bool = False  # Env: AGENT_CRON_ENABLED
    # Hard cap on the number of ENABLED scheduled_tasks rows. A defensive bound
    # against an agent that loops creating tasks (the cron-abuse guard). A
    # re-call with an existing name UPSERTs (does not consume a new slot), so
    # this bounds DISTINCT enabled tasks, not call count. Env: AGENT_CRON_MAX_TASKS.
    max_tasks: int = 25  # Env: AGENT_CRON_MAX_TASKS
    # Max goal length (chars) the tool accepts — bounds the stored row + the
    # enqueued RunJob.goal. Env: AGENT_CRON_MAX_GOAL_CHARS.
    max_goal_chars: int = 2000  # Env: AGENT_CRON_MAX_GOAL_CHARS
    # How often (seconds) the consumer reconciles scheduled_tasks rows ↔
    # APScheduler jobs (adds new, drops disabled/deleted, refreshes edits). A
    # short interval keeps agent-authored edits responsive; the per-task fire
    # is still gated by the task's own cron. Env: AGENT_CRON_SYNC_INTERVAL_S.
    sync_interval_s: int = 60  # Env: AGENT_CRON_SYNC_INTERVAL_S
    # Timezone for the reconcile IntervalTrigger + the daemon. (Per-task cron
    # triggers read each row's own ``timezone`` column, defaulting to UTC.) Mirrors
    # the ``timezone`` field on every other scheduler-adjacent settings class.
    # Env: AGENT_CRON_TIMEZONE.
    timezone: str = "UTC"  # Env: AGENT_CRON_TIMEZONE

    model_config = SettingsConfigDict(
        # env_prefix maps AGENT_CRON_* vars to these fields, mirroring the
        # Scheduler/Worker pattern (a missing prefix silently ignores vars).
        env_prefix="agent_cron_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class GitCloneSettings(BaseSettings):
    """Git-clone code indexer (Phase 5 I2).

    Lets the agent ``git_clone`` a PUBLIC repo into a confined workspace subdir
    and index its source into pgvector cold memory (``episode_type="code"``) so a
    later ``code_search`` recalls symbols by semantic similarity — the agent's
    "codebase memory" for repos it has read. Default-off: when disabled the
    handler is a no-op that tells the caller, so a host run is byte-identical to
    pre-I2 behavior (mirrors the corpus / agent-cron opt-in convention).

    The caps are the abuse/cost guardrail: a maliciously-large or pathologically-
    structured repo cannot exhaust disk (``max_files``/``max_file_bytes``/
    ``max_total_bytes``) or burn unbounded embedding cost (``max_chunks`` bounds
    the gateway calls, one per symbol). ``clone_timeout_s`` bounds a hanging
    remote so a node never wedges.
    """

    # Master opt-in. Default False so the tool no-ops on a clean host run. Env:
    # GIT_CLONE_ENABLED.
    enabled: bool = False  # Env: GIT_CLONE_ENABLED
    # Hard ceiling on the number of code files walked per clone. Bounds disk +
    # embed cost against a sprawling repo. Env: GIT_CLONE_MAX_FILES.
    max_files: int = 200  # Env: GIT_CLONE_MAX_FILES
    # A single file larger than this is skipped (never chunked/embedded). Guards
    # against a giant generated/minified blob swamping the index. Env:
    # GIT_CLONE_MAX_FILE_BYTES.
    max_file_bytes: int = 262144  # 256 KiB. Env: GIT_CLONE_MAX_FILE_BYTES
    # The walk stops once the CUMULATIVE file size exceeds this. Bounds the
    # whole-clone footprint + total embedding tokens. Env:
    # GIT_CLONE_MAX_TOTAL_BYTES.
    max_total_bytes: int = 20971520  # 20 MiB. Env: GIT_CLONE_MAX_TOTAL_BYTES
    # Hard ceiling on the number of chunks embedded + stored (one ColdMemory row
    # per symbol). Bounds gateway embedding calls. Env: GIT_CLONE_MAX_CHUNKS.
    max_chunks: int = 300  # Env: GIT_CLONE_MAX_CHUNKS
    # Wall-clock seconds the ``git clone`` subprocess may run before it is killed.
    # Bounds a slow/hanging remote so a node never wedges. Env:
    # GIT_CLONE_CLONE_TIMEOUT_S.
    clone_timeout_s: int = 120  # Env: GIT_CLONE_CLONE_TIMEOUT_S

    model_config = SettingsConfigDict(
        # env_prefix maps GIT_CLONE_* vars to these fields, mirroring the
        # AgentCron/Worker pattern (a missing prefix silently ignores vars).
        env_prefix="git_clone_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class LatsSettings(BaseSettings):
    """LATS / MCTS tree-search execution primitive (Phase 5 G3a).

    A reasoning-space MCTS that, on a stalled CRITICAL decision, explores
    alternative next-steps via gateway-only rollouts + an LLM value function
    (UCB1 lookahead), then commits the best branch for the single-trajectory
    ``execute`` to run. Default-off: when disabled ``route_after_reflect`` never
    routes to ``lats_search`` and the topology is byte-identical (mirrors the
    AgentCron/GitClone opt-in convention). Fail-safe: any error ⇒ the incumbent
    step runs unchanged.

    The caps are the cost guardrail — every CRITICAL retry is bounded by
    ``max_evaluations`` value calls (+ ``max_expansions`` one expand call +
    rollouts when ``rollout_depth > 0``), and the gateway's global budget
    hard-stop is the backstop.
    """

    # Master opt-in. Default False so a host run never engages LATS. Env:
    # LATS_ENABLED.
    enabled: bool = False  # Env: LATS_ENABLED
    # Number of DISTINCT alternative next-steps the expand call proposes (plus
    # the incumbent "stay-the-course" child). Bounds branching. Env:
    # LATS_MAX_EXPANSIONS.
    max_expansions: int = 3  # Env: LATS_MAX_EXPANSIONS
    # Per-child imagined-rollout depth (gateway-only, no tool calls). 0 = skip
    # rollouts (flat 1-ply value selection). Env: LATS_ROLLOUT_DEPTH.
    rollout_depth: int = 1  # Env: LATS_ROLLOUT_DEPTH
    # Hard ceiling on value-function calls across the whole per-call tree.
    # Bounds cost per CRITICAL retry. Env: LATS_MAX_EVALUATIONS.
    max_evaluations: int = 8  # Env: LATS_MAX_EVALUATIONS
    # Max MCTS search depth (plies). v1 builds a shallow frontier; this caps any
    # deeper expansion. Env: LATS_MAX_DEPTH.
    max_depth: int = 2  # Env: LATS_MAX_DEPTH
    # UCB1 exploration constant c (sqrt(2) ≈ 1.41). Env: LATS_EXPLORATION.
    exploration: float = 1.41  # Env: LATS_EXPLORATION
    # Engagement scope. "stall" (default) = only on CRITICAL + LOW/VERY_LOW
    # confidence (single-trajectory has stalled). "always" = any CRITICAL
    # decision. Env: LATS_SCOPE.
    scope: str = "stall"  # Env: LATS_SCOPE

    model_config = SettingsConfigDict(
        # env_prefix maps LATS_* vars to these fields, mirroring the
        # AgentCron/GitClone pattern (a missing prefix silently ignores vars).
        env_prefix="lats_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class AflowSettings(BaseSettings):
    """AFlow / ADAS workflow-topology optimization (Phase 5 G3b).

    An OFFLINE optimizer that searches which prompting TECHNIQUES get wired into
    each graph node, in what order, PER TASK CATEGORY — the one rewirable
    topology surface (``TechniqueSelector``), distinct from C2's single-node
    prompt-text optimizer and the mutation engine. Learns a per-(node, category)
    override via baseline→propose→evaluate→keep-if-better over an injected
    fitness ``run_fn`` (full agent runs reading correctness), persisted through
    an ``AflowPolicyStore`` pointer (mirrors ``PromotionGate``). Default-off:
    when disabled the builder hook short-circuits before any pointer read and
    ``optimize()`` is a no-op — selection is byte-identical (mirrors the LATS
    opt-in convention). The optimizer never raises (structured ``AflowResult``).

    The caps are the cost guardrail — fitness = real agent runs, bounded by
    ``max_candidates`` (× seed count) per (node, category); ``max_cost_usd`` is
    an advisory mid-search cap the CLI wires to the ledger, and the per-run
    budget hard-stop inside ``execute_run`` is the backstop.
    """

    # Master opt-in. Default False so a host run never installs an AFlow policy
    # (selection byte-identical). Env: AFLOW_ENABLED.
    enabled: bool = False  # Env: AFLOW_ENABLED
    # Comma-separated target nodes to optimize (e.g. "execute" or
    # "plan,execute"). Env: AFLOW_TARGET_NODES.
    target_nodes: str = "execute"  # Env: AFLOW_TARGET_NODES
    # Max DISTINCT candidate policies proposed per (node, category) (the dominant
    # cost lever — each is a full fitness evaluation × seeds). Env:
    # AFLOW_MAX_CANDIDATES.
    max_candidates: int = 3  # Env: AFLOW_MAX_CANDIDATES
    # Required improvement over the baseline to accept a candidate policy. 0.0
    # = any improvement; raise to require a clear margin. Env:
    # AFLOW_IMPROVEMENT_MARGIN.
    improvement_margin: float = 0.0  # Env: AFLOW_IMPROVEMENT_MARGIN
    # Advisory mid-search spend cap (USD). 0 = inert (the CLI wires a real ledger
    # read; max_candidates + the per-run hard-stop are the hard bounds). Env:
    # AFLOW_MAX_COST_USD.
    max_cost_usd: float = 1.0  # Env: AFLOW_MAX_COST_USD
    # Pre-flight C1 safety gate: refuse to consolidate during a known regression
    # OR insufficient curve data (mirrors the optimizer's require_curve_clear).
    # Env: AFLOW_PREFLIGHT_CURVE_CLEAR.
    preflight_curve_clear: bool = True  # Env: AFLOW_PREFLIGHT_CURVE_CLEAR

    model_config = SettingsConfigDict(
        # env_prefix maps AFLOW_* vars to these fields, mirroring the LATS
        # pattern (a missing prefix silently ignores vars).
        env_prefix="aflow_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class Neo4jSettings(BaseSettings):
    """Neo4j entity/relation graph — structured mirror (Phase 5 I3).

    An ADDITIVE, opt-in graph store: when enabled, the MemoryManager write hooks
    mirror STRUCTURED records (sub-agent defs, skills/procedures/workflows, facts)
    into Neo4j nodes/edges on write — a relationship substrate the relational +
    pgvector stores don't express (which skills depend on X, which sub-agent
    handles Y). Pure structured sync — NO LLM extraction (a later option).
    Best-effort + non-fatal (CostTracker-resilience pattern): a missing package,
    an unreachable Neo4j, or any driver error is caught, logged WARNING, and never
    re-raises — a graph hiccup can never abort a run. Default-off: when disabled
    the store is a no-op and nothing syncs (byte-identical to pre-I3).

    Env-var names are read via explicit per-field validation_alias (no env_prefix)
    so they are exactly GRAPH_ENABLED (the mirror FEATURE, conceptually independent
    of the Neo4j connection — you may have Neo4j up but the mirror off) and
    NEO4J_URI/USER/PASSWORD (the connection) — NOT NEO4J_ENABLED.
    """

    # Master opt-in for the entity/relation mirror. Env: GRAPH_ENABLED.
    enabled: bool = Field(default=False, validation_alias="GRAPH_ENABLED")
    # Bolt URI. Host mode: bolt://localhost:7687; compose: bolt://neo4j:7687.
    # Env: NEO4J_URI.
    uri: str = Field(default="bolt://localhost:7687", validation_alias="NEO4J_URI")
    # Neo4j user (the container's bootstrap user). Env: NEO4J_USER.
    user: str = Field(default="neo4j", validation_alias="NEO4J_USER")
    # Neo4j password — MUST match the container NEO4J_AUTH password. Env:
    # NEO4J_PASSWORD.
    password: str = Field(default="", validation_alias="NEO4J_PASSWORD")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class Settings(BaseSettings):
    """Root settings class that composes all settings groups."""

    # Nested settings groups
    llm: LLMProviderSettings = LLMProviderSettings()  # type: ignore[assignment]
    database: DatabaseSettings = DatabaseSettings()  # type: ignore[assignment]
    redis: RedisSettings = RedisSettings()  # type: ignore[assignment]
    budget: BudgetSettings = BudgetSettings()  # type: ignore[assignment]
    evolution: EvolutionSettings = EvolutionSettings()  # type: ignore[assignment]
    agent: AgentSettings = AgentSettings()  # type: ignore[assignment]
    logging: LoggingSettings = LoggingSettings()  # type: ignore[assignment]
    observability: ObservabilitySettings = ObservabilitySettings()  # type: ignore[assignment]
    langsmith: LangSmithSettings = LangSmithSettings()  # type: ignore[assignment]
    tool_cache: ToolCacheSettings = ToolCacheSettings()  # type: ignore[assignment]
    resilience: ResilienceSettings = ResilienceSettings()  # type: ignore[assignment]
    circuit_breaker: CircuitBreakerSettings = CircuitBreakerSettings()  # type: ignore[assignment]
    rate_limiter: RateLimiterSettings = RateLimiterSettings()  # type: ignore[assignment]
    routing: RoutingSettings = RoutingSettings()  # type: ignore[assignment]
    prompt_cache: PromptCacheControlSettings = PromptCacheControlSettings()  # type: ignore[assignment]
    batching: BatchingSettings = BatchingSettings()  # type: ignore[assignment]
    reasoning: ReasoningControlSettings = ReasoningControlSettings()  # type: ignore[assignment]
    native_structured: NativeStructuredSettings = NativeStructuredSettings()  # type: ignore[assignment]
    tools: ToolLimitsSettings = ToolLimitsSettings()  # type: ignore[assignment]
    tool_sandbox: ToolSandboxSettings = ToolSandboxSettings()  # type: ignore[assignment]
    eval: EvalSettings = EvalSettings()  # type: ignore[assignment]
    search: SearchSettings = SearchSettings()  # type: ignore[assignment]
    worker: WorkerSettings = WorkerSettings()  # type: ignore[assignment]
    runner: RunnerSettings = RunnerSettings()  # type: ignore[assignment]
    scheduler: SchedulerSettings = SchedulerSettings()  # type: ignore[assignment]
    capability_curve: CapabilityCurveSettings = CapabilityCurveSettings()  # type: ignore[assignment]
    governance_prune: GovernancePruneSettings = GovernancePruneSettings()  # type: ignore[assignment]
    optimizer: OptimizerSettings = OptimizerSettings()  # type: ignore[assignment]
    agent_cron: AgentCronSettings = AgentCronSettings()  # type: ignore[assignment]
    git_clone: GitCloneSettings = GitCloneSettings()  # type: ignore[assignment]
    lats: LatsSettings = LatsSettings()  # type: ignore[assignment]
    aflow: AflowSettings = AflowSettings()  # type: ignore[assignment]
    neo4j: Neo4jSettings = Neo4jSettings()  # type: ignore[assignment]

    # Environment metadata
    environment: Literal["development", "staging", "production"] = "development"
    deployment_id: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@functools.lru_cache
def get_settings() -> Settings:
    """Singleton getter for Settings.

    The @lru_cache decorator ensures Settings is instantiated only once per process.
    This prevents re-reading .env on every call and maintains consistency.

    Returns:
        Settings: The validated settings instance.

    Raises:
        ValidationError: If .env contains invalid values.
    """
    return Settings()
