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

from pydantic import field_validator, model_validator
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

    Previously the ``CircuitBreaker`` constructor defaults (failure_threshold=5,
    recovery_timeout=60, half_open_max_calls=1). Exposed so provider reliability
    tuning doesn't require editing ``src/llm/circuit_breaker.py``.
    """

    cb_failure_threshold: int = 5  # Env: CB_FAILURE_THRESHOLD
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
    code_executor_timeout: int = 30  # Env: CODE_EXECUTOR_TIMEOUT
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


# ─── Budget Settings ────────────────────────────────────────────────


class BudgetSettings(BaseSettings):
    """Token budget and cost control configuration."""

    daily_token_budget: int = 500000
    per_task_token_limit: int = 100000
    max_cost_usd: float = 10.0
    budget_warn_threshold: float = 0.70
    budget_critical_threshold: float = 0.90

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
    evolution_sandbox_mode: Literal["docker", "subprocess"] = "docker"
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
    # Directory holding promoted, versioned handler artifacts (prompts first).
    # Layout: ``<dir>/prompts/<node>.<sha>.json`` (immutable versions) +
    # ``<dir>/prompts/current.json`` (the live pointer manifest the builder reads).
    # Default lives under .turing/ (gitignored scratch), NOT in core src/. Env:
    # EVOLVED_HANDLERS_DIR.
    evolved_handlers_dir: str = ".turing/evolved"

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
    # Run caps — single source of truth for tool/sub-agent creation limits.
    # Enforcement sites (tool generator, agent_spawn, structure_analysis) read
    # these; the module-level MAX_TOOLS_PER_RUN / MAX_SUB_AGENTS_PER_RUN
    # constants remain only as the matching default, not the enforced value.
    # (max_sub_agents previously lived here but was never read — the live cap
    # was the MAX_SUB_AGENTS_PER_RUN constant — so it is folded into this
    # overridable field. Env: MAX_TOOLS_PER_RUN, MAX_SUB_AGENTS_PER_RUN.)
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
    # (success_rate < retire_success_floor over >= retire_min_runs), stale
    # (unused for retire_recency_days), or redundant (cosine >=
    # capability_redundancy_threshold with a better-scoring twin). Enforced on
    # load by SubAgentPersister.load_active_agents / ToolPersister.load_active_tools
    # when settings is passed. Env: MAX_ACTIVE_TOOLS, MAX_ACTIVE_SUB_AGENTS,
    # CAPABILITY_REDUNDANCY_THRESHOLD, RETIRE_MIN_RUNS, RETIRE_SUCCESS_FLOOR,
    # RETIRE_RECENCY_DAYS.
    max_active_tools: int = 25  # every active tool loads into the registry/selection prompt → keep tight
    # Sub-agents are delegated-to selectively (not all injected into every prompt),
    # so a higher stored cap is safe and preserves a richer specialist ecosystem.
    max_active_sub_agents: int = 60
    capability_redundancy_threshold: float = 0.92
    retire_min_runs: int = 20
    retire_success_floor: float = 0.25
    retire_recency_days: int = 30
    # Per-tool success-metrics recording (M4). When true, the execute chokepoint
    # records each tool invocation (success/empty/latency) to tool_call_metrics
    # and updates the running aggregates on tool_registrations, which the
    # performance-retirement path above (retire_min_runs/retire_success_floor)
    # scores. Gated so a DB hiccup in the recorder never breaks a run.
    # Env: TOOL_METRICS_ENABLED.
    tool_metrics_enabled: bool = True
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
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive integers."""
        if v < 1:
            raise ValueError(f"Must be a positive integer. Got: {v}")
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

    Eval is opt-in: ``eval_enabled`` gates the verify-node correctness checks
    (a normal goal with no registered GoalSpec is unaffected even when True).
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

    # Lightweight paid providers tried in order when SearXNG fails/throttles.
    # Each is only attempted if its API key is set. Heavy providers are excluded
    # from this chain (provisioned-only, see deep_crawl_enabled).
    search_fallback_providers: str = "tavily,serper,brave,serpapi,serpstack,llmlayer"  # Env: SEARCH_FALLBACK_PROVIDERS

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


# ─── Root Settings ──────────────────────────────────────────────────


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
    prompt_cache: PromptCacheControlSettings = PromptCacheControlSettings()  # type: ignore[assignment]
    batching: BatchingSettings = BatchingSettings()  # type: ignore[assignment]
    reasoning: ReasoningControlSettings = ReasoningControlSettings()  # type: ignore[assignment]
    native_structured: NativeStructuredSettings = NativeStructuredSettings()  # type: ignore[assignment]
    tools: ToolLimitsSettings = ToolLimitsSettings()  # type: ignore[assignment]
    eval: EvalSettings = EvalSettings()  # type: ignore[assignment]
    search: SearchSettings = SearchSettings()  # type: ignore[assignment]

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
