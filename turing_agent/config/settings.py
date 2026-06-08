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
    alibaba_workspace_id: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Default model selection
    default_llm_provider: str = "deepseek"
    default_llm_model: str = "deepseek-v4-flash"

    # Fast model for classification, routing, simple tasks
    fast_llm_provider: str = "openai"
    fast_llm_model: str = "gpt-4o-mini-2024-07-18"

    # Heavy model for reasoning, planning, complex analysis
    reasoning_llm_provider: str = "anthropic"
    reasoning_llm_model: str = "claude-sonnet-4-6"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
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
            "dashscope",
            "groq",
            "google",
            "openrouter",
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
            "dashscope": self.dashscope_api_key,
            "groq": self.groq_api_key,
            "google": self.google_api_key,
            "openrouter": self.openrouter_api_key,
            "ollama": None,  # Local, no key needed
        }
        return key_map.get(provider)

    def has_provider_key(self, provider: str) -> bool:
        """Check if a provider has a non-empty API key configured."""
        key = self.get_provider_key(provider)
        return key is not None and len(key.strip()) > 0


# ─── Database Settings ──────────────────────────────────────────────


class DatabaseSettings(BaseSettings):
    """PostgreSQL database connection configuration."""

    database_url: str = "postgresql+asyncpg://amiagarw@localhost:5432/turing_agent"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
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

    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_hot_memory: int = 86400  # 24 hours
    redis_ttl_session: int = 3600  # 1 hour
    redis_ttl_rate_limit: int = 60  # 1 minute

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Ensure REDIS_URL starts with redis://."""
        if not v.startswith("redis://"):
            raise ValueError(f"REDIS_URL must start with redis://. Got: {v}")
        return v


# ─── Budget Settings ────────────────────────────────────────────────


class BudgetSettings(BaseSettings):
    """Token budget and cost control configuration."""

    daily_token_budget: int = 500000
    per_task_token_limit: int = 100000
    budget_warn_threshold: float = 0.70
    budget_critical_threshold: float = 0.90

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
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

    evolution_enabled: bool = False
    evolution_interval: int = 10  # Every N tasks
    evolution_max_mutations: int = 5
    evolution_sandbox_timeout: int = 30  # Seconds
    evolution_require_human_approval: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator(
        "evolution_interval",
        "evolution_max_mutations",
        "evolution_sandbox_timeout",
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive integers."""
        if v < 1:
            raise ValueError(f"Must be a positive integer. Got: {v}")
        return v


# ─── Agent Settings ─────────────────────────────────────────────────


class AgentSettings(BaseSettings):
    """Core agent execution limits and safety controls."""

    max_iterations: int = 20
    max_sub_agents: int = 3
    context_window_reserve: float = 0.15  # 15% margin
    hitl_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("max_iterations", "max_sub_agents")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Ensure positive integers."""
        if v < 1:
            raise ValueError(f"Must be a positive integer. Got: {v}")
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
        case_sensitive=True,
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
        case_sensitive=True,
    )

    @field_validator("otel_sampling_rate")
    @classmethod
    def validate_sampling_rate(cls, v: float) -> float:
        """Ensure sampling rate is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Sampling rate must be between 0 and 1. Got: {v}")
        return v


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

    # Environment metadata
    environment: Literal["development", "staging", "production"] = "development"
    deployment_id: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
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
