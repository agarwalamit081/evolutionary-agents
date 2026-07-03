"""Observability package — logging, metrics, and tracing."""

from typing import Any

from src.observability.logging import (
    InterceptHandler,
    PIIRedactor,
    add_query_log_sink,
    get_logger,
    logger,
    pii_redaction_filter,
    remove_query_log_sink,
    reset_logging,
    setup_logging,
)

__all__ = [
    "setup_logging",
    "init_process_observability",
    "logger",
    "get_logger",
    "reset_logging",
    "PIIRedactor",
    "pii_redaction_filter",
    "InterceptHandler",
    "add_query_log_sink",
    "remove_query_log_sink",
]


def init_process_observability(obs: Any, *, component: str = "process") -> None:
    """Initialize OTel tracing + a Prometheus scrape server for a worker-style process.

    Shared by the worker / scheduler / optimizer bootstraps (each runs in its own
    container, so each binds its own ``PROMETHEUS_PORT`` in its own netns — no
    clash). The api does NOT use this — it exposes ``/metrics`` as a FastAPI route
    and instruments the app via ``instrument_fastapi_app`` in ``create_app``.

    Both subsystems are opt-in (OTEL_ENABLED / PROMETHEUS_ENABLED), idempotent,
    and best-effort: a missing exporter / prometheus_client / bound port disables
    only that piece and never aborts the process. Observability-only.

    Args:
        obs: An ``ObservabilitySettings`` instance (duck-typed — needs the
            otel_* / prometheus_* attributes).
        component: Tag for the startup log line (e.g. ``"worker"``).
    """
    if obs.otel_enabled:
        from src.observability.tracing import setup_tracing

        setup_tracing(
            service_name=obs.otel_service_name,
            endpoint=obs.otel_endpoint,
            sampling_rate=obs.otel_sampling_rate,
        )
    if obs.prometheus_enabled:
        from src.observability.metrics import start_metrics_server

        if start_metrics_server(obs.prometheus_port):
            logger.info(
                f"{component}: Prometheus /metrics server listening on :{obs.prometheus_port}"
            )
        else:
            logger.debug(
                f"{component}: Prometheus metrics server not started on "
                f":{obs.prometheus_port} (prometheus_client absent or port unavailable)"
            )
