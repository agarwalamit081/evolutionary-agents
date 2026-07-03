"""OpenTelemetry tracing setup for the Turing Agent.

Initializes a TracerProvider that exports spans over OTLP/HTTP (to Phoenix by
default), and auto-instruments the three libraries whose call paths matter for
bottleneck identification: **asyncpg** (DB queries), **httpx** (outbound HTTP,
incl. litellm's provider calls), and **FastAPI** (inbound request handling).

All opt-in via ``ObservabilitySettings.otel_enabled`` (default off). Idempotent —
a process may call ``setup_tracing`` / ``instrument_fastapi_app`` more than once
without double-instrumenting. Each auto-instrumentor is isolated so a missing or
incompatible library disables only itself, never the whole tracer.

LLM-call spans are emitted manually by ``LLMGateway`` (``src/llm/gateway.py``)
via ``get_tracer`` — the gateway is the single chokepoint that sees every
completion regardless of provider, so one manual span with OpenInference-
convention attributes captures each call (model/tokens/cost/latency).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# Idempotency: OTel's set_tracer_provider + the instrumentors must each run at
# most once per process (re-calling warns and is a no-op or double-wraps).
_PROVIDER_INSTALLED = False
_FASTAPI_INSTRUMENTED = False


def setup_tracing(
    service_name: str = "turing-agent",
    endpoint: str = "http://localhost:4318",
    sampling_rate: float = 0.1,
    *,
    instrument_http: bool = True,
) -> Any | None:
    """Initialize OpenTelemetry tracing + auto-instrument asyncpg/httpx.

    Args:
        service_name: Service name for traces (``service.name`` resource attr).
        endpoint: OTLP HTTP endpoint (Phoenix default ``http://localhost:4318``);
            ``/v1/traces`` is appended.
        sampling_rate: Trace sampling rate (0.0-1.0).
        instrument_http: Also auto-instrument asyncpg + httpx (DB + outbound HTTP
            spans). FastAPI is instrumented separately via ``instrument_fastapi_app``.

    Returns:
        The TracerProvider if setup succeeded, else None. A no-op (returns None)
        if already initialized this process.
    """
    global _PROVIDER_INSTALLED
    if _PROVIDER_INSTALLED:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        resource = Resource.create({"service.name": service_name})
        sampler = TraceIdRatioBased(rate=sampling_rate)
        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")

        provider = TracerProvider(resource=resource, sampler=sampler)
        provider.add_span_processor(_create_batch_processor(exporter))
        trace.set_tracer_provider(provider)

        if instrument_http:
            _instrument_asyncpg()
            _instrument_httpx()

        _PROVIDER_INSTALLED = True
        logger.info(
            f"OpenTelemetry tracing initialized (endpoint={endpoint}, "
            f"rate={sampling_rate}, auto_instrument={instrument_http})"
        )
        return provider
    except ImportError:
        logger.debug("OpenTelemetry not installed, tracing disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize tracing: {e}")
        return None


def instrument_fastapi_app(app: Any) -> None:
    """Instrument a specific FastAPI app for inbound-request spans.

    Called from ``create_app`` (the app instance is required, unlike the
    library-global asyncpg/httpx instrumentors). Idempotent + best-effort.
    """
    global _FASTAPI_INSTRUMENTED
    if _FASTAPI_INSTRUMENTED:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        _FASTAPI_INSTRUMENTED = True
        logger.debug("FastAPI instrumentation enabled")
    except ImportError:
        logger.debug("FastAPI instrumentor not installed, skipping")
    except Exception as e:  # noqa: BLE001 — instrumentation must never block app boot
        logger.warning(f"FastAPI instrumentation failed: {e}")


def _instrument_asyncpg() -> None:
    try:
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

        AsyncPGInstrumentor().instrument()
        logger.debug("asyncpg auto-instrumentation enabled")
    except ImportError:
        logger.debug("asyncpg instrumentor not installed, skipping")
    except Exception as e:  # noqa: BLE001 — per-library isolation
        logger.warning(f"asyncpg instrumentation failed: {e}")


def _instrument_httpx() -> None:
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
        logger.debug("httpx auto-instrumentation enabled")
    except ImportError:
        logger.debug("httpx instrumentor not installed, skipping")
    except Exception as e:  # noqa: BLE001 — per-library isolation
        logger.warning(f"httpx instrumentation failed: {e}")


def _create_batch_processor(exporter: Any) -> Any:
    """Create a BatchSpanProcessor for the exporter."""
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    return BatchSpanProcessor(exporter)


def get_tracer(name: str = "turing-agent") -> Any:
    """Get a tracer instance.

    Args:
        name: Tracer name (typically module name).

    Returns:
        OpenTelemetry Tracer, or a no-op tracer if tracing is disabled.
    """
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


class _NoOpTracer:
    """Fallback tracer when OpenTelemetry is not installed."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> Any:
        return _NoOpSpan()


class _NoOpSpan:
    """Fallback span that does nothing."""

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def end(self) -> None:
        pass
