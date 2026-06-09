"""OpenTelemetry tracing setup for the Turing Agent.

Configures tracing with automatic instrumentation for asyncpg, httpx,
and FastAPI. Provides helpers for creating spans around graph nodes.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def setup_tracing(
    service_name: str = "turing-agent",
    endpoint: str = "http://localhost:4318",
    sampling_rate: float = 0.1,
) -> Any | None:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for traces.
        endpoint: OTLP HTTP endpoint.
        sampling_rate: Trace sampling rate (0.0-1.0).

    Returns:
        TracerProvider if setup succeeded, None otherwise.
    """
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
        provider.add_span_processor(
            _create_batch_processor(exporter)
        )
        trace.set_tracer_provider(provider)

        logger.info(f"OpenTelemetry tracing initialized (endpoint={endpoint}, rate={sampling_rate})")
        return provider
    except ImportError:
        logger.debug("OpenTelemetry not installed, tracing disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize tracing: {e}")
        return None


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
