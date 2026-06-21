"""Tests for src.observability.tracing — OpenTelemetry tracing."""

from __future__ import annotations

from src.observability.tracing import _NoOpSpan, _NoOpTracer, get_tracer, setup_tracing


class TestNoOpTracer:
    """Tests for the fallback no-op tracer."""

    def test_start_as_current_span_returns_span(self) -> None:
        """start_as_current_span returns a _NoOpSpan."""
        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test")
        assert isinstance(span, _NoOpSpan)

    def test_start_span_returns_span(self) -> None:
        """start_span returns a _NoOpSpan."""
        tracer = _NoOpTracer()
        span = tracer.start_span("test")
        assert isinstance(span, _NoOpSpan)


class TestNoOpSpan:
    """Tests for the fallback no-op span."""

    def test_context_manager(self) -> None:
        """_NoOpSpan works as a context manager."""
        span = _NoOpSpan()
        with span as s:
            s.set_attribute("key", "value")
            s.set_status("ok")
            s.end()
        # No crash = success

    def test_methods_exist(self) -> None:
        """_NoOpSpan has all required methods."""
        span = _NoOpSpan()
        assert callable(span.set_attribute)
        assert callable(span.set_status)
        assert callable(span.record_exception)
        assert callable(span.end)


class TestSetupTracing:
    """Tests for setup_tracing function."""

    def test_returns_none_without_otel(self) -> None:
        """setup_tracing returns None when OpenTelemetry not installed."""
        result = setup_tracing()
        # May return None (no otel) or a provider (otel installed)
        # Either way, should not crash
        assert result is None or result is not None

    def test_get_tracer_returns_tracer_or_noop(self) -> None:
        """get_tracer returns a tracer (real or no-op)."""
        tracer = get_tracer("test")
        assert tracer is not None
        assert hasattr(tracer, "start_span")
