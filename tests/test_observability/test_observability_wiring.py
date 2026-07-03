"""Tests for the observability wiring landed in Track B (battery-04 follow-on):

- ``metrics_response`` returns ``(bytes, content_type)`` with the right shape, and
  a NON-EMPTY body when ``prometheus_client`` is installed (the recorder call
  sites populate the default registry).
- The api ``/metrics`` handler (``_metrics``) returns a 200 ``Response`` carrying
  that body — so the scrape endpoint is live without standing up uvicorn.
- ``init_process_observability`` is the worker/scheduler/optimizer façade: a
  no-op when both flags are off, and it actually wires tracing + the metrics
  server when they are on (verified via monkeypatch — no real OTel provider /
  bound port side effects in the unit suite).
- ``instrument_fastapi_app`` never raises on a None / repeat call (best-effort).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.app import _metrics
from src.observability import init_process_observability
from src.observability.metrics import metrics_response
from src.observability.tracing import instrument_fastapi_app


def _obs(*, otel: bool, prom: bool) -> SimpleNamespace:
    return SimpleNamespace(
        otel_enabled=otel,
        otel_service_name="turing-test",
        otel_endpoint="http://localhost:4318",
        otel_sampling_rate=0.1,
        prometheus_enabled=prom,
        prometheus_port=9100,
    )


# ─── metrics_response + /metrics handler ──────────────────────────────────────


def test_metrics_response_shape() -> None:
    body, content_type = metrics_response()
    assert isinstance(body, bytes)
    assert isinstance(content_type, str)
    assert content_type.startswith("text/plain")


def test_metrics_response_nonempty_when_prometheus_present() -> None:
    """prometheus_client is a pinned dep + recorder sites fire at runtime, so the
    default registry is populated; an empty body would mean it never recorded."""
    pytest.importorskip("prometheus_client")
    body, _ = metrics_response()
    assert len(body) > 0


@pytest.mark.asyncio
async def test_metrics_endpoint_handler_returns_200_with_body() -> None:
    """The /metrics handler returns 200 + the rendered registry (no uvicorn)."""
    response = await _metrics()
    assert response.status_code == 200
    assert isinstance(response.body, (bytes, bytearray))


# ─── init_process_observability façade ────────────────────────────────────────


def test_init_process_observability_is_noop_when_disabled() -> None:
    """Both flags off → no tracing/metrics setup, never raises."""
    init_process_observability(_obs(otel=False, prom=False), component="test")


def test_init_process_observability_wires_both_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flags on → setup_tracing + start_metrics_server are both invoked with the
    configured settings. Verified via monkeypatch (no real provider / bound port)."""
    tracing_calls: list[dict[str, object]] = []
    metrics_calls: list[int] = []

    def fake_setup_tracing(**kwargs: object) -> None:
        tracing_calls.append(kwargs)

    def fake_start(port: int, host: str = "0.0.0.0") -> bool:
        metrics_calls.append(port)
        return True

    monkeypatch.setattr("src.observability.tracing.setup_tracing", fake_setup_tracing)
    monkeypatch.setattr("src.observability.metrics.start_metrics_server", fake_start)

    init_process_observability(_obs(otel=True, prom=True), component="worker")

    assert len(tracing_calls) == 1
    assert tracing_calls[0]["service_name"] == "turing-test"
    assert tracing_calls[0]["endpoint"] == "http://localhost:4318"
    assert metrics_calls == [9100]


def test_init_process_observability_skips_metrics_when_only_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    """OTel on + Prometheus off → only tracing is wired."""
    metrics_calls: list[int] = []
    monkeypatch.setattr(
        "src.observability.metrics.start_metrics_server",
        lambda port, host="0.0.0.0": metrics_calls.append(port) or True,
    )
    monkeypatch.setattr("src.observability.tracing.setup_tracing", lambda **_kw: None)

    init_process_observability(_obs(otel=True, prom=False), component="worker")
    assert metrics_calls == []


# ─── instrument_fastapi_app best-effort safety ────────────────────────────────


def test_instrument_fastapi_app_never_raises_on_none_or_repeat() -> None:
    """instrument_fastapi_app is best-effort: a None app (instrumentor import fail
    path) and a repeat call must never raise into app boot."""
    instrument_fastapi_app(None)
    instrument_fastapi_app(None)  # repeat — still safe
