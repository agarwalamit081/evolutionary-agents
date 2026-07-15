"""Prometheus metrics export (ObservabilitySettings.prometheus_enabled) parity.

``src.observability.metrics`` creates the canonical counters/histograms/gauges
behind a ``prometheus_client`` import guard (no-op when the lib is absent). The
``prometheus_enabled`` flag controls the server scrape endpoint; the metric
OBJECTS + ``record_*`` helpers are wired unconditionally so a recorded event
increments the right counter whether or not the scrape server is up. These
tests assert the registry exposes the expected families and that a recorded
event lands in the right labeled child, plus the no-op-when-unavailable path.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.observability import metrics as m


def _samples() -> dict[tuple[str, frozenset[tuple[str, str]]], float]:
    """Flatten the registry into a {(name, frozenset(labels)): value} map."""
    out: dict[tuple[str, frozenset[tuple[str, str]]], float] = {}
    for family in _registry().collect():
        for s in family.samples:
            out[(s.name, frozenset(s.labels.items()))] = s.value
    return out


def _registry() -> Any:
    from prometheus_client import REGISTRY

    return REGISTRY


# ─── registry surface ───────────────────────────────────────────────────────


class TestMetricRegistrySurface:
    def test_should_expose_all_expected_counter_and_histogram_objects(self) -> None:
        # The module wires every canonical counter/histogram/gauge. When the lib
        # is present these are real metric objects (not None).
        assert m.LLM_REQUEST_TOKENS is not None
        assert m.LLM_REQUEST_COST is not None
        assert m.LLM_REQUEST_ERRORS is not None
        assert m.LLM_CACHE_HITS is not None
        assert m.LLM_CACHE_MISSES is not None
        assert m.LLM_PROMPT_CACHE_READ_TOKENS is not None
        assert m.LLM_PROMPT_CACHE_CREATION_TOKENS is not None
        assert m.CIRCUIT_BREAKER_STATE_TRANSITIONS is not None
        assert m.GRAPH_NODE_DURATION is not None
        assert m.GRAPH_ITERATIONS is not None
        assert m.GRAPH_TASKS_COMPLETED is not None
        assert m.MEMORY_OPERATIONS is not None
        assert m.TOOL_CALLS is not None
        assert m.TOOL_CALL_ERRORS is not None
        assert m.TOOL_CALL_DURATION is not None
        assert m.CAPABILITY_CURVE_SCORE is not None
        assert m.CAPABILITY_CURVE_REGRESSIONS is not None

    def test_counter_helpers_return_none_when_prometheus_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When prometheus_client is unavailable, every factory returns None so
        # the record_* helpers short-circuit (no raise).
        monkeypatch.setattr(m, "_PROMETHEUS_AVAILABLE", False)
        assert m._counter("x", "y") is None
        assert m._histogram("x", "y") is None
        assert m._gauge("x", "y") is None


# ─── record_* increment the right counter ───────────────────────────────────


class TestRecordIncrementsRightCounter:
    def test_record_llm_request_increments_tokens_and_cost(self) -> None:
        # Phase 6b: tokens/cost children now carry ``tier`` + ``complexity``
        # labels (Q105), so the label set must include them.
        labels_in = {
            "model": "glm-4.7", "provider": "zai", "token_type": "input",
            "tier": "moderate", "complexity": "complex",
        }
        labels_cost = {"model": "glm-4.7", "provider": "zai",
                       "tier": "moderate", "complexity": "complex"}
        before_in = _samples().get(
            ("llm_request_tokens_total", frozenset(labels_in.items())), 0.0)
        before_cost = _samples().get(
            ("llm_request_cost_usd_total", frozenset(labels_cost.items())), 0.0)
        m.record_llm_request("glm-4.7", "zai", 0.5,
                             input_tokens=7, output_tokens=3, cost_usd=0.02,
                             tier="moderate", complexity="complex")
        after = _samples()
        assert after[("llm_request_tokens_total",
                      frozenset(labels_in.items()))] == before_in + 7
        labels_out = {**labels_in, "token_type": "output"}
        assert after[("llm_request_tokens_total",
                      frozenset(labels_out.items()))] >= 3
        assert after[("llm_request_cost_usd_total",
                      frozenset(labels_cost.items()))] == before_cost + 0.02

    def test_record_llm_request_defaults_tier_complexity_to_unknown(self) -> None:
        # Pre-Phase-6b callers omit tier/complexity → they default to "unknown"
        # so the existing record_* surface stays backward-compatible.
        m.record_llm_request("legacy-model", "p", 0.1, input_tokens=1, cost_usd=0.01)
        labels = frozenset({"model": "legacy-model", "provider": "p", "token_type": "input",
                            "tier": "unknown", "complexity": "unknown"}.items())
        assert _samples()[("llm_request_tokens_total", labels)] >= 1

    def test_record_llm_request_label_cardinality_is_bounded(self) -> None:
        # Q105 guard: tier/complexity are bounded enums. Recording many distinct
        # (model, provider) pairs must NOT explode the cardinality of either
        # label beyond its enum — every child's tier ∈ the ModelTier values ∪
        # "unknown" and complexity ∈ TaskComplexity values ∪ "unknown".
        valid_tiers = {"very_cheap", "cheap", "moderate", "unknown"}
        valid_complexities = {"trivial", "simple", "complex", "critical", "unknown"}
        m.record_llm_request("a", "p1", 0.1, input_tokens=1,
                             tier="cheap", complexity="trivial")
        m.record_llm_request("b", "p2", 0.1, input_tokens=1,
                             tier="moderate", complexity="complex")
        for (name, labels), _value in _samples().items():
            if name != "llm_request_tokens_total":
                continue
            label_map = dict(labels)
            assert label_map["tier"] in valid_tiers
            assert label_map["complexity"] in valid_complexities

    def test_record_llm_request_skips_zero_cost(self) -> None:
        # A free call (cost_usd=0) must not create a cost child. Uses the full
        # Phase-6b label set so the assertion is meaningful (not vacuously true
        # on a 2-label child that can never exist now).
        labels = frozenset({"model": "free-model", "provider": "free",
                            "tier": "unknown", "complexity": "unknown"}.items())
        m.record_llm_request("free-model", "free", 0.1, cost_usd=0.0)
        assert ("llm_request_cost_usd_total", labels) not in _samples()

    def test_record_cache_lookup_routes_hit_and_miss(self) -> None:
        before_hit = _samples().get(
            ("llm_cache_hits_total", frozenset({"model": "cm"}.items())), 0.0)
        before_miss = _samples().get(
            ("llm_cache_misses_total", frozenset({"model": "cm"}.items())), 0.0)
        m.record_cache_lookup("cm", True)
        m.record_cache_lookup("cm", False)
        after = _samples()
        assert after[("llm_cache_hits_total", frozenset({"model": "cm"}.items()))] == before_hit + 1
        assert after[("llm_cache_misses_total", frozenset({"model": "cm"}.items()))] == before_miss + 1

    def test_record_prompt_cache_tokens_increments_read_and_creation(self) -> None:
        # #13 — provider-native cache token counts land in their own counters.
        labels = frozenset({"model": "glm-4.7", "provider": "zai"}.items())
        before_read = _samples().get(
            ("llm_prompt_cache_read_tokens_total", labels), 0.0)
        before_create = _samples().get(
            ("llm_prompt_cache_creation_tokens_total", labels), 0.0)
        m.record_prompt_cache_tokens("glm-4.7", "zai", 120, 40)
        after = _samples()
        assert after[("llm_prompt_cache_read_tokens_total", labels)] == before_read + 120
        assert after[("llm_prompt_cache_creation_tokens_total", labels)] == before_create + 40

    def test_record_prompt_cache_tokens_skips_zeros(self) -> None:
        # Zero cache tokens (caching off / provider doesn't report) ⇒ no child.
        m.record_prompt_cache_tokens("no-cache-model", "p", 0, 0)
        labels = frozenset({"model": "no-cache-model", "provider": "p"}.items())
        assert ("llm_prompt_cache_read_tokens_total", labels) not in _samples()
        assert ("llm_prompt_cache_creation_tokens_total", labels) not in _samples()

    def test_record_tool_call_increments_calls_and_errors_on_failure(self) -> None:
        before_calls = _samples().get(
            ("tool_calls_total", frozenset({"tool_name": "tx"}.items())), 0.0)
        before_err = _samples().get(
            ("tool_call_errors_total", frozenset({"tool_name": "tx"}.items())), 0.0)
        m.record_tool_call("tx", 1.0, success=False)
        after = _samples()
        assert after[("tool_calls_total", frozenset({"tool_name": "tx"}.items()))] == before_calls + 1
        assert after[("tool_call_errors_total", frozenset({"tool_name": "tx"}.items()))] == before_err + 1

    def test_record_tool_call_no_error_on_success(self) -> None:
        before_err = _samples().get(
            ("tool_call_errors_total", frozenset({"tool_name": "ts"}.items())), 0.0)
        m.record_tool_call("ts", 1.0, success=True)
        after = _samples()
        assert after.get(
            ("tool_call_errors_total", frozenset({"tool_name": "ts"}.items())), 0.0) == before_err

    def test_record_memory_operation_increments_labeled_counter(self) -> None:
        before = _samples().get(
            ("memory_operations_total", frozenset({"operation": "store", "tier": "warm"}.items())), 0.0)
        m.record_memory_operation("store", "warm")
        after = _samples()
        assert after[("memory_operations_total",
                      frozenset({"operation": "store", "tier": "warm"}.items()))] == before + 1

    def test_capability_curve_gauge_and_regression_counter_are_settable(self) -> None:
        m.CAPABILITY_CURVE_SCORE.set(0.55)
        assert _samples()[("capability_curve_score", frozenset())] == 0.55
        before = _samples().get(("capability_curve_regressions_total", frozenset()), 0.0)
        m.CAPABILITY_CURVE_REGRESSIONS.inc()
        assert _samples()[("capability_curve_regressions_total", frozenset())] == before + 1


# ─── disabled / no-op path ──────────────────────────────────────────────────


class TestDisabledNoOp:
    def test_record_helpers_never_raise_when_prometheus_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``prometheus_enabled`` off + lib absent ⇒ every metric object is None
        # and every record_* helper short-circuits (guard ``if X:``) without
        # raising. Simulate by nulling the module-level objects.
        monkeypatch.setattr(m, "LLM_REQUEST_DURATION", None)
        monkeypatch.setattr(m, "LLM_REQUEST_TOKENS", None)
        monkeypatch.setattr(m, "LLM_REQUEST_COST", None)
        monkeypatch.setattr(m, "TOOL_CALLS", None)
        monkeypatch.setattr(m, "TOOL_CALL_ERRORS", None)
        monkeypatch.setattr(m, "TOOL_CALL_DURATION", None)
        monkeypatch.setattr(m, "MEMORY_OPERATIONS", None)
        monkeypatch.setattr(m, "CAPABILITY_CURVE_SCORE", None)
        # None of these raise.
        m.record_llm_request("a", "b", 1.0, input_tokens=1, output_tokens=1, cost_usd=1.0)
        m.record_tool_call("a", 1.0, success=False)
        m.record_memory_operation("a", "b")
        m.record_node_duration("a", 1.0)
