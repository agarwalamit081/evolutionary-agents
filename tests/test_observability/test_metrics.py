"""Tests for src.observability.metrics — Prometheus metric recording."""

from __future__ import annotations

from src.observability.metrics import (
    record_llm_request,
    record_memory_operation,
    record_node_duration,
    record_tool_call,
)


class TestMetricsNoOp:
    """All record_* functions should work as no-ops without prometheus installed."""

    def test_record_llm_request_no_crash(self) -> None:
        """record_llm_request does not raise without prometheus."""
        record_llm_request(
            model="gpt-4o-mini",
            provider="openai",
            duration_seconds=1.5,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )

    def test_record_node_duration_no_crash(self) -> None:
        """record_node_duration does not raise without prometheus."""
        record_node_duration("classify", 0.5)

    def test_record_tool_call_no_crash(self) -> None:
        """record_tool_call does not raise without prometheus."""
        record_tool_call("code_executor", 2.0, success=True)

    def test_record_memory_operation_no_crash(self) -> None:
        """record_memory_operation does not raise without prometheus."""
        record_memory_operation("store", "hot")
