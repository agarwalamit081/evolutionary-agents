"""Tests for src.llm.cost_tracker — cost calculation."""

from __future__ import annotations

from src.llm.cost_tracker import CostTracker


class TestCalculateCost:
    """Tests for the static calculate_cost method."""

    def test_known_model_returns_fallback_cost(self) -> None:
        """Known model without cost fields uses fallback pricing."""
        # ModelSpec lacks input_cost_per_1k, so fallback pricing applies
        cost = CostTracker.calculate_cost("gpt-4o-mini-2024-07-18", 100, 50)
        assert cost > 0

    def test_unknown_model_uses_fallback(self) -> None:
        """Unknown model uses fallback pricing ($0.005/1K in, $0.015/1K out)."""
        cost = CostTracker.calculate_cost("unknown-model-xyz", 1000, 500)
        expected = (1000 * 0.005 / 1000) + (500 * 0.015 / 1000)
        assert abs(cost - expected) < 1e-10

    def test_zero_tokens_returns_zero(self) -> None:
        """Zero tokens → cost is 0.0."""
        cost = CostTracker.calculate_cost("any-model", 0, 0)
        assert cost == 0.0

    def test_cost_scales_linearly(self) -> None:
        """Cost scales linearly with token count."""
        cost_small = CostTracker.calculate_cost("test-model", 100, 50)
        cost_large = CostTracker.calculate_cost("test-model", 1000, 500)
        assert abs(cost_large - cost_small * 10) < 1e-10

    def test_fallback_pricing_formula(self) -> None:
        """Verify fallback pricing formula: (in * 0.005 + out * 0.015) / 1000."""
        cost = CostTracker.calculate_cost("nonexistent", 2000, 3000)
        expected = (2000 * 0.005 + 3000 * 0.015) / 1000
        assert abs(cost - expected) < 1e-10
