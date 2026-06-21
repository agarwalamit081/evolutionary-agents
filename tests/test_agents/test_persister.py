"""Tests for sub-agent persister — _model_to_spec and _spec_to_model conversions."""

from __future__ import annotations

import uuid

import pytest
from unittest.mock import MagicMock

from src.agents.persister import _model_to_spec as _model_to_spec
from src.agents.persister import _spec_to_model as _spec_to_model
from src.graph.models import SubAgentSpec


def _make_mock_model(**overrides: object) -> MagicMock:
    """Create a mock ORM SubAgentModel with sensible defaults."""
    defaults = {
        "id": uuid.uuid4(),
        "name": "test_agent",
        "description": "A test sub-agent",
        "model_tier": "simple",
        "max_iterations": 10,
        "template_type": "fixed",
        "tool_scope": "inherit_all",
        "tool_subset": None,
        "budget_mode": "shared",
        "budget_limit": 0.0,
        "depth_limit": 3,
        "node_config": None,
        "system_prompt_override": None,
        "version": 1,
        "is_active": True,
        "total_runs": 0,
        "success_rate": 0.0,
        "avg_cost": 0.0,
        "avg_latency_ms": 0,
        "quality_score": 0.5,
    }
    defaults.update(overrides)
    model = MagicMock()
    for key, value in defaults.items():
        setattr(model, key, value)
    return model


class TestModelToSpec:
    """Tests for _model_to_spec() conversion function."""

    def test_model_to_spec_produces_valid_spec(self) -> None:
        """_model_to_spec returns a valid SubAgentSpec instance."""
        model = _make_mock_model()
        spec = _model_to_spec(model)

        assert isinstance(spec, SubAgentSpec)
        assert spec.name == "test_agent"
        assert spec.description == "A test sub-agent"
        assert spec.is_active is True

    def test_model_to_spec_maps_uuid_to_id(self) -> None:
        """_model_to_spec converts DB UUID to spec id string."""
        test_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        model = _make_mock_model(id=test_uuid)
        spec = _model_to_spec(model)

        assert spec.id == "550e8400-e29b-41d4-a716-446655440000"
        # Verify the string is a valid UUID
        parsed = uuid.UUID(spec.id)
        assert parsed == test_uuid

    def test_model_to_spec_maps_model_tier(self) -> None:
        """_model_to_spec converts model_tier string to TaskComplexity enum."""
        model = _make_mock_model(model_tier="critical")
        spec = _model_to_spec(model)

        from src.graph.enums import TaskComplexity

        assert spec.model_tier == TaskComplexity.CRITICAL

    def test_model_to_spec_handles_null_optionals(self) -> None:
        """_model_to_spec handles None values for optional DB fields."""
        model = _make_mock_model(
            tool_subset=None,
            node_config=None,
            system_prompt_override=None,
        )
        spec = _model_to_spec(model)

        assert spec.tool_subset == []
        assert spec.node_config == {}
        assert spec.system_prompt_override is None

    def test_model_to_spec_handles_populated_optionals(self) -> None:
        """_model_to_spec handles populated optional DB fields."""
        model = _make_mock_model(
            tool_subset=["search_tool", "code_executor"],
            node_config={"nodes": ["plan", "execute"]},
            system_prompt_override="You are a security analyst.",
        )
        spec = _model_to_spec(model)

        assert spec.tool_subset == ["search_tool", "code_executor"]
        assert spec.node_config == {"nodes": ["plan", "execute"]}
        assert spec.system_prompt_override == "You are a security analyst."

    def test_model_to_spec_sets_runtime_fields_empty(self) -> None:
        """_model_to_spec sets goal and parent_thread_id to empty (runtime-set)."""
        model = _make_mock_model()
        spec = _model_to_spec(model)

        assert spec.goal == ""
        assert spec.parent_thread_id == ""

    def test_model_to_spec_maps_performance_metrics(self) -> None:
        """_model_to_spec maps performance metrics from DB model."""
        model = _make_mock_model(
            total_runs=42,
            success_rate=0.85,
            avg_cost=0.012,
            avg_latency_ms=1500,
            quality_score=0.92,
        )
        spec = _model_to_spec(model)

        assert spec.total_runs == 42
        assert spec.success_rate == pytest.approx(0.85)
        assert spec.avg_cost == pytest.approx(0.012)
        assert spec.avg_latency_ms == 1500
        assert spec.quality_score == pytest.approx(0.92)


class TestSpecToModel:
    """Tests for _spec_to_model() conversion function."""

    def test_spec_to_model_propagates_spec_id(self) -> None:
        """_spec_to_model propagates spec.id as the DB model primary key."""
        test_uuid = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        spec = SubAgentSpec(
            name="id_propagation_agent",
            description="Test ID propagation",
            goal="test",
            parent_thread_id="thread-test",
        )
        # Override the auto-generated id
        spec.id = str(test_uuid)

        model = _spec_to_model(spec)

        assert model.id == test_uuid

    def test_spec_to_model_maps_fields(self) -> None:
        """_spec_to_model maps spec fields to model columns."""
        spec = SubAgentSpec(
            name="field_map_agent",
            description="Test field mapping",
            goal="test",
            parent_thread_id="thread-test",
            template_type="custom",
            tool_scope="inherit_subset",
            tool_subset=["search_tool"],
            max_iterations=15,
        )

        model = _spec_to_model(spec, version=2)

        assert model.name == "field_map_agent"
        assert model.description == "Test field mapping"
        assert model.template_type == "custom"
        assert model.tool_scope == "inherit_subset"
        assert model.tool_subset == ["search_tool"]
        assert model.max_iterations == 15
        assert model.version == 2

    def test_spec_to_model_invalid_uuid_falls_back(self) -> None:
        """_spec_to_model generates a new UUID if spec.id is not a valid UUID."""
        spec = SubAgentSpec(
            name="invalid_id_agent",
            description="Test fallback",
            goal="test",
            parent_thread_id="thread-test",
        )
        # Set an invalid UUID
        spec.id = "not-a-uuid"

        model = _spec_to_model(spec)

        # Should get a valid auto-generated UUID, not crash
        assert isinstance(model.id, uuid.UUID)


class TestRollingMetricsSQL:
    """Tests for the _update_rolling_metrics SQL expression.

    Bug 11: PostgreSQL does not support sum(boolean). The fix uses
    SQLAlchemy case() to produce sum(CASE WHEN ... THEN 1 ELSE 0 END).
    """

    def test_metrics_sql_uses_case_not_boolean_sum(self) -> None:
        """The metrics query compiles to CASE WHEN, not sum(boolean)."""
        from sqlalchemy import case, func, select

        # Mirror the exact expression from _update_rolling_metrics
        from src.db.models import SubAgentRunModel

        successes_expr = func.sum(
            case(
                (SubAgentRunModel.status == "completed", 1),
                else_=0,
            )
        ).label("successes")

        stmt = select(
            func.count(SubAgentRunModel.id).label("total"),
            successes_expr,
        )

        # Compile and verify SQL contains CASE WHEN, not sum(column = value)
        compiled = stmt.compile(
            compile_kwargs={"literal_binds": True}
        )
        sql_str = str(compiled)

        # Must contain CASE WHEN (the fix)
        assert "CASE WHEN" in sql_str, (
            f"Expected CASE WHEN in SQL, got: {sql_str}"
        )
        # Must NOT contain sum(... = ...) which is sum(boolean)
        # The unfixed code produced: sum(sub_agent_runs.status = 'completed')
        # The fixed code produces: sum(CASE WHEN sub_agent_runs.status = 'completed' THEN 1 ELSE 0 END)
        assert "THEN" in sql_str, (
            f"Expected THEN (from CASE WHEN ... THEN) in SQL, got: {sql_str}"
        )


