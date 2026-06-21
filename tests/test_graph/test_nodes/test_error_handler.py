"""Tests for src.graph.nodes.error_handler — error handler node function."""

from __future__ import annotations

import pytest

from src.graph.enums import Phase
from src.graph.factory import initial_state
from src.graph.nodes.error_handler import error_handler_node


class TestErrorHandlerNode:
    """Tests for the error_handler_node async function."""

    @pytest.mark.asyncio
    async def test_error_handler_budget_routes_to_hitl(self) -> None:
        """Budget exhausted → phase=HITL_GATE."""
        state = initial_state("test goal", "thread-budget")
        state["errors"] = ["budget limit reached"]
        result = await error_handler_node(state)

        assert result["phase"] == Phase.HITL_GATE

    @pytest.mark.asyncio
    async def test_error_handler_max_iterations_routes_to_complete(self) -> None:
        """Max iterations exceeded → phase=COMPLETE."""
        state = initial_state("test goal", "thread-maxiter")
        state["errors"] = ["too many attempts"]
        state["iteration_count"] = 25
        state["max_iterations"] = 25
        result = await error_handler_node(state)

        assert result["phase"] == Phase.COMPLETE
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_error_handler_auth_error_routes_to_classify(self) -> None:
        """Auth error (401) → phase=CLASSIFY."""
        state = initial_state("test goal", "thread-auth")
        state["errors"] = ["401 unauthorized: invalid API key"]
        result = await error_handler_node(state)

        assert result["phase"] == Phase.CLASSIFY

    @pytest.mark.asyncio
    async def test_error_handler_rate_limit_routes_to_execute(self) -> None:
        """Rate limit (429) → phase=EXECUTE."""
        state = initial_state("test goal", "thread-rate")
        state["errors"] = ["429 rate limit exceeded"]
        result = await error_handler_node(state)

        assert result["phase"] == Phase.EXECUTE

    @pytest.mark.asyncio
    async def test_error_handler_generic_routes_to_execute(self) -> None:
        """Generic error → phase=EXECUTE (retry)."""
        state = initial_state("test goal", "thread-generic")
        state["errors"] = ["something went wrong unexpectedly"]
        result = await error_handler_node(state)

        assert result["phase"] == Phase.EXECUTE

    @pytest.mark.asyncio
    async def test_error_handler_no_errors_routes_to_verify(self) -> None:
        """No errors is a routing anomaly, not a success — route to verify so the
        actual state is judged honestly. Previously this declared is_complete=True
        for a run that may have produced nothing (F14)."""
        state = initial_state("test goal", "thread-noerr")
        state["errors"] = []
        result = await error_handler_node(state)

        assert result["phase"] == Phase.VERIFY
        assert result.get("is_complete") is not True

    @pytest.mark.asyncio
    async def test_error_handler_403_routes_to_classify(self) -> None:
        """403 forbidden → phase=CLASSIFY."""
        state = initial_state("test goal", "thread-403")
        state["errors"] = ["403 forbidden: access denied"]
        result = await error_handler_node(state)

        assert result["phase"] == Phase.CLASSIFY
