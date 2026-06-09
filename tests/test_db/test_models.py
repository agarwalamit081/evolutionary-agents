"""Tests for src.db.models — SQLAlchemy ORM model defaults."""

from __future__ import annotations

from src.db.models import ColdMemory, CostLedger, ExecutionStep, TaskExecution, WarmMemory


class TestTaskExecution:
    """Tests for TaskExecution model defaults."""

    def test_table_name(self) -> None:
        assert TaskExecution.__tablename__ == "task_executions"


class TestExecutionStep:
    """Tests for ExecutionStep model defaults."""

    def test_table_name(self) -> None:
        assert ExecutionStep.__tablename__ == "execution_steps"


class TestWarmMemory:
    """Tests for WarmMemory model defaults."""

    def test_table_name(self) -> None:
        assert WarmMemory.__tablename__ == "warm_memories"


class TestColdMemory:
    """Tests for ColdMemory model defaults."""

    def test_table_name(self) -> None:
        assert ColdMemory.__tablename__ == "cold_memories"


class TestCostLedger:
    """Tests for CostLedger model defaults."""

    def test_table_name(self) -> None:
        assert CostLedger.__tablename__ == "cost_ledger"
