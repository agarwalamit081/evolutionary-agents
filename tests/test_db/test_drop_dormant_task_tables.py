"""Regression tests for the dormant-table drop migration (m7e8f9a0b1c2).

``task_executions`` + ``feedback_events`` were fully dormant (0 rows, 0 readers,
0 writers — ``execution_steps`` is the live per-node-timing table keyed on
``run_id``, and ``RunStatusStore`` is the live run-status store). Migration
``m7e8f9a0b1c2`` drops both tables plus the four now-orphaned FK columns
(``execution_steps.task_id``, ``cost_ledger.task_id``, ``warm_memories.
source_task_id``, ``sub_agent_runs.parent_task_id``) and the FK constraints /
task_id indexes that pointed at them.

These tests guard the *regression*: the dormant classes/columns must not
silently re-appear in the ORM, and the migration must chain onto the prior head
(``l4d5e6f7a8b9``). The migration's upgrade/downgrade are additionally exercised
end-to-end on the live Postgres container during the rebuild step
(``alembic upgrade head`` → ``downgrade -1`` → ``upgrade head``); the FK
``drop_constraint`` step is not SQLite-portable, so the live DB is the authority
for execution while this file is the CI guard for schema shape.
"""

from __future__ import annotations

import importlib

import pytest

from src.db import models
from src.db.models import Base


MIGRATION_MODULE = "src.db.migrations.versions.m7e8f9a0b1c2_drop_dormant_task_tables"


def _col_names(table_name: str) -> set[str]:
    table = Base.metadata.tables[table_name]
    return {c.name for c in table.columns}


def _index_names(table_name: str) -> set[str]:
    table = Base.metadata.tables[table_name]
    return {str(ix.name) for ix in table.indexes}


class TestDormantTablesRemoved:
    """The two dormant tables + their ORM classes are gone for good."""

    def test_task_executions_table_removed_from_metadata(self) -> None:
        assert "task_executions" not in Base.metadata.tables

    def test_feedback_events_table_removed_from_metadata(self) -> None:
        assert "feedback_events" not in Base.metadata.tables

    def test_task_execution_class_removed(self) -> None:
        assert not hasattr(models, "TaskExecution")

    def test_feedback_event_class_removed(self) -> None:
        assert not hasattr(models, "FeedbackEvent")


class TestOrphanColumnsRemoved:
    """The four FK columns that pointed at task_executions are dropped."""

    def test_execution_steps_task_id_removed(self) -> None:
        assert "task_id" not in _col_names("execution_steps")
        # run_id (the live attribution key) is unaffected.
        assert "run_id" in _col_names("execution_steps")

    def test_cost_ledger_task_id_removed(self) -> None:
        assert "task_id" not in _col_names("cost_ledger")
        assert "run_id" in _col_names("cost_ledger")

    def test_warm_memories_source_task_id_removed(self) -> None:
        assert "source_task_id" not in _col_names("warm_memories")

    def test_sub_agent_runs_parent_task_id_removed(self) -> None:
        assert "parent_task_id" not in _col_names("sub_agent_runs")
        # parent_thread_id (the live sub-agent attribution key) is unaffected.
        assert "parent_thread_id" in _col_names("sub_agent_runs")


class TestOrphanIndexesRemoved:
    """Indexes that referenced the dropped columns are gone."""

    def test_execution_steps_task_indexes_removed(self) -> None:
        idxs = _index_names("execution_steps")
        assert "idx_execution_steps_task_number" not in idxs
        assert "idx_execution_steps_failed" not in idxs
        # The phase index is unrelated and must survive.
        assert "idx_execution_steps_phase" in idxs

    def test_cost_ledger_task_index_removed(self) -> None:
        idxs = _index_names("cost_ledger")
        assert "idx_cost_ledger_task" not in idxs
        # The run_id attribution index must survive.
        assert "idx_cost_ledger_run" in idxs


class TestMigrationChain:
    """The drop migration chains onto the prior head and is reversible."""

    def test_revision_and_down_revision(self) -> None:
        migration = importlib.import_module(MIGRATION_MODULE)
        assert migration.down_revision == "l4d5e6f7a8b9"
        assert migration.revision == "m7e8f9a0b1c2"

    def test_upgrade_and_downgrade_are_callable(self) -> None:
        migration = importlib.import_module(MIGRATION_MODULE)
        assert callable(migration.upgrade)
        assert callable(migration.downgrade)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
