"""Tests for src.evolution.persister — chain/mutation/telemetry persistence.

Mock-session unit tests mirroring the cost_tracker mock_session pattern. The
real ORM models (MutationChain/Mutation/EvolutionTelemetry) are instantiated
(constructors are DB-free); only get_session() is faked so no DB is touched.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.models import EvolutionTelemetry, Mutation, MutationChain
from src.evolution.persister import (
    EvolutionPersister,
    _coerce_mutation_type,
    _json_safe,
)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestJsonSafe:
    """_json_safe round-trips primitives and coerces non-serializable values."""

    def test_primitives_pass_through(self) -> None:
        payload = {"a": 1, "b": [1, 2], "c": "x", "d": None, "e": True}
        assert _json_safe(payload) == payload

    def test_enum_coerced_safely(self) -> None:
        from src.graph.enums import MutationType

        result = _json_safe({"type": MutationType.CODE})
        # Enum renders to a str regardless of whether it is str-mixin or plain.
        assert isinstance(result["type"], str)

    def test_none_passes_through(self) -> None:
        assert _json_safe(None) is None


class TestCoerceMutationType:
    """_coerce_mutation_type maps enum/None/str to a plain string."""

    def test_enum_yields_value(self) -> None:
        from src.graph.enums import MutationType

        assert _coerce_mutation_type(MutationType.CODE) == MutationType.CODE.value

    def test_none_becomes_unknown(self) -> None:
        assert _coerce_mutation_type(None) == "unknown"

    def test_str_unchanged(self) -> None:
        assert _coerce_mutation_type("custom_type") == "custom_type"


# ---------------------------------------------------------------------------
# Mock-session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session() -> MagicMock:
    """Mock AsyncSession. ``add`` simulates flush populating the PK default."""
    session = MagicMock()

    def _add(obj: Any) -> None:
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()

    session.add = MagicMock(side_effect=_add)
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def persister(mock_session: MagicMock) -> Generator[EvolutionPersister, None, None]:
    """An EvolutionPersister whose get_session() yields the mock session."""

    @asynccontextmanager
    async def _fake_get_session() -> AsyncGenerator[MagicMock, None]:
        yield mock_session

    with patch("src.db.session.get_session", new=_fake_get_session):
        yield EvolutionPersister()


# ---------------------------------------------------------------------------
# create_chain
# ---------------------------------------------------------------------------


class TestCreateChain:
    @pytest.mark.asyncio
    async def test_creates_in_progress_chain(
        self, persister: EvolutionPersister, mock_session: MagicMock
    ) -> None:
        chain_id = await persister.create_chain("test trigger", {"priority": "high"})

        assert isinstance(chain_id, uuid.UUID)
        mock_session.add.assert_called_once()
        added = mock_session.add.call_args[0][0]
        assert isinstance(added, MutationChain)
        assert added.status == "in_progress"
        assert added.trigger_reason == "test trigger"
        assert added.extra_data == {"priority": "high"}
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exception_returns_none(self, mock_session: MagicMock) -> None:
        mock_session.flush = AsyncMock(side_effect=RuntimeError("db down"))

        @asynccontextmanager
        async def _fake() -> AsyncGenerator[MagicMock, None]:
            yield mock_session

        with patch("src.db.session.get_session", new=_fake):
            result = await EvolutionPersister().create_chain("trigger")

        assert result is None


# ---------------------------------------------------------------------------
# record_mutation
# ---------------------------------------------------------------------------


class TestRecordMutation:
    @pytest.mark.asyncio
    async def test_none_chain_returns_none(
        self, persister: EvolutionPersister, mock_session: MagicMock
    ) -> None:
        """Mutation.chain_id is NOT NULL — a None chain is a no-op."""
        result = await persister.record_mutation(
            None, {"mutation_type": "code", "mutated_content": "x"}
        )
        assert result is None
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_records_mutation_with_coerced_enum(
        self,
        persister: EvolutionPersister,
        mock_session: MagicMock,
    ) -> None:
        from src.graph.enums import MutationType

        chain_id = uuid.uuid4()
        proposal = {
            "mutation_type": MutationType.CODE,
            "target_path": "src/x.py",
            "description": "improve",
            "original_content": "old",
            "mutated_content": "new",
            "model_used": "test-model",
            "tokens_used": 42,
        }
        result = await persister.record_mutation(chain_id, proposal, status="rejected")

        assert isinstance(result, uuid.UUID)
        added = mock_session.add.call_args[0][0]
        assert isinstance(added, Mutation)
        assert added.chain_id == chain_id
        assert added.mutation_type == MutationType.CODE.value  # enum → .value
        assert added.status == "rejected"
        assert added.tokens_used == 42
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_string_model_used_becomes_none(
        self,
        persister: EvolutionPersister,
        mock_session: MagicMock,
    ) -> None:
        """A non-str model_used (e.g. None from heuristic) is stored as None."""
        await persister.record_mutation(
            uuid.uuid4(),
            {"mutation_type": "code", "mutated_content": "x", "model_used": None},
        )
        added = mock_session.add.call_args[0][0]
        assert added.model_used is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(
        self, persister: EvolutionPersister, mock_session: MagicMock
    ) -> None:
        mock_session.flush = AsyncMock(side_effect=RuntimeError("db down"))
        result = await persister.record_mutation(
            uuid.uuid4(), {"mutation_type": "code", "mutated_content": "x"}
        )
        assert result is None


# ---------------------------------------------------------------------------
# update_mutation_status
# ---------------------------------------------------------------------------


class TestUpdateMutationStatus:
    @pytest.mark.asyncio
    async def test_none_mutation_returns_false(
        self, persister: EvolutionPersister, mock_session: MagicMock
    ) -> None:
        assert await persister.update_mutation_status(None, "deployed") is False
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_updates_status(
        self, persister: EvolutionPersister, mock_session: MagicMock
    ) -> None:
        assert await persister.update_mutation_status(uuid.uuid4(), "deployed") is True
        mock_session.execute.assert_awaited_once()
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exception_returns_false(
        self, persister: EvolutionPersister, mock_session: MagicMock
    ) -> None:
        mock_session.execute = AsyncMock(side_effect=RuntimeError("db down"))
        assert await persister.update_mutation_status(uuid.uuid4(), "deployed") is False


# ---------------------------------------------------------------------------
# record_event
# ---------------------------------------------------------------------------


class TestRecordEvent:
    @pytest.mark.asyncio
    async def test_records_event_with_chain(
        self,
        persister: EvolutionPersister,
        mock_session: MagicMock,
    ) -> None:
        chain_id = uuid.uuid4()
        await persister.record_event(chain_id, "generation_attempt", {"attempt": 1})

        added = mock_session.add.call_args[0][0]
        assert isinstance(added, EvolutionTelemetry)
        assert added.chain_id == chain_id
        assert added.event_type == "generation_attempt"
        assert added.event_data == {"attempt": 1}

    @pytest.mark.asyncio
    async def test_accepts_none_chain_id(
        self,
        persister: EvolutionPersister,
        mock_session: MagicMock,
    ) -> None:
        """EvolutionTelemetry.chain_id is nullable — None is allowed."""
        await persister.record_event(None, "validation_result", {"passed": False})
        added = mock_session.add.call_args[0][0]
        assert added.chain_id is None

    @pytest.mark.asyncio
    async def test_default_event_data_empty_dict(
        self,
        persister: EvolutionPersister,
        mock_session: MagicMock,
    ) -> None:
        await persister.record_event(uuid.uuid4(), "deployed")
        added = mock_session.add.call_args[0][0]
        assert added.event_data == {}

    @pytest.mark.asyncio
    async def test_exception_swallowed(
        self, persister: EvolutionPersister, mock_session: MagicMock
    ) -> None:
        """Telemetry failures must never propagate."""
        mock_session.flush = AsyncMock(side_effect=RuntimeError("db down"))
        # Should not raise.
        await persister.record_event(uuid.uuid4(), "deployed", {"x": 1})


# ---------------------------------------------------------------------------
# complete_chain
# ---------------------------------------------------------------------------


class TestCompleteChain:
    @pytest.mark.asyncio
    async def test_none_chain_returns_false(
        self, persister: EvolutionPersister, mock_session: MagicMock
    ) -> None:
        assert await persister.complete_chain(None, "deployed") is False
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_completes_chain(
        self, persister: EvolutionPersister, mock_session: MagicMock
    ) -> None:
        assert await persister.complete_chain(uuid.uuid4(), "deployed") is True
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exception_returns_false(
        self, persister: EvolutionPersister, mock_session: MagicMock
    ) -> None:
        mock_session.execute = AsyncMock(side_effect=RuntimeError("db down"))
        assert await persister.complete_chain(uuid.uuid4(), "deployed") is False
