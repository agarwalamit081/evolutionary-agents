"""Tests for src.graph.nodes.memory — retrieve_memory and store_memory nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.enums import Phase
from src.graph.factory import initial_state
from src.graph.models import ReflectionResult
from src.graph.nodes.memory import retrieve_memory_node, store_memory_node

# PlanStep imported indirectly via conftest fixtures


class TestRetrieveMemoryNode:
    """Tests for retrieve_memory_node."""

    @pytest.mark.asyncio
    async def test_retrieve_no_memory_returns_empty(self) -> None:
        """No MemoryManager → empty retrieved_memories, phase=EXECUTE."""
        state = initial_state("test goal", "thread-no-mem")
        result = await retrieve_memory_node(state)

        assert result["phase"] == Phase.EXECUTE
        assert result["retrieved_memories"] == []

    @pytest.mark.asyncio
    async def test_retrieve_with_mock_returns_results(self, mock_memory: MagicMock) -> None:
        """MemoryManager returns results → retrieved_memories populated."""
        state = initial_state("test goal", "thread-mem")
        result = await retrieve_memory_node(state, memory=mock_memory)

        assert result["phase"] == Phase.EXECUTE
        assert len(result["retrieved_memories"]) > 0

    @pytest.mark.asyncio
    async def test_retrieve_memory_failure_no_crash(self) -> None:
        """Memory failure → no crash, empty memories returned."""
        failing_memory = MagicMock()
        failing_memory.retrieve_context = AsyncMock(side_effect=RuntimeError("connection lost"))

        state = initial_state("test goal", "thread-fail")
        result = await retrieve_memory_node(state, memory=failing_memory)

        assert result["phase"] == Phase.EXECUTE
        assert result["retrieved_memories"] == []

    @pytest.mark.asyncio
    async def test_retrieve_no_goal_returns_execute(self) -> None:
        """No goal → still returns EXECUTE phase with empty list."""
        state = initial_state("goal", "thread-nogoal")
        state["current_goal"] = None
        result = await retrieve_memory_node(state)

        assert result["phase"] == Phase.EXECUTE
        assert result["retrieved_memories"] == []

    @pytest.mark.asyncio
    async def test_retrieve_recalls_facts_into_memories(self) -> None:
        """Phase 5: durable facts are recalled and surfaced with tier='fact'."""
        memory = MagicMock()
        memory.retrieve_context = AsyncMock(return_value=[])
        memory.warm = MagicMock()
        memory.warm.retrieve = AsyncMock(return_value=[])  # evolved/folded empty
        memory.retrieve_skills = AsyncMock(return_value=[])  # skills empty
        memory.retrieve_facts = AsyncMock(
            return_value=[
                {"key": "row_count", "value": "1024 rows", "confidence": 0.9},
            ]
        )
        state = initial_state("how many rows", "thread-facts")
        result = await retrieve_memory_node(state, memory=memory)

        assert result["phase"] == Phase.EXECUTE
        fact_entries = [m for m in result["retrieved_memories"] if m.get("tier") == "fact"]
        assert len(fact_entries) == 1
        assert fact_entries[0]["content"] == "row_count: 1024 rows"
        assert fact_entries[0]["score"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_retrieve_recalls_skills_into_memories(self) -> None:
        """findings-05 C: skills are recalled semantically and surfaced with tier='skill'."""
        memory = MagicMock()
        memory.retrieve_context = AsyncMock(return_value=[])
        memory.warm = MagicMock()
        memory.warm.retrieve = AsyncMock(return_value=[])  # evolved/folded empty
        memory.retrieve_facts = AsyncMock(return_value=[])  # facts empty
        memory.retrieve_skills = AsyncMock(
            return_value=[
                {
                    "id": "skill-uuid-1",
                    "type": "skill",
                    "name": "utc_normalize",
                    "content": "pd.to_datetime(ts, utc=True)",
                    "fitness_score": 0.8,
                },
            ]
        )
        state = initial_state("convert timestamps to UTC", "thread-skills")
        result = await retrieve_memory_node(state, memory=memory)

        assert result["phase"] == Phase.EXECUTE
        skill_entries = [m for m in result["retrieved_memories"] if m.get("tier") == "skill"]
        assert len(skill_entries) == 1
        assert skill_entries[0]["content"] == "pd.to_datetime(ts, utc=True)"
        assert skill_entries[0]["score"] == pytest.approx(0.8)
        # The recalled skill's id is surfaced so store_memory_node can feed the
        # skill-fitness EMA (findings-05 D).
        assert result["recalled_skill_ids"] == ["skill-uuid-1"]



class TestStoreMemoryNode:
    """Tests for store_memory_node."""

    @pytest.mark.asyncio
    async def test_store_no_memory_complete(self, state_complete: dict) -> None:
        """No MemoryManager, is_complete=True → phase=COMPLETE."""
        result = await store_memory_node(state_complete)

        assert result["phase"] == Phase.COMPLETE

    @pytest.mark.asyncio
    async def test_store_no_memory_incomplete(self, sample_state: dict) -> None:
        """No MemoryManager, is_complete=False → phase=HITL_GATE."""
        state = dict(sample_state)
        state["is_complete"] = False
        result = await store_memory_node(state)

        assert result["phase"] == Phase.HITL_GATE

    @pytest.mark.asyncio
    async def test_store_with_observations(self, mock_memory: MagicMock, sample_state: dict) -> None:
        """Observations stored via memory.store_observation for each item."""
        state = dict(sample_state)
        state["memory_observations"] = ["obs1", "obs2", "obs3"]
        state["reflection"] = None
        state["is_complete"] = True

        result = await store_memory_node(state, memory=mock_memory)

        assert mock_memory.store_observation.call_count == 3
        assert result["phase"] == Phase.COMPLETE

    @pytest.mark.asyncio
    async def test_store_with_reflection_lessons(self, mock_memory: MagicMock, sample_state: dict) -> None:
        """Reflection with lessons → store_skill called."""
        state = dict(sample_state)
        state["memory_observations"] = []
        state["reflection"] = ReflectionResult(
            summary="test",
            lessons_learned=["lesson1", "lesson2"],
            confidence="medium",
            should_evolve=False,
            should_replan=False,
            memory_observations=[],
            cost_efficiency=1.0,
        )
        state["is_complete"] = True

        result = await store_memory_node(state, memory=mock_memory)

        mock_memory.store_skill.assert_called_once()
        assert result["phase"] == Phase.COMPLETE

    @pytest.mark.asyncio
    async def test_store_updates_skill_fitness_on_success(self, sample_state: dict) -> None:
        """findings-05 D: recalled skills get success=True when the run completes."""
        memory = MagicMock()
        memory.store_observation = AsyncMock(return_value=None)
        memory.store_skill = AsyncMock(return_value="uuid")
        update_fitness = AsyncMock(return_value=None)
        memory.update_skill_fitness = update_fitness

        state = dict(sample_state)
        state["memory_observations"] = []
        state["reflection"] = None
        state["is_complete"] = True
        state["recalled_skill_ids"] = ["s1", "s2"]

        result = await store_memory_node(state, memory=memory)

        assert result["phase"] == Phase.COMPLETE
        # Each recalled skill credited with the run's success.
        assert update_fitness.await_count == 2
        successes = [call.kwargs["success"] for call in update_fitness.await_args_list]
        assert successes == [True, True]

    @pytest.mark.asyncio
    async def test_store_updates_skill_fitness_on_failure(self, sample_state: dict) -> None:
        """findings-05 D: recalled skills get success=False (decay) on an incomplete run."""
        memory = MagicMock()
        memory.store_observation = AsyncMock(return_value=None)
        memory.store_skill = AsyncMock(return_value="uuid")
        update_fitness = AsyncMock(return_value=None)
        memory.update_skill_fitness = update_fitness

        state = dict(sample_state)
        state["memory_observations"] = []
        state["reflection"] = None
        state["is_complete"] = False
        state["recalled_skill_ids"] = ["s1"]

        result = await store_memory_node(state, memory=memory)

        assert result["phase"] == Phase.HITL_GATE
        assert update_fitness.await_count == 1
        assert update_fitness.await_args.kwargs["success"] is False

    @pytest.mark.asyncio
    async def test_store_no_recalled_skills_skips_fitness(self, sample_state: dict) -> None:
        """No recalled skills → update_skill_fitness never called."""
        memory = MagicMock()
        memory.store_observation = AsyncMock(return_value=None)
        memory.store_skill = AsyncMock(return_value="uuid")
        memory.update_skill_fitness = AsyncMock(return_value=None)

        state = dict(sample_state)
        state["memory_observations"] = []
        state["reflection"] = None
        state["is_complete"] = True
        # No recalled_skill_ids in state.

        result = await store_memory_node(state, memory=memory)

        assert result["phase"] == Phase.COMPLETE
        memory.update_skill_fitness.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_observation_failure_graceful(self, sample_state: dict) -> None:
        """First store_observation raises → no crash, remaining stored."""
        failing_memory = MagicMock()
        call_count = 0

        async def _store_obs(**_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("write failed")

        failing_memory.store_observation = AsyncMock(side_effect=_store_obs)
        failing_memory.store_skill = AsyncMock(return_value="uuid")

        state = dict(sample_state)
        state["memory_observations"] = ["obs1", "obs2"]
        state["reflection"] = None
        state["is_complete"] = True

        result = await store_memory_node(state, memory=failing_memory)

        assert result["phase"] == Phase.COMPLETE

    @pytest.mark.asyncio
    async def test_store_empty_observations_no_reflection(self, sample_state: dict) -> None:
        """Empty observations + no reflection → succeeds with correct phase."""
        state = dict(sample_state)
        state["memory_observations"] = []
        state["reflection"] = None
        state["is_complete"] = True

        result = await store_memory_node(state)

        assert result["phase"] == Phase.COMPLETE


class TestStoreMemoryIntentFacts:
    """Feature E: persist classify's refined_intent as a durable fact (opt-in)."""

    def _memory(self) -> MagicMock:
        memory = MagicMock()
        memory.store_observation = AsyncMock(return_value=None)
        memory.store_skill = AsyncMock(return_value="uuid")
        memory.update_skill_fitness = AsyncMock(return_value=None)
        memory.store_fact = AsyncMock(return_value="fact-uuid")
        return memory

    @pytest.mark.asyncio
    async def test_persists_when_flag_on(
        self, sample_state: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import get_settings

        monkeypatch.setattr(get_settings().agent, "persist_intent_facts", True)

        memory = self._memory()
        state = dict(sample_state)
        state["memory_observations"] = []
        state["reflection"] = None
        state["is_complete"] = True
        state["refined_intent"] = "the user wants a UTC-normalized event log"
        state["ambiguity_type"] = "scope"

        await store_memory_node(state, memory=memory)

        memory.store_fact.assert_awaited_once()
        _, kwargs = memory.store_fact.call_args
        assert kwargs["value"] == "the user wants a UTC-normalized event log"
        assert kwargs["source"] == "classify"
        assert kwargs["tags"] == ["intent", "scope"]
        assert kwargs["key"].startswith("intent::")

    @pytest.mark.asyncio
    async def test_noop_when_flag_off(
        self, sample_state: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import get_settings

        monkeypatch.setattr(get_settings().agent, "persist_intent_facts", False)

        memory = self._memory()
        state = dict(sample_state)
        state["memory_observations"] = []
        state["reflection"] = None
        state["is_complete"] = True
        state["refined_intent"] = "should not be persisted with the flag off"

        await store_memory_node(state, memory=memory)

        memory.store_fact.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_refined_intent_empty(
        self, sample_state: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The heuristic classify path yields an empty refined_intent — nothing
        to persist even with the flag on."""
        from src.config import get_settings

        monkeypatch.setattr(get_settings().agent, "persist_intent_facts", True)

        memory = self._memory()
        state = dict(sample_state)
        state["memory_observations"] = []
        state["reflection"] = None
        state["is_complete"] = True
        state["refined_intent"] = ""

        await store_memory_node(state, memory=memory)

        memory.store_fact.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_store_fact_failure_isolated(
        self, sample_state: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fact-store hiccup never aborts the terminal sink (resilience)."""
        from src.config import get_settings

        monkeypatch.setattr(get_settings().agent, "persist_intent_facts", True)

        memory = self._memory()
        memory.store_fact = AsyncMock(side_effect=RuntimeError("db down"))
        state = dict(sample_state)
        state["memory_observations"] = []
        state["reflection"] = None
        state["is_complete"] = True
        state["refined_intent"] = "intent that cannot be stored"

        result = await store_memory_node(state, memory=memory)

        assert result["phase"] == Phase.COMPLETE
