"""Tests for src.graph.nodes.evolve — evolve node function."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.enums import Confidence, MutationType, Phase
from src.graph.factory import initial_state
from src.graph.models import ReflectionResult
from src.graph.nodes.evolve import (
    _derive_input_schema,
    _sanitize_tool_name,
    _synthesize_self_test,
    _try_register_deployed_tool,
    evolve_node,
)


class TestEvolveNode:
    """Tests for the evolve_node async function."""

    @pytest.mark.asyncio
    async def test_evolve_no_gateway_skips(self) -> None:
        """No gateway -> phase=STORE_MEMORY, outcome=skipped_no_gateway."""
        state = initial_state("test goal", "thread-nogw")
        state["generation"] = 0
        result = await evolve_node(state)

        assert result["phase"] == Phase.STORE_MEMORY
        record = result["evolution_history"][0]
        assert record["outcome"] == "skipped_no_gateway"

    @pytest.mark.asyncio
    async def test_evolve_increments_generation(self) -> None:
        """Each call increments generation by 1."""
        state = initial_state("test goal", "thread-gen")
        state["generation"] = 5
        result = await evolve_node(state)

        assert result["generation"] == 6

    @pytest.mark.asyncio
    async def test_evolve_records_reflection_summary(self) -> None:
        """Reflection summary is recorded in evolution history."""
        state = initial_state("test goal", "thread-refl")
        state["generation"] = 0
        state["reflection"] = ReflectionResult(
            summary="Reflection summary text",
            lessons_learned=["lesson1"],
            confidence="high",
            should_evolve=True,
            should_replan=False,
            memory_observations=[],
            cost_efficiency=1.0,
        )
        result = await evolve_node(state)

        record = result["evolution_history"][0]
        assert record["summary"] == "Reflection summary text"
        assert record["lessons"] == ["lesson1"]

    @pytest.mark.asyncio
    async def test_evolve_no_reflection_uses_default(self) -> None:
        """No reflection -> summary defaults to 'no reflection'."""
        state = initial_state("test goal", "thread-norefl")
        state["generation"] = 0
        state["reflection"] = None
        result = await evolve_node(state)

        record = result["evolution_history"][0]
        assert record["summary"] == "no reflection"
        assert record["lessons"] == []

    @pytest.mark.asyncio
    async def test_evolve_always_returns_store_memory_phase(self) -> None:
        """All paths return phase=STORE_MEMORY."""
        state = initial_state("test goal", "thread-phase")
        state["generation"] = 0
        result = await evolve_node(state)

        assert result["phase"] == Phase.STORE_MEMORY

    @pytest.mark.asyncio
    async def test_evolve_no_gateway_records_skip(self) -> None:
        """With no gateway, records 'skipped_no_gateway' evolution outcome."""
        state = initial_state("test goal", "thread-skip")
        state["generation"] = 2
        state["reflection"] = ReflectionResult(
            summary="good run",
            lessons_learned=["tried fast path"],
            confidence=Confidence.HIGH,
            should_evolve=True,
            should_replan=False,
            memory_observations=[],
            cost_efficiency=1.0,
        )
        result = await evolve_node(state)

        record = result["evolution_history"][0]
        assert record["outcome"] == "skipped_no_gateway"
        assert record["generation"] == 2
        assert record["summary"] == "good run"
        assert record["lessons"] == ["tried fast path"]

    @pytest.mark.asyncio
    async def test_evolve_with_mock_gateway(self) -> None:
        """Mock gateway + patched SelfEvolutionEngine verifies engine is called correctly."""
        mock_gateway = MagicMock()

        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 1,
            "mutations_deployed": 1,
            "proposal": {"rationale": "improve speed"},
            "deployment": {"commit_hash": "abc123def456"},
        }

        mock_engine_instance = MagicMock()
        mock_engine_instance.run_cycle = AsyncMock(return_value=cycle_result)

        with patch("src.evolution.engine.SelfEvolutionEngine", return_value=mock_engine_instance), \
             patch("src.safety.pipeline.SafetyPipeline"), \
             patch("src.sandbox.executor.SandboxExecutor", side_effect=Exception("no sandbox")), \
             patch("src.evolution.git_tracker.GitTracker", side_effect=Exception("no git")), \
             patch("src.config.get_settings") as mock_settings:

            mock_settings.return_value = MagicMock()

            state = initial_state("test goal", "thread-mockgw")
            state["generation"] = 3
            state["execution_history"] = [{"tool": "search", "duration_ms": 100}]
            state["reflection"] = ReflectionResult(
                summary="decent performance",
                lessons_learned=["optimize search"],
                confidence=Confidence.MEDIUM,
                should_evolve=True,
                should_replan=False,
                memory_observations=[],
                cost_efficiency=0.9,
            )

            result = await evolve_node(state, gateway=mock_gateway)

            assert result["phase"] == Phase.STORE_MEMORY
            assert result["generation"] == 4
            # Verify engine.run_cycle was called with the correct params
            mock_engine_instance.run_cycle.assert_awaited_once()
            call_kwargs = mock_engine_instance.run_cycle.call_args
            assert call_kwargs.kwargs["execution_history"] == state["execution_history"]
            assert call_kwargs.kwargs["reflection"] == state["reflection"]
            assert call_kwargs.kwargs["sandbox"] is None
            assert call_kwargs.kwargs["git_tracker"] is None

    @pytest.mark.asyncio
    async def test_evolve_records_commit_hash(self) -> None:
        """Mock engine returns a cycle result with commit_hash, verify it's in evolution_history."""
        mock_gateway = MagicMock()

        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 1,
            "mutations_deployed": 1,
            "proposal": {"rationale": "speed up classify"},
            "deployment": {"commit_hash": "deadbeef1234"},
        }

        mock_engine_instance = MagicMock()
        mock_engine_instance.run_cycle = AsyncMock(return_value=cycle_result)

        with patch("src.evolution.engine.SelfEvolutionEngine", return_value=mock_engine_instance), \
             patch("src.safety.pipeline.SafetyPipeline"), \
             patch("src.sandbox.executor.SandboxExecutor", side_effect=Exception("no sandbox")), \
             patch("src.evolution.git_tracker.GitTracker", side_effect=Exception("no git")), \
             patch("src.config.get_settings") as mock_settings:

            mock_settings.return_value = MagicMock()

            state = initial_state("test goal", "thread-commit")
            state["generation"] = 1
            state["execution_history"] = []

            result = await evolve_node(state, gateway=mock_gateway)

            record = result["evolution_history"][0]
            assert record["commit_hash"] == "deadbeef1234"
            assert record["rationale"] == "speed up classify"

    @pytest.mark.asyncio
    async def test_evolve_records_outcome(self) -> None:
        """Outcome from cycle result is recorded in evolution_history."""
        mock_gateway = MagicMock()

        cycle_result = {
            "status": "validation_failed",
            "deployed": False,
            "mutations_proposed": 1,
            "mutations_deployed": 0,
            "proposal": {"rationale": "try tool optimization"},
            "deployment": {},
        }

        mock_engine_instance = MagicMock()
        mock_engine_instance.run_cycle = AsyncMock(return_value=cycle_result)

        with patch("src.evolution.engine.SelfEvolutionEngine", return_value=mock_engine_instance), \
             patch("src.safety.pipeline.SafetyPipeline"), \
             patch("src.sandbox.executor.SandboxExecutor", side_effect=Exception("no sandbox")), \
             patch("src.evolution.git_tracker.GitTracker", side_effect=Exception("no git")), \
             patch("src.config.get_settings") as mock_settings:

            mock_settings.return_value = MagicMock()

            state = initial_state("test goal", "thread-outcome")
            state["generation"] = 4
            state["execution_history"] = []

            result = await evolve_node(state, gateway=mock_gateway)

            record = result["evolution_history"][0]
            assert record["outcome"] == "validation_failed"
            assert record["mutations_proposed"] == 1
            assert record["mutations_deployed"] == 0

    @pytest.mark.asyncio
    async def test_evolve_engine_failure_falls_back(self) -> None:
        """When engine raises exception, falls back to skip record."""
        mock_gateway = MagicMock()

        mock_engine_instance = MagicMock()
        mock_engine_instance.run_cycle = AsyncMock(side_effect=RuntimeError("database connection failed"))

        with patch("src.evolution.engine.SelfEvolutionEngine", return_value=mock_engine_instance), \
             patch("src.safety.pipeline.SafetyPipeline"), \
             patch("src.sandbox.executor.SandboxExecutor", side_effect=Exception("no sandbox")), \
             patch("src.evolution.git_tracker.GitTracker", side_effect=Exception("no git")), \
             patch("src.config.get_settings") as mock_settings:

            mock_settings.return_value = MagicMock()

            state = initial_state("test goal", "thread-enginefail")
            state["generation"] = 7
            state["execution_history"] = []

            result = await evolve_node(state, gateway=mock_gateway)

            # Falls back to the no-gateway path when engine throws
            assert result["phase"] == Phase.STORE_MEMORY
            record = result["evolution_history"][0]
            assert record["outcome"] == "skipped_no_gateway"
            assert result["generation"] == 8

    @pytest.mark.asyncio
    async def test_evolve_records_mutations_counts_from_cycle(self) -> None:
        """Mutations proposed and deployed counts are recorded from cycle result."""
        mock_gateway = MagicMock()

        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 3,
            "mutations_deployed": 2,
            "proposal": {"rationale": "batch optimization"},
            "deployment": {"commit_hash": "beefcafe5678"},
        }

        mock_engine_instance = MagicMock()
        mock_engine_instance.run_cycle = AsyncMock(return_value=cycle_result)

        with patch("src.evolution.engine.SelfEvolutionEngine", return_value=mock_engine_instance), \
             patch("src.safety.pipeline.SafetyPipeline"), \
             patch("src.sandbox.executor.SandboxExecutor", side_effect=Exception("no sandbox")), \
             patch("src.evolution.git_tracker.GitTracker", side_effect=Exception("no git")), \
             patch("src.config.get_settings") as mock_settings:

            mock_settings.return_value = MagicMock()

            state = initial_state("test goal", "thread-counts")
            state["generation"] = 0
            state["execution_history"] = []

            result = await evolve_node(state, gateway=mock_gateway)

            record = result["evolution_history"][0]
            assert record["mutations_proposed"] == 3
            assert record["mutations_deployed"] == 2
            assert record["commit_hash"] == "beefcafe5678"


# A deployed TOOL mutation's runnable artifact: exactly ONE async def (+ helpers),
# target_path ending in .py — the shape the LLM-gen path emits (see
# engine._CODE_EMITTING_MUTATIONS). Used by the Phase-4-E node tests below.
_TOOL_HANDLER = (
    "async def csv_normalizer(path: str, delimiter: str = ','):\n"
    "    return path\n"
)
_TOOL_TARGET = "evolution/tools/csv_normalizer.py"


@contextmanager
def _patch_engine(cycle_result: dict[str, Any], *, reexecute_flag: bool):
    """Patch the evolution engine + gates for deterministic node tests.

    Yields ``(engine_instance, mock_settings)``. ``reexecute_flag`` sets the
    opt-in ``EvolutionSettings.evolution_reexecute_tool`` value the node reads.
    """
    with patch("src.evolution.engine.SelfEvolutionEngine") as mock_engine_cls, \
         patch("src.safety.pipeline.SafetyPipeline"), \
         patch("src.sandbox.executor.SandboxExecutor", side_effect=Exception("no sandbox")), \
         patch("src.evolution.git_tracker.GitTracker", side_effect=Exception("no git")), \
         patch("src.config.get_settings") as mock_settings:
        instance = MagicMock()
        instance.run_cycle = AsyncMock(return_value=cycle_result)
        mock_engine_cls.return_value = instance
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.evolution.evolution_reexecute_tool = reexecute_flag
        yield instance, mock_settings


class TestEvolveReexecuteEdge:
    """Phase 4 E — evolve→execute edge for deployed TOOL mutations."""

    @pytest.mark.asyncio
    async def test_tool_deploy_live_registers_and_signals_reexecute(self) -> None:
        """TOOL .py deploy + registered + guard unset → offered & done True."""
        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 1,
            "mutations_deployed": 1,
            "proposal": {
                "mutation_type": MutationType.TOOL,
                "target_path": _TOOL_TARGET,
                "mutated_content": _TOOL_HANDLER,
                "description": "normalize CSV columns",
                "rationale": "fill a capability gap",
            },
            "deployment": {"commit_hash": "abc123"},
        }
        with _patch_engine(cycle_result, reexecute_flag=True), \
                patch(
                    "src.graph.nodes.evolve._try_register_deployed_tool",
                    new=AsyncMock(return_value=True),
                ) as mock_register:
            state = initial_state("clean a csv", "thread-reexec-ok")
            result = await evolve_node(state, gateway=MagicMock(), tools=MagicMock())

        # The deployed tool was handed to the live-registration helper.
        mock_register.assert_awaited_once()
        assert result["evolve_reexecute_offered"] is True
        assert result["evolve_reexecute_done"] is True
        record = result["evolution_history"][0]
        assert record["mutation_type"] == "tool"
        assert record["reexecute_registered_tool"] == _TOOL_TARGET

    @pytest.mark.asyncio
    async def test_tool_deploy_registration_failure_no_reexecute(self) -> None:
        """Live registration returns False → no offer, guard stays unset."""
        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 1,
            "mutations_deployed": 1,
            "proposal": {
                "mutation_type": MutationType.TOOL,
                "target_path": _TOOL_TARGET,
                "mutated_content": _TOOL_HANDLER,
                "description": "normalize CSV columns",
            },
            "deployment": {"commit_hash": "abc123"},
        }
        with _patch_engine(cycle_result, reexecute_flag=True), \
                patch(
                    "src.graph.nodes.evolve._try_register_deployed_tool",
                    new=AsyncMock(return_value=False),
                ) as mock_register:
            state = initial_state("clean a csv", "thread-reexec-fail")
            result = await evolve_node(state, gateway=MagicMock(), tools=MagicMock())

        mock_register.assert_awaited_once()
        assert result["evolve_reexecute_offered"] is False
        assert result["evolve_reexecute_done"] is False
        assert "reexecute_registered_tool" not in result["evolution_history"][0]

    @pytest.mark.asyncio
    async def test_prompt_deploy_never_reexecutes(self) -> None:
        """PROMPT mutations never reach live registration → store_memory path."""
        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 1,
            "mutations_deployed": 1,
            "proposal": {
                "mutation_type": MutationType.PROMPT,
                "target_path": "prompts/system_prompt.md",
                "mutated_content": "refined prompt",
                "description": "sharpen execute prompt",
            },
            "deployment": {"commit_hash": "prompt-hash"},
        }
        with _patch_engine(cycle_result, reexecute_flag=True), \
                patch(
                    "src.graph.nodes.evolve._try_register_deployed_tool",
                    new=AsyncMock(return_value=True),
                ) as mock_register:
            state = initial_state("any", "thread-prompt")
            result = await evolve_node(state, gateway=MagicMock(), tools=MagicMock())

        mock_register.assert_not_awaited()
        assert result["evolve_reexecute_offered"] is False
        assert result["evolve_reexecute_done"] is False
        assert result["evolution_history"][0]["mutation_type"] == "prompt"

    @pytest.mark.asyncio
    async def test_config_json_tool_target_not_py_no_reexecute(self) -> None:
        """Heuristic TOOL (config-JSON target) is excluded from re-execution."""
        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 1,
            "mutations_deployed": 1,
            "proposal": {
                "mutation_type": MutationType.TOOL,
                "target_path": "evolution/tool_config.json",  # not a runnable module
                "mutated_content": '{"tool": "x"}',
                "description": "tweak config",
            },
            "deployment": {"commit_hash": "cfg-hash"},
        }
        with _patch_engine(cycle_result, reexecute_flag=True), \
                patch(
                    "src.graph.nodes.evolve._try_register_deployed_tool",
                    new=AsyncMock(return_value=True),
                ) as mock_register:
            state = initial_state("any", "thread-cfgjson")
            result = await evolve_node(state, gateway=MagicMock(), tools=MagicMock())

        mock_register.assert_not_awaited()
        assert result["evolve_reexecute_offered"] is False
        assert result["evolve_reexecute_done"] is False

    @pytest.mark.asyncio
    async def test_reexecute_done_guard_blocks_second_offer(self) -> None:
        """Once evolve_reexecute_done is True, a later cycle never re-offers."""
        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 1,
            "mutations_deployed": 1,
            "proposal": {
                "mutation_type": MutationType.TOOL,
                "target_path": _TOOL_TARGET,
                "mutated_content": _TOOL_HANDLER,
                "description": "normalize CSV columns",
            },
            "deployment": {"commit_hash": "abc123"},
        }
        with _patch_engine(cycle_result, reexecute_flag=True), \
                patch(
                    "src.graph.nodes.evolve._try_register_deployed_tool",
                    new=AsyncMock(return_value=True),
                ) as mock_register:
            state = initial_state("clean a csv", "thread-reexec-done")
            state["evolve_reexecute_done"] = True  # guard already spent
            result = await evolve_node(state, gateway=MagicMock(), tools=MagicMock())

        mock_register.assert_not_awaited()
        assert result["evolve_reexecute_offered"] is False
        # Guard is monotonic-True: never reset back to False.
        assert result["evolve_reexecute_done"] is True

    @pytest.mark.asyncio
    async def test_reexecute_disabled_by_default_flag(self) -> None:
        """Opt-in flag off (production default) → never live-register."""
        cycle_result = {
            "status": "deployed",
            "deployed": True,
            "mutations_proposed": 1,
            "mutations_deployed": 1,
            "proposal": {
                "mutation_type": MutationType.TOOL,
                "target_path": _TOOL_TARGET,
                "mutated_content": _TOOL_HANDLER,
                "description": "normalize CSV columns",
            },
            "deployment": {"commit_hash": "abc123"},
        }
        with _patch_engine(cycle_result, reexecute_flag=False), \
                patch(
                    "src.graph.nodes.evolve._try_register_deployed_tool",
                    new=AsyncMock(return_value=True),
                ) as mock_register:
            state = initial_state("clean a csv", "thread-reexec-off")
            result = await evolve_node(state, gateway=MagicMock(), tools=MagicMock())

        mock_register.assert_not_awaited()
        assert result["evolve_reexecute_offered"] is False
        assert result["evolve_reexecute_done"] is False


class TestEvolveHelpers:
    """Unit tests for the Phase-4-E live-registration helpers."""

    def test_sanitize_tool_name(self) -> None:
        assert _sanitize_tool_name("csv-normalizer.py") == "csv_normalizer_py"
        assert _sanitize_tool_name("My Tool!") == "my_tool"
        assert _sanitize_tool_name("") == "evolved_tool"
        assert _sanitize_tool_name("9ways") == "t_9ways"  # leading-alpha guarantee

    def test_derive_input_schema_respects_defaults_and_self(self) -> None:
        code = (
            "async def csv_normalizer(self, path: str, delimiter: str = ','):\n"
            "    return path\n"
        )
        schema = _derive_input_schema(code)
        assert schema["type"] == "object"
        # self skipped; path required; delimiter (has default) NOT required.
        assert set(schema["properties"]) == {"path", "delimiter"}
        assert schema["required"] == ["path"]

    def test_derive_input_schema_invalid_syntax_falls_back(self) -> None:
        assert _derive_input_schema("def broken(:") == {
            "type": "object",
            "properties": {},
        }

    def test_synthesize_self_test_is_valid_for_d9_gate(self) -> None:
        """D9: the evolved-handler self-test must clear the shared code gate —
        non-empty, contains an ``assert``, and the handler is referenced by its
        defined name so combined source has no undefined name (ruff F821)."""
        test_code = _synthesize_self_test(_TOOL_HANDLER)
        # Non-empty + asserts something.
        assert test_code.strip()
        assert "assert " in test_code
        # Calls the evolved handler by its real name, passing the one required
        # positional (path) as a sample string; the defaulted arg is omitted.
        assert "asyncio.run(csv_normalizer(path=''))" in test_code

    def test_synthesize_self_test_no_required_args(self) -> None:
        """A handler with only defaulted params is called with no args."""
        handler = "async def ping(flag: bool = False):\n    return 'pong'\n"
        test_code = _synthesize_self_test(handler)
        assert "asyncio.run(ping())" in test_code
        assert "assert " in test_code

    @pytest.mark.asyncio
    async def test_try_register_success_persists_and_returns_true(self) -> None:
        """Successful validate_and_register → True + best-effort persist."""
        proposal = {
            "mutation_type": MutationType.TOOL,
            "target_path": _TOOL_TARGET,
            "mutated_content": _TOOL_HANDLER,
            "description": "normalize CSV columns",
        }
        registry = MagicMock()
        fake_gen = MagicMock()
        fake_gen.validate_and_register = AsyncMock(
            return_value={"success": True, "tool_name": "csv_normalizer"}
        )
        with patch("src.tools.dynamic.generator.ToolGenerator", return_value=fake_gen), \
                patch("src.safety.pipeline.SafetyPipeline"), \
                patch("src.graph.nodes.evolve._persist_deployed_tool", new=AsyncMock()) as mock_persist:
            ok = await _try_register_deployed_tool(proposal, registry, MagicMock())

        assert ok is True
        fake_gen.validate_and_register.assert_awaited_once()
        # The registered tool name is the sanitized module stem (Path.stem
        # strips the .py, so the name has no extension).
        gen_tool = fake_gen.validate_and_register.call_args.args[0]
        assert gen_tool.tool_name == "csv_normalizer"
        assert gen_tool.handler_code == _TOOL_HANDLER
        mock_persist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_try_register_safety_failure_returns_false(self) -> None:
        """validate_and_register failure → False, no persistence."""
        proposal = {
            "mutation_type": MutationType.TOOL,
            "target_path": _TOOL_TARGET,
            "mutated_content": _TOOL_HANDLER,
            "description": "normalize CSV columns",
        }
        fake_gen = MagicMock()
        fake_gen.validate_and_register = AsyncMock(
            return_value={"success": False, "reason": "Safety validation failed"}
        )
        with patch("src.tools.dynamic.generator.ToolGenerator", return_value=fake_gen), \
                patch("src.safety.pipeline.SafetyPipeline"), \
                patch("src.graph.nodes.evolve._persist_deployed_tool", new=AsyncMock()) as mock_persist:
            ok = await _try_register_deployed_tool(proposal, MagicMock(), MagicMock())

        assert ok is False
        mock_persist.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_try_register_non_py_target_short_circuits(self) -> None:
        """Config-JSON target → False before constructing any generator."""
        proposal = {
            "mutation_type": MutationType.TOOL,
            "target_path": "evolution/tool_config.json",
            "mutated_content": '{"tool": "x"}',
            "description": "tweak config",
        }
        with patch("src.tools.dynamic.generator.ToolGenerator") as mock_gen_cls, \
                patch("src.graph.nodes.evolve._persist_deployed_tool", new=AsyncMock()) as mock_persist:
            ok = await _try_register_deployed_tool(proposal, MagicMock(), MagicMock())

        assert ok is False
        mock_gen_cls.assert_not_called()
        mock_persist.assert_not_awaited()
