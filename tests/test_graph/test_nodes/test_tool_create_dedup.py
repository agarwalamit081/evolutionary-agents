"""Tests for the semantic dedup block in ``_create_single_tool`` (B3).

Before spending an LLM generation call, the node embeds the capability gap and
reuses an existing active tool whose capability is semantically identical
(cosine >= capability_dedup_threshold) and already loaded in the in-memory
registry. These cover the four contract points:

  * reuse-above-threshold   -> returns ``reused`` True, skips generation
  * create-no-match         -> generates, persists the embedding (api source)
  * candidate-not-in-registry -> above threshold but not loaded -> still generates
  * skip-on-hash-fallback   -> hash vectors are not deduped; persistence gets None
  * failure-degrades        -> a dedup error never blocks generation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.nodes.tool_create import _create_single_tool


def _gen_instance(tool_name: str = "new_tool") -> MagicMock:
    """A mock ToolGenerator instance whose one generation+validate succeeds."""
    inst = MagicMock()
    inst.generate = AsyncMock(
        return_value=MagicMock(
            tool_name=tool_name,
            description="d",
            input_schema={},
            handler_code="async def x(): ...",
            test_code="",
        )
    )
    inst.validate_and_register = AsyncMock(
        return_value={"success": True, "sandbox_result": {"passed": True}}
    )
    return inst


class TestCreateSingleToolDedup:
    @pytest.mark.asyncio
    async def test_reuses_existing_tool_above_threshold(self) -> None:
        embed_mock = AsyncMock(return_value=([0.1] * 768, "api"))
        persister_inst = MagicMock()
        persister_inst.find_similar = AsyncMock(
            return_value=[{"tool_name": "existing", "description": "d", "similarity": 0.95}]
        )
        gen_instance = _gen_instance()

        registry = MagicMock()
        registry.list_names = MagicMock(return_value=[])
        registry.has = MagicMock(return_value=True)  # existing tool is loaded

        persist_mock = AsyncMock()
        with (
            patch("src.memory.embeddings.embed_capability", embed_mock),
            patch("src.tools.dynamic.persister.ToolPersister", return_value=persister_inst),
            patch("src.tools.dynamic.generator.ToolGenerator", return_value=gen_instance),
            patch("src.graph.nodes.tool_create._persist_tool", new=persist_mock),
        ):
            result = await _create_single_tool(
                gateway=MagicMock(),
                registry=registry,
                gap_description="fetch URLs over HTTP",
                goal_text="g",
                tool_results=[],
            )

        assert result["success"] is True
        assert result.get("reused") is True
        assert result["tool_name"] == "existing"
        # Generation was skipped entirely (no LLM call spent).
        gen_instance.generate.assert_not_awaited()
        persist_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_creates_when_no_match_above_threshold(self) -> None:
        embed_mock = AsyncMock(return_value=([0.1] * 768, "api"))
        persister_inst = MagicMock()
        persister_inst.find_similar = AsyncMock(return_value=[])  # nothing similar
        gen_instance = _gen_instance(tool_name="csv_parser")

        registry = MagicMock()
        registry.list_names = MagicMock(return_value=[])

        persist_mock = AsyncMock()
        with (
            patch("src.memory.embeddings.embed_capability", embed_mock),
            patch("src.tools.dynamic.persister.ToolPersister", return_value=persister_inst),
            patch("src.tools.dynamic.generator.ToolGenerator", return_value=gen_instance),
            patch("src.graph.nodes.tool_create._persist_tool", new=persist_mock),
        ):
            result = await _create_single_tool(
                gateway=MagicMock(),
                registry=registry,
                gap_description="parse CSV files",
                goal_text="g",
                tool_results=[],
            )

        assert result["success"] is True
        assert "reused" not in result
        assert result["tool_name"] == "csv_parser"
        gen_instance.generate.assert_awaited_once()
        # The real "api" embedding is persisted so future gaps reuse this tool.
        persist_mock.assert_awaited_once()
        assert persist_mock.await_args_list[-1].kwargs["capability_embedding"] == [0.1] * 768
        assert persist_mock.await_args_list[-1].kwargs["capability_text"] == "parse CSV files"

    @pytest.mark.asyncio
    async def test_creates_when_candidate_not_in_registry(self) -> None:
        """Above threshold but the similar tool isn't loaded this run -> generate."""
        embed_mock = AsyncMock(return_value=([0.1] * 768, "api"))
        persister_inst = MagicMock()
        persister_inst.find_similar = AsyncMock(
            return_value=[{"tool_name": "missing", "description": "d", "similarity": 0.95}]
        )
        gen_instance = _gen_instance(tool_name="fresh_tool")

        registry = MagicMock()
        registry.list_names = MagicMock(return_value=[])
        registry.has = MagicMock(return_value=False)  # not loadable -> can't reuse

        with (
            patch("src.memory.embeddings.embed_capability", embed_mock),
            patch("src.tools.dynamic.persister.ToolPersister", return_value=persister_inst),
            patch("src.tools.dynamic.generator.ToolGenerator", return_value=gen_instance),
            patch("src.graph.nodes.tool_create._persist_tool", new=AsyncMock()),
        ):
            result = await _create_single_tool(
                gateway=MagicMock(),
                registry=registry,
                gap_description="need a fetcher",
                goal_text="g",
                tool_results=[],
            )

        assert result["success"] is True
        assert "reused" not in result
        assert result["tool_name"] == "fresh_tool"
        gen_instance.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_dedup_on_hash_fallback(self) -> None:
        """Hash-fallback vectors are not deduped or stored."""
        embed_mock = AsyncMock(return_value=([0.2] * 768, "hash"))
        persister_inst = MagicMock()
        persister_inst.find_similar = AsyncMock(
            return_value=[{"tool_name": "existing", "similarity": 0.95}]
        )
        gen_instance = _gen_instance()

        registry = MagicMock()
        registry.list_names = MagicMock(return_value=[])

        persist_mock = AsyncMock()
        with (
            patch("src.memory.embeddings.embed_capability", embed_mock),
            patch("src.tools.dynamic.persister.ToolPersister", return_value=persister_inst),
            patch("src.tools.dynamic.generator.ToolGenerator", return_value=gen_instance),
            patch("src.graph.nodes.tool_create._persist_tool", new=persist_mock),
        ):
            result = await _create_single_tool(
                gateway=MagicMock(),
                registry=registry,
                gap_description="offline gap",
                goal_text="g",
                tool_results=[],
            )

        assert result["success"] is True
        persister_inst.find_similar.assert_not_awaited()
        # Persistence must NOT receive a hash vector.
        assert persist_mock.await_args_list[-1].kwargs["capability_embedding"] is None
        assert persist_mock.await_args_list[-1].kwargs["capability_text"] is None

    @pytest.mark.asyncio
    async def test_dedup_failure_degrades_to_generation(self) -> None:
        """A dedup error must never block generation."""
        embed_mock = AsyncMock(return_value=([0.1] * 768, "api"))
        persister_inst = MagicMock()
        persister_inst.find_similar = AsyncMock(side_effect=RuntimeError("db down"))
        gen_instance = _gen_instance()

        registry = MagicMock()
        registry.list_names = MagicMock(return_value=[])

        with (
            patch("src.memory.embeddings.embed_capability", embed_mock),
            patch("src.tools.dynamic.persister.ToolPersister", return_value=persister_inst),
            patch("src.tools.dynamic.generator.ToolGenerator", return_value=gen_instance),
            patch("src.graph.nodes.tool_create._persist_tool", new=AsyncMock()),
        ):
            result = await _create_single_tool(
                gateway=MagicMock(),
                registry=registry,
                gap_description="resilient gap",
                goal_text="g",
                tool_results=[],
            )

        assert result["success"] is True
        assert result["tool_name"] == "new_tool"
        gen_instance.generate.assert_awaited_once()
