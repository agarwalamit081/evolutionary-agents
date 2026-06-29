"""#4 mid-run cap-enforce WIRING in ``tool_create_node`` / ``agent_spawn_node``.

Pins the contract between the two creation nodes and the cadence-gated prune:

  * after a MEANINGFUL round (a capability was created OR the cap was hit) the
    node calls ``maybe_enforce_caps_mid_run`` with the right ``fire`` trigger and
    the live ``iteration_count`` / ``mid_run_cap_last_enforced_iter``, and stamps
    its non-None return into ``mid_run_cap_last_enforced_iter``;
  * the early-return no-gap path must NOT call it (no needless DB work).

The prune fn itself is mocked here — its logic is pinned directly in
``tests/test_governance/test_prune.py``; these tests assert only the node↔prune
contract. ``_create_single_tool`` / ``_spawn_single_agent`` are mocked so the
node reaches the enforce call without an LLM generation.

NOTE on the default-off design: ``maybe_enforce_caps_mid_run`` returns ``None``
when ``MID_RUN_CAP_ENFORCE_ENABLED`` is off (the live default), so by default no
``mid_run_cap_last_enforced_iter`` key is ever written — these wiring tests force
the mock to return a sentinel so the stamping path is exercised regardless.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.nodes.agent_spawn import agent_spawn_node
from src.graph.nodes.tool_create import tool_create_node
from src.graph.state import AgentState


class TestToolCreateCapEnforceWiring:
    @pytest.mark.asyncio
    async def test_creation_round_calls_enforce_and_stamps_iter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # created_tools non-empty ⇒ fire=True. The mock returns a sentinel so the
        # stamping path is exercised (real fn returns None when the flag is off).
        enforce = AsyncMock(return_value=42)
        monkeypatch.setattr("src.governance.prune.maybe_enforce_caps_mid_run", enforce)
        create_single = AsyncMock(
            return_value={"success": True, "tool_name": "t", "description": "d"}
        )
        monkeypatch.setattr("src.graph.nodes.tool_create._create_single_tool", create_single)

        state: AgentState = {
            "pending_tool_gaps": ["normalize and dedupe event records"],
            "iteration_count": 7,
            "mid_run_cap_last_enforced_iter": 0,
        }
        result = await tool_create_node(state, gateway=MagicMock(), tools=MagicMock())

        enforce.assert_awaited_once()
        assert enforce.call_args.kwargs["fire"] is True
        assert enforce.call_args.kwargs["current_iter"] == 7
        assert enforce.call_args.kwargs["last_enforced_iter"] == 0
        assert result["mid_run_cap_last_enforced_iter"] == 42

    @pytest.mark.asyncio
    async def test_no_gaps_path_does_not_enforce(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        enforce = AsyncMock(return_value=99)
        monkeypatch.setattr("src.governance.prune.maybe_enforce_caps_mid_run", enforce)

        result = await tool_create_node(
            {"pending_tool_gaps": []}, gateway=MagicMock(), tools=MagicMock()
        )

        enforce.assert_not_awaited()
        assert "mid_run_cap_last_enforced_iter" not in result


class TestAgentSpawnCapEnforceWiring:
    @pytest.mark.asyncio
    async def test_spawn_round_calls_enforce_and_stamps_iter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        enforce = AsyncMock(return_value=42)
        monkeypatch.setattr("src.governance.prune.maybe_enforce_caps_mid_run", enforce)
        spawn_single = AsyncMock(
            return_value={
                "name": "event_analyst",
                "description": "d",
                "template_type": "custom",
                "tool_scope": "inherit_all",
                "id": "x",
            }
        )
        monkeypatch.setattr("src.graph.nodes.agent_spawn._spawn_single_agent", spawn_single)

        registry = MagicMock()
        registry.active_count = 0  # below the active-population cap so spawn proceeds
        state: dict[str, Any] = {
            "pending_agent_gaps": ["analyze events for outliers"],
            "iteration_count": 9,
            "mid_run_cap_last_enforced_iter": 0,
        }
        result = await agent_spawn_node(
            state, gateway=MagicMock(), tools=MagicMock(), sub_agent_registry=registry
        )

        enforce.assert_awaited_once()
        assert enforce.call_args.kwargs["fire"] is True
        assert enforce.call_args.kwargs["current_iter"] == 9
        assert enforce.call_args.kwargs["last_enforced_iter"] == 0
        assert result["mid_run_cap_last_enforced_iter"] == 42

    @pytest.mark.asyncio
    async def test_no_gaps_path_does_not_enforce(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        enforce = AsyncMock(return_value=99)
        monkeypatch.setattr("src.governance.prune.maybe_enforce_caps_mid_run", enforce)

        result = await agent_spawn_node({"pending_agent_gaps": []})

        enforce.assert_not_awaited()
        assert "mid_run_cap_last_enforced_iter" not in result
