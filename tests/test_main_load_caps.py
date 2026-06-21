"""Bug E regression — main.py capability loaders pass settings so B3 governance runs.

Before the fix, ``_load_dynamic_tools`` / ``_load_sub_agents`` called the
persisters WITHOUT ``settings``. Both loaders gate the entire B3 governance
layer (semantic dedup ``retire_redundant``, cumulative caps ``enforce_caps`` /
``_retire_excess_tools``, retirement) behind ``if settings is not None``, so the
governance — built and unit-tested in the roadmap — never executed at the only
runtime entry point. The active sub-agent population bloated to 83 against a
``max_active_sub_agents`` cap. These tests pin the wiring: the runtime loaders
must forward ``settings.agent`` (a non-None ``AgentSettings``) into the
persisters.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import main as main_mod
from src.config.settings import get_settings


class TestLoadersPassSettings:
    """main.py's capability loaders forward settings.agent to the persisters."""

    @pytest.mark.asyncio
    async def test_load_sub_agents_passes_settings(self) -> None:
        """_load_sub_agents forwards settings.agent to load_active_agents."""
        from src.agents.registry import SubAgentRegistry

        registry = SubAgentRegistry()
        settings = get_settings()

        with patch("src.agents.persister.SubAgentPersister") as persister_cls:
            persister = persister_cls.return_value
            persister.load_active_agents = AsyncMock(return_value=[])
            await main_mod._load_sub_agents(registry, settings)

            persister.load_active_agents.assert_awaited_once()
            _, kwargs = persister.load_active_agents.call_args
            assert "settings" in kwargs, "settings must be forwarded to enable B3 governance"
            assert kwargs["settings"] is settings.agent

    @pytest.mark.asyncio
    async def test_load_dynamic_tools_passes_settings(self) -> None:
        """_load_dynamic_tools forwards settings.agent to load_active_tools."""
        from src.tools.registry import ToolRegistry

        tools = ToolRegistry()
        settings = get_settings()

        with patch("src.tools.dynamic.persister.ToolPersister") as persister_cls:
            persister = persister_cls.return_value
            persister.load_active_tools = AsyncMock(return_value=[])
            await main_mod._load_dynamic_tools(tools, settings)

            persister.load_active_tools.assert_awaited_once()
            _, kwargs = persister.load_active_tools.call_args
            assert "settings" in kwargs, "settings must be forwarded to enable B3 governance"
            assert kwargs["settings"] is settings.agent

    def test_max_active_sub_agents_raised(self) -> None:
        """The sub-agent cap is 60 (raised from 15) — selective delegation tolerates it."""
        assert get_settings().agent.max_active_sub_agents == 60
