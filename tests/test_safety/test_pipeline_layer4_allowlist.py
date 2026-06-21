"""Layer 4 (imports) tests — context ``required_modules`` allowlist (M4/B2).

``context["required_modules"]`` extends the import allowlist, so an evolution
CODE mutation that legitimately needs a normally-blocked module (e.g. socket)
isn't falsely rejected — layered on top of the caller's explicit allowlist.
"""

from __future__ import annotations

import pytest

from src.safety.pipeline import SafetyPipeline


@pytest.fixture
def pipeline() -> SafetyPipeline:
    return SafetyPipeline()


def _socket_code() -> str:
    return (
        "import socket\n"
        "async def conn() -> None:\n"
        "    socket.socket()\n"
    )


class TestLayer4RequiredModules:
    @pytest.mark.asyncio
    async def test_dangerous_import_blocked_without_exemption(
        self, pipeline: SafetyPipeline
    ) -> None:
        result = await pipeline.validate(_socket_code())
        assert result["layers"]["imports"]["passed"] is False
        assert any("socket" in i for i in result["layers"]["imports"]["issues"])

    @pytest.mark.asyncio
    async def test_required_modules_exempts_dangerous_import(
        self, pipeline: SafetyPipeline
    ) -> None:
        result = await pipeline.validate(
            _socket_code(), context={"required_modules": ["socket"]}
        )
        assert result["layers"]["imports"]["passed"] is True

    @pytest.mark.asyncio
    async def test_required_modules_merges_with_explicit_allowlist(
        self, pipeline: SafetyPipeline
    ) -> None:
        """The context set unions with the caller's allowlisted_modules."""
        code = (
            "import socket\n"
            "import os\n"
            "async def conn() -> None:\n"
            "    socket.socket()\n"
        )
        result = await pipeline.validate(
            code,
            context={"required_modules": ["socket"]},
            allowlisted_modules=set(),
        )
        issues = result["layers"]["imports"]["issues"]
        assert any("socket" not in i and "os" in i for i in issues)
        assert not any("socket" in i for i in issues)

    @pytest.mark.asyncio
    async def test_required_modules_accepts_set(
        self, pipeline: SafetyPipeline
    ) -> None:
        result = await pipeline.validate(
            _socket_code(), context={"required_modules": {"socket"}}
        )
        assert result["layers"]["imports"]["passed"] is True
