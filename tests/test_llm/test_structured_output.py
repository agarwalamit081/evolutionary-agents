"""Tests for src.llm.structured_output.StructuredOutputManager."""

from __future__ import annotations

import pytest

from pydantic import BaseModel

from src.llm.structured_output import StructuredOutputManager


# ─── Fixtures ─────────────────────────────────────────────────────────


class SampleOutput(BaseModel):
    """Minimal Pydantic model for structured output tests."""

    name: str
    value: int


@pytest.fixture
def manager() -> StructuredOutputManager:
    """Return a fresh StructuredOutputManager instance."""
    return StructuredOutputManager()


# ─── StructuredOutputManager Tests ────────────────────────────────────


class TestStructuredOutputManager:
    """Unit tests for StructuredOutputManager.extract."""

    @pytest.mark.asyncio
    async def test_extract_valid_json(self, manager: StructuredOutputManager) -> None:
        """Valid JSON matching the Pydantic model extracts successfully."""
        raw = '{"name": "test", "value": 42}'
        result = await manager.extract(raw, SampleOutput)

        assert result is not None
        assert isinstance(result, SampleOutput)
        assert result.name == "test"
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_extract_invalid_json_returns_none(
        self, manager: StructuredOutputManager
    ) -> None:
        """Garbage input that cannot be parsed returns None."""
        raw = "this is not json at all <<<>>>}"
        result = await manager.extract(raw, SampleOutput)

        assert result is None

    @pytest.mark.asyncio
    async def test_extract_json_with_extra_fields(
        self, manager: StructuredOutputManager
    ) -> None:
        """Extra fields beyond the model schema are silently ignored."""
        raw = '{"name": "test", "value": 7, "extra": "ignored", "another": 99}'
        result = await manager.extract(raw, SampleOutput)

        assert result is not None
        assert isinstance(result, SampleOutput)
        assert result.name == "test"
        assert result.value == 7

    @pytest.mark.asyncio
    async def test_extract_json_wrapped_in_code_fences(
        self, manager: StructuredOutputManager
    ) -> None:
        """JSON wrapped in ```json...``` code fences extracts successfully."""
        raw = '```json\n{"name": "fenced", "value": 99}\n```'
        result = await manager.extract(raw, SampleOutput)

        assert result is not None
        assert result.name == "fenced"
        assert result.value == 99

    @pytest.mark.asyncio
    async def test_build_structured_prompt_contains_schema(self) -> None:
        """build_structured_prompt returns a string with JSON schema info."""
        from src.llm.structured_output import StructuredOutputManager as SOM

        prompt = SOM.build_structured_prompt("Respond in JSON.", SampleOutput)
        assert isinstance(prompt, str)
        assert "JSON" in prompt
        assert "name" in prompt
        assert "value" in prompt
