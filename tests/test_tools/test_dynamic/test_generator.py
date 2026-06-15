"""Tests for the dynamic tool generator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.safety.pipeline import SafetyPipeline
from src.tools.dynamic.generator import GeneratedTool, ToolGenerator
from src.tools.registry import ToolRegistry


# Silence Pyright unused-import warnings — used in tests below
assert json  # noqa: S101


def _make_gateway(response_content: str) -> MagicMock:
    """Create a mock gateway with a canned LLM response."""
    from src.llm.models import LLMResponse

    gateway = MagicMock()
    gateway.acompletion = AsyncMock(return_value=LLMResponse(
        content=response_content,
        model="test-model",
        provider="test",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        cost_usd=0.0,
    ))
    return gateway


def _safe_tool_json() -> str:
    """Return a valid GeneratedTool JSON response."""
    return json.dumps({
        "tool_name": "json_parser",
        "description": "Parse and validate JSON strings",
        "input_schema": {
            "type": "object",
            "properties": {
                "json_string": {"type": "string", "description": "JSON string to parse"},
            },
            "required": ["json_string"],
        },
        "handler_code": (
            "import json\n\n"
            "async def json_parser(json_string: str) -> str:\n"
            "    '''Parse and validate a JSON string.'''\n"
            "    try:\n"
            "        result = json.loads(json_string)\n"
            "        return json.dumps(result, indent=2)\n"
            "    except Exception as e:\n"
            "        return f'ERROR: {e}'\n"
        ),
        "test_code": (
            "import asyncio\n"
            "result = asyncio.get_event_loop().run_until_complete("
            "json_parser('{\"key\": \"value\"}'))\n"
            "assert 'ERROR' not in result\n"
            "print('Test passed')\n"
        ),
    })





class TestGeneratedTool:
    """Tests for the GeneratedTool model."""

    def test_valid_tool(self) -> None:
        tool = GeneratedTool(
            tool_name="my_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            handler_code="async def my_tool() -> str:\n    return 'ok'",
        )
        assert tool.tool_name == "my_tool"
        assert tool.description == "A test tool"

    def test_default_test_code(self) -> None:
        tool = GeneratedTool(
            tool_name="t",
            description="d",
            input_schema={},
            handler_code="async def t() -> str:\n    return 'ok'",
        )
        assert tool.test_code == ""


class TestToolGeneratorMaterialize:
    """Tests for handler materialization."""

    @pytest.mark.asyncio
    async def test_materialize_valid_handler(self) -> None:
        gateway = _make_gateway("")
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())

        handler = gen._materialize_handler(
            "async def greet(name: str) -> str:\n    return f'Hello {name}'"
        )
        result = await handler("world")
        assert result == "Hello world"

    def test_materialize_rejects_no_function(self) -> None:
        gateway = _make_gateway("")
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())

        with pytest.raises(ValueError, match="exactly one async function"):
            gen._materialize_handler("x = 1")

    def test_materialize_rejects_multiple_functions(self) -> None:
        gateway = _make_gateway("")
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())

        with pytest.raises(ValueError, match="exactly one async function"):
            gen._materialize_handler(
                "async def f1() -> str:\n    return 'a'\n\n"
                "async def f2() -> str:\n    return 'b'"
            )

    def test_materialize_rejects_syntax_error(self) -> None:
        gateway = _make_gateway("")
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())

        with pytest.raises(ValueError, match="Syntax error"):
            gen._materialize_handler("async def f( -> str:\n    return 'x'")

    @pytest.mark.asyncio
    async def test_materialize_with_json_import(self) -> None:
        gateway = _make_gateway("")
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())

        code = (
            "import json\n\n"
            "async def parse_json(data: str) -> str:\n"
            "    try:\n"
            "        return json.dumps(json.loads(data))\n"
            "    except Exception as e:\n"
            "        return f'ERROR: {e}'\n"
        )
        handler = gen._materialize_handler(code)
        result = await handler('{"key": "val"}')
        assert "key" in result


class TestToolGeneratorValidate:
    """Tests for validate_and_register."""

    @pytest.mark.asyncio
    async def test_rejects_dangerous_code(self) -> None:
        gateway = _make_gateway("")
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())
        registry = ToolRegistry()

        tool = GeneratedTool(
            tool_name="system_tool",
            description="Run system commands",
            input_schema={},
            handler_code=(
                "import os\n\n"
                "async def system_tool(cmd: str) -> str:\n"
                "    return os.popen(cmd).read()\n"
            ),
        )
        result = await gen.validate_and_register(tool, registry)
        assert result["success"] is False
        assert "Safety" in result["reason"] or "safety" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_accepts_safe_code(self) -> None:
        gateway = _make_gateway("")
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())
        registry = ToolRegistry()

        tool = GeneratedTool(
            tool_name="adder",
            description="Add two numbers",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            handler_code=(
                "async def adder(a: int, b: int) -> str:\n"
                "    return str(a + b)\n"
            ),
        )
        result = await gen.validate_and_register(tool, registry)
        assert result["success"] is True
        assert result["tool_name"] == "adder"

        # Verify tool is in registry and works
        handler = registry.get_handler("adder")
        assert handler is not None
        output = await handler(3, 4)
        assert output == "7"

    @pytest.mark.asyncio
    async def test_max_tools_per_run(self) -> None:
        gateway = _make_gateway("")
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())
        registry = ToolRegistry()

        for i in range(4):
            tool = GeneratedTool(
                tool_name=f"tool_{i}",
                description=f"Tool {i}",
                input_schema={},
                handler_code=f"async def tool_{i}() -> str:\n    return '{i}'",
            )
            result = await gen.validate_and_register(tool, registry)
            if i < 3:
                assert result["success"] is True, f"Tool {i} should succeed"
            else:
                # 4th tool should fail due to rate limit (max 3)
                # Actually, validate_and_register doesn't check the limit —
                # generate() does. So this should still succeed.
                # The limit is in generate(), not validate_and_register().
                assert result["success"] is True


class TestToolGeneratorGenerate:
    """Tests for the generate method."""

    @pytest.mark.asyncio
    async def test_generate_returns_tool(self) -> None:
        gateway = _make_gateway(_safe_tool_json())
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())

        tool = await gen.generate(
            "parse JSON data",
            {"goal_text": "test", "failed_tools": "none", "existing_tools": []},
        )
        assert tool is not None
        assert tool.tool_name == "json_parser"
        assert "json" in tool.handler_code

    @pytest.mark.asyncio
    async def test_generate_returns_none_on_malformed(self) -> None:
        gateway = _make_gateway("not valid json at all")
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())

        tool = await gen.generate(
            "do something",
            {"goal_text": "test", "failed_tools": "none", "existing_tools": []},
        )
        assert tool is None

    @pytest.mark.asyncio
    async def test_generate_respects_rate_limit(self) -> None:
        from src.config import get_settings

        cap = get_settings().agent.max_tools_per_run
        gateway = _make_gateway(_safe_tool_json())
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())

        # Up to the configured cap should work
        for _ in range(cap):
            tool = await gen.generate(
                "parse JSON",
                {"goal_text": "test", "failed_tools": "none", "existing_tools": []},
            )
            # Simulate counting as if registered
            gen._tools_created += 1

        # One past the cap is rate-limited
        tool = await gen.generate(
            "parse JSON",
            {"goal_text": "test", "failed_tools": "none", "existing_tools": []},
        )
        assert tool is None
