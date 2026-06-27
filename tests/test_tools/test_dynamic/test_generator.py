"""Tests for the dynamic tool generator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.safety.pipeline import SafetyPipeline
from src.tools.dynamic.generator import GeneratedTool, ToolGenerator
from src.tools.dynamic.validation import _run_sandbox_smoke, dedupe_imports, validate_tool_code
from src.tools.registry import ToolRegistry


# Silence Pyright unused-import warnings — used in tests below
assert json  # noqa: S101


async def _dummy_handler() -> str:
    """Minimal async handler for tool registrations in cap-population tests."""
    return "ok"


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
            "result = asyncio.run("
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
            test_code="assert True\n",
        )
        assert tool.tool_name == "my_tool"
        assert tool.description == "A test tool"

    def test_test_code_is_required(self) -> None:
        # D9: test_code is mandatory (min_length=1) — a tool with no test cannot
        # be registered. Construction without it, or with an empty string, raises.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GeneratedTool(
                tool_name="t",
                description="d",
                input_schema={},
                handler_code="async def t() -> str:\n    return 'ok'",
            )
        with pytest.raises(ValidationError):
            GeneratedTool(
                tool_name="t",
                description="d",
                input_schema={},
                handler_code="async def t() -> str:\n    return 'ok'",
                test_code="",
            )


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
            test_code=(
                "import asyncio\n"
                "result = asyncio.run(system_tool('echo hi'))\n"
                "assert isinstance(result, str)\n"
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
            test_code=(
                "import asyncio\n"
                "result = asyncio.run(adder(3, 4))\n"
                "assert result == '7'\n"
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
                test_code="assert True\n",
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

    @pytest.mark.asyncio
    async def test_skips_register_when_active_population_at_cap(self) -> None:
        """A3: when generated_count >= max_active_tools, validate_and_register's
        pre-register gate skips registration — returns success=False with the cap
        reason, WITHOUT running safety/sandbox/materialize or incrementing the
        per-run counter. The active population never grows past the cap mid-run.
        """
        from src.config import get_settings

        cap = get_settings().agent.max_active_tools
        gateway = _make_gateway("")
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())
        registry = ToolRegistry()
        # Pre-fill the active generated population to the cap (as if loaded from
        # prior runs). Builtin-style tools (generated=False) do not count.
        for i in range(cap):
            registry.register(name=f"gen_{i}", handler=_dummy_handler, generated=True)
        assert registry.generated_count == cap

        tool = GeneratedTool(
            tool_name="adder",
            description="Add two numbers",
            input_schema={"type": "object", "properties": {}},
            handler_code=(
                "async def adder(a: int, b: int) -> str:\n"
                "    return str(a + b)\n"
            ),
            test_code="assert True\n",
        )
        result = await gen.validate_and_register(tool, registry)

        assert result["success"] is False
        assert "Active tool cap" in result["reason"]
        # No new tool registered; per-run counter unchanged (gate ran before step 1)
        assert registry.generated_count == cap
        assert gen._tools_created == 0


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


class TestToolGeneratorCodegenModel:
    """Verify generate() routes code-gen at the configured code-strong model
    (default deepseek-v4-pro) instead of the CHEAP tier (complexity=SIMPLE →
    Haiku) that truncates non-trivial handlers (battery-02 N5).

    NOTE: ``AgentSettings`` is a nested model whose default instance is fixed at
    import time, so ``monkeypatch.setenv`` + ``cache_clear`` does NOT re-read it
    (a real ``.env``/env set before process start does). We therefore patch the
    live settings instance attribute directly.
    """

    @staticmethod
    def _patch_codegen_model(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        from src.config import get_settings

        monkeypatch.setattr(get_settings().agent, "tool_generation_model", value)

    @pytest.mark.asyncio
    async def test_generate_pins_codegen_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_codegen_model(monkeypatch, "deepseek-v4-pro")
        gateway = _make_gateway(_safe_tool_json())
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())

        await gen.generate(
            "parse JSON",
            {"goal_text": "test", "failed_tools": "none", "existing_tools": []},
        )

        kwargs = gateway.acompletion.await_args.kwargs
        assert kwargs.get("model") == "deepseek-v4-pro"
        # complexity routing must NOT fire when a model is pinned
        assert "complexity" not in kwargs
        # JSON mode is forced so a multi-line handler can't be truncated by
        # json_repair (the root cause of battery-02 N5's 156-char handlers).
        assert kwargs.get("response_format") == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_generate_respects_model_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_codegen_model(monkeypatch, "gpt-4.1-mini-2025-04-14")
        gateway = _make_gateway(_safe_tool_json())
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())

        await gen.generate(
            "parse JSON",
            {"goal_text": "test", "failed_tools": "none", "existing_tools": []},
        )

        kwargs = gateway.acompletion.await_args.kwargs
        assert kwargs.get("model") == "gpt-4.1-mini-2025-04-14"
        assert "complexity" not in kwargs
        assert kwargs.get("response_format") == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_generate_falls_back_to_complexity_when_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.graph.enums import TaskComplexity

        self._patch_codegen_model(monkeypatch, "")
        gateway = _make_gateway(_safe_tool_json())
        gen = ToolGenerator(gateway=gateway, safety_pipeline=SafetyPipeline())

        await gen.generate(
            "parse JSON",
            {"goal_text": "test", "failed_tools": "none", "existing_tools": []},
        )

        kwargs = gateway.acompletion.await_args.kwargs
        # Empty setting → legacy complexity-based routing (no explicit model pin)
        assert kwargs.get("complexity") == TaskComplexity.SIMPLE
        assert "model" not in kwargs
        assert kwargs.get("response_format") == {"type": "json_object"}


class TestValidateToolCode:
    """D9: the shared code gate (assertion presence + ruff lint + 7-layer safety)
    that BOTH the runtime generator and the D10 edit→approve route call. The
    sandbox smoke is covered separately by ``TestRunSandboxTestEnvAlignment``;
    these tests pass ``sandbox=None`` to isolate the static gates.
    """

    @pytest.mark.asyncio
    async def test_rejects_empty_test_code(self) -> None:
        result = await validate_tool_code(
            handler_code="async def t() -> str:\n    return 'ok'\n",
            test_code="   \n",
            tool_name="t",
            safety_pipeline=SafetyPipeline(),
        )
        assert not result.passed
        assert "empty" in result.reason

    @pytest.mark.asyncio
    async def test_rejects_test_without_assert(self) -> None:
        result = await validate_tool_code(
            handler_code="async def t() -> str:\n    return 'ok'\n",
            test_code="import asyncio\nresult = asyncio.run(t())\nprint(result)\n",
            tool_name="t",
            safety_pipeline=SafetyPipeline(),
        )
        assert not result.passed
        assert "assert" in result.reason

    @pytest.mark.asyncio
    async def test_rejects_undefined_name_via_lint(self) -> None:
        # ruff --select F,E9 flags the undefined name (F821) — a genuine bug.
        result = await validate_tool_code(
            handler_code="async def broken() -> str:\n    return undefined_value\n",
            test_code=(
                "import asyncio\n"
                "result = asyncio.run(broken())\n"
                "assert result is not None\n"
            ),
            tool_name="broken",
            safety_pipeline=SafetyPipeline(),
        )
        assert not result.passed
        assert result.reason.startswith("Lint failed")
        assert not result.lint_result["passed"]

    @pytest.mark.asyncio
    async def test_rejects_dangerous_import_via_safety(self) -> None:
        result = await validate_tool_code(
            handler_code=(
                "import os\n\n"
                "async def sys_tool(cmd: str) -> str:\n"
                "    return os.popen(cmd).read()\n"
            ),
            test_code=(
                "import asyncio\n"
                "result = asyncio.run(sys_tool('echo x'))\n"
                "assert isinstance(result, str)\n"
            ),
            tool_name="sys_tool",
            safety_pipeline=SafetyPipeline(),
        )
        assert not result.passed
        assert result.reason.startswith("Safety validation failed")

    @pytest.mark.asyncio
    async def test_accepts_valid_tool(self) -> None:
        result = await validate_tool_code(
            handler_code="async def echo(msg: str) -> str:\n    return msg\n",
            test_code=(
                "import asyncio\n"
                "result = asyncio.run(echo('hi'))\n"
                "assert result == 'hi'\n"
            ),
            tool_name="echo",
            safety_pipeline=SafetyPipeline(),
        )
        assert result.passed
        assert result.reason == ""
        assert result.safety_result.get("passed") is True
        assert result.lint_result.get("passed") is True


class TestDedupeImportsF811:
    """Regress the systematic F811 defect: LLM-generated handlers re-emit the
    same top-level ``import json`` twice. Ruff/pyflakes flags the second binding
    F811 and the D9 lint gate rejected the tool, but the retry-feedback loop
    never converged (the model kept re-emitting the duplicate). The deterministic
    ``dedupe_imports`` sanitizer — applied in ``validate_tool_code`` (gate
    tolerance) AND ``validate_and_register`` (clean persistence) — drops the
    later exact-duplicate so the tool registers on the first attempt.
    """

    def test_dedupe_drops_exact_duplicate_top_level_import(self) -> None:
        src = (
            "import json\n"
            "\n"
            "import json\n"  # exact duplicate → dropped
            "\n"
            "async def f() -> str:\n"
            "    return json.dumps({})\n"
        )
        out = dedupe_imports(src)
        assert sum(1 for ln in out.splitlines() if ln.rstrip() == "import json") == 1
        # Non-import content survives byte-identical.
        assert "async def f() -> str:" in out
        assert "return json.dumps({})" in out

    def test_dedupe_keeps_distinct_imports(self) -> None:
        # ``import json`` (duplicated) collapses to one; ``import os`` is distinct.
        src = "import json\nimport os\nimport json\n"
        out = dedupe_imports(src)
        assert sum(1 for ln in out.splitlines() if ln.rstrip() == "import json") == 1
        assert sum(1 for ln in out.splitlines() if ln.rstrip() == "import os") == 1

    def test_dedupe_tolerates_trailing_whitespace_variant(self) -> None:
        # ``import json`` and ``import json `` (trailing space) are the same import.
        src = "import json \nimport json\n"
        out = dedupe_imports(src)
        assert sum(1 for ln in out.splitlines() if ln.rstrip() == "import json") == 1

    def test_dedupe_leaves_indented_import_untouched(self) -> None:
        # A function-body re-import is a different binding scope → never deduped.
        src = (
            "import json\n"
            "async def f() -> str:\n"
            "    import json\n"  # indented → kept
            "    return json.dumps({})\n"
        )
        out = dedupe_imports(src)
        assert out == src

    def test_dedupe_idempotent_on_clean_source(self) -> None:
        src = "import json\nimport asyncio\n\nx = 1\n"
        assert dedupe_imports(src) == src

    @pytest.mark.asyncio
    async def test_validate_tool_code_accepts_duplicate_import(self) -> None:
        # Previously F811 → lint failed → rejected; now deduped pre-gate → accepted.
        result = await validate_tool_code(
            handler_code=(
                "import json\n"
                "\n"
                "import json\n"
                "\n"
                "async def parse_json(payload: str) -> str:\n"
                "    '''Parse a JSON string and re-emit it.'''\n"
                "    try:\n"
                "        data = json.loads(payload)\n"
                "    except Exception as e:\n"
                "        return f'ERR: {e}'\n"
                "    return json.dumps(data)\n"
            ),
            test_code=(
                "import asyncio\n"
                "result = asyncio.run(parse_json('{\"a\": 1}'))\n"
                "assert isinstance(result, str)\n"
            ),
            tool_name="parse_json",
            safety_pipeline=SafetyPipeline(),
        )
        assert result.passed, result.reason
        assert result.lint_result.get("passed") is True

    @pytest.mark.asyncio
    async def test_validate_and_register_persists_deduped_handler(self) -> None:
        # The registry persists ``tool.handler_code`` directly, so the generator
        # must dedupe the field before register — the stored handler ends up clean.
        gen = ToolGenerator(gateway=_make_gateway("ok"), safety_pipeline=SafetyPipeline())
        registry = ToolRegistry()
        tool = GeneratedTool(
            tool_name="parse_json",
            description="Parse and re-emit a JSON string",
            input_schema={
                "type": "object",
                "properties": {"payload": {"type": "string"}},
                "required": ["payload"],
            },
            handler_code=(
                "import json\n"
                "\n"
                "import json\n"
                "\n"
                "async def parse_json(payload: str) -> str:\n"
                "    '''Parse a JSON string and re-emit it.'''\n"
                "    try:\n"
                "        data = json.loads(payload)\n"
                "    except Exception as e:\n"
                "        return f'ERR: {e}'\n"
                "    return json.dumps(data)\n"
            ),
            test_code=(
                "import asyncio\n"
                "result = asyncio.run(parse_json('{\"a\": 1}'))\n"
                "assert isinstance(result, str)\n"
            ),
        )
        outcome = await gen.validate_and_register(tool, registry)
        assert outcome["success"], outcome.get("reason")
        # The tool object was mutated to the deduped handler (exactly one import).
        assert sum(
            1 for ln in tool.handler_code.splitlines() if ln.rstrip() == "import json"
        ) == 1
        # And the registered, materialized handler is callable on the clean code.
        handler = registry.get_handler("parse_json")
        assert handler is not None
        out = await handler('{"a": 1}')
        assert isinstance(out, str)


class TestRunSandboxTestEnvAlignment:
    """Finding #2: the tool-gen self-test must run in the host subprocess env
    (``sys.executable`` + host-venv allowlisted deps), matching the in-process
    materialization env. A tool whose only "offense" is importing an allowlisted
    host-installed dep (loguru) must PASS its self-test — the stripped
    ``python:3.12-slim`` docker self-test would ``ModuleNotFoundError`` and
    false-reject it (the exact q06 failure mode for sha256_file et al.).
    """

    @staticmethod
    def _subprocess_sandbox() -> object:
        """A real SandboxExecutor pinned to subprocess mode (sys.executable)."""
        from src.sandbox.executor import SandboxExecutor

        class _Settings:
            evolution_sandbox_mode = "subprocess"
            evolution_sandbox_image = "python:3.12-slim"
            evolution_sandbox_memory_mb = 256
            evolution_sandbox_timeout = 15

        return SandboxExecutor(_Settings())

    @pytest.mark.asyncio
    async def test_self_test_passes_for_host_dep_tool(self) -> None:
        """A handler importing loguru + an asyncio.run self-test passes end-to-end
        through the shared sandbox smoke (the same path ``validate_tool_code``
        runs) — Finding #2: an allowlisted host-installed dep must not be
        false-rejected by a stripped docker image."""
        handler_code = (
            "from loguru import logger\n\n"
            "async def loguru_echo(msg: str) -> str:\n"
            "    '''Echo via loguru.'''\n"
            "    try:\n"
            "        logger.info(msg)\n"
            "        return f'echo:{msg}'\n"
            "    except Exception as e:\n"
            "        return f'ERROR: {e}'\n"
        )
        test_code = (
            "import asyncio\n"
            "result = asyncio.run(loguru_echo('hello'))\n"
            "assert 'ERROR' not in result\n"
            "print('Test passed')\n"
        )
        result = await _run_sandbox_smoke(
            self._subprocess_sandbox(),  # type: ignore[arg-type]
            handler_code,
            test_code,
        )
        assert result["passed"] is True, f"issues={result.get('issues')}"

    @pytest.mark.asyncio
    async def test_self_test_passes_writes_tempfile_fixture(self) -> None:
        """A handler writing a file fixture under the system temp dir passes —
        the read-only docker FS would have failed the write (Finding #2).
        Exercised through the shared sandbox smoke path."""
        handler_code = (
            "import tempfile, pathlib\n\n"
            "async def scratch_writer(payload: str) -> str:\n"
            "    '''Write payload to a temp scratch file and read it back.'''\n"
            "    try:\n"
            "        p = pathlib.Path(tempfile.gettempdir()) / 'turing_scratch.txt'\n"
            "        p.write_text(payload)\n"
            "        return p.read_text()\n"
            "    except Exception as e:\n"
            "        return f'ERROR: {e}'\n"
        )
        test_code = (
            "import asyncio\n"
            "result = asyncio.run(scratch_writer('fixture-ok'))\n"
            "assert result == 'fixture-ok'\n"
            "print('Test passed')\n"
        )
        result = await _run_sandbox_smoke(
            self._subprocess_sandbox(),  # type: ignore[arg-type]
            handler_code,
            test_code,
        )
        assert result["passed"] is True, f"issues={result.get('issues')}"

    def test_prompt_template_uses_asyncio_run(self) -> None:
        """The generated-tool system prompt must teach asyncio.run, not the
        deprecated asyncio.get_event_loop().run_until_complete (Finding #2).

        The template may still NAME ``get_event_loop`` in its "do NOT use" guidance
        (that is intentional), so we assert the deprecated *call pattern*
        ``run_until_complete`` is gone and the recommended ``asyncio.run(...)`` is
        present in the example.
        """
        from src.graph.prompts import TOOL_GENERATE_SYSTEM

        rendered = str(TOOL_GENERATE_SYSTEM)
        assert "asyncio.run(" in rendered
        assert "run_until_complete" not in rendered
