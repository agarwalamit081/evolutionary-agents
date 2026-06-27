"""Security / gap / error-path tests for the dynamic-tool + MCP subsystem.

These tests deliberately complement the existing ``test_dynamic/`` suite (which
covers allowlist *membership*, persister caps/dedup/recall/review happy paths,
and tool_create wiring). The focus here is the gap/error/security surface that
must never regress:

* the **double-barrier** that validates untrusted LLM-generated tool code — the
  safety pipeline must REJECT dangerous constructs (``os.system`` / ``subprocess``
  / ``eval`` / ``exec`` / ``__import__("os")`` / ``open()`` path escapes) while
  ALLOWING safe stdlib (``math`` / ``json`` / ``statistics`` / ``re``);
* the generator's LLM-parse / rate-cap failure paths (graceful ``None``, never raise);
* the persister's DB-failure non-fatal path (``None``, never raise);
* the MCP adapter's schema conversion + connection-failure path (``[]``, never raise).

All I/O is mocked deterministically (litellm via mock_gateway, DB sessions, the
MCP client). No real network/LLM calls. ``monkeypatch`` is used for every
``get_settings()`` field so the process-global cached singleton is never mutated.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import get_settings
from src.safety.pipeline import SafetyPipeline
from src.tools.dynamic.allowlist import ALLOWED_MODULES, get_materializer_namespace
from src.tools.dynamic.generator import GeneratedTool, ToolGenerator
from src.tools.dynamic.persister import ToolPersister
from src.tools.mcp_adapter import MCPToolAdapter


# ─── helpers ────────────────────────────────────────────────────────────


def _security_layer(code: str) -> dict[str, Any]:
    """Run only Layer 3 (forbidden-pattern scan) on a code snippet."""
    return SafetyPipeline()._check_security(code)


def _imports_layer(code: str) -> dict[str, Any]:
    """Run only Layer 4 (dangerous-import scan) against the real allowlist."""
    return SafetyPipeline()._check_imports(code, allowlisted=set(ALLOWED_MODULES))


async def _full_validate(code: str) -> dict[str, Any]:
    """Run the full 7-layer pipeline as the dynamic-tool validator does."""
    return await SafetyPipeline().validate(
        code=code,
        context={"mutation_type": "tool", "tool_name": "t_under_test"},
        sandbox_executor=None,
        allowlisted_modules=set(ALLOWED_MODULES),
    )


# ─── Double-barrier security: REJECT dangerous constructs ───────────────


class TestSecurityBarrierRejectsDangerousPatterns:
    """Layer 3 must flag every forbidden pattern a generated tool might emit."""

    def test_rejects_os_system_call(self) -> None:
        result = _security_layer("import os\nos.system('rm -rf /')\n")
        assert result["passed"] is False
        # The issue text embeds the regex pattern (``os\.system\s*\(``); assert on
        # the stable ``system`` token rather than the literal regex.
        assert any("system" in i for i in result["issues"])

    def test_rejects_bare_os_system_without_import(self) -> None:
        # The pattern scan is textual — it must fire even without an import line.
        result = _security_layer("yield os.system(cmd)\n")
        assert result["passed"] is False

    def test_rejects_dunder_import_of_os(self) -> None:
        result = _security_layer('x = __import__("os")\n')
        assert result["passed"] is False
        assert any("__import__" in i for i in result["issues"])

    def test_rejects_eval(self) -> None:
        result = _security_layer("return eval(user_input)\n")
        assert result["passed"] is False
        assert any("eval" in i for i in result["issues"])

    def test_rejects_exec(self) -> None:
        result = _security_layer("exec(payload)\n")
        assert result["passed"] is False
        assert any("exec" in i for i in result["issues"])

    def test_rejects_compile_exec_mode(self) -> None:
        result = _security_layer("compile(src, '<s>', 'exec')\n")
        assert result["passed"] is False

    def test_rejects_subprocess_shell_true(self) -> None:
        result = _security_layer('subprocess.call(cmd, shell=True)\n')
        assert result["passed"] is False

    def test_rejects_shutil_rmtree(self) -> None:
        result = _security_layer("shutil.rmtree('/x')\n")
        assert result["passed"] is False

    def test_rejects_pickle_load(self) -> None:
        result = _security_layer("pickle.loads(blob)\n")
        assert result["passed"] is False

    def test_rejects_open_write_mode_then_sensitive_path(self) -> None:
        # The path-escape regex requires the write-mode literal to precede the
        # sensitive-path token within the open() call.
        result = _security_layer('open("w", "/etc/passwd")\n')
        assert result["passed"] is False

    def test_open_path_first_ordering_now_caught_by_layer3(self) -> None:
        """FIXED: Layer 3's open() regex is now ordering-independent — it flags
        any open() touching a sensitive path (/etc/passwd, /etc/shadow, .ssh,
        .env) regardless of whether the mode literal precedes or follows the
        path. Reading a credential file is a leak too, so dropping the
        write-mode requirement is the correct, more conservative posture. The
        realistic path-first escape ``open("/etc/passwd", "w")`` is now caught
        by the textual Layer 3 directly (previously relied on Layer 5)."""
        result = _security_layer('open("/etc/passwd", "w")\n')
        assert result["passed"] is False  # now caught by Layer 3 directly


class TestImportBarrierRejectsDangerousModules:
    """Layer 4 must block dangerous imports even when Layer 3's text scan misses
    them (e.g. ``subprocess.run([...])`` without ``shell=True``)."""

    def test_rejects_subprocess_import(self) -> None:
        # subprocess.run([...]) is NOT caught by the Layer-3 text scan…
        assert _security_layer('import subprocess\nsubprocess.run(["ls"])\n')["passed"] is True
        # …but Layer 4 blocks the import itself.
        result = _imports_layer('import subprocess\n')
        assert result["passed"] is False
        assert any("subprocess" in i for i in result["issues"])

    def test_rejects_os_import(self) -> None:
        result = _imports_layer('import os\n')
        assert result["passed"] is False

    def test_rejects_sys_import(self) -> None:
        result = _imports_layer('import sys\n')
        assert result["passed"] is False

    def test_rejects_socket_import(self) -> None:
        result = _imports_layer('import socket\n')
        assert result["passed"] is False

    def test_rejects_ctypes_import(self) -> None:
        result = _imports_layer('import ctypes\n')
        assert result["passed"] is False

    def test_rejects_from_import_of_dangerous_module(self) -> None:
        result = _imports_layer('from os import path\n')
        assert result["passed"] is False
        assert any("os" in i for i in result["issues"])

    def test_rejects_pickle_import(self) -> None:
        result = _imports_layer('import pickle\n')
        assert result["passed"] is False


class TestFullPipelineEndToEndRejection:
    """The end-to-end pipeline (the actual ``validate_tool_code`` gate input)
    must reject a complete dangerous async handler, not just a fragment."""

    async def test_pipeline_rejects_handler_with_os_system(self) -> None:
        code = (
            "import os\n\n\nasync def bad():\n"
            "    return os.system('whoami')\n"
        )
        result = await _full_validate(code)
        assert result["passed"] is False
        assert result["issues"]

    async def test_pipeline_rejects_handler_importing_subprocess(self) -> None:
        code = (
            "import subprocess\n\n\nasync def bad():\n"
            "    return subprocess.run(['ls'])\n"
        )
        result = await _full_validate(code)
        assert result["passed"] is False

    async def test_pipeline_rejects_eval_in_handler(self) -> None:
        code = "async def bad(expr):\n    return eval(expr)\n"
        result = await _full_validate(code)
        assert result["passed"] is False


# ─── Double-barrier security: ALLOW safe stdlib ─────────────────────────


class TestBarrierAllowsSafeStdlib:
    """A generated tool using only safe stdlib must clear BOTH layers."""

    @pytest.mark.parametrize("mod", ["math", "json", "statistics", "re"])
    def test_safe_import_passes_import_layer(self, mod: str) -> None:
        result = _imports_layer(f"import {mod}\n")
        assert result["passed"] is True, result["issues"]
        assert result["issues"] == []

    @pytest.mark.parametrize("mod", ["math", "json", "statistics", "re"])
    def test_safe_from_import_passes_import_layer(self, mod: str) -> None:
        result = _imports_layer(f"from {mod} import something\n")
        assert result["passed"] is True, result["issues"]

    def test_safe_handler_has_no_forbidden_patterns(self) -> None:
        code = (
            "import math\nimport json\nimport statistics\n\n\n"
            "async def good(values):\n"
            "    return json.dumps({'mean': statistics.mean(values)})\n"
        )
        assert _security_layer(code)["passed"] is True

    async def test_safe_stdlib_handler_passes_full_pipeline(self) -> None:
        code = (
            "import math\nimport json\n\n\n"
            "async def good(n):\n"
            "    return json.dumps({'sqrt': math.sqrt(n)})\n"
        )
        result = await _full_validate(code)
        assert result["passed"] is True, result.get("issues")
        assert result["issues"] == []

    def test_materializer_namespace_physically_lacks_dangerous_modules(self) -> None:
        """The constrained namespace is the second barrier — it must not carry
        ``os``/``subprocess``/``sys`` even if a materialized handler tried to
        reference them as bare names."""
        ns = get_materializer_namespace()
        for dangerous in ("os", "subprocess", "sys", "socket", "shutil", "ctypes"):
            assert dangerous not in ns, f"{dangerous} leaked into materializer namespace"
        # and the safe stdlib it DOES carry are real module objects
        for safe in ("math", "json", "re", "statistics"):
            assert safe in ns
            assert hasattr(ns[safe], "__name__")


# ─── Generator: LLM-parse failure + rate-cap paths (never raise) ────────


def _make_generator() -> ToolGenerator:
    """A ToolGenerator wired with a no-op safety pipeline (the generate() path
    under test never reaches validation — it returns first)."""
    safety = MagicMock()
    safety.validate = AsyncMock(return_value={"passed": True, "issues": []})
    return ToolGenerator(gateway=MagicMock(), safety_pipeline=safety, sandbox=None)


class TestGeneratorFailurePaths:
    """generate() must degrade to None on every failure mode, never raise."""

    async def test_returns_none_when_max_tools_per_run_reached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drive the per-instance counter to the configured cap without touching
        # the process-global settings singleton.
        monkeypatch.setattr(get_settings().agent, "max_tools_per_run", 1)
        gen = _make_generator()
        gen._tools_created = 1  # already at cap
        result = await gen.generate("do something", {"goal_text": "g"})
        assert result is None
        # The gateway must never have been called once the cap is hit.
        gateway: Any = gen._gateway
        gateway.acompletion.assert_not_called()

    async def test_returns_none_on_malformed_llm_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The LLM-parse failure path: garbled JSON that cannot yield a
        GeneratedTool degrades to None, not an exception."""
        monkeypatch.setattr(get_settings().agent, "tool_generation_model", "")
        gen = _make_generator()
        gen._gateway.acompletion = AsyncMock(return_value=MagicMock(content="not json at all {{{"))
        result = await gen.generate("gap", {"goal_text": "g"})
        assert result is None

    async def test_returns_none_when_extract_yields_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If StructuredOutputManager.extract returns None (parse-and-repair
        exhausted), generate() propagates None gracefully."""
        monkeypatch.setattr(get_settings().agent, "tool_generation_model", "")
        gen = _make_generator()
        gen._gateway.acompletion = AsyncMock(return_value=MagicMock(content="{}"))
        with patch(
            "src.llm.structured_output.StructuredOutputManager.extract",
            new=AsyncMock(return_value=None),
        ):
            result = await gen.generate("gap", {"goal_text": "g"})
        assert result is None

    async def test_returns_none_on_invalid_tool_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A spec with a non-alphanumeric name (after stripping underscores) is
        rejected — defense against a handler named e.g. ``bad-name``."""
        monkeypatch.setattr(get_settings().agent, "tool_generation_model", "")
        gen = _make_generator()
        bad_spec = GeneratedTool(
            tool_name="bad-name!",  # '-' and '!' are not alphanumeric
            description="d",
            input_schema={},
            handler_code="async def f():\n    return 1\n",
            test_code="assert True\n",
        )
        with patch(
            "src.llm.structured_output.StructuredOutputManager.extract",
            new=AsyncMock(return_value=bad_spec),
        ):
            result = await gen.generate("gap", {"goal_text": "g"})
        assert result is None

    async def test_returns_none_when_gateway_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A gateway exception (timeout / auth / network) is swallowed → None."""
        monkeypatch.setattr(get_settings().agent, "tool_generation_model", "")
        gen = _make_generator()
        gen._gateway.acompletion = AsyncMock(side_effect=RuntimeError("boom"))
        result = await gen.generate("gap", {"goal_text": "g"})
        assert result is None

    async def test_valid_spec_is_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sanity: a well-formed parsed spec round-trips through generate()."""
        monkeypatch.setattr(get_settings().agent, "tool_generation_model", "")
        gen = _make_generator()
        good = GeneratedTool(
            tool_name="good_tool",
            description="d",
            input_schema={"type": "object"},
            handler_code="async def good_tool():\n    return 1\n",
            test_code="assert True\n",
        )
        gen._gateway.acompletion = AsyncMock(return_value=MagicMock(content="{}"))
        with patch(
            "src.llm.structured_output.StructuredOutputManager.extract",
            new=AsyncMock(return_value=good),
        ):
            result = await gen.generate("gap", {"goal_text": "g"})
        assert result is not None
        assert result.tool_name == "good_tool"


class TestGeneratorMaterializeHandler:
    """_materialize_handler is the namespace-constrained exec barrier."""

    def test_rejects_zero_async_functions(self) -> None:
        gen = _make_generator()
        with pytest.raises(ValueError, match="exactly one async function"):
            gen._materialize_handler("x = 1\n")

    def test_rejects_multiple_async_functions(self) -> None:
        gen = _make_generator()
        code = "async def a():\n    return 1\n\nasync def b():\n    return 2\n"
        with pytest.raises(ValueError, match="found 2"):
            gen._materialize_handler(code)

    def test_rejects_syntax_error(self) -> None:
        gen = _make_generator()
        with pytest.raises(ValueError, match="Syntax error"):
            gen._materialize_handler("async def (\n")

    def test_materializes_safe_handler_using_only_namespace(self) -> None:
        """A handler using a namespaced safe module materializes and runs."""
        gen = _make_generator()
        code = (
            "import math\n\n\nasync def square(n):\n"
            "    return math.isqrt(n)\n"
        )
        handler = gen._materialize_handler(code)
        import asyncio
        assert asyncio.iscoroutinefunction(handler)
        assert asyncio.run(handler(16)) == 4


# ─── validate_and_register: validation-gate rejection ───────────────────


class TestValidateAndRegisterGate:
    """validate_and_register must refuse to register a tool whose code fails
    the shared safety gate, and must surface the failing reason."""

    async def test_rejects_dangerous_handler_before_registering(self) -> None:
        safety = SafetyPipeline()
        gen = ToolGenerator(gateway=MagicMock(), safety_pipeline=safety, sandbox=None)
        registry = MagicMock()
        registry.generated_count = 0
        registry.register = MagicMock()

        tool = GeneratedTool(
            tool_name="evil",
            description="d",
            input_schema={},
            handler_code="import os\n\n\nasync def evil():\n    return os.system('id')\n",
            test_code="assert True\n",
        )
        result = await gen.validate_and_register(tool, registry)
        assert result["success"] is False
        # The register call must never have fired for a dangerous tool.
        registry.register.assert_not_called()

    async def test_active_cap_short_circuits_before_validation(self) -> None:
        """When the active-tool population is at cap, registration is skipped
        BEFORE the expensive validation gate runs."""
        safety = MagicMock()
        safety.validate = AsyncMock(return_value={"passed": True, "issues": []})
        gen = ToolGenerator(gateway=MagicMock(), safety_pipeline=safety, sandbox=None)
        registry = MagicMock()
        registry.generated_count = 25  # at the default cap
        tool = GeneratedTool(
            tool_name="t",
            description="d",
            input_schema={},
            handler_code="async def t():\n    return 1\n",
            test_code="assert True\n",
        )
        result = await gen.validate_and_register(tool, registry)
        assert result["success"] is False
        assert "cap" in result["reason"].lower()
        safety.validate.assert_not_called()


# ─── Persister: DB-failure non-fatal path ───────────────────────────────


class TestPersisterDBFailureNonFatal:
    """Every persister write path must return None/[]/False on DB error — a DB
    hiccup can never abort the run (CostTracker-resilience pattern)."""

    async def test_persist_returns_none_on_db_exception(self) -> None:
        persister = ToolPersister()
        # Inject a get_session whose __aenter__ raises (simulates a poisoned
        # connection / unreachable DB).
        fake_session = MagicMock()
        fake_session.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
        fake_session.__aexit__ = AsyncMock(return_value=None)
        with patch("src.db.session.get_session", return_value=fake_session):
            result = await persister.persist(
                tool_name="t",
                description="d",
                input_schema={},
                handler_code="async def t():\n    return 1\n",
            )
        assert result is None  # not raised

    async def test_persist_propagates_real_exception_when_swallowed(
        self,
    ) -> None:
        """The except is broad — confirm a SELECT failure inside the session is
        also caught (mid-transaction error), not just session-entry."""
        persister = ToolPersister()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(side_effect=RuntimeError("select blew up"))
        with patch("src.db.session.get_session", return_value=session):
            result = await persister.persist(
                tool_name="t",
                description="d",
                input_schema={},
                handler_code="async def t():\n    return 1\n",
            )
        assert result is None

    async def test_find_similar_returns_empty_on_db_error(self) -> None:
        persister = ToolPersister()
        fake_session = MagicMock()
        fake_session.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
        fake_session.__aexit__ = AsyncMock(return_value=None)
        with patch("src.db.session.get_session", return_value=fake_session):
            result = await persister.find_similar(embedding=[0.1] * 8)
        assert result == []

    async def test_retrieve_tools_returns_empty_on_db_error(self) -> None:
        persister = ToolPersister()
        fake_session = MagicMock()
        fake_session.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
        fake_session.__aexit__ = AsyncMock(return_value=None)
        with patch("src.db.session.get_session", return_value=fake_session):
            result = await persister.retrieve_tools(query_embedding=[0.1] * 8)
        assert result == []

    async def test_retire_returns_zero_on_db_error(self) -> None:
        persister = ToolPersister()
        fake_session = MagicMock()
        fake_session.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
        fake_session.__aexit__ = AsyncMock(return_value=None)
        with patch("src.db.session.get_session", return_value=fake_session):
            result = await persister.retire(["t1", "t2"])
        assert result == 0

    async def test_list_tools_returns_empty_on_db_error(self) -> None:
        persister = ToolPersister()
        fake_session = MagicMock()
        fake_session.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
        fake_session.__aexit__ = AsyncMock(return_value=None)
        with patch("src.db.session.get_session", return_value=fake_session):
            result = await persister.list_tools()
        assert result == []

    async def test_load_active_tools_returns_empty_on_db_error(self) -> None:
        """load_active_tools must never raise even if the whole DB is down."""
        persister = ToolPersister()
        fake_session = MagicMock()
        fake_session.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
        fake_session.__aexit__ = AsyncMock(return_value=None)
        registry = MagicMock()
        with patch("src.db.session.get_session", return_value=fake_session):
            result = await persister.load_active_tools(registry, settings=None)
        assert result == []


# ─── MCP adapter: schema conversion + connection-failure paths ──────────


class TestMCPAdapterSchemaConversion:
    """The adapter converts an MCP tool's inputSchema into a registered tool."""

    async def test_registers_tools_with_prefixed_names_and_schema(self) -> None:
        registry = MagicMock()
        adapter = MCPToolAdapter(registry)

        # Build a fake MCP tool + client as langchain-mcp-adapters would expose.
        fake_tool = MagicMock()
        fake_tool.name = "search"
        fake_tool.description = "Search the web"
        fake_tool.inputSchema = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
        fake_client = MagicMock()
        fake_client.get_tools = AsyncMock(return_value=[fake_tool])

        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient", return_value=fake_client
        ):
            registered = await adapter.load_server(
                server_command=["npx", "srv"], server_name="web"
            )

        assert registered == ["mcp_web_search"]
        registry.register.assert_called_once()
        kwargs = registry.register.call_args.kwargs
        assert kwargs["name"] == "mcp_web_search"
        assert kwargs["description"] == "Search the web"
        assert kwargs["parameters"] == {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
        # The handler is an awaitable produced by _wrap_mcp_tool.
        assert callable(kwargs["handler"])

    async def test_falls_back_to_args_schema_when_no_input_schema(self) -> None:
        """Older MCP tool objects expose ``args_schema`` instead of inputSchema."""
        registry = MagicMock()
        adapter = MCPToolAdapter(registry)
        fake_tool = MagicMock(spec=["name", "description", "args_schema"])
        fake_tool.name = "ping"
        fake_tool.description = None  # exercises the fallback description too
        fake_tool.args_schema = {"type": "object"}
        fake_client = MagicMock()
        fake_client.get_tools = AsyncMock(return_value=[fake_tool])
        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient", return_value=fake_client
        ):
            registered = await adapter.load_server(["cmd"], "svc")
        assert registered == ["mcp_svc_ping"]
        kwargs = registry.register.call_args.kwargs
        assert kwargs["description"] == "MCP tool: ping"  # None → fallback
        assert kwargs["parameters"] == {"type": "object"}

    async def test_wrapped_handler_joins_text_results(self) -> None:
        """The produced handler flattens a list of MCP content items into text."""
        item_a = MagicMock()
        item_a.text = "alpha"
        item_b = MagicMock()
        item_b.text = "beta"
        bare = MagicMock(spec=[])  # no .text attribute → str() fallback
        client = MagicMock()
        client.call_tool = AsyncMock(return_value=[item_a, item_b, bare])

        handler = MCPToolAdapter._wrap_mcp_tool(client, "t")
        out = await handler(q="x")
        assert "alpha" in out and "beta" in out
        assert out.count("\n") >= 2  # three parts joined by newline

    async def test_wrapped_handler_returns_string_for_scalar_result(self) -> None:
        client = MagicMock()
        client.call_tool = AsyncMock(return_value=42)
        handler = MCPToolAdapter._wrap_mcp_tool(client, "t")
        assert await handler() == "42"

    async def test_wrapped_handler_swallows_call_error(self) -> None:
        """A tool-call exception becomes a sanitized string, never raised."""
        client = MagicMock()
        client.call_tool = AsyncMock(side_effect=RuntimeError("conn reset"))
        handler = MCPToolAdapter._wrap_mcp_tool(client, "t")
        out = await handler()
        assert "Error calling MCP tool t" in out


class TestMCPAdapterConnectionFailure:
    """load_server must degrade to [] on every failure mode, never raise."""

    async def test_returns_empty_when_adapter_not_installed(self) -> None:
        registry = MagicMock()
        adapter = MCPToolAdapter(registry)
        with patch.dict("sys.modules", {"langchain_mcp_adapters.client": None}):
            with patch(
                "langchain_mcp_adapters.client.MultiServerMCPClient",
                create=True,
                side_effect=ImportError("no module"),
            ):
                # Force the import inside load_server to fail.
                import builtins

                real_import = builtins.__import__

                def _fail(name: str, *a: Any, **k: Any) -> Any:
                    if name.startswith("langchain_mcp_adapters"):
                        raise ImportError("not installed")
                    return real_import(name, *a, **k)

                with patch("builtins.__import__", side_effect=_fail):
                    result = await adapter.load_server(["cmd"], "svc")
        assert result == []

    async def test_returns_empty_on_client_construction_failure(self) -> None:
        registry = MagicMock()
        adapter = MCPToolAdapter(registry)
        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            side_effect=RuntimeError("bad server config"),
        ):
            result = await adapter.load_server(["cmd"], "svc")
        assert result == []

    async def test_returns_empty_on_get_tools_failure(self) -> None:
        registry = MagicMock()
        adapter = MCPToolAdapter(registry)
        fake_client = MagicMock()
        fake_client.get_tools = AsyncMock(side_effect=RuntimeError("transport closed"))
        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient", return_value=fake_client
        ):
            result = await adapter.load_server(["cmd"], "svc")
        assert result == []
        registry.register.assert_not_called()

    async def test_empty_command_list_is_handled_gracefully(self) -> None:
        """An empty server_command (no command[0]) must not raise an IndexError
        out of load_server — the construction failure is caught → []."""
        registry = MagicMock()
        adapter = MCPToolAdapter(registry)
        fake_client = MagicMock()
        fake_client.get_tools = AsyncMock(return_value=[])
        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient", return_value=fake_client
        ):
            result = await adapter.load_server([], "svc")
        assert result == []


# ─── Allowlist-vs-barrier consistency (regression guard) ────────────────


class TestAllowlistBarrierConsistency:
    """The allowlist set and the Layer-4 dangerous-set must stay consistent:
    nothing in ALLOWED_MODULES may be a hardcoded dangerous module, and the
    canonical dangerous modules must all be absent from ALLOWED_MODULES."""

    DANGEROUS = {
        "os", "sys", "subprocess", "shutil", "ctypes",
        "multiprocessing", "threading", "socket",
        "pickle", "marshal",
        "importlib", "pkgutil", "code", "codeop",
    }

    def test_no_allowed_module_is_in_the_dangerous_set(self) -> None:
        overlap = self.DANGEROUS & set(ALLOWED_MODULES)
        assert overlap == set(), f"allowlist admits dangerous modules: {overlap}"

    @pytest.mark.parametrize("mod", list(DANGEROUS))
    def test_each_dangerous_module_is_rejected_by_layer4(self, mod: str) -> None:
        result = _imports_layer(f"import {mod}\n")
        assert result["passed"] is False, f"{mod} slipped past Layer 4"

    def test_http_server_slips_past_layer4_root_split(self) -> None:
        """KNOWN GAP (documented): ``http.server`` is in the hardcoded dangerous
        set, but Layer 4 keys on the import's ROOT module (split on ``.``) →
        ``http``, which is not dangerous. So ``import http.server`` is NOT
        blocked by Layer 4 today. Pinned so a future fix is a visible change."""
        result = _imports_layer("import http.server\n")
        assert result["passed"] is True  # root ``http`` is not in the dangerous set
