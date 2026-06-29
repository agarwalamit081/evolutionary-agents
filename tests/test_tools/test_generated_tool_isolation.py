"""Regression tests: route LLM-generated tool handlers through the code-exec sandbox.

Closes the runner-isolation gap (#288): a dynamically-created tool's
``handler_code`` (untrusted LLM output) otherwise runs in-process in the worker
with full DB/Redis/FS access, bypassing the no-DinD runner sandbox. These tests
lock:

- the registry tags generated tools with their source and exposes it;
- a generated tool's invocation is routed through the sandbox in docker/runner
  mode (the in-process handler is NOT called) and fails CLOSED on a sandbox
  outage / handler exception (never falls back to in-process);
- subprocess mode (the dev default) is unchanged — the in-process handler runs;
- the generated flag propagates through the sub-agent scoped-registry copy.

No docker/runner infrastructure is required: ``_run_driver_in_sandbox`` is
monkeypatched, and the mode is forced via ``sandbox_dispatch._code_exec_mode``.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.graph.models import ToolResult
from src.sandbox.executor import SandboxResult, SandboxUnavailable
from src.tools.dynamic import sandbox_dispatch
from src.tools.registry import ToolRegistry

# A minimal untrusted-handler source (one async function, writes a deliverable).
_HANDLER_SRC = (
    "async def normalize(path):\n"
    "    import json as j\n"
    "    with open(path) as f:\n"
    "        rows = [j.loads(l) for l in f if l.strip()]\n"
    "    return {'normalized': len(rows)}\n"
)


def _registry_with_generated() -> ToolRegistry:
    """Registry holding one generated tool + one builtin (in-process) tool."""

    async def _inproc(**_kw: Any) -> str:  # noqa: ANN401 — test stand-in
        return "inproc-ok"

    reg = ToolRegistry()
    reg.register(
        name="normalize_rows",
        handler=_inproc,
        description="normalize",
        parameters={"type": "object"},
        generated=True,
        handler_code=_HANDLER_SRC,
    )
    reg.register(name="builtin_tool", handler=_inproc, description="b", parameters={})
    return reg


def _sandbox_success(payload_output: str = "5") -> SandboxResult:
    """A SandboxResult whose stdout carries a parsed result sentinel."""
    import json

    stdout = (
        f"\n{sandbox_dispatch._RESULT_BEGIN}\n"
        + json.dumps({"output": payload_output})
        + f"\n{sandbox_dispatch._RESULT_END}\n"
    )
    return SandboxResult(
        success=True, exit_code=0, stdout=stdout, stderr="",
        duration_seconds=0.1, memory_mb=None, timed_out=False,
    )


# ─── Registry tagging ────────────────────────────────────────────────


class TestRegistryTagging:
    """Generated tools are tagged with their source; builtins are not."""

    def test_generated_flag_and_source_round_trip(self) -> None:
        reg = _registry_with_generated()
        assert reg.is_generated("normalize_rows") is True
        assert reg.get_handler_code("normalize_rows") == _HANDLER_SRC

    def test_builtin_is_not_generated(self) -> None:
        reg = _registry_with_generated()
        assert reg.is_generated("builtin_tool") is False
        assert reg.get_handler_code("builtin_tool") is None

    def test_unknown_tool_is_not_generated(self) -> None:
        reg = _registry_with_generated()
        assert reg.is_generated("nope") is False
        assert reg.get_handler_code("nope") is None

    def test_generated_false_discards_handler_code(self) -> None:
        """``handler_code`` is only retained when ``generated=True``."""
        reg = ToolRegistry()

        async def h(**_kw: Any) -> str:  # noqa: ANN401
            return "x"

        reg.register(name="t", handler=h, generated=False, handler_code="should-be-ignored")
        assert reg.is_generated("t") is False
        assert reg.get_handler_code("t") is None


# ─── Driver builder + extraction ─────────────────────────────────────


class TestDriverAndExtraction:
    """The synthesized driver + result-marker parsing."""

    def test_extract_async_func_name(self) -> None:
        assert sandbox_dispatch._extract_async_func_name(_HANDLER_SRC) == "normalize"

    def test_extract_async_func_name_syntax_error(self) -> None:
        assert sandbox_dispatch._extract_async_func_name("async def (:\n") is None

    def test_extract_async_func_name_no_async(self) -> None:
        assert sandbox_dispatch._extract_async_func_name("def sync():\n    pass\n") is None

    def test_build_driver_contains_handler_and_harness(self) -> None:
        drv = sandbox_dispatch._build_driver(_HANDLER_SRC, "normalize", {"path": "x.jsonl"})
        # The untrusted handler source is present verbatim.
        assert "async def normalize(path):" in drv
        # The args travel as a repr-escaped JSON literal (safe against quotes).
        assert "_TURING_ARGS = _turing_json.loads(" in drv
        assert "normalize(**_TURING_ARGS)" in drv
        assert sandbox_dispatch._RESULT_BEGIN in drv
        assert sandbox_dispatch._RESULT_END in drv

    def test_build_driver_embeds_quote_bearing_arg_safely(self) -> None:
        """An arg containing a quote must not break out of the JSON literal."""
        drv = sandbox_dispatch._build_driver(
            _HANDLER_SRC, "normalize", {"path": "it's a \"trap\".jsonl"}
        )
        # The driver is valid Python (compiles), proving no literal breakout.
        compile(drv, "<drv>", "exec")

    def test_extract_output_success(self) -> None:
        success, output, error = sandbox_dispatch._extract_output(
            _sandbox_success("ok-value"), "normalize"
        )
        assert (success, output, error) == (True, "ok-value", None)

    def test_extract_output_handler_raised(self) -> None:
        res = SandboxResult(
            success=False, exit_code=1, stdout="", stderr="ValueError: bad",
            duration_seconds=0.1, memory_mb=None, timed_out=False,
        )
        success, output, error = sandbox_dispatch._extract_output(res, "normalize")
        assert success is False
        assert output == ""
        assert "ValueError: bad" in (error or "")

    def test_extract_output_timeout(self) -> None:
        res = SandboxResult(
            success=False, exit_code=None, stdout="", stderr="",
            duration_seconds=5.0, memory_mb=None, timed_out=True,
        )
        success, _output, error = sandbox_dispatch._extract_output(res, "normalize")
        assert success is False
        assert "timed out" in (error or "").lower()


# ─── invoke_generated_tool dispatch policy ───────────────────────────


class TestInvokePolicy:
    """Mode-gated routing + fail-closed behavior."""

    @pytest.mark.asyncio
    async def test_subprocess_mode_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dev default: not isolated — caller runs the in-process handler."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "subprocess")
        reg = _registry_with_generated()
        assert await sandbox_dispatch.invoke_generated_tool("normalize_rows", reg, {"path": "x"}) is None

    @pytest.mark.asyncio
    async def test_runner_mode_routes_through_sandbox(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Runner mode: the driver is sandboxed and the result parsed back."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "runner")
        captured: dict[str, Any] = {}

        async def _fake_run(driver: str, timeout: int) -> SandboxResult:
            captured["driver"] = driver
            captured["timeout"] = timeout
            return _sandbox_success("12 rows")

        monkeypatch.setattr(sandbox_dispatch, "_run_driver_in_sandbox", _fake_run)
        reg = _registry_with_generated()
        result = await sandbox_dispatch.invoke_generated_tool(
            "normalize_rows", reg, {"path": "seed.jsonl"}
        )
        assert result is not None
        assert result.success is True
        assert result.output == "12 rows"
        assert result.metadata.get("isolated") is True
        assert result.metadata.get("mode") == "runner"
        # The untrusted handler source was carried into the sandbox driver.
        assert "async def normalize(path):" in captured["driver"]
        assert "seed.jsonl" in captured["driver"]  # args threaded through

    @pytest.mark.asyncio
    async def test_runner_mode_fail_closed_on_sandbox_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sandbox outage FAILS CLOSED — never falls back to in-process."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "runner")

        async def _boom(_driver: str, _timeout: int) -> SandboxResult:
            raise SandboxUnavailable("runner unreachable")

        monkeypatch.setattr(sandbox_dispatch, "_run_driver_in_sandbox", _boom)
        reg = _registry_with_generated()
        result = await sandbox_dispatch.invoke_generated_tool(
            "normalize_rows", reg, {"path": "x"}
        )
        assert result is not None
        assert result.success is False
        assert "sandbox unavailable" in (result.error or "")
        assert "in-process" in (result.error or "")

    @pytest.mark.asyncio
    async def test_runner_mode_fail_closed_on_handler_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A handler that raises is a fail-closed error, never re-run in-process."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "runner")
        res = SandboxResult(
            success=False, exit_code=1, stdout="", stderr="ZeroDivisionError",
            duration_seconds=0.1, memory_mb=None, timed_out=False,
        )

        async def _run(_driver: str, _timeout: int) -> SandboxResult:
            return res

        monkeypatch.setattr(sandbox_dispatch, "_run_driver_in_sandbox", _run)
        reg = _registry_with_generated()
        result = await sandbox_dispatch.invoke_generated_tool(
            "normalize_rows", reg, {"path": "x"}
        )
        assert result is not None and result.success is False
        assert "ZeroDivisionError" in (result.error or "")

    @pytest.mark.asyncio
    async def test_runner_mode_missing_source_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A generated tool with no recoverable source refuses in-process."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "runner")
        reg = ToolRegistry()

        async def h(**_kw: Any) -> str:  # noqa: ANN401
            return "x"

        # generated=True but no handler_code (shouldn't happen, but be safe).
        reg.register(name="broken", handler=h, generated=True, handler_code=None)
        result = await sandbox_dispatch.invoke_generated_tool("broken", reg, {})
        assert result is not None and result.success is False
        assert "could not be isolated" in (result.error or "")


# ─── execute._execute_tool_call integration ──────────────────────────


class TestExecuteDispatchHook:
    """The execute node routes generated tools through the sandbox in runner mode."""

    @pytest.mark.asyncio
    async def test_runner_mode_does_not_call_inproc_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In runner mode the in-process handler is NEVER called for a generated tool."""
        from src.graph.nodes import execute as execute_mod

        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "runner")

        async def _fake_run(_d: str, _t: int) -> SandboxResult:
            return _sandbox_success("sandboxed-result")

        monkeypatch.setattr(sandbox_dispatch, "_run_driver_in_sandbox", _fake_run)
        # Neutralize the metrics recorder (no DB).
        monkeypatch.setattr(execute_mod, "_record_tool_metric", _noop_metric())

        called: list[str] = []

        async def _inproc(**_kw: Any) -> str:  # noqa: ANN401
            called.append("inproc")
            return "MUST-NOT-HAPPEN"

        reg = ToolRegistry()
        reg.register(
            name="normalize_rows", handler=_inproc, description="d", parameters={},
            generated=True, handler_code=_HANDLER_SRC,
        )
        tc = {"function": {"name": "normalize_rows", "arguments": '{"path": "s.jsonl"}'}}
        result = await execute_mod._execute_tool_call(tc, reg)
        assert called == []  # in-process handler never invoked
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "sandboxed-result"
        assert result.metadata.get("isolated") is True

    @pytest.mark.asyncio
    async def test_subprocess_mode_calls_inproc_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dev default: the generated tool runs in-process, unchanged."""
        from src.graph.nodes import execute as execute_mod

        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "subprocess")
        monkeypatch.setattr(execute_mod, "_record_tool_metric", _noop_metric())

        async def _inproc(path: str = "") -> str:
            return f"inproc:{path}"

        reg = ToolRegistry()
        reg.register(
            name="normalize_rows", handler=_inproc, description="d", parameters={},
            generated=True, handler_code=_HANDLER_SRC,
        )
        tc = {"function": {"name": "normalize_rows", "arguments": '{"path": "s.jsonl"}'}}
        result = await execute_mod._execute_tool_call(tc, reg)
        assert result.success is True
        assert result.output == "inproc:s.jsonl"

    @pytest.mark.asyncio
    async def test_builtin_tool_always_inproc_in_runner_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hand-written builtin runs in-process even in runner mode."""
        from src.graph.nodes import execute as execute_mod

        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "runner")
        monkeypatch.setattr(execute_mod, "_record_tool_metric", _noop_metric())

        async def _inproc(query: str = "") -> str:
            return f"builtin:{query}"

        reg = ToolRegistry()
        reg.register(name="web_search", handler=_inproc, description="d", parameters={})
        tc = {"function": {"name": "web_search", "arguments": '{"query": "q"}'}}
        result = await execute_mod._execute_tool_call(tc, reg)
        assert result.success is True
        assert result.output == "builtin:q"


# ─── Sub-agent scoped-registry propagation ───────────────────────────


class TestScopedPropagation:
    """The generated flag survives the sub-agent scoped copy."""

    def test_inherit_all_propagates_generated_flag(self) -> None:
        from src.agents.subgraph import scope_tools
        from src.graph.models import SubAgentSpec

        parent = _registry_with_generated()
        spec = SubAgentSpec(
            name="sub", goal="g", tool_scope="inherit_all", parent_thread_id="parent-run"
        )
        scoped = scope_tools(spec, parent)
        assert scoped.is_generated("normalize_rows") is True
        assert scoped.get_handler_code("normalize_rows") == _HANDLER_SRC
        # Builtin stays non-generated in the scoped copy.
        assert scoped.is_generated("builtin_tool") is False

    def test_inherit_subset_propagates_generated_flag(self) -> None:
        from src.agents.subgraph import scope_tools
        from src.graph.models import SubAgentSpec

        parent = _registry_with_generated()
        spec = SubAgentSpec(
            name="sub", goal="g", tool_scope="inherit_subset", tool_subset=["normalize_rows"],
            parent_thread_id="parent-run",
        )
        scoped = scope_tools(spec, parent)
        assert scoped.is_generated("normalize_rows") is True
        assert scoped.get_handler_code("normalize_rows") == _HANDLER_SRC


# ─── #2 worker-default isolation (#2) ─────────────────────────────────


def _iso_settings(
    *,
    worker: bool = False,
    default: bool = True,
    promote: bool = True,
) -> Any:
    """A ToolSandboxSettings stand-in carrying only the #2 isolation knobs.

    Returned by a monkeypatched ``sandbox_dispatch._isolation_settings`` so the
    gate is exercised without touching ``.env`` (which the host test suite
    inherits from the live deployment).
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        worker_process=worker,
        isolation_default_to_sandbox=default,
        auto_promote_subprocess_to_runner=promote,
    )


class TestIsolationRuntimeGate:
    """``_is_isolated_runtime`` / ``_effective_isolation_mode`` — pure policy."""

    def test_explicit_runner_is_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "runner")
        monkeypatch.setattr(sandbox_dispatch, "_isolation_settings", lambda: _iso_settings(worker=False))
        assert sandbox_dispatch._is_isolated_runtime() is True

    def test_explicit_docker_is_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "docker")
        assert sandbox_dispatch._is_isolated_runtime() is True

    def test_subprocess_no_worker_not_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Host CLI default: subprocess + no worker flag → in-process."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "subprocess")
        monkeypatch.setattr(sandbox_dispatch, "_isolation_settings", lambda: _iso_settings(worker=False))
        assert sandbox_dispatch._is_isolated_runtime() is False

    def test_subprocess_worker_default_is_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The #2 gap: a worker in subprocess mode STILL isolates."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "subprocess")
        monkeypatch.setattr(sandbox_dispatch, "_isolation_settings", lambda: _iso_settings(worker=True))
        assert sandbox_dispatch._is_isolated_runtime() is True

    def test_master_switch_off_disables_worker_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "subprocess")
        monkeypatch.setattr(
            sandbox_dispatch, "_isolation_settings", lambda: _iso_settings(worker=True, default=False)
        )
        assert sandbox_dispatch._is_isolated_runtime() is False

    def test_promotion_off_disables_worker_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No safe surface without promotion → leave the subprocess gap as-is."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "subprocess")
        monkeypatch.setattr(
            sandbox_dispatch, "_isolation_settings", lambda: _iso_settings(worker=True, promote=False)
        )
        assert sandbox_dispatch._is_isolated_runtime() is False

    def test_explicit_mode_wins_over_worker_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """runner mode isolates even when the worker knobs would say otherwise."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "runner")
        monkeypatch.setattr(
            sandbox_dispatch,
            "_isolation_settings",
            lambda: _iso_settings(worker=False, default=False, promote=False),
        )
        assert sandbox_dispatch._is_isolated_runtime() is True

    def test_effective_mode_promotes_subprocess_to_runner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "subprocess")
        assert sandbox_dispatch._effective_isolation_mode() == "runner"

    def test_effective_mode_keeps_explicit_runner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "runner")
        assert sandbox_dispatch._effective_isolation_mode() == "runner"

    def test_effective_mode_keeps_explicit_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "docker")
        assert sandbox_dispatch._effective_isolation_mode() == "docker"


class TestWorkerDefaultDispatch:
    """End-to-end: a subprocess-mode worker isolates via the runner surface."""

    @pytest.mark.asyncio
    async def test_subprocess_worker_isolates_via_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#2: worker + subprocess mode → driver runs under the RUNNER surface."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "subprocess")
        monkeypatch.setattr(
            sandbox_dispatch, "_isolation_settings", lambda: _iso_settings(worker=True)
        )
        captured: dict[str, Any] = {}

        async def _fake_run(driver: str, timeout: int) -> SandboxResult:
            captured["driver"] = driver
            captured["timeout"] = timeout
            return _sandbox_success("worker-promoted")

        monkeypatch.setattr(sandbox_dispatch, "_run_driver_in_sandbox", _fake_run)
        # Reset the one-time promotion notice so this test is order-independent.
        monkeypatch.setattr(sandbox_dispatch, "_subprocess_promotion_warned", False)
        reg = _registry_with_generated()

        result = await sandbox_dispatch.invoke_generated_tool(
            "normalize_rows", reg, {"path": "seed.jsonl"}
        )

        assert result is not None
        assert result.success is True
        assert result.output == "worker-promoted"
        # The metadata reports the EFFECTIVE surface (runner), not subprocess.
        assert result.metadata.get("isolated") is True
        assert result.metadata.get("mode") == "runner"
        assert "async def normalize(path):" in captured["driver"]
        # The promotion fired (one-time flag flipped).
        assert sandbox_dispatch._subprocess_promotion_warned is True

    @pytest.mark.asyncio
    async def test_subprocess_worker_fails_closed_on_runner_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A promoted worker whose runner is unreachable FAILS CLOSED."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "subprocess")
        monkeypatch.setattr(
            sandbox_dispatch, "_isolation_settings", lambda: _iso_settings(worker=True)
        )

        async def _boom(_driver: str, _timeout: int) -> SandboxResult:
            raise SandboxUnavailable("runner unreachable")

        monkeypatch.setattr(sandbox_dispatch, "_run_driver_in_sandbox", _boom)
        reg = _registry_with_generated()

        result = await sandbox_dispatch.invoke_generated_tool("normalize_rows", reg, {"path": "x"})
        assert result is not None and result.success is False
        assert "sandbox unavailable" in (result.error or "")

    @pytest.mark.asyncio
    async def test_subprocess_no_worker_still_inproc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A host CLI in subprocess mode is unaffected — in-process (None)."""
        monkeypatch.setattr(sandbox_dispatch, "_code_exec_mode", lambda: "subprocess")
        monkeypatch.setattr(sandbox_dispatch, "_isolation_settings", lambda: _iso_settings(worker=False))

        async def _should_not_run(_d: str, _t: int) -> SandboxResult:
            raise AssertionError("sandbox must not run for a non-isolated runtime")

        monkeypatch.setattr(sandbox_dispatch, "_run_driver_in_sandbox", _should_not_run)
        reg = _registry_with_generated()
        assert await sandbox_dispatch.invoke_generated_tool("normalize_rows", reg, {}) is None


# ─── helpers ─────────────────────────────────────────────────────────


def _noop_metric() -> Any:
    """A no-op replacement for ``_record_tool_metric`` (avoids the DB)."""

    async def _record(*_a: Any, **_k: Any) -> None:  # noqa: ANN401
        return None

    return _record
