"""#11 — per-tool success contract: pure evaluator + registry + execute wiring.

Three layers, mirroring the implementation:

1. ``evaluate_success`` (``src/tools/success.py``) — pure, no I/O / no settings
   read, so pinned without a registry/gateway. Covers: no-contract ⇒ today's
   behavior, the ``nonempty`` / ``exclude_prefixes`` / ``regex`` clauses, their
   AND semantics, malformed-pattern fail-open, and non-str output coercion.
2. ``ToolRegistry.get_success_contract`` + ``create_default_registry`` — the
   contract is sourced from ``TOOL_ANNOTATIONS`` and round-trips as a copy.
3. The execute chokepoint — the handler-success path records the REAL success
   (contract) to ``tool_call_metrics`` WITHOUT mutating the model-facing
   ``ToolResult``; flag-off / no-contract / bad-regex all stay today's behavior.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.graph.nodes.execute import _evaluate_tool_success, _execute_tool_call
from src.tools import create_default_registry
from src.tools.registry import ToolRegistry
from src.tools.success import SuccessContract, evaluate_success

#: The git_clone contract as wired in TOOL_ANNOTATIONS — mirrors the production
#: annotation so a contract-shape drift here fails loudly.
GIT_CLONE_CONTRACT: SuccessContract = {
    "mode": "nonempty",
    "exclude_prefixes": ["ERROR:", "git_clone is disabled", "git_clone needs"],
}


# ─── Helpers ─────────────────────────────────────────────────────────


def _tc(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """OpenAI-style tool_call dict (function.name + JSON arguments)."""
    return {"function": {"name": name, "arguments": json.dumps(args or {})}}


def _settings(*, contract_enabled: bool = True) -> SimpleNamespace:
    """Minimal settings fake exposing the two seams the execute path reads.

    ``agent.tool_success_contract_enabled`` drives ``_evaluate_tool_success``;
    ``tool_sandbox.code_executor_mode`` is read by ``invoke_generated_tool``'s
    mode probe (which short-circuits to ``None`` for ``"subprocess"`` since it
    is not in ``_SANDBOXED_MODES``, so the hand-registered tool runs in-process).
    """
    return SimpleNamespace(
        agent=SimpleNamespace(tool_success_contract_enabled=contract_enabled),
        tool_sandbox=SimpleNamespace(code_executor_mode="subprocess"),
    )


def _registry(
    name: str,
    handler: Any,
    *,
    contract: SuccessContract | None = None,
) -> ToolRegistry:
    """A one-tool registry; the handler's return value is what's under test."""
    reg = ToolRegistry()
    reg.register(
        name,
        handler,
        description="test tool",
        parameters={},
        success_contract=contract,
    )
    return reg


def _make_handler(return_value: str) -> Any:
    async def _h(**_: Any) -> str:
        return return_value

    return _h


# ─── 1. evaluate_success (pure) ──────────────────────────────────────


class TestEvaluateSuccess:
    def test_no_contract_is_today_behavior(self) -> None:
        assert evaluate_success(None, "anything") is True
        assert evaluate_success({}, "anything") is True

    def test_nonempty_mode(self) -> None:
        contract: SuccessContract = {"mode": "nonempty"}
        assert evaluate_success(contract, "Cloned repo X") is True
        assert evaluate_success(contract, "") is False
        assert evaluate_success(contract, "   ") is False  # stripped

    def test_exclude_prefixes(self) -> None:
        contract: SuccessContract = {"exclude_prefixes": ["ERROR:", "DISABLED"]}
        assert evaluate_success(contract, "Cloned repo X") is True
        assert evaluate_success(contract, "ERROR: boom") is False
        assert evaluate_success(contract, "DISABLED by config") is False
        # Only a matching prefix fails; a non-listed prefix passes.
        assert evaluate_success(contract, "WARN: something") is True

    def test_exclude_prefixes_ignores_non_str_entries(self) -> None:
        contract: SuccessContract = {"exclude_prefixes": ["ERROR:", None, "", 7]}
        # Non-str / empty entries are skipped; only the real prefix applies.
        assert evaluate_success(contract, "ERROR: boom") is False
        assert evaluate_success(contract, "Cloned repo X") is True

    def test_regex_clause(self) -> None:
        contract: SuccessContract = {"regex": r"\b\d+ file\(s\)"}
        assert evaluate_success(contract, "Cloned and indexed from 12 file(s)") is True
        assert evaluate_success(contract, "Cloned and indexed from files") is False

    def test_clauses_are_anded(self) -> None:
        # nonempty + exclude + regex must ALL hold.
        contract: SuccessContract = {
            "mode": "nonempty",
            "exclude_prefixes": ["ERROR:"],
            "regex": r"index",
        }
        assert evaluate_success(contract, "Cloned and indexed 12 file(s)") is True
        assert evaluate_success(contract, "") is False  # nonempty fails
        assert evaluate_success(contract, "ERROR: indexed") is False  # prefix fails
        assert evaluate_success(contract, "Cloned and parsed") is False  # regex fails

    def test_malformed_regex_is_fail_open(self) -> None:
        # A broken pattern must never break a tool call → treated as no-regex
        # clause (the other clauses still apply).
        contract: SuccessContract = {"mode": "nonempty", "regex": "("}
        assert evaluate_success(contract, "Cloned repo X") is True
        # nonempty still catches the empty case even with a bad regex.
        assert evaluate_success(contract, "") is False

    def test_non_str_output_is_coerced(self) -> None:
        contract: SuccessContract = {"mode": "nonempty"}
        assert evaluate_success(contract, 0) is True  # str(0) == "0" is non-empty
        assert evaluate_success(contract, {"k": "v"}) is True  # non-empty
        # An int whose str() starts with a listed prefix proves coercion-to-str
        # happens BEFORE the prefix check (not after a bool short-circuit).
        prefix_contract: SuccessContract = {"exclude_prefixes": ["0"]}
        assert evaluate_success(prefix_contract, 0) is False
        # A whitespace-only str is stripped → empty → nonempty catches it.
        ws_contract: SuccessContract = {"mode": "nonempty"}
        assert evaluate_success(ws_contract, "   ") is False


# ─── 2. registry get_success_contract + create_default_registry ───────


class TestRegistrySuccessContract:
    def test_unknown_tool_returns_none(self) -> None:
        reg = ToolRegistry()
        assert reg.get_success_contract("nope") is None

    def test_registered_contract_round_trips_as_copy(self) -> None:
        reg = ToolRegistry()
        reg.register(
            "t", _make_handler("ok"), description="d", parameters={},
            success_contract=GIT_CLONE_CONTRACT,
        )
        got = reg.get_success_contract("t")
        assert got == GIT_CLONE_CONTRACT
        # Returned dict is a COPY — mutating it must not corrupt the registry.
        assert got is not None
        got["mode"] = "tampered"
        assert reg.get_success_contract("t") == GIT_CLONE_CONTRACT

    def test_registered_without_contract_returns_none(self) -> None:
        reg = ToolRegistry()
        reg.register("t", _make_handler("ok"), description="d", parameters={})
        assert reg.get_success_contract("t") is None


class TestCreateDefaultRegistrySourcesContract:
    def test_git_clone_carries_contract_file_reader_does_not(self) -> None:
        reg = create_default_registry()
        gc = reg.get_success_contract("git_clone")
        assert gc == GIT_CLONE_CONTRACT
        # A tool with no annotation entry for a contract is today's behavior.
        assert reg.get_success_contract("file_reader") is None
        assert reg.get_success_contract("web_search") is None


# ─── 3. execute chokepoint wiring ────────────────────────────────────


class TestExecuteRecordsRealSuccess:
    """The handler-success path records the CONTRACT success to metrics without
    mutating the model-facing ToolResult."""

    @pytest.mark.asyncio
    async def test_error_surface_recorded_as_failure_tr_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.config.settings.get_settings", lambda: _settings(contract_enabled=True)
        )
        recorded = AsyncMock()
        monkeypatch.setattr("src.graph.nodes.execute._record_tool_metric", recorded)

        reg = _registry(
            "fake_clone", _make_handler("ERROR: clone failed"), contract=GIT_CLONE_CONTRACT
        )
        tr = await _execute_tool_call(_tc("fake_clone"), reg, None)

        # Model-facing ToolResult is UNCHANGED — the agent still sees success.
        assert tr.success is True
        assert tr.output == "ERROR: clone failed"
        # But the recorded metric reflects the REAL (contract) outcome.
        recorded.assert_awaited_once()
        assert recorded.call_args.args == ("fake_clone",)
        assert recorded.call_args.kwargs["success"] is False

    @pytest.mark.asyncio
    async def test_success_surface_recorded_as_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.config.settings.get_settings", lambda: _settings(contract_enabled=True)
        )
        recorded = AsyncMock()
        monkeypatch.setattr("src.graph.nodes.execute._record_tool_metric", recorded)

        reg = _registry(
            "fake_clone",
            _make_handler("Cloned repo X and indexed 12 code chunk(s)."),
            contract=GIT_CLONE_CONTRACT,
        )
        tr = await _execute_tool_call(_tc("fake_clone"), reg, None)

        assert tr.success is True
        recorded.assert_awaited_once()
        assert recorded.call_args.kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_flag_off_keeps_today_behavior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # TOOL_SUCCESS_CONTRACT_ENABLED=false → an ERROR surface is recorded as
        # success (today's behavior); the contract is bypassed entirely.
        monkeypatch.setattr(
            "src.config.settings.get_settings", lambda: _settings(contract_enabled=False)
        )
        recorded = AsyncMock()
        monkeypatch.setattr("src.graph.nodes.execute._record_tool_metric", recorded)

        reg = _registry(
            "fake_clone", _make_handler("ERROR: clone failed"), contract=GIT_CLONE_CONTRACT
        )
        await _execute_tool_call(_tc("fake_clone"), reg, None)

        recorded.assert_awaited_once()
        assert recorded.call_args.kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_no_contract_tool_keeps_today_behavior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A tool with NO contract (the overwhelming majority) is unaffected.
        monkeypatch.setattr(
            "src.config.settings.get_settings", lambda: _settings(contract_enabled=True)
        )
        recorded = AsyncMock()
        monkeypatch.setattr("src.graph.nodes.execute._record_tool_metric", recorded)

        reg = _registry("plain", _make_handler("ERROR: whatever"))  # no contract
        tr = await _execute_tool_call(_tc("plain"), reg, None)

        assert tr.success is True
        recorded.assert_awaited_once()
        assert recorded.call_args.kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_bad_regex_contract_is_fail_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A malformed regex in the contract must never break the tool call;
        # evaluate_success drops the regex clause (other clauses still apply).
        monkeypatch.setattr(
            "src.config.settings.get_settings", lambda: _settings(contract_enabled=True)
        )
        recorded = AsyncMock()
        monkeypatch.setattr("src.graph.nodes.execute._record_tool_metric", recorded)

        reg = _registry(
            "bad_regex",
            _make_handler("ok output"),
            contract={"mode": "nonempty", "regex": "("},  # malformed
        )
        tr = await _execute_tool_call(_tc("bad_regex"), reg, None)

        assert tr.success is True
        recorded.assert_awaited_once()
        # nonempty holds ("ok output" is non-empty) → recorded success even
        # though the regex clause is unparseable.
        assert recorded.call_args.kwargs["success"] is True


class TestEvaluateToolSuccessFailOpen:
    """``_evaluate_tool_success`` must fail-open on any internal error."""

    def test_settings_error_is_fail_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If get_settings itself blows up, the helper MUST return True rather
        # than break the tool call.
        def _boom() -> Any:
            raise RuntimeError("settings exploded")

        monkeypatch.setattr("src.config.settings.get_settings", _boom)
        reg = _registry("fake_clone", _make_handler("ERROR: x"), contract=GIT_CLONE_CONTRACT)
        assert _evaluate_tool_success("fake_clone", "ERROR: x", reg) is True
