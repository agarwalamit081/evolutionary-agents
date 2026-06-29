"""Route an LLM-generated tool's invocation through the code-exec sandbox.

THE GAP THIS CLOSES. A dynamically-created tool (``tool_create`` →
``ToolGenerator`` → ``ToolRegistry.register(generated=True, handler_code=…)``)
registers its handler as an **in-process async callable** materialized from
untrusted LLM source via ``exec``. Without this module the execute node calls
that callable directly inside the worker — so generated code runs with the
worker's FULL access (DATABASE_URL, REDIS_URL, the process FS, the host
network) and the no-DinD runner sandbox (network off, no creds) is bypassed
entirely. Only the ``code_executor`` builtin routed through the sandbox; a
created tool did not.

THE FIX. When the operator has opted into a sandboxed code-exec mode
(``CODE_EXECUTOR_MODE`` ∈ {``docker``, ``runner``}), the execute node routes a
generated tool's invocation HERE instead of calling the in-process handler.
We synthesize a small *driver* — the handler source plus an ``asyncio.run``
harness that calls it with the tool-call args and prints the return value
under sentinel markers — and run that driver through the SAME
``SandboxExecutor.execute_runtime_code`` surface ``code_executor`` uses. So a
generated tool is isolated exactly like a one-off code snippet: network off,
no creds, capped. A hand-written builtin is never generated and always runs
in-process (it is trusted and needs gateway/Redis access).

FAIL-CLOSED. If the sandbox is unavailable (``SandboxUnavailable`` — docker
missing / daemon down / image absent / runner down) we return a failed
``ToolResult`` and DO NOT fall back to the in-process handler. Silently
running untrusted LLM code in the worker when the sandbox is down would
re-open the exact gap this module closes — and an attacker who can influence
tool creation (prompt injection) could deliberately DoS the runner to force
the in-process path. The run surfaces the error and re-plans instead.

SUBPROCESS MODE IS UNCHANGED FOR THE HOST CLI. When ``CODE_EXECUTOR_MODE=subprocess``
(the dev default; the whole agent runs on the host with full access anyway) this
module returns ``None`` and the execute node calls the in-process handler as
before — zero behavior change for host-run / test / non-isolated deployments.

#2 — WORKER-DEFAULT ISOLATION. A worker can land in subprocess mode (operator
override, stale image) yet still be the long-lived process accumulating
generated tools, so isolation there must NOT hinge solely on the mode knob.
When ``TURING_WORKER_PROCESS`` marks this process as the worker AND
``ISOLATION_DEFAULT_TO_SANDBOX`` is on, isolation engages even in subprocess
mode: ``AUTO_PROMOTE_SUBPROCESS_TO_RUNNER`` routes the driver to the runner
surface (the only no-DinD sandbox available without docker), fail-closed if the
runner is unreachable. The local CLI sets no worker flag, so it is unaffected.
"""


from __future__ import annotations

import ast
import json
import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config.settings import ToolSandboxSettings, get_settings
from src.graph.models import ToolResult
from src.sandbox.executor import SandboxResult, SandboxUnavailable
from src.tools._paths import results_root

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry

# Result sentinels. The handler may print to stdout itself (logging, debug), so
# the driver writes the RETURN value between these markers and the dispatcher
# extracts only that slice — mirroring the in-process contract (``str(result)``
# only, not the handler's stray prints). json-encoded so a multi-line / quote-
# bearing return value survives the stdout round-trip.
_RESULT_BEGIN = "__TURING_TOOL_RESULT_BEGIN__"
_RESULT_END = "__TURING_TOOL_RESULT_END__"
_RESULT_SLICE_RE = re.compile(
    re.escape(_RESULT_BEGIN) + r"\s*(\{.*?\})\s*" + re.escape(_RESULT_END),
    re.DOTALL,
)

# Modes in which generated-tool invocation is isolated. ``subprocess`` (the
# default) is intentionally excluded: there is no isolation concept on a
# host-run agent, and generated code there is no less trusted than the
# one-off ``code_executor`` snippets that also run in a host subprocess.
_SANDBOXED_MODES = frozenset({"docker", "runner"})

# One-time guard so the subprocess→runner promotion WARNING (#2) logs once per
# process, not on every generated-tool call. Module-level by design: the
# promotion is a steady-state condition (the worker is misconfigured), so a
# single notice is enough; per-call spam would bury real signal in the logs.
_subprocess_promotion_warned: bool = False


def _code_exec_mode() -> str:
    """Call-time read of the code-exec isolation mode (never capture at import)."""
    return get_settings().tool_sandbox.code_executor_mode


def _sandbox_timeout() -> int:
    """Mirror code_executor's sandbox timeout for generated-tool dispatch."""
    return get_settings().tool_sandbox.code_executor_sandbox_timeout


def _isolation_settings() -> ToolSandboxSettings:
    """The ToolSandboxSettings slice governing generated-tool isolation (#2).

    Wrapped so unit tests can stub the worker-default knobs without touching
    ``.env``. Read at call time (never captured at import) like ``_code_exec_mode``.
    """
    return get_settings().tool_sandbox


def _is_isolated_runtime() -> bool:
    """True when a generated tool's invocation should be isolated by default (#2).

    Isolation applies when EITHER:

    * the operator explicitly opted into a sandboxed code-exec mode
      (``CODE_EXECUTOR_MODE`` ∈ {docker, runner}); OR
    * this process IS the long-lived worker/runner (``TURING_WORKER_PROCESS``)
      AND the master ``ISOLATION_DEFAULT_TO_SANDBOX`` switch is on AND
      ``AUTO_PROMOTE_SUBPROCESS_TO_RUNNER`` can route it to a safe surface — so a
      worker that lands in subprocess mode (operator override / stale image)
      STILL isolates untrusted LLM handler code instead of running it in-process
      with full DB/Redis/FS access (the #288 gap).

    The local host CLI sets none of the worker knobs, so it stays
    subprocess / in-process — unchanged. Promotion is a hard gate here: without it
    there is no safe isolation surface in subprocess mode, so the gap is left
    as-is (an explicit operator choice) rather than failing every call.
    """
    if _code_exec_mode() != "subprocess":
        return True  # explicit opt-in (docker/runner) — today's behavior
    ts = _isolation_settings()
    return bool(
        ts.isolation_default_to_sandbox
        and ts.auto_promote_subprocess_to_runner
        and ts.worker_process
    )


def _effective_isolation_mode() -> str:
    """The sandbox surface to run an isolated generated tool through (#2).

    Reached only after ``_is_isolated_runtime()`` returned True. Returns the
    configured ``CODE_EXECUTOR_MODE`` when it is itself a sandboxed mode;
    otherwise '``runner``' — the only no-DinD surface available without docker.
    A subprocess value here means the worker-process default engaged isolation
    and ``AUTO_PROMOTE_SUBPROCESS_TO_RUNNER`` promoted it to the runner.
    """
    mode = _code_exec_mode()
    return mode if mode in _SANDBOXED_MODES else "runner"


def _extract_async_func_name(handler_code: str) -> str | None:
    """Return the name of the (first) ``async def`` in ``handler_code``.

    The materializer requires exactly one async function; we reuse the same AST
    walk to name it in the driver. Returns None on a syntax error / no async
    func (the caller treats that as a dispatch failure rather than running the
    untrusted source blindly).
    """
    try:
        tree = ast.parse(handler_code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            return node.name
    return None


def _build_driver(handler_code: str, func_name: str, args: dict[str, Any]) -> str:
    """Assemble the sandboxed driver script.

    Layout: the write-bootstrap (auto-creates ``results/<file>`` parent dirs,
    identical to ``code_executor``) + the handler source + a harness that
    JSON-decodes the args, runs the async handler, and prints the return value
    under the result sentinels. The args travel as a ``repr``-escaped JSON
    literal — a Python string literal that round-trips exactly, so a tool-call
    arg containing quotes / newlines cannot break out of the literal.
    """
    from src.tools.builtin.code_executor import _write_bootstrap

    args_json = json.dumps(args)  # JSON (controlled) -> safe to embed as a literal
    parts: list[str] = [
        _write_bootstrap(""),  # mkdir-only shim; handler runs inside the sandbox
        handler_code,
        "\n",
        "# --- turing generated-tool sandbox dispatch driver (untrusted handler) ---\n",
        "import asyncio as _turing_aio\n",
        "import json as _turing_json\n",
        f"_TURING_ARGS = _turing_json.loads({args_json!r})\n",
        "def _turing_main():\n",
        f"    return _turing_aio.run({func_name}(**_TURING_ARGS))\n",
        "_result = _turing_main()\n",
        "import sys as _turing_sys\n",
        f"_turing_sys.stdout.write('\\n{_RESULT_BEGIN}\\n')\n",
        '_turing_sys.stdout.write(_turing_json.dumps({"output": str(_result)}))\n',
        f"_turing_sys.stdout.write('\\n{_RESULT_END}\\n')\n",
    ]
    return "".join(parts)


def _extract_output(result: SandboxResult, tool_name: str) -> tuple[bool, str, str | None]:
    """Map a ``SandboxResult`` to (success, output, error).

    On success the driver prints the return value under the sentinels; we parse
    that slice. A non-zero exit / timeout means the handler raised before
    printing — the traceback lives in stderr and becomes the error (the call is
    NEVER re-run in-process). A success with no sentinel (a handler that
    ``os._exit``-ed or similar) degrades to full stdout so the LLM still gets a
    useful message rather than an empty one.
    """
    if result.timed_out:
        return False, "", f"Execution timed out after {result.duration_seconds}s"

    if not result.success:
        # The handler raised: stderr holds the traceback. Cap it so a verbose
        # error doesn't blow the tool-message budget.
        err = (result.stderr or "").strip() or f"exit code {result.exit_code}"
        return False, "", f"generated tool raised: {err[:1500]}"

    stdout = result.stdout or ""
    match = _RESULT_SLICE_RE.search(stdout)
    if match:
        try:
            payload = json.loads(match.group(1))
            return True, str(payload.get("output", ""))[:2000], None
        except (ValueError, TypeError) as exc:
            logger.warning(
                "generated tool {}: result sentinel held non-JSON payload ({}); "
                "falling back to full stdout",
                tool_name, exc,
            )
    # Success but no parseable sentinel — surface whatever stdout there is.
    return True, stdout.strip()[:2000], None


async def _run_driver_in_sandbox(driver: str, timeout: int) -> SandboxResult:
    """Run the driver through the SAME docker/runner surface code_executor uses.

    Mirrors ``code_executor._run_in_sandbox``: a ``SandboxExecutor`` wired from
    ``ToolSandboxSettings`` with the results dir mounted read-write, so a
    handler's ``results/<file>`` writes persist on the shared volume. Raises
    ``SandboxUnavailable`` on any infrastructure problem (caller fail-closes).
    """
    from types import SimpleNamespace

    from src.sandbox.executor import SandboxExecutor

    ts = get_settings().tool_sandbox
    mount_src = ts.code_executor_results_mount or str(results_root())
    # Ensure the host mount target exists so docker can bind it (harmless for
    # runner mode, which uses its own results dir under the shared volume).
    from pathlib import Path

    Path(mount_src).mkdir(parents=True, exist_ok=True)

    sandbox = SandboxExecutor(
        SimpleNamespace(
            # #2: use the EFFECTIVE isolation mode, not the raw configured mode,
            # so a worker-process default that promoted subprocess→runner routes
            # the driver to the runner surface (not a host subprocess).
            evolution_sandbox_mode=_effective_isolation_mode(),
            evolution_sandbox_image=ts.code_executor_sandbox_image,
            evolution_sandbox_memory_mb=ts.code_executor_sandbox_memory_mb,
            evolution_sandbox_timeout=timeout,
        )
    )
    return await sandbox.execute_runtime_code(
        driver,
        timeout=timeout,
        workdir=mount_src,
        workdir_dest=ts.code_executor_sandbox_workdir_dest,
    )


async def invoke_generated_tool(
    tool_name: str,
    tools: ToolRegistry,
    args: dict[str, Any],
) -> ToolResult | None:
    """Isolate a generated tool's invocation; return None if not applicable.

    Returns a ``ToolResult`` when the call was handled — either a successful
    isolated run or a fail-closed error (sandbox down / handler raised) — so the
    execute node never falls back to the in-process handler for untrusted code.
    Returns ``None`` when isolation does not apply (subprocess mode), signaling
    the caller to run the in-process handler as before.

    Args:
        tool_name: The generated tool being invoked.
        tools: The registry (for the handler source lookup).
        args: Parsed tool-call arguments (JSON-native).

    Returns:
        A ``ToolResult`` if isolated/dispatched, else ``None``.
    """
    if not _is_isolated_runtime():
        return None  # not an isolated runtime — caller runs the in-process handler

    # Only GENERATED tools are isolated. A hand-written builtin is trusted code
    # (and typically needs gateway/Redis/DB access a sandbox denies), so it runs
    # in-process regardless of mode — returning None signals the caller to do so.
    if not tools.is_generated(tool_name):
        return None

    handler_code = tools.get_handler_code(tool_name)
    if not handler_code:
        # A generated tool with no recoverable source cannot be sandboxed; rather
        # than run an untrusted in-process handler, fail closed. (Should not
        # happen: the generator/persister always store handler_code.)
        logger.warning(
            "generated tool '{}' has no handler_code in a sandboxed mode; "
            "failing closed instead of running in-process",
            tool_name,
        )
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output="",
            error=(
                f"generated tool '{tool_name}' could not be isolated "
                "(source unavailable); refusing to run in-process"
            ),
        )

    func_name = _extract_async_func_name(handler_code)
    if not func_name:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output="",
            error=f"generated tool '{tool_name}' handler has no async function",
        )

    driver = _build_driver(handler_code, func_name, args)
    timeout = _sandbox_timeout()
    # #2: the actual surface may differ from the configured mode when the
    # worker-process default promoted subprocess→runner. Log the EFFECTIVE mode
    # (what the untrusted code actually ran under), and warn ONCE on promotion
    # so a misconfigured worker surfaces without per-call spam.
    effective_mode = _effective_isolation_mode()
    global _subprocess_promotion_warned
    if effective_mode != _code_exec_mode() and not _subprocess_promotion_warned:
        _subprocess_promotion_warned = True
        logger.warning(
            "Worker-process default isolation engaged for generated tools in "
            "subprocess mode — promoting to the '{}' surface "
            "(AUTO_PROMOTE_SUBPROCESS_TO_RUNNER). Set CODE_EXECUTOR_MODE=runner "
            "to make isolation explicit.",
            effective_mode,
        )
    logger.info(
        "Isolating generated tool '{}' via {} sandbox (handler {} chars)",
        tool_name, effective_mode, len(handler_code),
    )

    try:
        result = await _run_driver_in_sandbox(driver, timeout)
    except SandboxUnavailable as exc:
        # FAIL-CLOSED: never fall back to the in-process handler. The operator
        # opted into isolation; running untrusted LLM code in the worker when
        # the sandbox is down re-opens the gap. The run surfaces this and the
        # planner can retry / re-plan.
        logger.warning(
            "generated tool '{}' sandbox unavailable ({}); failing closed",
            tool_name, exc,
        )
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output="",
            error=(
                f"generated tool '{tool_name}' sandbox unavailable ({exc}); "
                "isolation opted-in, refusing in-process fallback"
            ),
        )

    success, output, error = _extract_output(result, tool_name)
    return ToolResult(
        tool_name=tool_name,
        success=success,
        output=output,
        error=error,
        metadata={"isolated": True, "mode": effective_mode},
    )
