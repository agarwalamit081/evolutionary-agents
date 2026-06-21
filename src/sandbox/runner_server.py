"""HTTP server: the remote no-DinD code-execution runner (Phase 3b/c).

This runs IN THE RUNNER CONTAINER — the agent's single sink for executing
generated code. It exposes a tiny HTTP API::

    POST /execute  {"code": str, "timeout": float, "test_code"?: str}
                  -> {"success", "exit_code", "stdout", "stderr",
                      "duration_seconds", "timed_out"}   (SandboxResult shape)
    GET  /health   -> {"status": "ok"}

The runner executes submitted Python as a constrained subprocess IN ITS OWN
container. It has NO Docker socket (so no Docker-in-Docker), NO
DATABASE/REDIS/search credentials, and (in compose) NO internet egress — it
lives on an ``internal: true`` network. The disposable container itself is the
isolation boundary (per-invocation gVisor/Kata hardening is a Phase-5 doc item).

This module is intentionally SELF-CONTAINED: it has NO ``src.*`` imports, so
``Dockerfile.runner`` copies only this one file into a least-privilege image
that holds neither the DB models, the gateway, nor Settings-with-DB-URLs. It
runs standalone (``python /app/runner_server.py``); the companion client is
:mod:`src.sandbox.runner_client`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import aiofiles
from aiohttp import web
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class RunnerServerSettings(BaseSettings):
    """Runtime config for the runner server, read from the process env ONLY.

    ``env_file=None`` deliberately: the runner container holds NO ``.env`` (it
    has no DB/Redis/search creds) and we never want it to silently load a stray
    ``.env`` if one were bind-mounted. It reads exactly the env vars compose
    sets (``RUNNER_*``).
    """

    runner_host: str = "0.0.0.0"
    runner_port: int = 8090
    # RESULTS_ROOT shared with the worker via the turing-workspace volume. The
    # subprocess CWD is this path's PARENT so a script's relative
    # ``results/<file>`` write lands on the shared volume — the same contract
    # as the host code_executor and the docker sandbox's /workspace mount.
    runner_results_root: str = "/home/turing/.turing/results"
    # Hard cap on a requested timeout. The client also caps; this is the
    # server-side enforcement (defense in depth — a hand-crafted request cannot
    # bypass the client to hold a subprocess open for hours).
    runner_max_timeout_s: int = 300
    # Interpreter for the executed subprocess. Empty -> sys.executable at
    # runtime (the runner image's python, whose site-packages mirror the
    # allowlist).
    runner_python: str = ""

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )


def _python(settings: RunnerServerSettings) -> str:
    """Resolve the interpreter (config override or the running one)."""
    return settings.runner_python or sys.executable


# aiohttp 3.9+ deprecates bare-string app keys (NotAppKeyWarning); a typed
# AppKey also gives static type-checking on the stored settings.
_SETTINGS_KEY = web.AppKey("settings", RunnerServerSettings)


async def _run_subprocess(
    settings: RunnerServerSettings,
    full_script: str,
    timeout: float,
) -> dict[str, Any]:
    """Execute ``full_script`` as a python subprocess; return the result dict.

    CWD = parent of ``runner_results_root`` so relative ``results/<file>``
    writes land on the shared volume. On timeout the proc is killed and a
    ``timed_out`` result is returned (never raised — a timeout is a result, not
    an infrastructure failure).
    """
    cwd = str(Path(settings.runner_results_root).parent)
    Path(settings.runner_results_root).mkdir(parents=True, exist_ok=True)
    Path(cwd).mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    tmp_dir = tempfile.mkdtemp(prefix="runner_exec_")
    script_path = Path(tmp_dir) / "script.py"
    try:
        async with aiofiles.open(str(script_path), mode="w") as f:
            await f.write(full_script)

        proc = await asyncio.create_subprocess_exec(
            _python(settings),
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            duration = time.monotonic() - start
            exit_code = proc.returncode
            return {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "stdout": stdout_b.decode("utf-8", errors="replace"),
                "stderr": stderr_b.decode("utf-8", errors="replace"),
                "duration_seconds": round(duration, 4),
                "timed_out": False,
            }
        except asyncio.TimeoutError:
            # Kill the runaway proc so it cannot outlive the request / leak.
            proc.kill()
            await proc.wait()
            duration = time.monotonic() - start
            logger.warning("Runner subprocess timed out after {:.1f}s", duration)
            return {
                "success": False,
                "exit_code": None,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout}s",
                "duration_seconds": round(duration, 4),
                "timed_out": True,
            }
    finally:
        try:
            script_path.unlink(missing_ok=True)
            Path(tmp_dir).rmdir()
        except OSError:
            pass


async def handle_execute(request: web.Request) -> web.Response:
    """POST /execute — run submitted code in an isolated subprocess."""
    settings: RunnerServerSettings = request.app[_SETTINGS_KEY]

    try:
        body = await request.json()
    except json.JSONDecodeError:
        logger.warning("POST /execute rejected: request body is not JSON")
        raise web.HTTPBadRequest(text="request body must be JSON")

    if not isinstance(body, dict) or not isinstance(body.get("code"), str):
        logger.warning("POST /execute rejected: missing or non-string 'code'")
        raise web.HTTPBadRequest(text="missing or non-string 'code'")

    code: str = body["code"]
    test_code = body.get("test_code")
    if test_code is not None and not isinstance(test_code, str):
        logger.warning("POST /execute rejected: 'test_code' is not a string")
        raise web.HTTPBadRequest(text="'test_code' must be a string")

    try:
        timeout = float(body.get("timeout", 60))
    except (TypeError, ValueError):
        logger.warning("POST /execute rejected: 'timeout' is not a number")
        raise web.HTTPBadRequest(text="'timeout' must be a number")
    # Server-side cap (defense in depth; the client also caps). Floor at 0.1s.
    timeout = max(0.1, min(timeout, float(settings.runner_max_timeout_s)))

    full_script = (
        code if test_code is None else f"{code}\n\n# --- test ---\n{test_code}"
    )
    # Per-execute audit trace (observability): the runner is the agent's single
    # sink for LLM-generated code, so each execution MUST be attributable — both
    # to prove the no-DinD path fired (vs a silent host-subprocess fallback in
    # the worker's code_executor) and to make a sandbox that runs untrusted code
    # auditable. Logs METADATA only (code size + outcome), never the code body,
    # which can be large / may echo tool input.
    logger.info(
        "POST /execute → running (code_chars={}, has_test={}, timeout={:.1f}s)",
        len(code),
        test_code is not None,
        timeout,
    )
    result = await _run_subprocess(settings, full_script, timeout)
    logger.info(
        "POST /execute ← done (success={}, exit_code={}, duration={:.3f}s, "
        "timed_out={})",
        result.get("success"),
        result.get("exit_code"),
        float(result.get("duration_seconds", 0.0)),
        result.get("timed_out"),
    )
    return web.json_response(result)


async def handle_health(_request: web.Request) -> web.Response:
    """GET /health — liveness/readiness for the compose healthcheck."""
    return web.json_response({"status": "ok"})


def build_app(settings: RunnerServerSettings | None = None) -> web.Application:
    """Construct the aiohttp Application (tests call this directly)."""
    settings = settings or RunnerServerSettings()
    app = web.Application(client_max_size=10 * 1024 * 1024)
    app[_SETTINGS_KEY] = settings
    app.router.add_post("/execute", handle_execute)
    app.router.add_get("/health", handle_health)
    return app


def main() -> None:
    """Entrypoint: ``python /app/runner_server.py`` (or ``-m`` form in dev)."""
    settings = RunnerServerSettings()
    logger.info(
        "runner server starting on {}:{} (results_root={}, python={})",
        settings.runner_host,
        settings.runner_port,
        settings.runner_results_root,
        _python(settings),
    )
    web.run_app(
        build_app(settings),
        host=settings.runner_host,
        port=settings.runner_port,
    )


if __name__ == "__main__":
    main()
