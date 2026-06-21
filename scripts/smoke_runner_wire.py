#!/usr/bin/env python
"""Cross-process wire smoke for the no-DinD runner (Phase 3b/c).

The one wire gap not covered by the unit suites: ``RunnerClient`` (httpx) → the
REAL ``runner_server`` (aiohttp) over actual HTTP, in separate processes.
``test_runner_client`` mocks httpx's transport; ``test_runner_server`` uses an
in-process aiohttp TestClient. This proves the two ends agree on the wire
contract end-to-end with REAL subprocess execution — the way the worker talks to
the runner in compose.

No Docker, no LLM API key. Starts the server on an ephemeral port, runs three
cases (success / script-failure / timeout) through the client, then tears down.

Usage:
    python scripts/smoke_runner_wire.py
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER = REPO_ROOT / "src" / "sandbox" / "runner_server.py"
# `python scripts/...` puts scripts/ (not the repo root) on sys.path[0]; add the
# repo root so `from src.sandbox.runner_client import RunnerClient` resolves. The
# server subprocess is self-contained (no `src` imports) so it needs no such help.
sys.path.insert(0, str(REPO_ROOT))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_health(base_url: str, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{base_url}/health")
                if r.status_code == 200:
                    return
            except httpx.HTTPError as exc:
                last_exc = exc
            await asyncio.sleep(0.2)
    raise RuntimeError(f"runner server never became healthy: {last_exc}")


async def _main() -> int:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    results_root = tempfile.mkdtemp(prefix="runner_wire_smoke_")

    env = {
        **os.environ,
        "RUNNER_HOST": "127.0.0.1",
        "RUNNER_PORT": str(port),
        "RUNNER_RESULTS_ROOT": str(Path(results_root) / "results"),
        "RUNNER_MAX_TIMEOUT_S": "5",
    }
    proc = subprocess.Popen(
        [sys.executable, str(SERVER)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )
    try:
        await _wait_for_health(base_url)
        print(f"[smoke] runner healthy at {base_url}")

        from src.sandbox.runner_client import RunnerClient

        client = RunnerClient(base_url, max_timeout_s=5)

        # 1. success
        r = await client.execute("print(6 * 7)")
        assert r.success and r.exit_code == 0, r
        assert r.stdout.strip() == "42", r.stdout
        print(f"[smoke] success     -> exit={r.exit_code} stdout={r.stdout.strip()!r}  OK")

        # 2. script failure (non-zero) is a RESULT, not raised
        r = await client.execute("raise ValueError('boom-wire')")
        assert not r.success and r.exit_code != 0, r
        assert "boom-wire" in r.stderr, r.stderr
        print(f"[smoke] script-fail -> exit={r.exit_code} stderr-has-boom  OK")

        # 3. timeout is a timed_out RESULT, not raised
        r = await client.execute("while True:\n    pass", timeout=1)
        assert r.timed_out and not r.success, r
        print(f"[smoke] timeout     -> timed_out={r.timed_out}  OK")

        print("[smoke] ALL CASES PASSED — httpx client <-> aiohttp server wire is sound")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
