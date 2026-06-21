"""HTTP client for the remote no-DinD code-execution runner (Phase 3b/c).

The runner is a dedicated container exposing a tiny HTTP server that executes
submitted Python as a constrained subprocess in its OWN container — no Docker
socket (so no Docker-in-Docker), no DATABASE/REDIS/search credentials, and no
internet egress (it lives on a compose ``internal: true`` network). The worker
POSTs code here instead of calling ``docker.from_env()``, so the worker needs
NO Docker access at all and its compose service drops the ``docker.sock`` mount.

This module is the wire protocol between the caller side (``SandboxExecutor`` /
``code_executor`` in the worker) and ``src.sandbox.runner_server`` (the runner
container). It raises :class:`SandboxUnavailable` on ANY infrastructure problem
(connection refused, connect timeout, HTTP 5xx, malformed JSON) so callers apply
their OWN fallback policy — ``code_executor`` falls back to its host subprocess,
the evolution path degrades to a subprocess — EXACTLY mirroring the docker-mode
``SandboxUnavailable`` contract (see ``SandboxExecutor._run_docker``). A script
that runs and exits non-zero, or times out, is a normal (failed)
:class:`SandboxResult` and is NEVER raised — re-running untrusted code on the
host would defeat the isolation an operator opted into.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from src.sandbox.executor import SandboxResult, SandboxUnavailable

# Used only when the caller passes ``timeout=None`` (the executor always passes
# a concrete value; this is the defensive default for direct callers).
_DEFAULT_TIMEOUT_S = 60.0


class RunnerClient:
    """Async HTTP client for the runner's ``POST /execute`` endpoint.

    Construct with an explicit ``base_url`` (tests) or
    :meth:`from_settings` (production reads ``Settings.runner``). Each
    :meth:`execute` call uses a short-lived ``httpx.AsyncClient`` — the runner
    is a hot service and a fresh client per call keeps the worker side
    connection-light without managing a shared client lifecycle across the
    long-lived worker loop. Tests inject an ``httpx.MockTransport`` via the
    ``transport`` kwarg (the production path leaves it ``None``).
    """

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout_s: float = 5.0,
        max_timeout_s: int = 300,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._connect_timeout_s = connect_timeout_s
        self._max_timeout_s = max_timeout_s
        self._transport = transport

    @classmethod
    def from_settings(cls, settings: Any) -> RunnerClient:
        """Build from a ``RunnerSettings`` (or a root ``Settings`` with ``.runner``)."""
        s = getattr(settings, "runner", settings)
        return cls(
            base_url=s.runner_url,
            connect_timeout_s=s.runner_connect_timeout_s,
            max_timeout_s=s.runner_max_timeout_s,
        )

    async def execute(
        self,
        code: str,
        *,
        timeout: float | None = None,
        test_code: str | None = None,
    ) -> SandboxResult:
        """POST code to the runner and return its :class:`SandboxResult`.

        Raises :class:`SandboxUnavailable` on any infrastructure problem
        (runner down / connect timeout / 5xx / malformed JSON). A non-zero exit
        or a subprocess timeout is a normal ``SandboxResult`` (``success=False``).
        """
        effective_timeout = (
            timeout if timeout is not None else _DEFAULT_TIMEOUT_S
        )
        # Cap a runaway caller so the runner never holds a subprocess for hours.
        if effective_timeout > self._max_timeout_s:
            logger.debug(
                "Runner execute timeout {}s capped to max {}s",
                effective_timeout,
                self._max_timeout_s,
            )
            effective_timeout = self._max_timeout_s

        payload: dict[str, Any] = {
            "code": code,
            "timeout": effective_timeout,
        }
        if test_code is not None:
            payload["test_code"] = test_code

        # The httpx READ timeout must exceed the runner's subprocess timeout,
        # otherwise httpx would abort the call before the runner can report a
        # (legitimate) subprocess-timeout result. ``connect`` bounds the
        # handshake so a down runner fails fast; read/write carry timeout+slack.
        request_timeout = self._connect_timeout_s + effective_timeout + 5.0
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._connect_timeout_s,
                    read=request_timeout,
                    write=request_timeout,
                    pool=self._connect_timeout_s,
                ),
                transport=self._transport,
            ) as client:
                resp = await client.post(
                    f"{self._base_url}/execute", json=payload
                )
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
        ) as exc:
            raise SandboxUnavailable(f"runner unreachable: {exc}") from exc
        except httpx.HTTPError as exc:
            raise SandboxUnavailable(f"runner HTTP error: {exc}") from exc

        if resp.status_code >= 500:
            raise SandboxUnavailable(
                f"runner returned HTTP {resp.status_code}"
            )
        if resp.status_code != 200:
            raise SandboxUnavailable(
                f"runner returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise SandboxUnavailable(
                f"runner returned non-JSON body: {exc}"
            ) from exc

        return self._parse(data)

    @staticmethod
    def _parse(data: dict[str, Any]) -> SandboxResult:
        """Map the runner's JSON payload to a :class:`SandboxResult`."""
        try:
            return SandboxResult(
                success=bool(data.get("success", False)),
                exit_code=(
                    int(data["exit_code"])
                    if data.get("exit_code") is not None
                    else None
                ),
                stdout=str(data.get("stdout", "")),
                stderr=str(data.get("stderr", "")),
                duration_seconds=float(data.get("duration_seconds", 0.0)),
                memory_mb=None,
                timed_out=bool(data.get("timed_out", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SandboxUnavailable(
                f"runner returned malformed payload: {exc}"
            ) from exc
