"""Unit tests for the worker entrypoint ``python -m src.worker`` (Phase 3 staged).

The entrypoint wires the Redis-backed queue + status store + consumer and drains
``turing:runs`` forever. Its one testable, role-split-critical piece is
``_resolve_consumer_name``: under ``deploy.replicas``, each worker container MUST
own a DISTINCT consumer name, or multiple processes share one pending-entries list
and break ``XAUTOCLAIM`` crash-recovery. These tests pin the two behaviors —

- explicit ``WORKER_CONSUMER_NAME`` honored verbatim (single-worker opt-in);
- env unset → a per-process-unique ``worker-{hostname}-{pid}`` (replica-safe).

The full ``_run`` loop is exercised in the live e2e (P3e-staged); here we keep the
test hermetic — no Redis, no agent, no LLM — by targeting the pure function.
"""

from __future__ import annotations

import os
import socket
from importlib import reload

import pytest

from src.config.settings import WorkerSettings


def _fresh_worker_settings() -> WorkerSettings:
    """Build a WorkerSettings WITHOUT reading the repo ``.env`` (hermetic).

    pydantic-settings reads ``.env`` by default; ``_env_file=None`` disables file
    loading so the test is driven solely by the monkeypatched env / explicit args.
    """
    return WorkerSettings(_env_file=None)


class TestResolveConsumerName:
    def test_explicit_env_is_honored_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WORKER_CONSUMER_NAME=pinned → settings value returned as-is (single-worker)."""
        monkeypatch.setenv("WORKER_CONSUMER_NAME", "pinned")
        worker = _fresh_worker_settings()
        assert worker.consumer_name == "pinned"

        # Imported lazily so the worker package (not yet under test elsewhere) is fresh.
        from src.worker import __main__ as worker_main

        assert worker_main._resolve_consumer_name(worker) == "pinned"

    def test_env_unset_derives_hostname_pid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No WORKER_CONSUMER_NAME → worker-{hostname}-{pid} (default per-process name)."""
        monkeypatch.delenv("WORKER_CONSUMER_NAME", raising=False)
        worker = _fresh_worker_settings()
        from src.worker import __main__ as worker_main

        name = worker_main._resolve_consumer_name(worker)
        expected = f"worker-{socket.gethostname()}-{os.getpid()}"
        assert name == expected
        # Must NOT leak the static default — a replicated worker would collide on it.
        assert name != "worker-1"

    def test_derived_name_distinguishes_replicas_by_hostname(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two containers (different hostnames) → different consumer names.

        This is the replica-uniqueness guarantee for ``deploy.replicas``: with the
        env unset, the hostname dimension alone makes each replica's name distinct
        even if pids happened to coincide.
        """
        monkeypatch.delenv("WORKER_CONSUMER_NAME", raising=False)
        worker = _fresh_worker_settings()
        from src.worker import __main__ as worker_main

        monkeypatch.setattr(worker_main.socket, "gethostname", lambda: "replica-a")
        name_a = worker_main._resolve_consumer_name(worker)
        monkeypatch.setattr(worker_main.socket, "gethostname", lambda: "replica-b")
        name_b = worker_main._resolve_consumer_name(worker)

        assert name_a.startswith("worker-replica-a-")
        assert name_b.startswith("worker-replica-b-")
        assert name_a != name_b

    def test_derived_name_distinguishes_restarts_by_pid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same hostname, different pid → different names (stable for a process life)."""
        monkeypatch.delenv("WORKER_CONSUMER_NAME", raising=False)
        worker = _fresh_worker_settings()
        from src.worker import __main__ as worker_main

        monkeypatch.setattr(worker_main.os, "getpid", lambda: 111)
        name_1 = worker_main._resolve_consumer_name(worker)
        monkeypatch.setattr(worker_main.os, "getpid", lambda: 222)
        name_2 = worker_main._resolve_consumer_name(worker)

        assert name_1.endswith("-111")
        assert name_2.endswith("-222")
        assert name_1 != name_2


class TestModuleImportability:
    def test_module_imports_and_main_is_callable(self) -> None:
        """``python -m src.worker`` needs a callable ``main`` and the executor wired in."""
        import src.worker.__main__ as worker_main

        # Reload to be robust to the lazy-import-order in other tests.
        reload(worker_main)
        assert callable(worker_main.main)
        # The executor must be the canonical run path (no main.py/Click coupling).
        from src.worker.executors import default_agent_executor

        assert worker_main.default_agent_executor is default_agent_executor
