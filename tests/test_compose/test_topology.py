"""docker-compose.yml topology invariants (Phase 3b/c — no-DinD runner, #228).

Pins the security-critical shape of the compose topology so a future edit cannot
silently regress it:

- the worker has NO Docker socket mount and NO supplementary docker group (no
  Docker-in-Docker / no host-root-equivalent socket on the worker);
- generated code instead runs in the remote no-DinD runner, so the worker routes
  there (CODE_EXECUTOR_MODE=runner, RUNNER_URL) and is attached to the runner's
  network;
- the runner holds NO DB/Redis/search credentials, exposes only :8090 on the
  internal network (no host port), and is egress-isolated on an
  ``internal: true`` network it does not share with postgres/redis/search.

These are source-level assertions over docker-compose.yml (PyYAML resolves the
``<<: *agent-common`` merge, so the worker's inherited env is visible). No Docker
daemon is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose() -> dict:
    assert COMPOSE.exists(), f"docker-compose.yml not found at {COMPOSE}"
    with COMPOSE.open() as f:
        return yaml.safe_load(f)


def _volumes(service: dict) -> list[str]:
    """Normalize a service's volumes to a list of mount strings."""
    vols = service.get("volumes") or []
    out: list[str] = []
    for v in vols:
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            out.append(f"{v.get('source', '')}:{v.get('target', '')}")
    return out


# ── Worker: no Docker socket, routes to runner ──────────────────────────


def test_worker_has_no_docker_socket_mount(compose: dict) -> None:
    """The worker must NOT mount /var/run/docker.sock — it has no Docker access
    (generated code runs in the remote no-DinD runner)."""
    worker = compose["services"]["worker"]
    mounts = _volumes(worker)
    assert not any("docker.sock" in m for m in mounts), (
        f"worker mounts the Docker socket: {mounts}"
    )


def test_worker_has_no_docker_group_add(compose: dict) -> None:
    """No supplementary docker gid (group_add) — the socket is gone, so the gid
    grant that let the non-root user read it is gone too."""
    worker = compose["services"]["worker"]
    assert worker.get("group_add") is None, (
        f"worker still has group_add: {worker.get('group_add')}"
    )


def test_worker_routes_code_exec_to_runner(compose: dict) -> None:
    """The worker forces runner mode + the internal runner URL (inherited from
    the anchor env)."""
    worker = compose["services"]["worker"]
    env = worker["environment"]
    assert env["CODE_EXECUTOR_MODE"] == "runner"
    assert env["EVOLUTION_SANDBOX_MODE"] == "runner"
    assert env["RUNNER_URL"] == "http://runner:8090"


def test_worker_attached_to_runner_network(compose: dict) -> None:
    """The worker must be able to REACH the runner, so it joins turing-runner-net
    (and keeps turing-net for the DB/Redis/search services)."""
    worker = compose["services"]["worker"]
    nets = worker["networks"]
    assert "turing-runner-net" in nets, f"worker not on turing-runner-net: {nets}"
    assert "turing-net" in nets, f"worker lost turing-net: {nets}"


# ── Runner: least-privilege, egress-isolated ────────────────────────────


def test_runner_service_exists(compose: dict) -> None:
    assert "runner" in compose["services"], "runner service missing"


def test_runner_builds_from_runner_dockerfile(compose: dict) -> None:
    """The runner image is built from Dockerfile.runner (least-privilege image:
    only runner_server.py copied in, no DB models / gateway / Settings)."""
    runner = compose["services"]["runner"]
    build = runner.get("build") or {}
    assert build.get("dockerfile") == "Dockerfile.runner", build


def test_runner_has_no_db_redis_search_credentials(compose: dict) -> None:
    """The runner holds NO DATABASE_URL / REDIS_URL / search keys — it does not
    use env_file and its environment block carries only RUNNER_* knobs plus a
    closed allowlist of benign Python runtime knobs (image hygiene under the
    read-only rootfs, never credentials)."""
    runner = compose["services"]["runner"]
    assert runner.get("env_file") is None, "runner must not load an env_file"
    env = runner.get("environment") or {}
    forbidden = {"DATABASE_URL", "REDIS_URL", "SEARXNG_URL", "MEILISEARCH_URL", "MEILISEARCH_KEY"}
    leaked = forbidden & set(env)
    assert not leaked, f"runner leaked credentials: {leaked}"
    # benign Python interpreter knobs the read-only-rootfs image needs (NOT secrets).
    # Adding any OTHER non-RUNNER_ key must fail here → forces a conscious review.
    benign = {"PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "PYTHONHASHSEED"}
    offenders = {k for k in env if not k.startswith("RUNNER_") and k not in benign}
    assert not offenders, f"runner has unexpected non-RUNNER_ env: {offenders}"


def test_runner_exposes_internal_port_only(compose: dict) -> None:
    """The runner exposes :8090 only to the internal network — it must NOT map a
    host port (no egress path off the internal net)."""
    runner = compose["services"]["runner"]
    assert "8090" in (runner.get("expose") or []), runner.get("expose")
    assert "ports" not in runner, f"runner maps a host port: {runner['ports']}"


def test_runner_is_egress_isolated(compose: dict) -> None:
    """The runner attaches ONLY to the internal turing-runner-net (not turing-net,
    where postgres/redis/searxng/meili live) — so it can neither reach the
    internet nor the data services."""
    runner = compose["services"]["runner"]
    nets = runner.get("networks") or []
    assert "turing-runner-net" in nets, nets
    assert "turing-net" not in nets, f"runner on turing-net (data-plane leak): {nets}"


# ── Network + start-order ───────────────────────────────────────────────


def test_runner_network_is_internal(compose: dict) -> None:
    """turing-runner-net is internal:true → no internet egress for the runner."""
    net = compose["networks"]["turing-runner-net"]
    assert net.get("internal") is True, net


def test_code_exec_roles_depend_on_runner_healthy(compose: dict) -> None:
    """api/worker/agent (the code-exec roles) wait for the runner to be healthy
    before starting, so the first code-exec doesn't fall back to a host
    subprocess at boot."""
    for role in ("api", "worker", "agent"):
        deps = compose["services"][role].get("depends_on") or {}
        assert "runner" in deps, f"{role} does not depend_on runner"


def test_agent_cli_reaches_runner(compose: dict) -> None:
    """The CLI-in-container smoke path inherits runner mode, so it too must be on
    turing-runner-net to reach the runner."""
    agent = compose["services"]["agent"]
    nets = agent.get("networks") or []
    assert "turing-runner-net" in nets, f"agent-cli not on turing-runner-net: {nets}"


# ── Healthcheck correctness ─────────────────────────────────────────────


def test_meilisearch_healthcheck_uses_ipv4_loopback(compose: dict) -> None:
    """meilisearch binds 0.0.0.0:7700 (IPv4-ONLY — `--http-addr 0.0.0.0:7700` is not
    IPv6 `::`). A healthcheck using ``localhost`` resolves to IPv6 ``[::1]`` under
    busybox wget, which has no listener there → the probe is refused and compose
    marks the service permanently unhealthy. That blocks every role with
    `depends_on: meilisearch: service_healthy` (api/worker/agent) — so the whole
    stack never starts. The probe MUST pin the IPv4 loopback 127.0.0.1."""
    meili = compose["services"]["meilisearch"]
    test_cmd = meili.get("healthcheck", {}).get("test") or []
    joined = " ".join(str(p) for p in test_cmd)
    assert "127.0.0.1" in joined, (
        f"meilisearch healthcheck not pinned to IPv4 loopback: {test_cmd}"
    )
    assert "localhost" not in joined, (
        f"meilisearch healthcheck uses 'localhost' (IPv6 ::1 refused vs IPv4-only "
        f"bind → perpetual unhealthy → blocks dependent roles): {test_cmd}"
    )


# ── API host-port: avoid the 8000 conflict magnet ────────────────────────


def test_api_host_port_is_not_the_conflict_magnet_8000(compose: dict) -> None:
    """The api maps a HOST port other than 8000.

    8000 is a magnet for conflicts with other dev apps (another project's API
    already collided on it during bring-up). The non-default host port matches
    the stack's pattern (5433/6380/8081/7701 all avoid the canonic port). The
    container still LISTENS on 8000 internally (uvicorn ``--port 8000``); only
    the HOST side of the mapping is non-8000 — so this asserts the left (host)
    side, not the right (container) side.
    """
    api = compose["services"]["api"]
    ports = api.get("ports") or []
    assert ports, "api has no host port mapping"
    mapping = ports[0]
    if isinstance(mapping, dict):
        host = str(mapping.get("published", ""))
        container = str(mapping.get("target", ""))
    else:
        host, _, container = str(mapping).partition(":")
    assert container == "8000", f"api container port drifted from uvicorn 8000: {mapping}"
    assert host, "api host port is empty/unmapped"
    assert host != "8000", f"api host port is still 8000 (conflict magnet): {mapping}"


# ── Scheduler: nightly capability-curve battery feeder (#197) ─────────


def test_scheduler_service_exists_and_is_profile_gated(compose: dict) -> None:
    """The nightly capability-curve battery feeder is opt-in behind the
    `scheduler` profile so `docker compose up -d` (the default stack) does NOT
    start it — an operator brings it up explicitly:
    `docker compose --profile scheduler up -d`."""
    assert "scheduler" in compose["services"], "scheduler service missing"
    profiles = compose["services"]["scheduler"].get("profiles")
    assert profiles == ["scheduler"], f"scheduler not profile-gated: {profiles}"


def test_scheduler_runs_the_scheduler_module(compose: dict) -> None:
    """The sidecar runs `python -m src.scheduler`."""
    scheduler = compose["services"]["scheduler"]
    assert scheduler.get("command") == ["python", "-m", "src.scheduler"], (
        scheduler.get("command")
    )


def test_scheduler_is_cred_minimal(compose: dict) -> None:
    """The scheduler is a pure Redis PRODUCER — it must NOT hold DATABASE_URL /
    search keys / runner URL. It neither executes code nor touches the DB; the
    worker draining these jobs does the work. Mirrors the runner's least-privilege
    posture (see test_runner_has_no_db_redis_search_credentials)."""
    scheduler = compose["services"]["scheduler"]
    # No env_file: cred-minimal (does NOT load .env → no inherited DB/search creds).
    assert scheduler.get("env_file") is None, "scheduler loads an env_file (cred leak)"
    env = scheduler.get("environment") or {}
    forbidden = {
        "DATABASE_URL",
        "SEARXNG_URL",
        "MEILISEARCH_URL",
        "MEILISEARCH_KEY",
        "RUNNER_URL",
    }
    leaked = forbidden & set(env)
    assert not leaked, f"scheduler leaked credentials: {leaked}"
    # It MUST have Redis (to enqueue) + SCHEDULER_ENABLED forced on.
    assert env.get("REDIS_URL") == "redis://redis:6379/0", env.get("REDIS_URL")
    assert env.get("SCHEDULER_ENABLED") == "true", env.get("SCHEDULER_ENABLED")


def test_scheduler_depends_only_on_redis(compose: dict) -> None:
    """The scheduler talks to Redis ONLY — it must not depend on postgres/searxng/
    meilisearch/runner (it neither reads the DB nor executes code)."""
    scheduler = compose["services"]["scheduler"]
    deps = scheduler.get("depends_on") or {}
    assert "redis" in deps, f"scheduler must depend_on redis: {deps}"
    unwanted = {"postgres", "searxng", "meilisearch", "runner"} & set(deps)
    assert not unwanted, (
        f"scheduler depends on data/code services it doesn't need: {unwanted}"
    )


def test_scheduler_is_isolated_to_turing_net(compose: dict) -> None:
    """The scheduler is on turing-net (to reach redis) ONLY — it must NOT be on
    turing-runner-net (it has no code to route to the runner)."""
    scheduler = compose["services"]["scheduler"]
    nets = scheduler.get("networks") or []
    assert "turing-net" in nets, f"scheduler not on turing-net: {nets}"
    assert "turing-runner-net" not in nets, (
        f"scheduler on runner net (it has no code to run): {nets}"
    )


def test_scheduler_drops_all_caps(compose: dict) -> None:
    """Defense-in-depth: the scheduler drops all Linux caps + blocks privilege
    escalation (the discipline every app role carries)."""
    scheduler = compose["services"]["scheduler"]
    assert "ALL" in (scheduler.get("cap_drop") or []), (
        f"scheduler did not cap_drop: [ALL]: {scheduler.get('cap_drop')}"
    )
    assert "no-new-privileges:true" in (scheduler.get("security_opt") or []), (
        scheduler.get("security_opt")
    )
