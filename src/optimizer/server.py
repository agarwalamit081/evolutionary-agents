"""HTTP server for the metric-driven prompt optimizer sidecar (Phase 2 C2).

Runs IN THE OPTIMIZER CONTAINER. Unlike the deliberately self-contained
:mod:`src.sandbox.runner_server` (least-privilege, no ``src.*`` imports, no
DB/search creds), the optimizer IS "app + ML stack": it imports
:mod:`src.optimizer.engine` (the DSPy/GEPA compile + canary validate path) and
reads the SHARED ``.env`` because it runs the full
:class:`~src.evolution.promote.GoldenCanary` (provider keys) and writes to the
shared ``cost_ledger`` (DATABASE_URL). So this server uses the app's
:class:`~src.config.settings.Settings`, not a self-contained settings class.

Tiny internal-only HTTP API on ``turing-net`` (no host port is published — the
scheduler calls it by service name)::

    POST /optimize  {"node"?, "backend"?, "budget_hint"?}  -> OptimizeResponse
                   (ConfigurationError -> 400; bad body -> 422; other -> 500 generic)
    GET  /healthz   -> {"status": "ok"}

Error discipline (security.md): a 422 returns Pydantic's field-level validation
errors (input feedback, not internals); a caller bug
(:class:`~src.optimizer.models.ConfigurationError` — e.g. ``backend="textgrad"``)
is a 400; every other engine failure is a generic 500 with the detail logged
server-side only — never a stack trace, file path, or DB error to the client.
"""

from __future__ import annotations

import json

from aiohttp import web
from loguru import logger
from pydantic import ValidationError

from src.config import get_settings
from src.optimizer.engine import PromptOptimizer
from src.optimizer.models import ConfigurationError, OptimizeRequest, OptimizeResponse

# aiohttp 3.9+ deprecates bare-string app keys (NotAppKeyWarning); a typed
# AppKey also gives static type-checking on the stored optimizer. ``build_app``
# always stores a non-None instance (it resolves ``optimizer or PromptOptimizer()``).
_OPTIMIZER_KEY = web.AppKey("optimizer", PromptOptimizer)

# The /optimize request body is tiny (node/backend/budget_hint); a 1 MiB cap is
# generous and bounds a hand-crafted oversized body (defense in depth).
_CLIENT_MAX_SIZE = 1024 * 1024


async def handle_optimize(request: web.Request) -> web.Response:
    """POST /optimize — run one optimization attempt; return the outcome.

    The engine returns a structured :class:`OptimizeResponse` for every runtime
    condition (curve guard, budget, no signal, no improvement, promoted); it
    only RAISES :class:`ConfigurationError` for an unsupported backend/node.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        logger.warning("POST /optimize rejected: request body is not JSON")
        raise web.HTTPBadRequest(text="request body must be JSON")

    try:
        req = OptimizeRequest.model_validate(body)
    except ValidationError as exc:
        # Field-level validation errors are safe input feedback (not internals).
        logger.warning("POST /optimize rejected: body failed validation")
        raise web.HTTPUnprocessableEntity(
            text=exc.json(), content_type="application/json"
        )

    optimizer: PromptOptimizer = request.app[_OPTIMIZER_KEY]
    try:
        resp: OptimizeResponse = await optimizer.optimize(req)
    except ConfigurationError as exc:
        # Caller bug (bad backend/node) — surface the reason so the scheduler
        # can log/act on it; this is not an internal failure.
        logger.warning("POST /optimize rejected: configuration error ({})", exc)
        raise web.HTTPBadRequest(text=str(exc))
    except Exception:
        # Any other failure is internal — log the full detail server-side, return
        # a generic message (never expose a stack trace / path / DB error).
        logger.exception("POST /optimize failed")
        raise web.HTTPInternalServerError(text="Something went wrong")

    return web.json_response(resp.model_dump())


async def handle_health(_request: web.Request) -> web.Response:
    """GET /healthz — liveness/readiness for the compose healthcheck."""
    return web.json_response({"status": "ok"})


def build_app(optimizer: PromptOptimizer | None = None) -> web.Application:
    """Construct the aiohttp Application (tests call this directly).

    ``optimizer`` is optional so a test can inject a fake without touching
    ``get_settings()`` or the shared DB — when a fake is passed the ``or`` below
    short-circuits on it and a real optimizer is never constructed; ``None`` (the
    prod path) constructs one eagerly (cheap — it only stores settings).
    """
    app = web.Application(client_max_size=_CLIENT_MAX_SIZE)
    app[_OPTIMIZER_KEY] = optimizer or PromptOptimizer()
    app.router.add_post("/optimize", handle_optimize)
    app.router.add_get("/healthz", handle_health)
    return app


def main() -> None:
    """Entrypoint: ``python -m src.optimizer`` (in the optimizer container)."""
    settings = get_settings()
    opt = settings.optimizer
    logger.info(
        "optimizer sidecar starting on {}:{} (target={}, backend={}, enabled={})",
        opt.host,
        opt.port,
        opt.target_node,
        opt.backend,
        opt.enabled,
    )
    web.run_app(build_app(), host=opt.host, port=opt.port)
