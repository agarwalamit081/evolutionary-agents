"""Operator dashboard routes (Phase 5) — server-rendered (FastAPI + Jinja2).

Five read-only HTML views over the existing data (no new schema, no writes):
  GET /dashboard              — index: summary cards + recent runs + recent mutations
  GET /dashboard/runs         — run observer (Redis ``turing:run:*``), auto-refreshing
  GET /dashboard/runs/{id}    — run detail: status, cost-by-model, review card, live steps
  GET /dashboard/runs/{id}/steps — polled partial: per-node execution-step rows (poll.js)
  GET /dashboard/curve        — per-run × per-goal eval matrix + summary chart
  GET /dashboard/mutations    — mutation/promotion timeline (with Phase-3 diff + Phase-4 A/B)

Auto-refresh is a tiny vanilla-JS polling swap (``static/poll.js``) — no SSE, no
build step. Each list endpoint honors ``?partial=1`` to return just its table
fragment so the poller can swap a ``<tbody>`` without re-rendering the page.

All views degrade gracefully: a Redis/DB hiccup in the data layer returns an
empty result, so a page renders "no data yet" rather than 500. The dashboard
mounts under ``/dashboard`` (no API prefix — it is a UI, not a programmatic API).
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from src.api.routes import dashboard_data as data
from src.config import get_settings
from src.db.session import get_session
from src.worker.status import RunStatusStore

def _dashboard_api_key() -> str:
    """Configured dashboard key, exposed as a (test-overridable) dependency.

    Empty by default → the UI is open (today's local-dev behavior).
    """
    return getattr(get_settings().dashboard, "api_key", "") or ""


async def _require_dashboard_key(
    configured: str = Depends(_dashboard_api_key),
    x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key"),
) -> None:
    """Opt-in ``X-Dashboard-Key`` gate applied to every ``/dashboard*`` route.

    When ``DASHBOARD_API_KEY`` is unset, the UI stays open (no header required —
    today's behavior). When set, every dashboard route 401s without a
    constant-time-matching ``X-Dashboard-Key`` header. Applied via the router's
    own ``dependencies`` so all five views are covered with one declaration.
    """
    if not configured:
        return
    if x_dashboard_key is None or not secrets.compare_digest(
        x_dashboard_key, configured
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="a valid X-Dashboard-Key header is required",
        )


router = APIRouter(dependencies=[Depends(_require_dashboard_key)])

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


async def _open_store() -> tuple[RunStatusStore, aioredis.Redis]:
    """Build a per-request RunStatusStore over a fresh Redis client.

    The caller closes the returned client in a ``finally`` so the connection
    returns to the pool (the agent routes leave this to GC; the dashboard polls
    frequently, so an explicit close avoids leaking connections under refresh).
    """
    settings = get_settings()
    redis_client = aioredis.from_url(settings.redis.redis_url)
    return RunStatusStore(redis_client, settings.worker), redis_client


def _is_partial(request: Request) -> bool:
    """``?partial=1`` → render just the table fragment (polled swap)."""
    return request.query_params.get("partial") == "1"


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_index(request: Request) -> HTMLResponse:
    """Index: summary cards + the most recent runs + most recent mutations."""
    store, redis_client = await _open_store()
    try:
        async with get_session() as session:
            runs, summary = await data.runs_with_cost(store, session, limit=10)
            mutations = await data.mutation_timeline(session, limit=5)
    except Exception as e:  # noqa: BLE001 — degrade, never 500
        logger.warning(f"Dashboard index data fetch failed: {e}")
        runs, summary, mutations = [], {"runs_total": 0, "runs_in_flight": 0, "runs_completed": 0, "total_cost_usd": 0.0}, []
    finally:
        await redis_client.aclose()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "summary": summary, "runs": runs, "mutations": mutations},
    )


@router.get("/dashboard/runs", response_class=HTMLResponse)
async def dashboard_runs(request: Request) -> HTMLResponse:
    """Run observer — the live run list (Redis), auto-refreshed by poll.js."""
    store, redis_client = await _open_store()
    try:
        async with get_session() as session:
            runs, summary = await data.runs_with_cost(store, session)
    except Exception as e:  # noqa: BLE001 — degrade
        logger.warning(f"Dashboard runs data fetch failed: {e}")
        runs, summary = [], {"runs_total": 0, "runs_in_flight": 0, "runs_completed": 0, "total_cost_usd": 0.0}
    finally:
        await redis_client.aclose()
    if _is_partial(request):
        return templates.TemplateResponse(
            request=request, name="_runs_rows.html", context={"request": request, "runs": runs}
        )
    return templates.TemplateResponse(
        request=request,
        name="runs.html",
        context={"request": request, "summary": summary, "runs": runs},
    )


@router.get("/dashboard/runs/{run_id}", response_class=HTMLResponse)
async def dashboard_run_detail(run_id: str, request: Request) -> HTMLResponse:
    """One run's status + per-model cost + the HITL review (Q100) card."""
    store, redis_client = await _open_store()
    run_view: dict[str, Any] | None = None
    cost_breakdown: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    token_split: dict[str, Any] = {
        "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
        "total_tokens": 0, "input_pct": 0.0, "cache_hit_pct": 0.0,
    }
    try:
        record = await store.get(run_id)
        if record is not None:
            run_view = data._run_to_view(record)  # noqa: SLF001 — view-shape helper
        async with get_session() as session:
            if run_view is not None:
                cost_breakdown = await data.run_cost_breakdown(session, run_view)
                steps = await data.execution_steps(session, run_view)
                token_split = await data.run_token_split(session, run_view)
    except Exception as e:  # noqa: BLE001 — degrade
        logger.warning(f"Dashboard run-detail fetch failed for {run_id}: {e}")
    finally:
        await redis_client.aclose()
    if run_view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown or expired run_id: {run_id}",
        )
    # The review card surfaces the full final output + any error (the structured
    # HumanMessage from the HITL node lives in the checkpoint; the Redis status
    # carries the rendered output/error the operator reviews).
    return templates.TemplateResponse(
        request=request,
        name="run_detail.html",
        context={
            "request": request,
            "run": run_view,
            "cost_breakdown": cost_breakdown,
            "total_cost": sum(c["cost_usd"] for c in cost_breakdown),
            "steps": steps,
            "token_split": token_split,
        },
    )


@router.get("/dashboard/runs/{run_id}/steps", response_class=HTMLResponse)
async def dashboard_run_steps(run_id: str, request: Request) -> HTMLResponse:
    """Polled partial — the per-node execution-step rows for one run.

    poll.js fetches this every ``data-poll-interval`` seconds and swaps the
    response into the run-detail steps ``<tbody>`` (HTMX-equivalent). Re-derives
    the run view from Redis so the ``run_id`` key candidates (thread_id / bare id)
    match, exactly as the run-detail route does. Best-effort: a Redis/DB hiccup
    renders the empty partial (never 500). The run-detail route 404s unknown ids;
    this partial degrades to "no steps" so a poll never errors mid-page.
    """
    store, redis_client = await _open_store()
    steps: list[dict[str, Any]] = []
    try:
        record = await store.get(run_id)
        if record is not None:
            run_view = data._run_to_view(record)  # noqa: SLF001 — view-shape helper
            async with get_session() as session:
                steps = await data.execution_steps(session, run_view)
    except Exception as e:  # noqa: BLE001 — degrade
        logger.warning(f"Dashboard steps fetch failed for {run_id}: {e}")
    finally:
        await redis_client.aclose()
    return templates.TemplateResponse(
        request=request,
        name="_steps_rows.html",
        context={"request": request, "steps": steps},
    )


@router.get("/dashboard/curve", response_class=HTMLResponse)
async def dashboard_curve(
    request: Request,
    suffix: str | None = Query(default=None, description="run_id substring filter"),
) -> HTMLResponse:
    """Per-run × per-goal eval terminal-state matrix + summary bar chart."""
    try:
        async with get_session() as session:
            curve = await data.generation_curve(session, suffix=suffix)
    except Exception as e:  # noqa: BLE001 — degrade
        logger.warning(f"Dashboard curve data fetch failed: {e}")
        curve = {"runs": [], "goals": [], "matrix": {}, "run_means": {}, "goal_means": {}}
    return templates.TemplateResponse(
        request=request,
        name="curve.html",
        context={"request": request, "curve": curve, "suffix": suffix or ""},
    )


@router.get("/dashboard/mutations", response_class=HTMLResponse)
async def dashboard_mutations(request: Request) -> HTMLResponse:
    """Mutation timeline + sub-agents + counts + evolution status + web-search use.

    Surfaces what the operator actually wants on a "did the app evolve?" page:
    the mutation timeline (Phase-3 diff + Phase-4 A/B) AND the sub-agents it
    spawned, summary counts (prompts/tools mutated), whether the runs behind each
    mutation used web search, and whether any promotion reached production
    (channel-B live PROMPT promotions + channel-A deployed tools/sub-agents).
    """
    mutations: list[dict[str, Any]] = []
    subagents: list[dict[str, Any]] = []
    counts: dict[str, Any] = {
        "by_type": {}, "prompts_mutated": 0, "tools_mutated": 0,
        "total_mutations": 0, "deployed_tools": 0, "active_subagents": 0, "active_configs": 0,
    }
    evolution: dict[str, Any] = {
        "live_prompt_promotions": [], "total_live_promotions": 0,
        "deployed_tools": 0, "active_subagents": 0, "active_configs": 0, "any_evolved": False,
    }
    web: dict[str, Any] = {"total_calls": 0, "runs_using_search": 0}
    ws_by_mutation: dict[str, bool] = {}
    sub_ws_runs: set[str] = set()
    try:
        async with get_session() as session:
            mutations = await data.mutation_timeline(session)
            subagents = await data.sub_agent_timeline(session)
            counts = await data.mutation_counts(session)
            evolution = await data.evolution_summary(session, counts)
            web = await data.web_search_summary(session)
            ws_by_mutation = await data.mutations_web_search(
                session, [m["id"] for m in mutations]
            )
            sub_ws_runs = await data.web_search_runs(
                session, [s["owner_run_id"] for s in subagents if s["owner_run_id"]]
            )
    except Exception as e:  # noqa: BLE001 — degrade, never 500
        logger.warning(f"Dashboard mutations data fetch failed: {e}")
    return templates.TemplateResponse(
        request=request,
        name="mutations.html",
        context={
            "request": request,
            "mutations": mutations,
            "mut_web_search": ws_by_mutation,
            "subagents": subagents,
            "sub_web_search": sub_ws_runs,
            "counts": counts,
            "evolution": evolution,
            "web": web,
        },
    )
