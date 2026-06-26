"""Tool that lets the agent author a durable cron task (Phase 5 I1).

Autonomy gap: the agent could react to a goal but could not ask for *future* work
to run on a schedule (e.g. "re-pull this report every weekday 09:00"). This tool
writes one row into the ``scheduled_tasks`` table; the scheduler consumer
(``src.scheduler.cron_consumer``) reconciles those rows against APScheduler and,
on each fire, enqueues a ``RunJob`` through the existing ``RunsQueue`` seam so the
scheduled run flows through the real deployed worker stack (lease-lock,
checkpoint, eval-resolution all apply unchanged).

``name`` is the stable upsert handle: re-calling with an existing name revises
the cron/goal/model instead of duplicating, so the agent can re-schedule
idempotently. Default-off (``AGENT_CRON_ENABLED``); when disabled the handler is
a no-op that tells the caller, so nothing changes until an operator opts in.
"""

from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger


async def create_scheduled_task(
    name: str,
    cron: str,
    goal: str,
    model: str = "",
    timezone: str = "UTC",
) -> str:
    """Create or revise a durable cron-scheduled agent run.

    Args:
        name: Stable handle for the task. Re-calling with an existing name
            revises the schedule in place instead of creating a duplicate.
        cron: Standard 5-field crontab expression (``minute hour day month
            weekday``), e.g. ``"0 9 * * 1-5"`` for every weekday at 09:00.
        goal: The objective text enqueued as the run's goal on each fire.
        model: Optional model id to pin every fired run to. Empty/omit ⇒ the
            scheduler's default routing applies.
        timezone: IANA timezone the cron expression is interpreted in (default
            ``"UTC"``). An unknown zone falls back to UTC.

    Returns:
        A human-readable confirmation including the action taken (created vs.
        revised), the resolved schedule, and the next projected fire time — or a
        short reason when the call was rejected (feature disabled, bad cron, cap
        reached).
    """
    # Lazy import so tests can patch src.config.settings.get_settings.
    from src.config.settings import get_settings

    cron_settings = get_settings().agent_cron
    if not cron_settings.enabled:
        return (
            "Scheduled-task tool is disabled (AGENT_CRON_ENABLED=false). Ask the "
            "operator to enable durable cron before scheduling future work."
        )

    name = (name or "").strip()
    goal = (goal or "").strip()
    cron = (cron or "").strip()
    model_val = (model or "").strip() or None
    if not name or not goal or not cron:
        return "name, cron, and goal are all required (non-empty)."
    if len(goal) > cron_settings.max_goal_chars:
        return (
            f"goal is {len(goal)} chars; cap is {cron_settings.max_goal_chars}. "
            "Shorten the goal text."
        )

    # Resolve timezone before cron so an invalid zone can't masquerade as a bad
    # cron expression (clearer rejection). Unknown zone ⇒ UTC, like get_current_time.
    resolved_tz = timezone.strip() or "UTC"
    try:
        ZoneInfo(resolved_tz)
    except (ZoneInfoNotFoundError, ValueError):
        resolved_tz = "UTC"

    # Validate the cron expression + compute the informational next-fire. APScheduler
    # is the authoritative scheduler; next_fire_at is just a hint for operators.
    try:
        from apscheduler.triggers.cron import CronTrigger

        trigger = CronTrigger.from_crontab(cron, timezone=resolved_tz)
        next_fire = trigger.get_next_fire_time(None, datetime.now(dt_timezone.utc))
    except ValueError as exc:
        return f"invalid cron expression {cron!r}: {exc}"

    from sqlalchemy import func, select

    from src.db.models import ScheduledTask
    from src.db.session import get_session
    from src.tools._paths import get_active_run_id

    owner = get_active_run_id()
    now_utc = datetime.now(dt_timezone.utc)

    # Lazily imported so the builtin module stays import-light (no apscheduler/DB
    # coupling at registry build time). get_session() commits on a clean exit and
    # rolls back on any error, so the upsert is one atomic transaction.
    async with get_session() as session:
        existing = (
            await session.execute(select(ScheduledTask).where(ScheduledTask.name == name))
        ).scalar_one_or_none()

        if existing is None:
            enabled_count = (
                await session.execute(
                    select(func.count())
                    .select_from(ScheduledTask)
                    .where(ScheduledTask.enabled.is_(True))
                )
            ).scalar_one()
            if enabled_count >= cron_settings.max_tasks:
                logger.warning(
                    f"schedule_task: cap reached ({enabled_count}/{cron_settings.max_tasks}) "
                    f"for new task name={name!r}"
                )
                return (
                    f"Enabled-task cap reached ({enabled_count}/{cron_settings.max_tasks}); "
                    "disable an existing task or reuse an existing name to revise it."
                )
            session.add(
                ScheduledTask(
                    name=name,
                    cron=cron,
                    goal=goal,
                    model=model_val,
                    owner_run_id=owner,
                    enabled=True,
                    timezone=resolved_tz,
                    next_fire_at=next_fire,
                )
            )
            action = "Created"
        else:
            existing.cron = cron
            existing.goal = goal
            existing.model = model_val
            existing.timezone = resolved_tz
            existing.enabled = True
            existing.next_fire_at = next_fire
            existing.updated_at = now_utc
            action = "Updated"

    logger.info(
        f"schedule_task: {action} name={name!r} cron={cron!r} tz={resolved_tz} "
        f"owner_run_id={'set' if owner else 'none'} model={'pinned' if model_val else 'default'}"
    )
    fire_str = next_fire.isoformat() if next_fire else "unknown"
    return (
        f"{action} scheduled task {name!r} (cron {cron!r} {resolved_tz}). "
        f"Next projected fire: {fire_str}. The scheduler consumer enqueues a run "
        "for this goal on each cron tick."
    )


TOOL_DEFINITION = {
    "name": "create_scheduled_task",
    "handler": create_scheduled_task,
    "description": (
        "Schedule a durable future agent run on a cron schedule (default-off; "
        "no-op when disabled). Persists one task identified by a stable `name` — "
        "re-calling with the same name revises the schedule in place. On each "
        "cron tick the scheduler enqueues a normal run for the given `goal`. Use "
        "this when a task should recur (e.g. 'refresh this report every weekday "
        "morning', 're-check the feed hourly') rather than run once now."
    ),
    # Each call mutates durable state (a DB row); never serve a cached result.
    "cacheable": False,
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Stable handle for the task. Re-calling with an existing name "
                    "revises the schedule instead of creating a duplicate "
                    "(e.g. 'weekday-report-refresh')."
                ),
            },
            "cron": {
                "type": "string",
                "description": (
                    "Standard 5-field crontab expression 'minute hour day month "
                    "weekday' (e.g. '0 9 * * 1-5' = every weekday at 09:00, "
                    "'*/30 * * * *' = every 30 minutes)."
                ),
            },
            "goal": {
                "type": "string",
                "description": (
                    "The objective text that becomes the run goal on every fire. "
                    "Keep it self-contained — it runs unattended on a schedule."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model id to pin every fired run to. Omit/empty to "
                    "let the scheduler use its default routing."
                ),
                "default": "",
            },
            "timezone": {
                "type": "string",
                "description": (
                    "IANA timezone the cron expression is interpreted in (e.g. "
                    "'UTC', 'America/New_York'). Defaults to 'UTC'."
                ),
                "default": "UTC",
            },
        },
        "required": ["name", "cron", "goal"],
    },
}
