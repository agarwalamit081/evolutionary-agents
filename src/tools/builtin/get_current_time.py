"""Tool returning the current wall-clock timestamp, timezone, and date.

Agents otherwise have no time awareness, so they cannot answer temporal queries
("check the logs from the last 2 hours") or produce correctly-dated artifacts.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger


async def get_current_time(timezone: str = "UTC") -> str:
    """Return the current wall-clock time in the requested timezone.

    Args:
        timezone: IANA timezone name (e.g. ``"UTC"``, ``"America/New_York"``,
            ``"Asia/Kolkata"``). Defaults to ``"UTC"``.

    Returns:
        ISO-8601 timestamp, timezone + UTC offset, calendar date, and weekday.
    """
    logger.info(f"get_current_time: tz={timezone}")

    used_tz = timezone
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        # Fall back to UTC rather than failing the whole call.
        tz = ZoneInfo("UTC")
        used_tz = "UTC"
        note = f" (unknown timezone '{timezone}', used UTC)"
    else:
        note = ""

    now = datetime.now(tz)
    offset = now.strftime("%z")
    offset_fmt = f"{offset[:3]}:{offset[3:]}" if offset else "Z"
    return (
        f"{now.isoformat(timespec='seconds')}\n"
        f"Timezone: {used_tz} (UTC offset {offset_fmt})\n"
        f"Date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')}){note}"
    )


TOOL_DEFINITION = {
    "name": "get_current_time",
    "handler": get_current_time,
    "description": (
        "Return the current wall-clock time as an ISO-8601 timestamp, with "
        "timezone, UTC offset, calendar date, and weekday. Use this before "
        "any time-relative reasoning (e.g. 'logs from the last 2 hours', "
        "'today', 'this week') or when dating generated artifacts. Time is "
        "never cached, so each call is current."
    ),
    # Wall-clock time changes every call — must never be cached.
    "cacheable": False,
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "IANA timezone name (e.g. 'UTC', 'America/New_York', "
                    "'Asia/Kolkata'). Defaults to 'UTC'."
                ),
                "default": "UTC",
            },
        },
        "required": [],
    },
}
