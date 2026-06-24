"""LLM-layer exceptions.

Currently houses the opt-in budget hard-stop signal. Kept in its own module so
the gateway can raise it and the worker can catch it without either depending
on the other's heavy imports (budget exhaustion is a cross-cutting run-control
concern, not a gateway-only concept).
"""

from __future__ import annotations


class BudgetExhaustedError(Exception):
    """Raised when the per-run token budget is exhausted AND the opt-in
    ``budget_hard_stop`` is set (``BudgetSettings.budget_hard_stop``).

    Without the flag the gateway instead *downgrades* to a cheaper fallback
    model and keeps going — which under downgrade can fabricate (battery-04 q09:
    the run degraded onto a free-tier provider and never completed). With the
    flag the run stops cleanly at a terminal-but-resumable BUDGET_EXHAUSTED
    status; the per-iteration AsyncPostgresSaver checkpoint means
    ``--resume <run_id>`` picks up from the last write.

    Caveat (documented, deferred): ``get_run_token_usage`` is cumulative, so a
    resume currently re-trips this immediately. The resume-window delta is a
    deferred follow-up; ``budget_hard_stop`` defaults off so the downgrade
    behavior is unchanged until explicitly opted in.
    """
