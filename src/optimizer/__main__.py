"""Optimizer sidecar entrypoint: ``python -m src.optimizer``.

Boots the aiohttp :mod:`src.optimizer.server` with structured logging configured
(the DSPy/GEPA compile + canary-validate runs in this one process). Mirrors the
``src.scheduler`` / ``src.worker`` entrypoints.
"""

from __future__ import annotations

from src.config import get_settings
from src.observability import init_process_observability
from src.observability.logging import setup_logging
from src.optimizer.server import main

if __name__ == "__main__":
    settings = get_settings()
    setup_logging(settings.logging)
    # Observability (OTel tracing + Prometheus scrape server); opt-in, idempotent.
    init_process_observability(settings.observability, component="optimizer")
    main()
