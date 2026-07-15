# syntax=docker/dockerfile:1
# ── Stage 1: Builder ─────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
# litellm==1.83.14 over-pins 9 shared packages to old exact versions (fastapi,
# uvicorn, orjson, rich, prometheus-client, opentelemetry, aiohttp, click, pypdf).
# The working env runs the NEWER versions (passes 848 tests); litellm is compatible
# with them at runtime — its pins are for its optional proxy server, which this
# project never uses (only litellm.acompletion). So install everything-but-litellm
# first (resolves cleanly), then litellm itself with --no-deps so its over-strict
# metadata never blocks resolution. fastuuid (litellm's only base dep not otherwise
# required) is pinned in requirements.txt and lands in the core install below.
#
# dspy==2.6.27 (the prompt-optimizer backend) depends on litellm, so it is ALSO
# excluded from req.core — installing it there would pull litellm transitively and
# re-trigger the over-pinned resolution. Installed AFTER the --no-deps litellm step
# (with deps) so its `litellm>=1.60.3` requirement is satisfied by the already-
# installed 1.83.14; its other deps (datasets/optuna/…) resolve cleanly.
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v -E '^\s*(litellm|dspy)\b' requirements.txt > /tmp/req.core.txt && \
    grep -E '^\s*litellm\b' requirements.txt > /tmp/req.litellm.txt && \
    grep -E '^\s*dspy\b' requirements.txt > /tmp/req.dspy.txt && \
    pip install --prefix=/install -r /tmp/req.core.txt && \
    pip install --no-deps --prefix=/install -r /tmp/req.litellm.txt && \
    pip install --prefix=/install -r /tmp/req.dspy.txt

# ── Stage 2: Runtime ───────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# System deps for asyncpg (libpq) + read-only CLI tools available to the
# terminal_command builtin tool (the code_executor sandbox is separately locked).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl git jq ripgrep tree file \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY main.py .
COPY alembic.ini .
# NOTE: tests/ + pytest.ini are intentionally NOT copied into the prod runtime
# image (uvicorn / worker / CLI). Pytest runs on the host via the aiml01 venv
# per CLAUDE.md; no runtime code under src/ or main.py imports tests/. Keeping
# them out shrinks the image and avoids shipping test fixtures to prod.

# Create non-root user for security. The results dir is pre-created (and
# owned) so the agent's persisted RESULTS_ROOT (set in docker-compose) is
# writable on first run — file_writer/code_executor deliverables land there.
RUN useradd -m -r turing && \
    mkdir -p /home/turing/.turing/workspace /home/turing/.turing/results /home/turing/logs && \
    chown -R turing:turing /app /home/turing/.turing /home/turing/logs

USER turing

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default: CLI interactive mode. Override for API: python main.py --api
CMD ["python", "main.py", "--interactive"]
