# ── Stage 1: Builder ─────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ───────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# System deps for asyncpg (libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY main.py .
COPY alembic.ini .
COPY pytest.ini .
COPY tests/ tests/

# Create non-root user for security
RUN useradd -m -r turing && \
    mkdir -p /home/turing/.turing/workspace /home/turing/logs && \
    chown -R turing:turing /app /home/turing/.turing /home/turing/logs

USER turing

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default: CLI interactive mode. Override for API: python main.py --api
CMD ["python", "main.py", "--interactive"]
