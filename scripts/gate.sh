#!/usr/bin/env bash
# Local pre-merge gate (Track-1): the deterministic quality bar a change must
# clear before commit/merge. Runs the same three checks CI would:
#   1. ruff   — lint (fast, deterministic)
#   2. pyright — type check on src/ (the project uses pyright, NOT mypy)
#   3. pytest — unit + integration, E2E excluded (E2E needs a provider key)
#
# Activate your project virtualenv FIRST (the repo uses the `aiml01` env, NOT
# `uv run` — blocked by an upstream vllm dep). This script does NOT hardcode an
# activation path so it stays portable; it assumes `ruff`, `pyright`, and
# `python` resolve on PATH in the active env.
#
# Usage:
#   scripts/gate.sh                # full gate (ruff + pyright + pytest)
#   scripts/gate.sh --no-pytest    # ruff + pyright only (the pre-commit subset)
#
# Exit non-zero on the first failing stage so a broken change never reports green.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_PYTEST=1
if [[ "${1:-}" == "--no-pytest" ]]; then
  RUN_PYTEST=0
fi

echo "── ruff check . ────────────────────────────────────────────"
ruff check .

echo "── pyright src/ ────────────────────────────────────────────"
pyright src/

if [[ "$RUN_PYTEST" -eq 1 ]]; then
  echo "── pytest (unit + integration, e2e excluded) ──────────────"
  python -m pytest tests/ -v -k "not e2e"
fi

echo "── gate: PASS ─────────────────────────────────────────────"
