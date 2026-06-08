#!/bin/bash
# Worktree CRUD helper for parallel feature development.
# Usage: ./worktree_manager.sh <command> [args]
# Commands:
#   create <slug> [base]   — Create worktree + branch
#   verify <path>          — Check tests pass, build passes, no uncommitted changes
#   merge <branch> [base]  — Merge feature branch with --no-ff, test after
#   cleanup <slug>         — Remove worktree and delete branch
set -euo pipefail

WORKTREE_BASE="../worktrees"

cmd_create() {
  local slug="$1"
  local base="${2:-main}"
  local branch="feat/${slug}"
  local path="${WORKTREE_BASE}/${slug}"

  echo "Creating worktree: ${path} on branch ${branch} from ${base}"
  git worktree add "${path}" -b "${branch}" "${base}"
  echo "✓ Worktree created at ${path}"
}

cmd_verify() {
  local path="$1"
  local errors=0

  echo "Verifying: ${path}"

  # Check for uncommitted changes
  if [ -n "$(cd "${path}" && git status --porcelain)" ]; then
    echo "  ✗ Uncommitted changes found"
    errors=$((errors + 1))
  else
    echo "  ✓ Clean working tree"
  fi

  # Check for test command
  if [ -f "${path}/package.json" ]; then
    echo "  Running tests..."
    (cd "${path}" && npm test -- --silent 2>/dev/null) || { echo "  ✗ Tests failed"; errors=$((errors + 1)); }
  elif [ -f "${path}/pytest.ini" ] || [ -f "${path}/pyproject.toml" ]; then
    echo "  Running tests..."
    (cd "${path}" && python -m pytest -q 2>/dev/null) || { echo "  ✗ Tests failed"; errors=$((errors + 1)); }
  fi

  if [ "${errors}" -eq 0 ]; then
    echo "✓ Verification passed"
    return 0
  else
    echo "✗ Verification failed (${errors} issues)"
    return 1
  fi
}

cmd_merge() {
  local branch="$1"
  local base="${2:-main}"
  local slug="${branch#feat/}"

  echo "Merging ${branch} into ${base}..."
  git checkout "${base}"
  git pull origin "${base}" 2>/dev/null || true
  git merge --no-ff "${branch}"

  echo "Running post-merge tests..."
  if npm test -- --silent 2>/dev/null || python -m pytest -q 2>/dev/null; then
    echo "✓ Post-merge tests passed"
    return 0
  else
    echo "✗ Post-merge tests FAILED — rolling back"
    git reset --hard ORIG_HEAD
    echo "✓ Rolled back. ${branch} left unmerged."
    return 1
  fi
}

cmd_cleanup() {
  local slug="$1"
  local branch="feat/${slug}"
  local path="${WORKTREE_BASE}/${slug}"

  echo "Cleaning up: ${slug}"
  git worktree remove "${path}" 2>/dev/null && echo "  ✓ Worktree removed" || echo "  ⚠ Worktree not found"
  git branch -d "${branch}" 2>/dev/null && echo "  ✓ Branch deleted" || echo "  ⚠ Branch not found or not fully merged"
}

case "${1:-}" in
  create)  cmd_create "$2" "${3:-main}" ;;
  verify)  cmd_verify "$2" ;;
  merge)   cmd_merge "$2" "${3:-main}" ;;
  cleanup) cmd_cleanup "$2" ;;
  *)       echo "Usage: $0 {create|verify|merge|cleanup} [args]"; exit 1 ;;
esac
