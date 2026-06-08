#!/bin/bash
# Diff wrapper for summarize-changes skill.
# Usage: ./summarize_diff.sh --scope [staged|unstaged|all|branch] [--base main] [--format summary|commit-msg|pr-body]
set -euo pipefail

SCOPE="all"
BASE="main"
FORMAT="summary"

while [[ $# -gt 0 ]]; do
  case $1 in
    --scope)   SCOPE="$2"; shift 2 ;;
    --base)    BASE="$2"; shift 2 ;;
    --format)  FORMAT="$2"; shift 2 ;;
    *)         echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "=== Diff Scope: $SCOPE | Base: $BASE | Format: $FORMAT ==="
echo ""

case "$SCOPE" in
  staged)     git diff --cached ;;
  unstaged)   git diff ;;
  all)        git diff HEAD ;;
  branch)     git diff "$BASE...HEAD" ;;
  *)          echo "Unknown scope: $SCOPE (use: staged|unstaged|all|branch)"; exit 1 ;;
esac

echo ""
echo "=== Stats ==="
case "$SCOPE" in
  staged|unstaged|all) git diff --stat ;;
  branch)              git diff "$BASE...HEAD" --stat ;;
esac

echo ""
echo "=== Recent Commits ==="
git log --oneline -5
