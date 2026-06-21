#!/usr/bin/env bash
# Live monitoring session for the battery. Idempotent: re-running kills any
# existing session first. The user can attach with:  tmux attach -t turing-mon
#
# Layout:
#   pane 0 (left, tall)  — main agent log (tail -F waits for the file to appear)
#   pane 1 (right top)   — category logs: tools / subagents / llm
#   pane 2 (right bot.)  — DB count poller → monitor/counts.tsv, tailed live
set -euo pipefail

SESSION="turing-mon"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/home/amiagarw/aiml01/bin/activate"
mkdir -p "$ROOT/logs" "$ROOT/monitor"

# Idempotent: tear down any prior session so re-running is clean.
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -c "$ROOT"
# pane 0 — main log
tmux send-keys -t "$SESSION:0.0" "tail -F logs/turing_agent.log 2>/dev/null" C-m
# split right column
tmux split-window -h -t "$SESSION:0.0" -c "$ROOT"
# pane 1 (right top) — category logs
tmux send-keys -t "$SESSION:0.1" \
  "tail -F logs/tools.log logs/subagents.log logs/llm.log 2>/dev/null" C-m
# split right column vertically → pane 2 (right bottom)
tmux split-window -v -t "$SESSION:0.1" -c "$ROOT"
# pane 2 — DB count poller (30s) appended to counts.tsv + tailed
tmux send-keys -t "$SESSION:0.2" \
  "bash -lc 'source $VENV && while true; do python scripts/_battery_counts.py --label TICK >> monitor/counts.tsv 2>/dev/null; sleep 30; done; tail -F monitor/counts.tsv'" C-m

echo "tmux monitor '$SESSION' started."
echo "  attach:   tmux attach -t $SESSION"
echo "  detach:   Ctrl-b d"
echo "  stop:     tmux kill-session -t $SESSION"
