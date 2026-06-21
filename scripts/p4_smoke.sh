#!/usr/bin/env bash
# Phase-4 cost-bounded live smoke — plan §"Verification (end-to-end, after G)".
#
# Confirms the four Phase-4 knobs fire against real LLMs (MAX_COST_USD=3 hard cap):
#   (a) a COMPLEX plan routes to a MODERATE model (glm-4.7) — proof = BOTH glm-4.7
#       (plan/reflect/verify via de-flat + NODE_TIER_MAP) AND deepseek-v4-flash
#       (execute via the CHEAP override) appear in one run; old flat routing showed
#       only deepseek-v4-flash.
#   (b) evolution still fires + deploys (objective-success gate: deliverable on disk
#       + all steps done).
#   (c) TOOL_RETRIEVAL_ENABLED=true shows a top-k subset (log: "Tool retrieval
#       selected N/M tools").
#   (d) a run that deploys a TOOL mutation re-executes exactly once then completes
#       (EVOLUTION_REEXECUTE_TOOL=true; NON-DETERMINISTIC — may not fire in one run).
#
# Usage:  bash scripts/p4_smoke.sh
# Then:   grep the log, or query cost_ledger by run_id cli-p4-smoke-<ts>.

set -uo pipefail

# Resolve the self-evolving-agent repo root from the script location (this is its
# own git repo, not the parent turing-agent-1).
cd "$(dirname "$0")/.."

source /home/amiagarw/aiml01/bin/activate

mkdir -p logs

TS="$(date +%Y%m%d-%H%M%S)"
RUN_ID="p4-smoke-${TS}"
echo "${RUN_ID}" > logs/p4-smoke-latest.txt

GOAL="Build an analytics pipeline for a synthetic online store. Step 1: generate a synthetic dataset of 20 orders with fields order_id, product, category, quantity, unit_price, and order_ts (ISO-8601 UTC timestamps). Step 2: compute total revenue, revenue per category, the top-3 products by revenue, and the average order value. Step 3: write a single well-formed JSON object to results/p4_smoke/analytics_report.json with keys total_revenue, revenue_by_category, top_3_products, average_order_value, and row_count. All order_ts timestamps must be valid UTC. Compute every number from the generated dataset and do not hardcode totals."

echo "Launching Phase-4 smoke  run_id=${RUN_ID}  goal_bytes=${#GOAL}"

# Intentionally NOT using `set -e` so the run's exit code is captured + reported,
# not swallowed by the shell. The MAX_COST_USD cap is the hard spend guard.
MAX_COST_USD=3 \
TOOL_RETRIEVAL_ENABLED=true \
TOOL_RETRIEVAL_TOP_K=8 \
EVOLUTION_REEXECUTE_TOOL=true \
RESULTS_PER_RUN_SUBDIR=true \
python main.py --verbose --run-id "${RUN_ID}" --goal "${GOAL}" \
  > "logs/p4-smoke-${RUN_ID}.log" 2>&1
RC=$?

echo "EXIT=${RC}  RUN_ID=${RUN_ID}  log=logs/p4-smoke-${RUN_ID}.log"
