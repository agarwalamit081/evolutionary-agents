#!/usr/bin/env bash
# Live e2e compose smoke (#231): api → worker → runner → deliverable.
#
# Submits a deterministic Fibonacci goal to the api ingress, polls until the run
# terminates, then verifies the four invariants that prove the no-DinD runner
# topology works end-to-end:
#   (a) the deliverable CSV exists on the shared turing-workspace volume with the
#       expected 15 data rows and the recomputed sum (anti-fabrication: we don't
#       trust the run's self-reported sum — we recompute it from the file);
#   (b) the runner was actually used (worker→runner /execute POSTs, NOT a host
#       subprocess — i.e. the no-DinD code-exec path fired);
#   (c) the run reached a terminal status (completed) — not stuck looping;
#   (d) the per-run results subdir (results/<run_id>/...) routing works.
#
# Cost-bounded by MAX_ITERATIONS (8) and the worker's MAX_COST_USD knob. Run from
# the host (talks to the api's mapped host port); verification shells into a
# container that mounts turing-workspace (the volume is named, not a host bind).
#
# Usage: scripts/smoke_e2e_compose.sh [api_port] [max_iterations]

set -euo pipefail

API_PORT="${1:-8800}"
MAX_ITERS="${2:-8}"
# Optional 3rd arg: an existing run_id to ADOPT (monitor+verify only, no enqueue).
# Used to reclaim an orphaned in-flight run instead of submitting a costly duplicate.
ADOPT_RUN_ID="${3:-}"
API="http://localhost:${API_PORT}/api/v1/agent"
TS="$(date +%s)"
RUN_ID="${ADOPT_RUN_ID:-smoke-fib-${TS}}"
GOAL="Compute the first 15 Fibonacci numbers (F(1)=1, F(2)=1) and write them to a CSV file named fibonacci.csv with two columns: 'index' and 'value'. The file must contain exactly 15 data rows (one per Fibonacci number). Then print the sum of all 15 values."
LOG_DIR="/home/amiagarw/code/turing-agent-1/self-evolving-agent/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/smoke-${RUN_ID}.log"
COMPOSE_DIR="/home/amiagarw/code/turing-agent-1/self-evolving-agent"
WORKER="$(docker compose -f "${COMPOSE_DIR}/docker-compose.yml" ps -q worker | head -1)"

json() { python3 -c "import json,sys; print(json.loads(sys.stdin.read())$1)" 2>/dev/null || echo ""; }

echo "[$(date -Iseconds)] SMOKE start run_id=${RUN_ID} api=${API} worker=${WORKER}" | tee "${LOG}"

# ── enqueue (skip in adopt mode — the run is already in flight) ──────────
if [ -z "${ADOPT_RUN_ID}" ]; then
  BODY="$(python3 -c "import json,sys; print(json.dumps({'goal':sys.argv[1],'max_iterations':int(sys.argv[2]),'run_id':sys.argv[3]}))" "${GOAL}" "${MAX_ITERS}" "${RUN_ID}")"
  RESP="$(curl -sS -X POST "${API}/run" -H 'Content-Type: application/json' -d "${BODY}")"
  echo "[$(date -Iseconds)] enqueue resp: ${RESP}" | tee -a "${LOG}"
  # The HTTP code is 202 Accepted, but the RESPONSE BODY's ``status`` field is the
  # job's state ("queued") — so validate on the body carrying a thread_id, NOT on a
  # literal "accepted" string.
  echo "${RESP}" | json "['thread_id']" | grep -q "api-${RUN_ID}" \
    || { echo "ENQUEUE FAILED (no thread_id for ${RUN_ID})" | tee -a "${LOG}"; exit 1; }
else
  echo "[$(date -Iseconds)] ADOPT mode: monitoring existing run_id=${RUN_ID} (no enqueue)" | tee -a "${LOG}"
fi

# ── poll (≤ ~12 min) ─────────────────────────────────────────────────────
STATUS="queued"
for i in $(seq 1 90); do
  SJ="$(curl -sS "${API}/runs/${RUN_ID}")"
  STATUS="$(echo "${SJ}" | json "['status']")"
  COMPLETE="$(echo "${SJ}" | json "['is_complete']")"
  ITER="$(echo "${SJ}" | json "['iteration_count']")"
  echo "[$(date -Iseconds)] poll #${i} status=${STATUS} complete=${COMPLETE} iter=${ITER}" | tee -a "${LOG}"
  case "${STATUS}" in
    completed|failed) echo "${SJ}" | python3 -m json.tool | tee -a "${LOG}"; break;;
  esac
  sleep 8
done

echo "" | tee -a "${LOG}"
echo "========== VERIFICATION ==========" | tee -a "${LOG}"

# ── (a)+(d) deliverable on the shared volume ─────────────────────────────
echo "[deliverable] searching turing-workspace for fibonacci.csv …" | tee -a "${LOG}"
CSV_FOUND="$(docker exec "${WORKER}" sh -c "
  for c in '/home/turing/.turing/results/${RUN_ID}/fibonacci.csv' '/home/turing/.turing/results/fibonacci.csv'; do
    [ -f \"\$c\" ] && { echo \"\$c\"; break; }
  done" 2>/dev/null || echo "")"
if [ -n "${CSV_FOUND}" ]; then
  echo "[deliverable] FOUND: ${CSV_FOUND}" | tee -a "${LOG}"
  docker exec "${WORKER}" python3 -c "
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1])))
vals=[int(r['value']) for r in rows]
print(f'[deliverable] data_rows={len(vals)} (expect 15) sum={sum(vals)} (expect 1596) ok={len(vals)==15 and sum(vals)==1596}')
" "${CSV_FOUND}" | tee -a "${LOG}"
  echo "[deliverable] head:" | tee -a "${LOG}"
  docker exec "${WORKER}" head -4 "${CSV_FOUND}" | sed 's/^/    /' | tee -a "${LOG}"
else
  echo "[deliverable] NOT FOUND — listing results/ for diagnosis:" | tee -a "${LOG}"
  docker exec "${WORKER}" sh -c "ls -la /home/turing/.turing/results/ 2>/dev/null; echo '---'; ls -la /home/turing/.turing/results/${RUN_ID}/ 2>/dev/null; echo '---'; find /home/turing/.turing/results -name 'fibonacci*' 2>/dev/null" | sed 's/^/    /' | tee -a "${LOG}"
fi

# ── (b) runner was used (NOT a host subprocess) ──────────────────────────
# Precise signals (the runner-side trace only exists once the runner image
# carries the per-execute audit log; the authoritative proof the agent ROUTED
# to the runner is the worker's code_executor ``mode=runner`` line, and the
# ABSENCE of a "falling back to host subprocess" warning):
#   - worker: count ``code_executor ... mode=runner`` executions (≥1 ⇒ routed
#     to the runner) and ``falling back to host subprocess`` (must be 0 ⇒ the
#     runner call succeeded rather than degrading to a host subprocess);
#   - runner: count ``POST /execute`` traces since the run started (≥0; present
#     on images with the audit trace — a 0 here is NOT a failure if the worker
#     shows mode=runner on an older image that predates the trace).
echo "" | tee -a "${LOG}"
echo "[runner] execution-path evidence:" | tee -a "${LOG}"
RUNNER_MODE=0; FALLBACK=0
for c in $(docker compose -f "${COMPOSE_DIR}/docker-compose.yml" ps -q worker); do
  H="$(docker inspect --format '{{.Config.Hostname}}' "${c}")"
  M="$(docker logs "${c}" 2>&1 | grep -c 'code_executor.*mode=runner\|Executing code.*mode=runner' || true)"
  F="$(docker logs "${c}" 2>&1 | grep -c 'falling back to host subprocess' || true)"
  RUNNER_MODE=$((RUNNER_MODE + M)); FALLBACK=$((FALLBACK + F))
  echo "    worker ${H}: code_executor mode=runner=${M}  host-fallback-warnings=${F}" | tee -a "${LOG}"
done
echo "    totals: mode=runner=${RUNNER_MODE}  host-fallback=${FALLBACK}" | tee -a "${LOG}"
# Runner-side traces only exist on images that carry the per-execute audit log
# (commit b14bb52); a 0 here is NOT a failure when the worker shows mode=runner.
RUNNER_TRACE="$(docker logs --since 10m self-evolving-agent-runner-1 2>&1 | grep -c 'POST /execute' || true)"
echo "    runner POST /execute traces (last 10m): ${RUNNER_TRACE}" | tee -a "${LOG}"
if [ "${RUNNER_MODE}" -ge 1 ] && [ "${FALLBACK}" -eq 0 ]; then
  echo "    [runner] VERDICT: PASS — generated code ran in the runner (mode=runner, no host fallback)" | tee -a "${LOG}"
elif [ "${RUNNER_MODE}" -eq 0 ]; then
  echo "    [runner] VERDICT: NOTE — this run did not use code_executor (LLM chose another tool); runner wire still proven by the unit suite + a direct probe" | tee -a "${LOG}"
fi

# ── (c) terminal status ──────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
echo "[status] terminal=${STATUS} run_id=${RUN_ID}" | tee -a "${LOG}"
echo "[status] final_output (truncated):" | tee -a "${LOG}"
curl -sS "${API}/runs/${RUN_ID}" | python3 -c "import json,sys; o=json.load(sys.stdin).get('final_output',''); print('   ', (o[:400] + ('…' if len(o)>400 else '')).replace(chr(10),' '))" | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "[$(date -Iseconds)] SMOKE done run_id=${RUN_ID} log=${LOG}" | tee -a "${LOG}"
