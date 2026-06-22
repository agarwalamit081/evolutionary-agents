#!/usr/bin/env bash
# Validate the DEPLOYED app on a complex query (api → worker → runner → deliverable),
# then verify the deliverables by RECOMPUTING the invariants from disk (anti-
# fabrication: we never trust the run's self-reported numbers). One attempt per
# invocation; the caller decides on retry.
#
# The query is deliberately multi-deliverable and forces code execution (the
# runner path): write primes.csv (first 25 primes) AND analysis.json (aggregates
# + a palindrome filter), both with exact, machine-checkable values.
#
# Usage: scripts/validate_complex.sh [api_port] [max_iterations]

set -euo pipefail

API_PORT="${1:-8800}"
MAX_ITERS="${2:-15}"
API="http://localhost:${API_PORT}/api/v1/agent"
TS="$(date +%s)"
RUN_ID="complex-primes-${TS}"
GOAL="Compute the first 25 prime numbers starting from 2. Write them to a CSV file named 'primes.csv' with two columns 'index' and 'prime'. Then create a JSON file named 'analysis.json' containing six keys: 'count', 'sum', 'mean', 'max', 'min', and 'palindromes' (the subset of the primes that read identically forwards and backwards, for example 2 and 11). Both files must be present, well-formed, and contain correct values."
LOG_DIR="/home/amiagarw/code/turing-agent-1/self-evolving-agent/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/complex-${RUN_ID}.log"
COMPOSE_DIR="/home/amiagarw/code/turing-agent-1/self-evolving-agent"
WORKER="$(docker compose -f "${COMPOSE_DIR}/docker-compose.yml" ps -q worker | head -1)"

json() { python3 -c "import json,sys; print(json.loads(sys.stdin.read())$1)" 2>/dev/null || echo ""; }

echo "[$(date -Iseconds)] COMPLEX start run_id=${RUN_ID} api=${API} worker=${WORKER}" | tee "${LOG}"

# ── enqueue ──────────────────────────────────────────────────────────────
BODY="$(python3 -c "import json,sys; print(json.dumps({'goal':sys.argv[1],'max_iterations':int(sys.argv[2]),'run_id':sys.argv[3]}))" "${GOAL}" "${MAX_ITERS}" "${RUN_ID}")"
RESP="$(curl -sS -X POST "${API}/run" -H 'Content-Type: application/json' -d "${BODY}")"
echo "[$(date -Iseconds)] enqueue resp: ${RESP}" | tee -a "${LOG}"
echo "${RESP}" | json "['thread_id']" | grep -q "api-${RUN_ID}" \
  || { echo "ENQUEUE FAILED (no thread_id for ${RUN_ID})" | tee -a "${LOG}"; exit 2; }

# ── poll (≤ ~16 min) ──────────────────────────────────────────────────────
STATUS="queued"
for i in $(seq 1 120); do
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
echo "========== VERIFICATION (recomputed from disk) ==========" | tee -a "${LOG}"

# Recompute every invariant from the on-disk files on the shared volume. NOTE:
# ``docker exec -i`` (the -i is mandatory) so the heredoc reaches python's stdin
# — without -i docker exec DISCARDS stdin and the verifier silently never runs.
VERIFY_RC=0
docker exec -i -e RUN_ID="${RUN_ID}" "${WORKER}" python3 - <<'PYEOF' | tee -a "${LOG}" || VERIFY_RC=${PIPESTATUS[0]}
import csv, json, glob, os

EXPECTED_PRIMES = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
EXPECTED_PALIN = [2,3,5,7,11]
BASE = "/home/turing/.turing/results"

def find(name):
    # Scope to THIS run's subdir first so a PRIOR run's same-named file cannot
    # poison the verdict (anti-fabrication is about THIS run's outputs). Fall
    # back to the recursive search only if the run subdir has nothing.
    RUN_ID = os.environ.get("RUN_ID", "")
    if RUN_ID:
        scoped = sorted(
            glob.glob(os.path.join(BASE, RUN_ID, "**", name), recursive=True)
        )
        if scoped:
            return scoped
    return sorted(glob.glob(os.path.join(BASE, "**", name), recursive=True))

ok = True
def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))

csv_hits = find("primes.csv")
if not csv_hits:
    check("primes.csv present", False, "not found under results/")
else:
    print(f"[deliverable] primes.csv → {csv_hits[0]}")
    try:
        rows = list(csv.DictReader(open(csv_hits[0])))
        vals = [int(r["prime"]) for r in rows]
        check("primes.csv has 25 rows", len(vals) == 25, f"got {len(vals)}")
        check("primes.csv == first 25 primes", vals == EXPECTED_PRIMES, f"got {vals}")
    except Exception as e:  # noqa: BLE001 — verifier must report, never crash
        check("primes.csv parse", False, repr(e))

json_hits = find("analysis.json")
if not json_hits:
    check("analysis.json present", False, "not found under results/")
else:
    print(f"[deliverable] analysis.json → {json_hits[0]}")
    try:
        a = json.load(open(json_hits[0]))
        check("count == 25", a.get("count") == 25, f"got {a.get('count')}")
        check("sum == 1060", a.get("sum") == 1060, f"got {a.get('sum')}")
        check("mean == 42.4", abs(float(a.get("mean", -999)) - 42.4) < 0.01, f"got {a.get('mean')}")
        check("max == 97", a.get("max") == 97, f"got {a.get('max')}")
        check("min == 2", a.get("min") == 2, f"got {a.get('min')}")
        palin = sorted(a.get("palindromes", []))
        check("palindromes == [2,3,5,7,11]", palin == EXPECTED_PALIN, f"got {palin}")
    except Exception as e:  # noqa: BLE001
        check("analysis.json parse", False, repr(e))

print(f"\nVERDICT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
raise SystemExit(0 if ok else 1)
PYEOF

# ── runner wire evidence ─────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
echo "[runner] execution-path evidence:" | tee -a "${LOG}"
for c in $(docker compose -f "${COMPOSE_DIR}/docker-compose.yml" ps -q worker); do
  H="$(docker inspect --format '{{.Config.Hostname}}' "${c}")"
  M="$(docker logs "${c}" 2>&1 | grep -c 'code_executor.*mode=runner\|Executing code.*mode=runner' || true)"
  F="$(docker logs "${c}" 2>&1 | grep -c 'falling back to host subprocess' || true)"
  echo "    worker ${H}: mode=runner=${M}  host-fallback=${F}" | tee -a "${LOG}"
done
RT="$(docker logs --since 20m self-evolving-agent-runner-1 2>&1 | grep -c 'POST /execute' || true)"
echo "    runner POST /execute traces (last 20m): ${RT}" | tee -a "${LOG}"

# ── cost ─────────────────────────────────────────────────────────────────
COST="$(docker exec self-evolving-agent-postgres-1 psql -U postgres -d turing_agent -t -A -c "SELECT COALESCE(round(sum(cost_usd)::numeric,4),0) FROM cost_ledger WHERE run_id='api-${RUN_ID}';")"
echo "[cost] run_id=api-${RUN_ID} cost_usd=\$${COST}" | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "[$(date -Iseconds)] COMPLEX done run_id=${RUN_ID} verify_rc=${VERIFY_RC} status=${STATUS}" | tee -a "${LOG}"
exit "${VERIFY_RC}"
