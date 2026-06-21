#!/usr/bin/env bash
# Validate the lease-lock fix (Bug C) on a query HARDER and LONGER-RUNNING than the
# primes task (api → worker → runner → deliverables). The run deliberately exceeds
# ``reclaim_min_idle_ms`` (30s) so a peer worker's XAUTOCLAIM fires mid-run —
# exactly the window where the old bug double-processed. The lease lock must make
# the peer SKIP instead.
#
# Two independent proofs:
#   (1) Deliverables recomputed from disk (anti-fabrication — we never trust the
#       run's self-reported numbers; we re-derive fib / primes / intersection).
#   (2) Claim-trail: EXACTLY ONE worker logs "Worker claimed run <id>" (was two
#       before the fix); any peer that tried mid-run logs "skipping duplicate
#       claim" (the lock working) instead of running the executor.
#
# Usage: scripts/validate_complex2.sh [api_port] [max_iterations]

set -euo pipefail

API_PORT="${1:-8800}"
MAX_ITERS="${2:-15}"
API="http://localhost:${API_PORT}/api/v1/agent"
TS="$(date +%s)"
RUN_ID="complex2-fibprimes-${TS}"
GOAL="Using the code execution tool, perform this multi-part computation and write FOUR deliverable files. (1) 'fib.csv': the first 40 Fibonacci numbers using the convention F(1)=1, F(2)=1, with columns 'index' and 'value'. (2) 'fib_analysis.json': JSON with keys 'total_count' (40), 'prime_fibonacci_numbers' (the list of those 40 Fibonacci numbers that are themselves prime), 'prime_count', 'sum', 'mean', 'max', and 'min'. (3) 'primes.csv': the first 40 prime numbers starting at 2, columns 'rank' and 'value'. (4) 'summary.json': JSON with keys 'intersection' (the Fibonacci numbers from part 1 that ALSO appear among the first 40 primes), 'intersection_count', 'largest_fib_prime', and 'mean_of_primes' (the mean of the first 40 primes). All four files must be present and contain correct, self-consistent values."
LOG_DIR="/home/amiagarw/code/turing-agent-1/self-evolving-agent/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/complex2-${RUN_ID}.log"
COMPOSE_DIR="/home/amiagarw/code/turing-agent-1/self-evolving-agent"
WORKER="$(docker compose -f "${COMPOSE_DIR}/docker-compose.yml" ps -q worker | head -1)"

json() { python3 -c "import json,sys; print(json.loads(sys.stdin.read())$1)" 2>/dev/null || echo ""; }

echo "[$(date -Iseconds)] COMPLEX2 start run_id=${RUN_ID} api=${API} worker=${WORKER}" | tee "${LOG}"

# ── enqueue ──────────────────────────────────────────────────────────────
BODY="$(python3 -c "import json,sys; print(json.dumps({'goal':sys.argv[1],'max_iterations':int(sys.argv[2]),'run_id':sys.argv[3]}))" "${GOAL}" "${MAX_ITERS}" "${RUN_ID}")"
RESP="$(curl -sS -X POST "${API}/run" -H 'Content-Type: application/json' -d "${BODY}")"
echo "[$(date -Iseconds)] enqueue resp: ${RESP}" | tee -a "${LOG}"
echo "${RESP}" | json "['thread_id']" | grep -q "api-${RUN_ID}" \
  || { echo "ENQUEUE FAILED (no thread_id for ${RUN_ID})" | tee -a "${LOG}"; exit 2; }

# ── poll (≤ ~16 min) ──────────────────────────────────────────────────────
STATUS="queued"
START_TS="$(date +%s)"
for i in $(seq 1 120); do
  SJ="$(curl -sS "${API}/runs/${RUN_ID}")"
  STATUS="$(echo "${SJ}" | json "['status']")"
  COMPLETE="$(echo "${SJ}" | json "['is_complete']")"
  ITER="$(echo "${SJ}" | json "['iteration_count']")"
  ELAPSED=$(( $(date +%s) - START_TS ))
  echo "[$(date -Iseconds)] poll #${i} status=${STATUS} complete=${COMPLETE} iter=${ITER} elapsed=${ELAPSED}s" | tee -a "${LOG}"
  case "${STATUS}" in
    completed|failed) echo "${SJ}" | python3 -m json.tool | tee -a "${LOG}"; break;;
  esac
  sleep 8
done
WALL=$(( $(date +%s) - START_TS ))
echo "[run] wall-clock duration: ${WALL}s (reclaim_min_idle_ms=30000 → lease exercised if ${WALL} > 30)" | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "========== VERIFICATION (recomputed from disk — anti-fabrication) ==========" | tee -a "${LOG}"

# Recompute EVERY invariant independently (fib sequence, primality, primes,
# intersection) and check the agent's four files against ground truth. ``docker
# exec -i`` (the -i is mandatory) so the heredoc reaches python's stdin.
VERIFY_RC=0
docker exec -i -e RUN_ID="${RUN_ID}" "${WORKER}" python3 - <<'PYEOF' | tee -a "${LOG}" || VERIFY_RC=${PIPESTATUS[0]}
import csv, json, glob, os, math

# ── ground truth, recomputed independently ───────────────────────────────
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

FIB = [1, 1]
while len(FIB) < 40:
    FIB.append(FIB[-1] + FIB[-2])
PRIME_FIBS = [x for x in FIB if is_prime(x)]
FIRST40_PRIMES = [n for n in range(2, 1000) if is_prime(n)][:40]
INTERSECTION = sorted(set(FIB) & set(FIRST40_PRIMES))
FIB_SUM = sum(FIB)

BASE = "/home/turing/.turing/results"
ok = True
def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))

def find(name):
    # Scope to THIS run's subdir first so a PRIOR run's same-named file cannot
    # poison the verdict (anti-fabrication is about THIS run's outputs). A prior
    # run left results/<other-run>/primes.csv with different columns → the
    # unscoped glob took it (hits[0]) and raised KeyError('value'). Fall back to
    # the recursive search only if the run subdir has nothing (defensive).
    RUN_ID = os.environ.get("RUN_ID", "")
    if RUN_ID:
        scoped = sorted(
            glob.glob(os.path.join(BASE, RUN_ID, "**", name), recursive=True)
        )
        if scoped:
            return scoped
    return sorted(glob.glob(os.path.join(BASE, "**", name), recursive=True))

# ── fib.csv ───────────────────────────────────────────────────────────────
hits = find("fib.csv")
if not hits:
    check("fib.csv present", False, "not found under results/")
else:
    print(f"[deliverable] fib.csv -> {hits[0]}")
    try:
        rows = list(csv.DictReader(open(hits[0])))
        vals = [int(r["value"]) for r in rows]
        check("fib.csv has 40 rows", len(vals) == 40, f"got {len(vals)}")
        check("fib.csv == first 40 fib (F1=1,F2=1)", vals == FIB, f"head={vals[:6]} tail={vals[-3:]}")
    except Exception as e:  # noqa: BLE001 — verifier reports, never crashes
        check("fib.csv parse", False, repr(e))

# ── fib_analysis.json ────────────────────────────────────────────────────
hits = find("fib_analysis.json")
if not hits:
    check("fib_analysis.json present", False, "not found under results/")
else:
    print(f"[deliverable] fib_analysis.json -> {hits[0]}")
    try:
        a = json.load(open(hits[0]))
        check("total_count == 40", a.get("total_count") == 40, f"got {a.get('total_count')}")
        pf = sorted(a.get("prime_fibonacci_numbers", []))
        check(f"prime_fibonacci_numbers == {sorted(PRIME_FIBS)}", pf == sorted(PRIME_FIBS), f"got {pf}")
        check("prime_count", a.get("prime_count") == len(PRIME_FIBS), f"got {a.get('prime_count')}")
        check("sum", a.get("sum") == FIB_SUM, f"got {a.get('sum')} expected {FIB_SUM}")
        check("mean", abs(float(a.get("mean", -1)) - FIB_SUM / 40) < 1e-6, f"got {a.get('mean')}")
        check("max", a.get("max") == max(FIB), f"got {a.get('max')}")
        check("min", a.get("min") == 1, f"got {a.get('min')}")  # F1=1
    except Exception as e:  # noqa: BLE001
        check("fib_analysis.json parse", False, repr(e))

# ── primes.csv ───────────────────────────────────────────────────────────
hits = find("primes.csv")
if not hits:
    check("primes.csv present", False, "not found under results/")
else:
    print(f"[deliverable] primes.csv -> {hits[0]}")
    try:
        rows = list(csv.DictReader(open(hits[0])))
        vals = [int(r["value"]) for r in rows]
        check("primes.csv has 40 rows", len(vals) == 40, f"got {len(vals)}")
        check("primes.csv == first 40 primes", vals == FIRST40_PRIMES, f"got {vals}")
    except Exception as e:  # noqa: BLE001
        check("primes.csv parse", False, repr(e))

# ── summary.json ─────────────────────────────────────────────────────────
hits = find("summary.json")
if not hits:
    check("summary.json present", False, "not found under results/")
else:
    print(f"[deliverable] summary.json -> {hits[0]}")
    try:
        s = json.load(open(hits[0]))
        inter = sorted(s.get("intersection", []))
        check(f"intersection == {INTERSECTION}", inter == INTERSECTION, f"got {inter}")
        check("intersection_count", s.get("intersection_count") == len(INTERSECTION), f"got {s.get('intersection_count')}")
        check("largest_fib_prime", s.get("largest_fib_prime") == max(PRIME_FIBS), f"got {s.get('largest_fib_prime')}")
        primes_mean = sum(FIRST40_PRIMES) / 40
        check("mean_of_primes", abs(float(s.get("mean_of_primes", -1)) - primes_mean) < 1e-6, f"got {s.get('mean_of_primes')}")
    except Exception as e:  # noqa: BLE001
        check("summary.json parse", False, repr(e))

print(f"\nVERDICT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
raise SystemExit(0 if ok else 1)
PYEOF

# ── LEASE-LOCK evidence (the Bug-C fix proof) ────────────────────────────
echo "" | tee -a "${LOG}"
echo "[lease-lock] claim-trail evidence for run ${RUN_ID} (the fix proof):" | tee -a "${LOG}"
TOTAL_CLAIMED=0
TOTAL_SKIPPED=0
for c in $(docker compose -f "${COMPOSE_DIR}/docker-compose.yml" ps -q worker); do
  H="$(docker inspect --format '{{.Config.Hostname}}' "${c}")"
  CL="$(docker logs "${c}" 2>&1 | grep -c "Worker claimed run ${RUN_ID}" || true)"
  SK="$(docker logs "${c}" 2>&1 | grep -cE "Run ${RUN_ID} .*already in-flight" || true)"
  CK="$(docker logs "${c}" 2>&1 | grep -c "Run ${RUN_ID} completed (acked=1)" || true)"
  TOTAL_CLAIMED=$(( TOTAL_CLAIMED + CL ))
  TOTAL_SKIPPED=$(( TOTAL_SKIPPED + SK ))
  echo "    worker ${H}: claimed=${CL}  skipped=${SK}  completed_acked1=${CK}" | tee -a "${LOG}"
done
echo "    TOTAL claimed=${TOTAL_CLAIMED} (MUST be 1 — was 2 before the fix)" | tee -a "${LOG}"
echo "    TOTAL skipped=${TOTAL_SKIPPED} (>0 proves the lease made a mid-run peer skip)" | tee -a "${LOG}"

# ── runner wire evidence ─────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
echo "[runner] execution-path evidence:" | tee -a "${LOG}"
for c in $(docker compose -f "${COMPOSE_DIR}/docker-compose.yml" ps -q worker); do
  H="$(docker inspect --format '{{.Config.Hostname}}' "${c}")"
  M="$(docker logs "${c}" 2>&1 | grep -c 'mode=runner' || true)"
  F="$(docker logs "${c}" 2>&1 | grep -c 'falling back to host subprocess' || true)"
  echo "    worker ${H}: mode=runner=${M}  host-fallback=${F}" | tee -a "${LOG}"
done
RT="$(docker logs --since 25m self-evolving-agent-runner-1 2>&1 | grep -c 'POST /execute' || true)"
echo "    runner POST /execute traces (last 25m): ${RT}" | tee -a "${LOG}"

# ── cost ─────────────────────────────────────────────────────────────────
COST="$(docker exec turing-postgres psql -U postgres -d turing_agent -t -A -c "SELECT COALESCE(round(sum(cost_usd)::numeric,4),0) FROM cost_ledger WHERE run_id='api-${RUN_ID}';")"
echo "[cost] run_id=api-${RUN_ID} cost_usd=\$${COST}" | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
LEASE_OK=1
[ "${TOTAL_CLAIMED}" -eq 1 ] || LEASE_OK=0
echo "[$(date -Iseconds)] COMPLEX2 done run_id=${RUN_ID} verify_rc=${VERIFY_RC} lease_ok=${LEASE_OK} (claimed==1) status=${STATUS}" | tee -a "${LOG}"
# Both must pass: deliverables correct AND exactly one claim.
[ "${VERIFY_RC}" -eq 0 ] && [ "${LEASE_OK}" -eq 1 ] && exit 0 || exit 1
