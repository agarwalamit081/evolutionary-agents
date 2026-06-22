#!/usr/bin/env bash
# Validate the lease-lock fix (Bug C) on a query HARDER than the fibprimes task:
# a 5-deliverable number-theory survey (Collatz over n=1..100 with step/peak
# tracking + primes <=500 + twin-prime stats + cross-referenced summary). The
# run is longer-running than the simple case, so it spans the
# ``reclaim_min_idle_ms`` (30s) window where the old bug double-processed — the
# per-run lease must keep EXACTLY ONE worker on the job.
#
# Three independent checks:
#   (1) Deliverables recomputed from disk (anti-fabrication — we never trust the
#       run's self-reported numbers; we re-derive Collatz steps/peaks, the prime
#       list, twin primes, and the cross-deliverable summary from scratch).
#   (2) Claim-trail: EXACTLY ONE worker logs "Worker claimed run <id>" (was two
#       before the fix); any peer that reclaimed mid-run logs "already in-flight"
#       instead of running the executor a second time.
#   (3) Peer-activity dump: what the non-claiming worker actually did during the
#       run window (so a skipped=0 result is explained, not silent).
#
# Usage: scripts/validate_complex3.sh [api_port] [max_iterations]

set -euo pipefail

API_PORT="${1:-8800}"
MAX_ITERS="${2:-15}"
API="http://localhost:${API_PORT}/api/v1/agent"
TS="$(date +%s)"
RUN_ID="complex3-collatz-${TS}"
GOAL="Using the code execution tool, perform a multi-part number-theory computation and write FIVE deliverable files. (1) 'collatz.csv': for each integer n from 1 to 100 (inclusive), compute its Collatz sequence (start at n; while the value is not 1: if even divide by 2, else multiply by 3 and add 1) and record one row with columns 'n', 'steps' (the number of iterations taken to reach 1; for n=1 this is 0), and 'peak' (the maximum value reached across the entire sequence, including the starting value n). There must be exactly 100 rows. (2) 'collatz_stats.json': JSON with keys 'n_with_most_steps' (the n in 1..100 with the largest steps value; if tied, the smallest n), 'most_steps' (that maximum steps value), 'global_peak' (the single largest peak value reached across all 100 sequences), 'mean_steps' (the arithmetic mean of the steps values over n=1..100), and 'all_reach_one' (true if every n from 1 to 100 reached 1). (3) 'primes.csv': every prime number less than or equal to 500, one row each, columns 'rank' (1-based, ascending) and 'value', sorted by value ascending. (4) 'prime_stats.json': JSON with keys 'prime_count' (number of primes <=500), 'twin_prime_count' (number of pairs (p, p+2) where both p and p+2 are prime and p+2 <= 500), 'sum_of_primes', 'mean_of_primes', and 'largest_prime'. (5) 'summary.json': JSON with keys 'collatz_peak_value' (equal to global_peak from part 2), 'collatz_peak_is_prime' (boolean: is global_peak a prime number?), 'collatz_n_at_peak' (the n in 1..100 whose sequence produced global_peak; if tied, the smallest n), 'overlap_count' (the count of n in 1..100 whose steps value is itself a prime number), and 'overlap_values' (those n, sorted ascending as a list). All five files must be present and contain correct, self-consistent values."
LOG_DIR="/home/amiagarw/code/turing-agent-1/self-evolving-agent/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/complex3-${RUN_ID}.log"
COMPOSE_DIR="/home/amiagarw/code/turing-agent-1/self-evolving-agent"

json() { python3 -c "import json,sys; print(json.loads(sys.stdin.read())$1)" 2>/dev/null || echo ""; }

echo "[$(date -Iseconds)] COMPLEX3 start run_id=${RUN_ID} api=${API}" | tee "${LOG}"

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

# Recompute EVERY invariant independently (Collatz steps/peaks, the prime list,
# twin primes, the cross-deliverable summary) and check the agent's five files
# against ground truth. ``docker exec -i`` (the -i is mandatory) + -e RUN_ID so
# the find() resolves THIS run's subdir only (a prior run's same-named file must
# not poison the verdict — the cross-run glob leak, fixed in validate_complex2).
WORKER="$(docker compose -f "${COMPOSE_DIR}/docker-compose.yml" ps -q worker | head -1)"
VERIFY_RC=0
docker exec -i -e RUN_ID="${RUN_ID}" "${WORKER}" python3 - <<'PYEOF' | tee -a "${LOG}" || VERIFY_RC=${PIPESTATUS[0]}
import csv, json, glob, os

# ── ground truth, recomputed independently ───────────────────────────────
def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    i = 3
    while i * i <= n:
        if n % i == 0: return False
        i += 2
    return True

def collatz(n):
    steps, peak, v = 0, n, n
    while v != 1:
        v = v // 2 if v % 2 == 0 else 3 * v + 1
        steps += 1
        if v > peak: peak = v
    return steps, peak

STEPS = {n: collatz(n)[0] for n in range(1, 101)}
PEAKS = {n: collatz(n)[1] for n in range(1, 101)}
MOST_STEPS = max(STEPS.values())
N_MOST_STEPS = min(n for n, s in STEPS.items() if s == MOST_STEPS)
GLOBAL_PEAK = max(PEAKS.values())
N_AT_PEAK = min(n for n, p in PEAKS.items() if p == GLOBAL_PEAK)
MEAN_STEPS = sum(STEPS.values()) / 100
PRIMES_500 = [n for n in range(2, 501) if is_prime(n)]
TWIN = sum(1 for p in PRIMES_500 if (p + 2) <= 500 and is_prime(p + 2))
OVERLAP = sorted(n for n in range(1, 101) if is_prime(STEPS[n]))

BASE = "/home/turing/.turing/results"
ok = True
def check(label, cond, detail=""):
    global ok; ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
def find(name):
    RUN_ID = os.environ.get("RUN_ID", "")
    if RUN_ID:
        scoped = sorted(glob.glob(os.path.join(BASE, RUN_ID, "**", name), recursive=True))
        if scoped: return scoped
    return sorted(glob.glob(os.path.join(BASE, "**", name), recursive=True))

# ── collatz.csv ───────────────────────────────────────────────────────────
hits = find("collatz.csv")
if not hits:
    check("collatz.csv present", False, "not found in run subdir")
else:
    print(f"[deliverable] collatz.csv -> {hits[0]}")
    try:
        rows = list(csv.DictReader(open(hits[0])))
        check("collatz.csv has 100 rows", len(rows) == 100, f"got {len(rows)}")
        got_steps = {int(r["n"]): int(r["steps"]) for r in rows}
        got_peaks = {int(r["n"]): int(r["peak"]) for r in rows}
        check("collatz steps match (all 100)", got_steps == STEPS,
              f"n=27 steps got={got_steps.get(27)} exp={STEPS[27]}")
        check("collatz peaks match (all 100)", got_peaks == PEAKS,
              f"n=27 peak got={got_peaks.get(27)} exp={PEAKS[27]}")
    except Exception as e:
        check("collatz.csv parse", False, repr(e))

# ── collatz_stats.json ────────────────────────────────────────────────────
hits = find("collatz_stats.json")
if not hits:
    check("collatz_stats.json present", False, "not found in run subdir")
else:
    print(f"[deliverable] collatz_stats.json -> {hits[0]}")
    try:
        a = json.load(open(hits[0]))
        check("n_with_most_steps", a.get("n_with_most_steps") == N_MOST_STEPS, f"got {a.get('n_with_most_steps')} exp {N_MOST_STEPS}")
        check("most_steps", a.get("most_steps") == MOST_STEPS, f"got {a.get('most_steps')} exp {MOST_STEPS}")
        check("global_peak", a.get("global_peak") == GLOBAL_PEAK, f"got {a.get('global_peak')} exp {GLOBAL_PEAK}")
        check("mean_steps", abs(float(a.get("mean_steps", -1)) - MEAN_STEPS) < 1e-6, f"got {a.get('mean_steps')} exp {MEAN_STEPS}")
        check("all_reach_one", a.get("all_reach_one") is True, f"got {a.get('all_reach_one')}")
    except Exception as e:
        check("collatz_stats.json parse", False, repr(e))

# ── primes.csv ────────────────────────────────────────────────────────────
hits = find("primes.csv")
if not hits:
    check("primes.csv present", False, "not found in run subdir")
else:
    print(f"[deliverable] primes.csv -> {hits[0]}")
    try:
        rows = list(csv.DictReader(open(hits[0])))
        vals = [int(r["value"]) for r in rows]
        check("primes.csv == primes <=500", vals == PRIMES_500, f"got {len(vals)} exp {len(PRIMES_500)}; head={vals[:5]} tail={vals[-3:]}")
    except Exception as e:
        check("primes.csv parse", False, repr(e))

# ── prime_stats.json ──────────────────────────────────────────────────────
hits = find("prime_stats.json")
if not hits:
    check("prime_stats.json present", False, "not found in run subdir")
else:
    print(f"[deliverable] prime_stats.json -> {hits[0]}")
    try:
        a = json.load(open(hits[0]))
        check("prime_count", a.get("prime_count") == len(PRIMES_500), f"got {a.get('prime_count')} exp {len(PRIMES_500)}")
        check("twin_prime_count", a.get("twin_prime_count") == TWIN, f"got {a.get('twin_prime_count')} exp {TWIN}")
        check("sum_of_primes", a.get("sum_of_primes") == sum(PRIMES_500), f"got {a.get('sum_of_primes')}")
        check("mean_of_primes", abs(float(a.get("mean_of_primes", -1)) - sum(PRIMES_500) / len(PRIMES_500)) < 1e-6, f"got {a.get('mean_of_primes')}")
        check("largest_prime", a.get("largest_prime") == 499, f"got {a.get('largest_prime')}")
    except Exception as e:
        check("prime_stats.json parse", False, repr(e))

# ── summary.json (cross-deliverable) ──────────────────────────────────────
hits = find("summary.json")
if not hits:
    check("summary.json present", False, "not found in run subdir")
else:
    print(f"[deliverable] summary.json -> {hits[0]}")
    try:
        s = json.load(open(hits[0]))
        check("collatz_peak_value", s.get("collatz_peak_value") == GLOBAL_PEAK, f"got {s.get('collatz_peak_value')} exp {GLOBAL_PEAK}")
        check("collatz_peak_is_prime", s.get("collatz_peak_is_prime") is is_prime(GLOBAL_PEAK), f"got {s.get('collatz_peak_is_prime')} exp {is_prime(GLOBAL_PEAK)}")
        check("collatz_n_at_peak", s.get("collatz_n_at_peak") == N_AT_PEAK, f"got {s.get('collatz_n_at_peak')} exp {N_AT_PEAK}")
        check("overlap_count", s.get("overlap_count") == len(OVERLAP), f"got {s.get('overlap_count')} exp {len(OVERLAP)}")
        check("overlap_values", sorted(s.get("overlap_values", [])) == OVERLAP, f"got {sorted(s.get('overlap_values', []))[:8]}… exp {OVERLAP[:8]}…")
    except Exception as e:
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
echo "    TOTAL skipped=${TOTAL_SKIPPED} (>0 proves a peer reclaimed mid-run and the lease made it skip)" | tee -a "${LOG}"

# ── peer-activity dump (explains a skipped=0, never silent) ───────────────
echo "" | tee -a "${LOG}"
echo "[peer-activity] non-claiming worker's activity during the run window:" | tee -a "${LOG}"
CLAIMANT=""
for c in $(docker compose -f "${COMPOSE_DIR}/docker-compose.yml" ps -q worker); do
  if docker logs "${c}" 2>&1 | grep -q "Worker claimed run ${RUN_ID}"; then CLAIMANT="${c}"; fi
done
for c in $(docker compose -f "${COMPOSE_DIR}/docker-compose.yml" ps -q worker); do
  H="$(docker inspect --format '{{.Config.Hostname}}' "${c}")"
  ROLE="claimant"; [ "${c}" != "${CLAIMANT}" ] && ROLE="peer"
  RE="$(docker logs --since 20m "${c}" 2>&1 | grep -cE 'reclaim_stale|XAUTOCLAIM|Recovering|reclaim' || true)"
  echo "    ${ROLE} ${H}: reclaim-related lines(last20m)=${RE}" | tee -a "${LOG}"
done

# ── runner wire evidence ─────────────────────────────────────────────────
echo "" | tee -a "${LOG}"
echo "[runner] execution-path evidence:" | tee -a "${LOG}"
for c in $(docker compose -f "${COMPOSE_DIR}/docker-compose.yml" ps -q worker); do
  H="$(docker inspect --format '{{.Config.Hostname}}' "${c}")"
  M="$(docker logs "${c}" 2>&1 | grep -c 'mode=runner' || true)"
  F="$(docker logs "${c}" 2>&1 | grep -c 'falling back to host subprocess' || true)"
  echo "    worker ${H}: mode=runner=${M}  host-fallback=${F}" | tee -a "${LOG}"
done
RT="$(docker logs --since 20m self-evolving-agent-runner-1 2>&1 | grep -c 'POST /execute' || true)"
echo "    runner POST /execute traces (last 20m): ${RT}" | tee -a "${LOG}"

# ── cost ─────────────────────────────────────────────────────────────────
COST="$(docker exec self-evolving-agent-postgres-1 psql -U postgres -d turing_agent -t -A -c "SELECT COALESCE(round(sum(cost_usd)::numeric,4),0) FROM cost_ledger WHERE run_id='api-${RUN_ID}';")"
echo "[cost] run_id=api-${RUN_ID} cost_usd=\$${COST}" | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
LEASE_OK=1
[ "${TOTAL_CLAIMED}" -eq 1 ] || LEASE_OK=0
echo "[$(date -Iseconds)] COMPLEX3 done run_id=${RUN_ID} verify_rc=${VERIFY_RC} lease_ok=${LEASE_OK} (claimed==1) status=${STATUS}" | tee -a "${LOG}"
# Both must pass: deliverables correct AND exactly one claim.
[ "${VERIFY_RC}" -eq 0 ] && [ "${LEASE_OK}" -eq 1 ] && exit 0 || exit 1
