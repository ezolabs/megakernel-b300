#!/bin/bash
# run_all.sh OUT TAG [suite ...] -- run the quality suites one after another against the engine on this box.
#   suites (default all, in this order): gsm8k needles json agent mgsm images load
#   e.g.  bash run_all.sh /workspace/mk29/raw/quality-final final
#         bash run_all.sh /workspace/mk29/raw/quality-mk mk gsm8k load
# Each suite writes OUT/<suite>-<cell>.jsonl + .summary.json and logs to OUT/<suite>.log. ENGINE_URL / MODEL /
# VLLM_API_KEY override the defaults (http://127.0.0.1:8002, DeepSeek-V4.1-Flash, /etc/environment).
# Expected wall on the final, all suites: ~50-80 min (each suite alone is under 1 h; see README.md).
set -u
OUT=${1:?usage: run_all.sh OUT TAG [suite ...]}
TAG=${2:?usage: run_all.sh OUT TAG [suite ...]}
shift 2
SUITES=${*:-gsm8k needles json agent mgsm images load}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
echo "quality run_all: tag=$TAG out=$OUT suites=[$SUITES] start $(date -u +%FT%TZ)" | tee -a "$OUT/run_all.log"
rc_all=0
for s in $SUITES; do
  case $s in
    gsm8k)   cmd=(gsm8k_full.py) ;;
    needles) cmd=(needles.py) ;;
    json)    cmd=(json_format.py) ;;
    agent)   cmd=(agent_sessions.py) ;;
    mgsm)    cmd=(mgsm.py) ;;
    images)  cmd=(images.py) ;;
    load)    cmd=(under_load.py) ;;
    *) echo "unknown suite $s" | tee -a "$OUT/run_all.log"; rc_all=2; continue ;;
  esac
  t0=$(date +%s)
  python3 "$HERE/${cmd[0]}" --out "$OUT" --tag "$TAG" > "$OUT/$s.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && rc_all=1
  echo "$s rc=$rc wall=$(( $(date +%s) - t0 ))s $(date -u +%FT%TZ)" | tee -a "$OUT/run_all.log"
  grep -E '^(==|   [a-z_@0-9.]+ )' "$OUT/$s.log" | grep -E '\*$|^==' | tee -a "$OUT/run_all.log"
done
echo "quality run_all: done $(date -u +%FT%TZ) rc=$rc_all" | tee -a "$OUT/run_all.log"
exit $rc_all
