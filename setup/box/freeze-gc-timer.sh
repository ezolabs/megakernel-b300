#!/bin/bash
# freeze-gc-timer.sh (box-side, loopback only): POST the local engine's /freeze_gc every --interval-s
# seconds, so the schedulers' Python gen2 collections never grow back to multi-second passes (#944/#958).
#
# Why: box-b's >1 s eviction passes were gen2 collections over the radix tree's objects. After one
# POST /freeze_gc (00:36:20Z 2026-11-05) tp0's >1 s count stayed flat while thousands of passes ran,
# and a 0.05-0.5 s band grew instead: the gen2 collections over the objects made since the freeze
# (qa/eviction-stall-20261105/NOTES.md). gc.freeze() is O(1) (a list splice); its cost is that cyclic
# garbage frozen with the tree is never collected, so every tick records each TP scheduler's RSS
# before and after the freeze. The HiCache host pool is fully resident from engine start, so
# RssAnon's growth between ticks is the rest of the process (the Python heap first of all).
#
# Every tick writes ONE content-free line to $OUT/timer.log: the HTTP code, tp0's eviction-pass
# histogram (count, le 0.05/0.1/0.5/1.0, sum), and each scheduler's VmRSS/RssAnon (kB) before and
# 5 s after. The engine's key is read from API_KEY_FILE on every tick (relaunch.sh rewrites it at
# each launch) and is never printed or logged. An engine that is down or booting is logged as such
# and the loop goes on, so a watchdog relaunch needs nothing from here (the timer runs in its own
# session, outside the engine's process group and the watchdog's pkill patterns); the engine freezes
# once by itself after warmup, and the next tick resumes the cadence. One engine per box: a box with
# ENGINES (box-c's 2xTP4) needs one timer per --port and --out.
#
# Install (stage.sh puts this file at $TOOLS_REMOTE): the box file's FREEZEGC_CMD names the command,
# and the box's /root/onstart.sh starts it after the engine launch on every boot. A second start on
# the same --out exits at once (flock), so a re-run onstart or a hand start never makes two timers.
#
# usage: freeze-gc-timer.sh [--interval-s 300] [--port 8002] [--out /workspace/freezegc] [--once]
# start: setsid nohup bash /workspace/tools/freeze-gc-timer.sh --interval-s 300 >> /workspace/freezegc/nohup.log 2>&1 < /dev/null &
# stop:  touch /workspace/freezegc/stop   (ends it at the next tick, within 10 s)
set -u
INTERVAL=300
PORT=8002
OUT=/workspace/freezegc
ONCE=0
API_KEY_FILE="${API_KEY_FILE:-/etc/sglang-api-key.yaml}"
while [ $# -gt 0 ]; do
  case "$1" in
    --interval-s) INTERVAL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    *) echo "usage: $0 [--interval-s N] [--port P] [--out DIR] [--once]" >&2; exit 2 ;;
  esac
done
[[ "$INTERVAL" =~ ^[0-9]+$ ]] && [ "$INTERVAL" -ge 30 ] || { echo "--interval-s must be an integer >= 30" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "--port must be an integer" >&2; exit 2; }
mkdir -p "$OUT"
LOG="$OUT/timer.log"

# One timer per --out: a second start exits at once.
exec 9> "$OUT/timer.lock"
flock -n 9 || { echo "$(date -u +%FT%TZ) another timer holds $OUT/timer.lock -- exiting" >&2; exit 0; }
rm -f "$OUT/stop"

ts() { date -u +%FT%TZ; }

# rss: "TP0=<VmRSS>/<RssAnon> TP1=..." over the running scheduler processes, kB.
rss() {
  local pid title v a out=""
  for pid in $(pgrep -f '^sglang::scheduler_TP' | sort -n); do
    title=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    title=${title%% *}
    v=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    a=$(awk '/^RssAnon:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    out="$out ${title#sglang::scheduler_}=${v:-?}/${a:-?}"
  done
  printf '%s' "${out# }"
}

# bands: tp0's eviction-pass histogram, the numbers only.
bands() {
  curl -s -m 15 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | awk '
    /^sglang:eviction_duration_seconds/ && /tp_rank="0"/ && /UnifiedRadixCache/ {
      if ($0 ~ /_count/) c = $NF
      else if ($0 ~ /_sum/) s = $NF
      else if ($0 ~ /le="0.05"/) a = $NF
      else if ($0 ~ /le="0.1"/) b = $NF
      else if ($0 ~ /le="0.5"/) d = $NF
      else if ($0 ~ /le="1.0"/) e = $NF
    }
    END { if (c == "") print "ev=none"; else printf "ev_count=%d le0.05=%d le0.1=%d le0.5=%d le1.0=%d ev_sum=%.1f", c, a, b, d, e, s }'
}

while :; do
  [ -e "$OUT/stop" ] && { echo "$(ts) stop file -- exiting" >> "$LOG"; rm -f "$OUT/stop"; exit 0; }
  before=$(rss)
  if [ -z "$before" ]; then
    echo "$(ts) engine=down (no scheduler process)" >> "$LOG"
  else
    b=$(bands)
    key=$(sed -n 's/^api-key: *"\{0,1\}\([^"]*\)"\{0,1\} *$/\1/p' "$API_KEY_FILE" 2>/dev/null | head -1)
    t0=$(date +%s%N)
    code=$(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST -H "Authorization: Bearer $key" \
      "http://127.0.0.1:$PORT/freeze_gc" 2>/dev/null)
    ms=$(( ($(date +%s%N) - t0) / 1000000 ))
    unset key
    sleep 5
    echo "$(ts) http=${code:-000} ms=$ms $b rss_before[$before] rss_after[$(rss)]" >> "$LOG"
  fi
  [ "$ONCE" = 1 ] && exit 0
  # Sleep in short steps so the stop file is read within 10 s.
  left=$INTERVAL
  while [ "$left" -gt 0 ] && [ ! -e "$OUT/stop" ]; do sleep $(( left < 10 ? left : 10 )); left=$((left - 10)); done
done
