#!/bin/bash
# Box watchdog (runs ON the box, staged to $TOOLS_REMOTE/watchdog.sh, started by relaunch.sh
# pgrep-guarded so exactly one instance survives across relaunches and container restarts).
# Ported from the earlier per-box v2 ladder, box-parametrized. A single box's restart is by construction
# an outage for the router, so the worst case must be bounded: detect + boot, never open-ended.
#
# Engine signals, checked every 30 s, in this order:
#   1. no `sglang serve` process                                        -> relaunch at once
#      (SGLang's own --watchdog-timeout, default 300 s, aborts the scheduler on a stuck forward --
#       a silent kernel hang ends here, as a dead process)
#   2. /health not 200 for HEALTH_FAILS (3) consecutive reads after BOOT_GRACE (2700 s) since
#      launch, AND a 1-token chat-completion probe (PROBE_TIMEOUT, 600 s) also fails/times out
#                                                                         -> relaunch
#
#      A cold boot is weight load, JIT, FlashInfer autotune and CUDA-graph captures. On box-e it took
#      13 min. box-a's first boot took 27 min, about 10 of them autotune.
#      BOOT_GRACE is the box file's WATCHDOG_BOOT_GRACE, else 2700 s (45 min). An exported BOOT_GRACE
#      wins over both.
#      A boot that outlives the grace is not killed while it still makes progress. Until the engine
#      first answers /health 200 after its launch, the miss ladder ends in boot_progressing(), not in
#      progress(). boot_progressing() is true while $SERVE_LOG was written within the last BOOT_STALL
#      s. BOOT_STALL is the box file's WATCHDOG_BOOT_STALL, else 900 s, because autotune and graph
#      capture can sit quiet for minutes. Such a boot is logged "booting, not dead (...)" and left
#      alone. A boot whose log has gone silent for BOOT_STALL s is relaunched as "boot stalled".
#      The CEILING bounds a boot whose log never stops. A boot that reaches BOOT_MAX s since launch
#      with no /health 200 is relaunched however busy its log is, logged "boot ceiling". A boot that
#      loops on errors writes its log forever, so the log alone cannot exempt it forever.
#      BOOT_MAX is the box file's WATCHDOG_BOOT_MAX, else 5400 s (90 min), over three times box-a's
#      27-min cold boot. An exported BOOT_MAX wins over both.
#      The ceiling is checked on the same miss ladder, so it never fires before BOOT_GRACE. It is a
#      boot rule only: once the engine has answered /health 200, the ceiling no longer applies.
#      NOTE: /health runs a real generation probe that queues FCFS behind user prefills. A lone slow
#      or failed read under load means busy, not dead. So a 200 on the confirming probe resets the
#      health-miss counter instead of relaunching.
#
#   3. serving, but a 1-token chat completion fails or exceeds PROBE_TIMEOUT, PROBE_FAILS (3)
#      checks in a row                                                   -> relaunch
#
#   Busy, not dead: once the engine has served (its first /health 200 since launch), the relaunches
#   in 2 and 3 wait on a progress() check, one level further out than the confirming probe above.
#   Signal 1's relaunch never does -- a dead process makes no progress -- and a boot that never
#   served has its own rule, in 2. The check: if $SERVE_LOG's mtime is under 60 s old, OR /metrics
#   still shows sglang:num_running_reqs + sglang:num_queue_reqs > 0, the engine is busy, not dead. Logs
#   "busy, not dead (running R queue Q, log age S s)" and resets the miss counters instead of
#   relaunching -- the probe itself can time out under PROBE_TIMEOUT while a long-tailed
#   decode is still producing tokens (log growing) or has requests admitted (queue/running
#   > 0), and killing that engine mid-generation is strictly worse than waiting.
#
#   Key rotation: a probe answered 401 is first a question about the key, not the engine. The
#   watchdog re-reads its bearer (key_reread: /etc/environment, then the engine's own key file) and,
#   on a different key, retries that probe once before anything is counted; a 401 with no other key
#   to try is a miss like any other code.
#
# Relaunch loop guard: this watchdog's OWN relaunch() calls are tracked (timestamps kept only
# for the life of this process -- an engine relaunch never restarts the watchdog, so only a
# watchdog restart or a box reboot resets this ring). It exists because an args file whose HTTP
# server dies at lifespan (a bad flag combination, not a crash -- e.g. R2's
# --tokenizer-worker-num 2) makes signal 1 above relaunch the IDENTICAL broken line forever, at
# a full weight-load cycle (~5-6 min) per attempt, GPUs churning, never serving, until a human
# notices. When the Nth relaunch (RELAUNCH_LOOP_N, default 3) would land within
# RELAUNCH_LOOP_WINDOW seconds (default 1800, i.e. 30 min) of the oldest one still in the
# window, the dying args are never relaunched again unexamined: the newest
# sglang-args.txt.pre-* backup on the box (by mtime -- a cutover.sh/restart.sh install target)
# is restored over sglang-args.txt first (the dying file kept alongside as
# sglang-args.txt.looping-<ts> for the operator, and the matching sglang-env.txt.pre-<tag>
# restored too, if one exists), then the relaunch proceeds once more against the restored args.
# With no backup on the box at all, the hold file is touched instead of relaunching at all, and
# the operator runs cutover.sh by hand. Both knobs are env overrides on watchdog.sh, same
# convention as BOOT_GRACE/HEALTH_FAILS/PROBE_FAILS above.
# A deliberate restart (cutover.sh / restart.sh) holds $TOOLS_REMOTE/watchdog.hold; this watchdog
# sleeps through it (stale after HOLD_MAX s) so a cutover never races a relaunch into two engines
# on the same GPUs -- and a relaunch already under way when the hold appears re-reads it (hold_fresh)
# on entry, before kill_engine's KILL and before its launch, and stands down (`RELAUNCH ABORTED`):
# the loop's own read at the top of a pass is 10+ s stale by the time a relaunch launches.
#
# Caddy (the direct-TLS front door, tls.sh): checked in the SAME loop, on the same 30 s cadence,
# independently of the engine signal above -- a dead Caddy is not an engine failure and must
# never trigger relaunch.sh, which is why its check and its own relaunch are their own branch,
# never folded into relaunch(). Caddy's launcher path is fixed: $TOOLS_REMOTE/caddy-relaunch.sh
# (written by tls.sh; since #1306 it reads no caddy.env and carries no token -- Caddy serves the
# certificate files the operator pushes in). The
# engine's BOOT_GRACE window does not apply to Caddy -- it has nothing analogous to weight-load
# JIT warmup, so its own health check is held to a fixed, much shorter grace (CADDY_BOOT_GRACE,
# default 180 s) after each of its own relaunches, tracked separately from LAST_LAUNCH.
# Caddy's /health is the ENGINE's /health proxied, so a Caddy that is up answers 502 (or 503/504, or
# holds the request past the probe's 10 s) whenever the engine is booting or down. That is not a
# Caddy miss: during box-a's 27-minute first boot (2026-10-26) counting it as one relaunched Caddy
# three times, each relaunch risking an in-flight ACME order and fixing nothing. caddy_verdict()
# separates the two: Caddy is DOWN on a refused connection, a failed TLS handshake (a bad or
# self-signed cert fails curl's verification, so the handshake never completes), any other code, or
# no Caddy process (the branch above it); Caddy is UP with the upstream not ready when the handshake
# completed (curl's time_appconnect > 0) AND the answer was 502/503/504 or no answer in time, AND
# the engine's own loopback /health is not 200 at that moment -- logged once per spell as
# "CADDY UP, upstream not ready", and the miss counter resets (Caddy just proved it answers). The
# same gateway answer while the engine's own /health IS 200 is Caddy failing to proxy: a miss.
# The warm gate (#379) is the one other 503 that is not a miss: relaunch.sh writes
# $TOOLS_REMOTE/engine.warming at every engine start and warm-short.py removes it once the short
# in-process warm has run (or at its timeout), and while it exists the door answers 503 over a
# healthy engine (tls.sh's @warming matcher). warm_gate_state() reads it as `warming` while its owner
# lives; caddy_verdict's fourth argument turns that 503 into `warming` (logged once per spell, the
# miss counter reset). A flag whose owner is gone -- warm-short.py's pid dead, or relaunch's own
# owner=relaunch record never taken over within WARM_GATE_TAKEOVER s (warm-short.py never started)
# -- is stale: removed with a `WARM GATE STALE` line, so the door never stays shut on a dead warm.
# Probed through the real hostname (--resolve to loopback), never `curl -k`: a `-k` probe would
# still read 200 even if the ACME DNS-01 cert silently expired or Caddy fell back to its own
# internal self-signed issuer -- exactly the failure this probe exists to catch.
#
# Caddy's own health-miss ladder is deliberately NOT an immediate relaunch, unlike the
# `process missing` branch above it. A relaunch never fixes a certificate problem -- it fixes a
# dead process -- and a relaunch mid-issuance actively cancels the in-flight ACME order
# (`finalizing order ... context canceled`, measured), turning a slow-but-completing DNS-01
# round into a never-completing one. So a health miss only increments CFAILS (logged as
# `CADDY MISS n/N -> <code>`); only once CFAILS reaches CADDY_FAILS (default 10, i.e. 5 min at
# this loop's 30 s cadence) does the branch relaunch, and CFAILS resets to 0 on any 200 or on
# any relaunch. `process missing` still relaunches at once -- there is no in-flight order to
# protect when the process itself is gone.
#
# Relaunch prewarm (#1256): every relaunch() of this watchdog's runs the full prewarm sweep
# (prewarm-box-d.py, every tail-chunk size to the args' chunk) on the new engine BEFORE the warm gate
# lifts. warm-short.py's short warm alone left a relaunched copy paying a one-time compile on its
# first long cold prefill (#1255: a 110k needle lost on box-c copy a after the provider reset's container
# restart). A subshell (relaunch_prewarm, documented at its definition) takes the gate from
# relaunch.sh / warm-short.py, waits for /health 200, sweeps under `timeout` (box-file
# WATCHDOG_RELAUNCH_PREWARM_TIMEOUT, default 1800 s), and only then opens the door; a failed or
# timed-out sweep is logged `RELAUNCH PREWARM FAILED ... FALLING BACK` and warm-short.py's short warm
# runs instead. A boot this watchdog did not launch (onstart's relaunch.sh after a container restart)
# is adopted at the watchdog's start while its gate is still relaunch.sh's or warm-short.py's and no
# hold is fresh. While it sweeps, the loop counts no /health or probe miss the normal way -- but
# HEALTH_FAILS /health misses IN A ROW (minutes; a compile fails it for 17-34 s) read as a wedged
# engine: they end the sweep and relaunch (prewarm_sweep_health, #1271), rather than waiting out
# the sweep's timeout and the fallback warm. The door answers 503 the whole time, so the router
# never sees the copy ready mid-sweep. WATCHDOG_RELAUNCH_PREWARM=off (box file) restores the short
# warm alone. Log lines: `RELAUNCH PREWARM ...` here; the sweep's own output in
# /workspace/prewarm-relaunch[-<id>].log.
#
# Every probe/health request carries a curl User-Agent: a UA-less/default client can be blocked
# at a front door bot rule -- harmless on loopback, required the moment this ever runs through one.
#
# Stop it: pkill -f 'watchdo[g].sh' -- and never name this file literally in the same ssh
# one-liner (write watch"dog".sh), or the pkill/pgrep matches that shell itself.
# Log: /workspace/$BOX-watchdog.log.
#
# Per engine: `watchdog.sh <id>` (a box file's ENGINES, lib.sh's engine_select; box-c's 2xTP4), one
# instance per engine, started by `relaunch.sh <id>`. With no id everything above holds byte for byte.
# With one, every path and probe is that engine's alone:
#   * log $WATCHDOG_LOG (/workspace/$BOX-<id>-watchdog.log), serve log $ENGINE_SERVE_LOG, hold
#     $WATCHDOG_HOLD (watchdog-<id>.hold), warm gate $ENGINE_WARMING, probes against $ENGINE_URL;
#   * liveness is `sglang serve ... --port $ENGINE_PORT`, never any engine on the box;
#   * kill_engine signals the engine's process group, named by $ENGINE_PIDFILE (relaunch.sh <id>
#     writes its own pid there before its exec under `setsid`, so that pid leads the group), TERM then
#     KILL -- falling back, when the pid file is stale, to the port-scoped pgrep's processes and their
#     descendants. Never a box-wide pkill: the other copy keeps serving. The GPU-free wait reads
#     `nvidia-smi --id=$ENGINE_GPUS` only;
#   * a relaunch is `relaunch.sh <id>` into $ENGINE_SERVE_LOG; the loop guard restores
#     sglang-args-<id>.txt from its own sglang-args-<id>.txt.pre-* (env likewise);
#   * Caddy is the FIRST engine's to check and relaunch (engine a on box-c): one door, one owner.
# Stop one: pkill -f 'watchdo[g].sh a( |$)'.
. "$(dirname "$0")/lib.sh"
# lib.sh runs under `set -euo pipefail`, which this script inherits (sourcing, not a subshell).
# A forever-loop daemon must never die from one transient nonzero (a curl timeout is exit 28, a
# missed pgrep is exit 1, both routine here) -- so drop every strict-mode flag once box.env has
# loaded; every failure path below is handled explicitly instead.
set +euo pipefail

WD_ID=""
case "${1:-}" in
  ""|--*) ;;
  *) WD_ID="$1"; shift; engine_select "$WD_ID" ;;   # dies on a box without ENGINES or an unknown id
esac

if [ -n "$WD_ID" ]; then
  # engine_select set every per-engine name; the WATCHDOG_TEST_* overrides exist for
  # tests/watchdog-engines.test.sh (WATCHDOG_LOG itself is engine_select's, exported to relaunch.sh).
  LOG="${WATCHDOG_TEST_LOG:-$WATCHDOG_LOG}"
  SERVE_LOG="${WATCHDOG_TEST_SERVE_LOG:-$ENGINE_SERVE_LOG}"
else
LOG="${WATCHDOG_LOG:-/workspace/$BOX-watchdog.log}"          # the overrides exist for tests/watchdog.test.sh
SERVE_LOG="${WATCHDOG_SERVE_LOG:-/workspace/sglang-serve.log}"
fi
BOOT_GRACE=${BOOT_GRACE:-${WATCHDOG_BOOT_GRACE:-2700}}   # box-file key WATCHDOG_BOOT_GRACE; signal 2 above
BOOT_STALL=${BOOT_STALL:-${WATCHDOG_BOOT_STALL:-900}}    # box-file key WATCHDOG_BOOT_STALL; signal 2 above
BOOT_MAX=${BOOT_MAX:-${WATCHDOG_BOOT_MAX:-5400}}         # box-file key WATCHDOG_BOOT_MAX; signal 2's ceiling
PROBE_TIMEOUT=${PROBE_TIMEOUT:-600}
PROBE_FAILS=${PROBE_FAILS:-3}
HEALTH_FAILS=${HEALTH_FAILS:-3}
HOLD="$TOOLS_REMOTE/watchdog.hold"
WARM_GATE="$TOOLS_REMOTE/engine.warming"   # relaunch.sh writes it, warm-short.py removes it (#379)
SFX=""                                     # the args/env file suffix: sglang-args$SFX.txt
ENGINE_PAT='sglang serv[e]'                # liveness: any engine on a one-engine box
CADDY_OWNER=1                              # this watchdog checks and relaunches Caddy
if [ -n "$WD_ID" ]; then
  HOLD="$WATCHDOG_HOLD"; WARM_GATE="$ENGINE_WARMING"; SFX="-$WD_ID"
  ENGINE_PAT="sglang serv[e].*--port $ENGINE_PORT( |\$)"
  read -r WD_FIRST _ <<< "$ENGINES"
  [ "${WD_FIRST%%:*}" = "$WD_ID" ] || CADDY_OWNER=0   # the first ENGINES entry owns Caddy
fi
WARM_GATE_TAKEOVER=${WARM_GATE_TAKEOVER:-120}   # s an owner=relaunch flag may wait for warm-short.py
# The relaunch prewarm (#1256, "Relaunch prewarm" above): box-file keys WATCHDOG_RELAUNCH_PREWARM
# (on|off) and WATCHDOG_RELAUNCH_PREWARM_TIMEOUT (s, restart.sh's PREWARM_TIMEOUT default); an
# exported RELAUNCH_PREWARM* wins. _FIND/_POLL/_LOG exist for tests/watchdog-relaunch-prewarm.test.sh.
RELAUNCH_PREWARM=${RELAUNCH_PREWARM:-${WATCHDOG_RELAUNCH_PREWARM:-on}}
RELAUNCH_PREWARM_TIMEOUT=${RELAUNCH_PREWARM_TIMEOUT:-${WATCHDOG_RELAUNCH_PREWARM_TIMEOUT:-1800}}
RELAUNCH_PREWARM_FIND=${RELAUNCH_PREWARM_FIND:-300}   # s to wait for this launch's gate flag to name its engine
RELAUNCH_PREWARM_POLL=${RELAUNCH_PREWARM_POLL:-5}
RELAUNCH_PREWARM_LOG=${RELAUNCH_PREWARM_LOG:-${LOG%/*}/prewarm-relaunch$SFX.log}
HOLD_MAX=${HOLD_MAX:-900}
RELAUNCH_LOOP_N=${RELAUNCH_LOOP_N:-3}
RELAUNCH_LOOP_WINDOW=${RELAUNCH_LOOP_WINDOW:-1800}
RELAUNCH_RING=""   # space-separated relaunch() timestamps, this process's lifetime only
CURL_UA="curl/8.7.1"
CADDY_LAUNCHER="$TOOLS_REMOTE/caddy-relaunch.sh"
CADDY_BOOT_GRACE=${CADDY_BOOT_GRACE:-180}  # no JIT/weight-load warmup analog, but a first DNS-01
                                            # issuance (measured ~15s) plus propagation slack needs
                                            # more than a bare "is it up" grace -- was 60, raised
                                            # after a relaunch inside that window cancelled an
                                            # in-flight ACME order
CADDY_FAILS=${CADDY_FAILS:-10}             # consecutive caddy_health misses (past CADDY_BOOT_GRACE)
                                            # before relaunch_caddy runs -- 10 x the 30s loop = 5
                                            # min tolerance for a cert renewal or a slow DNS-01
                                            # round to finish on its own before assuming Caddy is
                                            # actually stuck
# The bearer is read at start, and again whenever the probe answers 401 (key_reread below): a
# key rotation -- a stage writing a new VLLM_API_KEY into /etc/environment, the engine's next launch
# writing it into its key file -- otherwise left this daemon probing with the old key, every probe a
# 401, and PROBE_FAILS of them on an idle engine (quiet serve log, nothing running) a relaunch of a
# healthy one. Both files are overridable for tests/watchdog.test.sh only.
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-/etc/environment}"
API_KEY_FILE="${API_KEY_FILE:-/etc/sglang-api-key.yaml}"   # relaunch.sh's: the key the engine launched on
env_key() { grep -a '^VLLM_API_KEY=' "$ENVIRONMENT_FILE" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'; }
engine_key() { sed -n 's/^api-key: *"\(.*\)" *$/\1/p' "$API_KEY_FILE" 2>/dev/null | head -1; }
# env_key's read, spelled out with its default: tests/argv-bearer.test.sh lifts this one line.
KEY=$(grep -a '^VLLM_API_KEY=' "${ENVIRONMENT_FILE:-/etc/environment}" | head -1 | cut -d= -f2- | tr -d '"')

# The bearer never rides a curl argv (#3003): a `-H "Authorization: Bearer $KEY"` argument sits in
# /proc/<pid>/cmdline -- readable by every user on a box without hidepid -- for as long as the call
# runs, up to PROBE_TIMEOUT (600 s) for probe(). Each authed call below pipes this header into
# `curl -H @-` (header lines read from stdin, curl >= 7.55) instead; `printf` is a shell builtin, so
# the key is in no argv at all. An empty KEY sends no header, as an empty -H value never did either.
auth_hdr() { [ -z "$KEY" ] || printf 'Authorization: Bearer %s\n' "$KEY"; }

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

health() { curl -s -A "$CURL_UA" -m 30 -o /dev/null -w '%{http_code}' "$ENGINE_URL/health"; }

probe() {
  # "priority": 0 (SGLang's per-request priority) rides in the body unconditionally: a no-op
  # while --enable-priority-scheduling is off, but the moment R-PRIO ships this confirming
  # probe is answered ahead of the user queue instead of FCFS behind it.
  auth_hdr | curl -s -A "$CURL_UA" -m "$PROBE_TIMEOUT" -o /dev/null -w '%{http_code}' \
    -H @- -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"priority\":0}" \
    "$ENGINE_URL/v1/chat/completions"
}

# key_reread: after a 401, re-read the bearer -- $ENVIRONMENT_FILE's VLLM_API_KEY first, then the key
# the engine launched on ($API_KEY_FILE; a watchdog started after a rotation, ahead of the engine's
# next launch, holds the new key while the engine still answers the old) -- and switch KEY to the
# first one that differs from the key just refused. True when it switched; false when neither file
# holds another key, so the 401 is the engine's own answer and counts. Never logs a key, only where
# the new one came from.
key_reread() {
  local k
  k=$(env_key)
  if [ -n "$k" ] && [ "$k" != "$KEY" ]; then
    KEY=$k; log "API KEY: probe 401 -- $ENVIRONMENT_FILE holds a different VLLM_API_KEY (rotated), re-read; retrying once with it"; return 0
  fi
  k=$(engine_key)
  if [ -n "$k" ] && [ "$k" != "$KEY" ]; then
    KEY=$k; log "API KEY: probe 401 -- the engine's $API_KEY_FILE holds a different key (it launched before the rotation), re-read; retrying once with it"; return 0
  fi
  log "API KEY: probe 401 -- neither $ENVIRONMENT_FILE nor $API_KEY_FILE holds another key; counted"
  return 1
}

# probe_code: probe(), its code in PCODE -- in THIS shell, never a $(...) subshell, so a KEY that
# key_reread switched outlives the call. A 401 re-reads the key and, on a different one, retries once
# before the caller counts anything.
probe_code() {
  PCODE=$(probe)
  if [ "$PCODE" = 401 ] && key_reread; then PCODE=$(probe); fi
}

# progress(): is the engine still doing work despite a failing health/probe read? Echoes
# "R Q S" (running reqs, queue reqs, serve-log age in seconds) for the caller to log, and its
# own exit status is the verdict (0 = busy, not dead) -- true when $SERVE_LOG's mtime is under
# 60s old, or /metrics still adds sglang:num_running_reqs + sglang:num_queue_reqs > 0. A curl
# failure on /metrics (engine truly down) reads both as 0, same as an empty scrape.
serve_log_age() {
  local mtime
  if [ -f "$SERVE_LOG" ]; then
    mtime=$(stat -c %Y "$SERVE_LOG" 2>/dev/null)
    echo $(( $(date +%s) - ${mtime:-0} ))
  else
    echo 999999
  fi
}

progress() {
  local log_age metrics running queue active
  log_age=$(serve_log_age)
  metrics=$(auth_hdr | curl -s -A "$CURL_UA" -m 10 -H @- "$ENGINE_URL/metrics" 2>/dev/null)
  running=$(echo "$metrics" | awk '/^sglang:num_running_reqs/{print $NF; exit}')
  queue=$(echo "$metrics" | awk '/^sglang:num_queue_reqs/{print $NF; exit}')
  running=${running%.*}; queue=${queue%.*}
  active=$(( ${running:-0} + ${queue:-0} ))
  echo "$running $queue $log_age"
  [ "$log_age" -lt 60 ] || [ "$active" -gt 0 ]
}

# boot_progressing <serve-log age>: true while the engine has not answered /health 200 since its
# launch (BOOTED=0) and its serve log moved within BOOT_STALL s -- a boot still under way (signal 2).
boot_progressing() {
  [ "$BOOTED" = 0 ] && [ "$1" -lt "$BOOT_STALL" ]
}

# caddy_health echoes "<http code> <time_appconnect>": the second is 0.000000 unless the TLS
# handshake, certificate verification included, completed.
caddy_health() {
  auth_hdr | curl -s -A "$CURL_UA" -m 10 --resolve "$HOSTNAME:$DOOR_PORT:127.0.0.1" \
    -o /dev/null -w '%{http_code} %{time_appconnect}' -H @- \
    "https://$HOSTNAME:$DOOR_PORT/health"
}

# warm_gate_state: echoes `warming` while $WARM_GATE exists and its owner lives, else `-`. A stale flag
# (see the header) is removed here, with a log line, and reads as `-`.
warm_gate_state() {
  [ -e "$WARM_GATE" ] || { echo -; return; }
  local pid owner age
  pid=$(sed -n 's/^pid=//p' "$WARM_GATE" 2>/dev/null | head -1)
  owner=$(sed -n 's/^owner=//p' "$WARM_GATE" 2>/dev/null | head -1)
  age=$(( $(date +%s) - $(stat -c %Y "$WARM_GATE" 2>/dev/null || stat -f %m "$WARM_GATE" 2>/dev/null || date +%s) ))
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    log "WARM GATE STALE: $WARM_GATE names pid ${pid:-none} (owner ${owner:-?}), not running -- removed, door open"
    rm -f "$WARM_GATE"; echo -; return
  fi
  if [ "$owner" = relaunch ] && [ "$age" -gt "$WARM_GATE_TAKEOVER" ]; then
    log "WARM GATE STALE: $WARM_GATE still owner=relaunch after ${age}s (warm-short.py never took it over) -- removed, door open"
    rm -f "$WARM_GATE"; echo -; return
  fi
  echo warming
}

# caddy_verdict <code> <time_appconnect> <engine /health code, or - when not read> [<warm gate state>]:
# ok | upstream | warming | miss. Echoes "need-engine" when the answer depends on the engine's own
# /health and none was passed, so the loop reads that (a real generation) only for a gateway answer,
# never on a healthy pass. A 503 while the warm gate is `warming` is the gate's own answer.
caddy_verdict() {
  local code=$1 tls=$2 engine=$3 gate=${4:--}
  [ "$code" = 200 ] && { echo ok; return; }
  # The handshake never completed: refused, TLS/cert failure, or nothing listening -- Caddy is down.
  awk -v t="$tls" 'BEGIN { exit !(t + 0 > 0) }' || { echo miss; return; }
  [ "$code" = 503 ] && [ "$gate" = warming ] && { echo warming; return; }
  case "$code" in
    502|503|504|000) ;;
    *) echo miss; return ;;
  esac
  case "$engine" in
    -)   echo need-engine ;;
    200) echo miss ;;
    *)   echo upstream ;;
  esac
}

relaunch_caddy() {
  log "CADDY RELAUNCH: $1"
  bash "$CADDY_LAUNCHER" >> "$LOG" 2>&1
  CADDY_LAST_LAUNCH=$(date +%s)
  CFAILS=0
}

# engine_pgid (per-engine only): echoes the engine's process group from $ENGINE_PIDFILE while that
# group still exists and is this engine's -- its leader, if alive, is `sglang serve ... --port
# $ENGINE_PORT`; a dead leader's surviving members (sglang::scheduler_TP*, ...) keep the pgid and are
# reached through it. Fails on a missing, garbled or stale pid file.
engine_pgid() {
  local pid rows lead
  pid=$(head -1 "$ENGINE_PIDFILE" 2>/dev/null)
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  rows=$(ps -A -o pid=,pgid=,args= 2>/dev/null | awk -v g="$pid" '$2 == g')
  [ -n "$rows" ] || return 1
  lead=$(printf '%s\n' "$rows" | awk -v g="$pid" '$1 == g')
  if [ -n "$lead" ]; then
    printf '%s' "$lead" | grep -qE -- "$ENGINE_PAT" || return 1
  else
    printf '%s\n' "$rows" | grep -q 'sglang' || return 1
  fi
  echo "$pid"
}

# proc_tree <pid>: the pid and every descendant of it.
proc_tree() {
  ps -A -o pid=,ppid= 2>/dev/null | awk -v r="$1" '
    { kids[$2] = kids[$2] " " $1 }
    END { q = r; out = r; while (q != "") { split(q, a, " "); q = ""
            for (i in a) { n = split(kids[a[i]], k, " "); for (j = 1; j <= n; j++) { out = out " " k[j]; q = q " " k[j] } } }
          print out }'
}

# engine_signal <TERM|KILL> (per-engine only): this engine's processes and nothing else's.
engine_signal() {
  local sig=$1 pg p
  if pg=$(engine_pgid); then
    kill -"$sig" -- -"$pg" 2>/dev/null
    return
  fi
  for p in $(pgrep -f "$ENGINE_PAT" 2>/dev/null); do
    if [ "$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')" = "$p" ]; then
      kill -"$sig" -- -"$p" 2>/dev/null
    else
      kill -"$sig" $(proc_tree "$p") 2>/dev/null
    fi
  done
}

kill_engine() {
  if [ -n "$WD_ID" ]; then
    local pg
    pg=$(engine_pgid) && log "kill_engine: engine $WD_ID process group $pg ($ENGINE_PIDFILE)" \
      || log "kill_engine: $ENGINE_PIDFILE stale or missing -- the port-$ENGINE_PORT engine's processes"
    engine_signal TERM
  else
  pkill -TERM -f 'sglang serv[e]' 2>/dev/null
  pkill -TERM -f 'sglang:[:]' 2>/dev/null          # sglang::scheduler_TP*, sglang::detokenizer, ...
  fi
  sleep 8
  # A hold taken during that wait (relaunch()'s comment): its restart runs its own stop, and its own
  # relaunch can land inside these 8 s -- a KILL now would take down THAT engine. Leave the rest to it.
  if hold_fresh; then
    log "kill_engine: $HOLD appeared during the TERM wait -- leaving the KILL and the GPU wait to the restart that holds it"
    return
  fi
  if [ -n "$WD_ID" ]; then
    engine_signal KILL
  else
  pkill -KILL -f 'sglang serv[e]' 2>/dev/null
  pkill -KILL -f 'sglang:[:]' 2>/dev/null
  fi
  for _ in $(seq 1 60); do
    if [ -n "$WD_ID" ]; then
      used=$(nvidia-smi --id="$ENGINE_GPUS" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    else
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    fi
    [ "${used:-0}" -lt 8000 ] && break
    sleep 2
  done
  log "GPU memory after kill: ${used:-?} MiB"
}

# loop_guard_decision <count_in_window> <backup_or_empty>: pure function, prints one of
# restore|hold|relaunch. <count_in_window> is how many relaunch() calls (including the one
# about to happen) fall inside RELAUNCH_LOOP_WINDOW; <backup_or_empty> is the newest
# sglang-args.txt.pre-* path on the box, or "" if none exists.
loop_guard_decision() {
  local count=$1 backup=$2
  if [ "$count" -lt "$RELAUNCH_LOOP_N" ]; then
    echo relaunch
  elif [ -n "$backup" ]; then
    echo restore
  else
    echo hold
  fi
}

# hold_fresh: true while a deliberate restart holds the watchdog -- $HOLD exists and is no older than
# HOLD_MAX (the loop's own staleness rule; a stale one is the loop's to remove).
hold_fresh() {
  local m
  [ -e "$HOLD" ] || return 1
  m=$(stat -c %Y "$HOLD" 2>/dev/null) || return 0   # there, but unreadable: treat it as held
  [ $(( $(date +%s) - m )) -le "$HOLD_MAX" ]
}

# relaunch() re-reads the hold twice: on entry, and again after kill_engine, right before it
# launches. The loop reads it only at the top of a pass, and a relaunch spends 10+ s between that
# read and its launch (kill_engine's sleep 8 and GPU wait). On gpu1021 (2026-10-21 01:26-01:27Z) a
# restart.sh FATAL released the hold, this watchdog found no engine and began a relaunch, the
# operator's retry took the hold back and ran its own stop and relaunch inside that window, and
# both launches booted -- two engines, both dead on `Loaded weights leave no GPU memory for the KV
# cache`. A hold that appears during the relaunch means restart.sh/cutover.sh owns the engine now:
# this one stands down (`RELAUNCH ABORTED`) and launches nothing.
relaunch() {
  log "RELAUNCH: $1"
  local now_ts kept ts count backup decision window_start elapsed tag env_backup
  if hold_fresh; then
    log "RELAUNCH ABORTED: $HOLD appeared before this relaunch began -- a deliberate restart owns the engine; launching nothing"
    return
  fi
  now_ts=$(date +%s)
  kept=""
  for ts in $RELAUNCH_RING; do
    [ $((now_ts - ts)) -le "$RELAUNCH_LOOP_WINDOW" ] && kept="$kept $ts"
  done
  RELAUNCH_RING="${kept# } $now_ts"
  RELAUNCH_RING="${RELAUNCH_RING# }"
  count=$(set -- $RELAUNCH_RING; echo $#)
  backup=$(ls -t "$TOOLS_REMOTE"/sglang-args$SFX.txt.pre-* 2>/dev/null | head -1)
  decision=$(loop_guard_decision "$count" "$backup")

  if [ "$decision" = hold ]; then
    if [ -n "$WD_ID" ]; then
      log "RELAUNCH LOOP: no backup -- holding; run restart.sh --engine $WD_ID by hand"
    else
    log "RELAUNCH LOOP: no backup -- holding; run cutover.sh by hand"
    fi
    touch "$HOLD"
    RELAUNCH_RING=""
    return
  fi

  if [ "$decision" = restore ]; then
    window_start=$(set -- $RELAUNCH_RING; echo "$1")
    elapsed=$((now_ts - window_start))
    log "RELAUNCH LOOP: $count relaunches in ${elapsed}s -- restoring $backup"
    cp -f "$TOOLS_REMOTE/sglang-args$SFX.txt" "$TOOLS_REMOTE/sglang-args$SFX.txt.looping-$now_ts" 2>/dev/null
    cp -f "$backup" "$TOOLS_REMOTE/sglang-args$SFX.txt"
    tag=${backup##*sglang-args$SFX.txt.pre-}
    env_backup="$TOOLS_REMOTE/sglang-env$SFX.txt.pre-$tag"
    [ -f "$env_backup" ] && cp -f "$env_backup" "$TOOLS_REMOTE/sglang-env$SFX.txt"
    RELAUNCH_RING=""
  fi

  prewarm_stop
  kill_engine
  if hold_fresh; then
    log "RELAUNCH ABORTED: $HOLD appeared while this relaunch was stopping the engine -- a deliberate restart (restart.sh / cutover.sh) owns the engine now, and a second launch would boot two engines on one box; launching nothing"
    return
  fi
  # The serve log carries the key (the engine's `server_args=` line): created 0600 when absent (#3003),
  # in a subshell so the umask never reaches relaunch.sh or the engine; `>>` keeps an existing mode.
  ( umask 077; : >> "$SERVE_LOG" )
  echo "===== WATCHDOG RELAUNCH $(date -u): $1 =====" >> "$SERVE_LOG"
  nohup setsid bash "$TOOLS_REMOTE/relaunch.sh" ${WD_ID:+"$WD_ID"} >> "$SERVE_LOG" 2>&1 < /dev/null &
  LAST_LAUNCH=$(date +%s)
  FAILS=0
  HFAILS=0
  BOOTED=0
  prewarm_start
}

# ---- relaunch prewarm (#1256): every watchdog relaunch sweeps prewarm-box-d.py before the door opens ----
# A relaunch's own warm-up (warm-short.py: one ~60k fill, three extends) leaves most of the
# tail-chunk sizes a long cold prefill reaches uncompiled IN THIS PROCESS; the first one after the
# boot pays them on live traffic (#1255: a 110k fill's last 3,782-token chunk at 3,394 tok/s, ~23k
# warm, and its needle lost). prewarm_start runs relaunch_prewarm in a subshell of this watchdog
# (relaunch.sh's watchdog count skips a match whose parent is one: wd_pids):
#   1. find the engine this launch started: relaunch.sh's flag (owner=relaunch, pid=<engine>) or
#      warm-short.py's takeover of it (owner=warm-short, engine=<engine>), naming a live pid, within
#      RELAUNCH_PREWARM_FIND s; no flag (warm-short.py not staged: relaunch.sh left the door open)
#      -> the engine on ENGINE_PAT, and the gate is closed here;
#   2. take the gate: pid=<this subshell> owner=prewarm engine=<engine> phase=boot. warm-short.py
#      stands down (it opens the gate only while the flag names it, and quits its /health wait when
#      it does not); a takeover of warm-short.py's that raced ours is taken back while it names the
#      same engine. Anything else -- a newer launch's flag, the flag removed -- and this stands down;
#   3. wait for the engine's /health 200 (stand down if it dies, or past BOOT_MAX: the loop's boot
#      rules relaunch it, and a relaunch stops this subshell first: prewarm_stop);
#   4. phase=sweep: prewarm-box-d.py --base-url $ENGINE_URL (the same sweep restart.sh runs: every
#      tail-chunk size to the args' chunk) under `timeout RELAUNCH_PREWARM_TIMEOUT`, log
#      RELAUNCH_PREWARM_LOG. While phase is sweep|fallback the loop counts no /health or probe miss
#      (prewarm_sweeping): a compile fails /health for 17-34 s, and restart.sh holds its hold for the
#      same reason. The door answers 503 throughout -- the flag is the gate tls.sh's @warming reads;
#   5. exit 0 -> `RELAUNCH PREWARM DONE`; anything else -> `RELAUNCH PREWARM FAILED ... FALLING BACK`,
#      phase=fallback and warm-short.py's one-shot warm (no --gate), bounded WARM_TIMEOUT + 60 s;
#   6. open the gate (remove the flag) while it still names this subshell.
gate_field() { sed -n "s/^$1=//p" "$WARM_GATE" 2>/dev/null | head -1; }

prewarm_write_flag() {   # <me> <engine pid> <phase>: the gate, owned by the relaunch prewarm
  printf 'pid=%s\nowner=prewarm\nengine=%s\nphase=%s\nstarted=%s\n' "$1" "$2" "$3" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$WARM_GATE.tmp.$1" && mv -f "$WARM_GATE.tmp.$1" "$WARM_GATE"
}

prewarm_keep_gate() {   # <me> <engine pid> <phase>: 0 while the gate is ours (taken back from warm-short.py on the same engine)
  local owner eng
  [ -e "$WARM_GATE" ] || return 1
  owner=$(gate_field owner)
  [ "$owner" = prewarm ] && [ "$(gate_field pid)" = "$1" ] && return 0
  eng=$(gate_field engine)
  if [ "$owner" = warm-short ] && [ "$eng" = "$2" ]; then
    prewarm_write_flag "$1" "$2" "$3" 2>/dev/null && { log "RELAUNCH PREWARM: warm-short.py's takeover raced ours on engine pid $2 -- gate taken back"; return 0; }
  fi
  return 1
}

prewarm_sweeping() {   # 0 while a live relaunch prewarm's sweep or fallback warm holds the gate
  local pid
  [ "$(gate_field owner)" = prewarm ] || return 1
  case "$(gate_field phase)" in sweep|fallback) ;; *) return 1 ;; esac
  pid=$(gate_field pid)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# prewarm_sweep_health: the loop's turn while prewarm_sweeping (#1271), in place of its /health and
# probe rules. One /health read: a 200 clears the count (PWHFAILS); a miss adds one, logged. A
# compile fails /health for 17-34 s at a time, and one read is a 30 s curl plus the loop's 30 s
# sleep, so HEALTH_FAILS misses IN A ROW are minutes of a dead /health: a wedged-but-alive engine,
# not a compile. That ends the sweep and takes the normal relaunch path -- relaunch() stops the sweep
# (prewarm_stop) before kill_engine -- instead of waiting out RELAUNCH_PREWARM_TIMEOUT (1800 s) and
# the fallback warm (up to 240 s). Returns 0 when it relaunched, else 1.
prewarm_sweep_health() {
  local hc ph
  hc=$(health)
  if [ "$hc" = 200 ]; then PWHFAILS=0; return 1; fi
  PWHFAILS=$((PWHFAILS + 1))
  ph=$(gate_field phase)
  log "RELAUNCH PREWARM sweeping (phase $ph): health $hc (miss $PWHFAILS/$HEALTH_FAILS in a row; a compile fails it for 17-34 s, $HEALTH_FAILS in a row is a wedged engine)"
  [ "$PWHFAILS" -ge "$HEALTH_FAILS" ] || return 1
  PWHFAILS=0
  log "RELAUNCH PREWARM ENDED: /health missed ${HEALTH_FAILS}x in a row mid-sweep (phase $ph) -- the engine is wedged, not compiling; relaunching it rather than waiting out the sweep"
  relaunch "health missed ${HEALTH_FAILS}x in a row during the relaunch prewarm (phase $ph) -- wedged mid-sweep"
  return 0
}

prewarm_run() {   # <me> <engine pid> <phase> <log> <cmd...>: PW_RC its exit; 1 = stood down mid-run
  local me=$1 epid=$2 phase=$3 out=$4
  shift 4
  "$@" >> "$out" 2>&1 < /dev/null &
  PW_KID=$!
  while kill -0 "$PW_KID" 2>/dev/null; do
    if ! kill -0 "$epid" 2>/dev/null || ! prewarm_keep_gate "$me" "$epid" "$phase"; then
      kill "$PW_KID" 2>/dev/null; wait "$PW_KID" 2>/dev/null; PW_KID=""
      return 1
    fi
    sleep "$RELAUNCH_PREWARM_POLL"
  done
  wait "$PW_KID"; PW_RC=$?; PW_KID=""
  return 0
}

relaunch_prewarm() {
  local me=$BASHPID t0 t1 epid="" owner="" eng waited=0 py what
  PW_KID=""; PW_RC=""
  trap '[ -n "$PW_KID" ] && kill "$PW_KID" 2>/dev/null; exit 143' TERM INT HUP
  t0=$(date +%s)
  while [ "$waited" -lt "$RELAUNCH_PREWARM_FIND" ]; do
    owner=$(gate_field owner)
    case "$owner" in relaunch) eng=$(gate_field pid) ;; warm-short) eng=$(gate_field engine) ;; *) eng="" ;; esac
    if [ -n "$eng" ] && kill -0 "$eng" 2>/dev/null; then epid=$eng; break; fi
    sleep 2; waited=$((waited + 2))
  done
  if [ -z "$epid" ]; then
    owner=""
    epid=$(pgrep -o -f "$ENGINE_PAT" 2>/dev/null)
    if [ -z "$epid" ]; then
      log "RELAUNCH PREWARM NOT RUN: no gate flag naming a live engine and no '$ENGINE_PAT' process within ${RELAUNCH_PREWARM_FIND}s of the launch -- nothing to prewarm (the loop's own rules see the launch)"
      exit 1
    fi
    log "RELAUNCH PREWARM: no gate flag naming this launch's engine within ${RELAUNCH_PREWARM_FIND}s (warm-short.py not staged?) -- closing $WARM_GATE here for engine pid $epid"
  fi
  if ! prewarm_write_flag "$me" "$epid" boot 2>/dev/null; then
    log "RELAUNCH PREWARM NOT RUN: could not write $WARM_GATE -- this boot keeps the short warm only (gate owner: ${owner:-none}); the first long cold prefill pays its compiles on live traffic"
    exit 1
  fi
  log "RELAUNCH PREWARM: warm gate $WARM_GATE taken from ${owner:-nobody (it was open)} for engine pid $epid -- warm-short.py stands down; prewarm-box-d.py runs once /health answers 200, and the door answers 503 until it ends"
  while :; do
    if ! kill -0 "$epid" 2>/dev/null; then
      log "RELAUNCH PREWARM: engine pid $epid gone before its /health 200 -- standing down"; exit 0
    fi
    if ! prewarm_keep_gate "$me" "$epid" boot; then
      log "RELAUNCH PREWARM: $WARM_GATE no longer ours ($( { tr '\n' ' ' < "$WARM_GATE"; } 2>/dev/null || echo removed)) -- standing down"; exit 0
    fi
    [ "$(health)" = 200 ] && break
    if [ $(( $(date +%s) - t0 )) -ge "$BOOT_MAX" ]; then
      log "RELAUNCH PREWARM: no /health 200 from engine pid $epid in ${BOOT_MAX}s -- standing down (the loop's boot ceiling owns it)"; exit 0
    fi
    sleep "$RELAUNCH_PREWARM_POLL"
  done
  py="${VENV:-/nonexistent}/bin/python3"; [ -x "$py" ] || py=python3
  prewarm_write_flag "$me" "$epid" sweep 2>/dev/null
  log "RELAUNCH PREWARM: engine pid $epid up ($(( $(date +%s) - t0 ))s after launch): prewarm-box-d.py --base-url $ENGINE_URL, bounded ${RELAUNCH_PREWARM_TIMEOUT}s (log $RELAUNCH_PREWARM_LOG)"
  : > "$RELAUNCH_PREWARM_LOG"
  t1=$(date +%s)
  if ! prewarm_run "$me" "$epid" sweep "$RELAUNCH_PREWARM_LOG" \
         env BOX="$BOX" timeout "$RELAUNCH_PREWARM_TIMEOUT" "$py" "$TOOLS_REMOTE/prewarm-box-d.py" --base-url "$ENGINE_URL"; then
    log "RELAUNCH PREWARM ABORTED after $(( $(date +%s) - t1 ))s: engine pid $epid gone or $WARM_GATE no longer ours mid-sweep -- standing down"; exit 0
  fi
  if [ "$PW_RC" -eq 0 ]; then
    log "RELAUNCH PREWARM DONE in $(( $(date +%s) - t1 ))s: $(grep -a '^PREWARM DONE' "$RELAUNCH_PREWARM_LOG" 2>/dev/null | tail -1)"
  else
    what="exit $PW_RC"; [ "$PW_RC" -eq 124 ] && what="timed out at ${RELAUNCH_PREWARM_TIMEOUT}s"
    log "RELAUNCH PREWARM FAILED: prewarm-box-d.py $what after $(( $(date +%s) - t1 ))s (log $RELAUNCH_PREWARM_LOG) -- FALLING BACK to warm-short.py's short warm; the compiles the sweep would have paid fall on the first long cold prefills: sweep it under a router cut (restart.sh${WD_ID:+ --engine $WD_ID} with PREWARM=force)"
    prewarm_write_flag "$me" "$epid" fallback 2>/dev/null
    if ! prewarm_run "$me" "$epid" fallback "$RELAUNCH_PREWARM_LOG" \
           timeout $(( ${WARM_TIMEOUT:-180} + 60 )) python3 "$TOOLS_REMOTE/warm-short.py" --base-url "$ENGINE_URL" \
           --environment-file "${ENVIRONMENT_FILE:-/etc/environment}" --timeout "${WARM_TIMEOUT:-180}"; then
      log "RELAUNCH PREWARM ABORTED: engine pid $epid gone or $WARM_GATE no longer ours mid-fallback -- standing down"; exit 0
    fi
    log "RELAUNCH PREWARM FALLBACK: warm-short.py exit $PW_RC (log $RELAUNCH_PREWARM_LOG)"
  fi
  if prewarm_keep_gate "$me" "$epid" open; then
    rm -f "$WARM_GATE"
    log "RELAUNCH PREWARM: warm gate OPEN ($WARM_GATE removed) $(( $(date +%s) - t0 ))s after the launch"
  else
    log "RELAUNCH PREWARM: $WARM_GATE no longer ours at the end -- left alone"
  fi
  exit 0
}

prewarm_boot_unswept() {   # 0 when the gate is still relaunch.sh's or warm-short.py's, on a live engine: a boot whose short warm has not lifted it
  local eng
  case "$(gate_field owner)" in
    relaunch) eng=$(gate_field pid) ;;
    warm-short) eng=$(gate_field engine) ;;
    *) return 1 ;;
  esac
  [ -n "$eng" ] && kill -0 "$eng" 2>/dev/null
}

prewarm_start() {   # after relaunch()'s launch: the sweep, in a subshell, unless off or unstaged
  if [ "$RELAUNCH_PREWARM" = off ]; then
    log "RELAUNCH PREWARM off (WATCHDOG_RELAUNCH_PREWARM=off): this boot gets warm-short.py's short warm only"; return
  fi
  if [ ! -f "$TOOLS_REMOTE/prewarm-box-d.py" ]; then
    log "RELAUNCH PREWARM NOT RUN: $TOOLS_REMOTE/prewarm-box-d.py not staged -- this boot gets warm-short.py's short warm only; the first long cold prefill pays its compiles on live traffic"; return
  fi
  ( relaunch_prewarm ) >> "$LOG" 2>&1 < /dev/null &
  PW_PID=$!
}

prewarm_stop() {   # before relaunch()'s kill_engine: the last launch's sweep never outlives its engine
  [ -n "$PW_PID" ] || return 0
  if kill -0 "$PW_PID" 2>/dev/null; then
    kill "$PW_PID" 2>/dev/null; wait "$PW_PID" 2>/dev/null
    log "RELAUNCH PREWARM stopped (subshell $PW_PID): this relaunch replaces its engine"
  fi
  PW_PID=""
}
# ---- end relaunch prewarm ----

# --loop-guard-test: print loop_guard_decision's three fixed outcomes and exit -- never the
# loop, never a pkill/relaunch, no state.
if [ "${1:-}" = --caddy-verdict-test ]; then
  for c in "200 0.120000 -" "502 0.004000 000" "502 0.004000 200" "503 0.004000 503" "000 0.004000 000" \
           "000 0.000000 000" "502 0.000000 000" "404 0.004000 000" "502 0.004000 -"; do
    set -- $c
    echo "($1, $2, $3) -> $(caddy_verdict "$1" "$2" "$3")"
  done
  for c in "503 0.004000 - warming" "503 0.004000 200 warming" "502 0.004000 200 warming" \
           "503 0.000000 - warming" "503 0.004000 200 -"; do
    set -- $c
    echo "($1, $2, $3, $4) -> $(caddy_verdict "$1" "$2" "$3" "$4")"
  done
  exit 0
fi

if [ "${1:-}" = --loop-guard-test ]; then
  echo "(2, backup) -> $(loop_guard_decision 2 backup)"
  echo "(3, backup) -> $(loop_guard_decision 3 backup)"
  echo "(3, ) -> $(loop_guard_decision 3 "")"
  exit 0
fi

# --self-test: a ONE-SHOT read-only health+probe check, never the loop, never a pkill/relaunch.
if [ "${1:-}" = --self-test ]; then
  [ -n "$KEY" ] || { echo "no VLLM_API_KEY in $ENVIRONMENT_FILE"; exit 3; }
  hc=$(health); probe_code; pc=$PCODE
  echo "$BOX: health=$hc probe=$pc"
  [ "$hc" = 200 ] && [ "$pc" = 200 ] && { echo "SELF-TEST PASS"; exit 0; }
  echo "SELF-TEST FAIL"; exit 1
fi

LAST_LAUNCH=$(date +%s)
FAILS=0
HFAILS=0
CADDY_LAST_LAUNCH=$(date +%s)
CFAILS=0
CUPSTREAM=0   # 1 while Caddy answers but the engine behind it is not ready (logged once per spell)
CWARM=0       # 1 while the door answers the warm gate's 503 (logged once per spell)
BOOTED=0      # 1 once the engine answered /health 200 since LAST_LAUNCH (signal 2's boot rule)
PW_PID=""     # the running relaunch prewarm's subshell (prewarm_start), if any
PWSWEEP=0     # 1 while a relaunch prewarm's sweep holds the gate (logged once per spell)
PWHFAILS=0    # /health misses in a row while it sweeps (prewarm_sweep_health)
sleep 20   # relaunch.sh starts this watchdog just before its own exec into `sglang serve`
log "watchdog start pid=$$ box=$BOX${WD_ID:+ id=$WD_ID gpus=$ENGINE_GPUS caddy_owner=$CADDY_OWNER} engine=$ENGINE_URL boot_grace=${BOOT_GRACE}s boot_stall=${BOOT_STALL}s boot_max=${BOOT_MAX}s probe_timeout=${PROBE_TIMEOUT}s fails=${PROBE_FAILS} hold_max=${HOLD_MAX}s caddy_boot_grace=${CADDY_BOOT_GRACE}s caddy_fails=${CADDY_FAILS} relaunch_loop_n=${RELAUNCH_LOOP_N} relaunch_loop_window=${RELAUNCH_LOOP_WINDOW}s"

# A boot this watchdog did not launch -- onstart's relaunch.sh after a container restart, a hand
# relaunch.sh -- starts this watchdog too (relaunch.sh's guarded start). While its gate is still
# relaunch.sh's or warm-short.py's and no deliberate restart holds the engine (restart.sh and
# cutover.sh run their own sweep), it gets the same relaunch prewarm (#1256).
if ! hold_fresh && prewarm_boot_unswept; then
  log "RELAUNCH PREWARM: adopting the boot behind $WARM_GATE (owner $(gate_field owner), launched before this watchdog) -- the sweep a relaunch runs"
  prewarm_start
fi

while true; do
  now=$(date +%s)
  if [ -e "$HOLD" ]; then
    if [ $((now - $(stat -c %Y "$HOLD"))) -gt "$HOLD_MAX" ]; then
      log "stale hold file (> ${HOLD_MAX}s), removing"; rm -f "$HOLD"
    else
      LAST_LAUNCH=$now; BOOTED=0; sleep 15; continue   # a deliberate restart boots a fresh engine
    fi
  fi

  if [ "$CADDY_OWNER" = 1 ] && [ -x "$CADDY_LAUNCHER" ]; then
    if ! pgrep -f "$TOOLS_REMOTE/cadd[y]-relaunch.sh|cadd[y] run" > /dev/null; then
      relaunch_caddy "process missing"
      sleep 60; continue
    fi
    if [ $((now - CADDY_LAST_LAUNCH)) -gt "$CADDY_BOOT_GRACE" ]; then
      read -r cc ctls <<< "$(caddy_health)"
      wg=$(warm_gate_state)
      cv=$(caddy_verdict "$cc" "${ctls:-0}" - "$wg")
      if [ "$cv" = need-engine ]; then
        ec=$(health)
        cv=$(caddy_verdict "$cc" "${ctls:-0}" "$ec" "$wg")
      fi
      [ "$cv" = warming ] || CWARM=0
      if [ "$cv" = warming ]; then
        [ "$CWARM" = 1 ] || log "CADDY UP, engine warming (door -> $cc behind the warm gate $WARM_GATE) -- not a Caddy miss; the door opens when warm-short.py, or a relaunch's prewarm sweep, finishes"
        CWARM=1
        CUPSTREAM=0
        CFAILS=0
      elif [ "$cv" = upstream ]; then
        [ "$CUPSTREAM" = 1 ] || log "CADDY UP, upstream not ready (door -> $cc, engine /health -> $ec) -- not a Caddy miss; Caddy is left alone while the engine boots or recovers"
        CUPSTREAM=1
        CFAILS=0
      elif [ "$cv" = miss ]; then
        CUPSTREAM=0
        CFAILS=$((CFAILS + 1))
        log "CADDY MISS $CFAILS/$CADDY_FAILS -> $cc"
        if [ "$CFAILS" -ge "$CADDY_FAILS" ]; then
          relaunch_caddy "/health (through hostname, real cert) -> $cc after $CADDY_FAILS misses"
          sleep 60; continue
        fi
      else
        CUPSTREAM=0
        CFAILS=0
      fi
    fi
  fi

  if ! pgrep -f "$ENGINE_PAT" > /dev/null; then
    relaunch "no sglang serve process"
    sleep 60; continue
  fi

  if prewarm_sweeping; then
    [ "$PWSWEEP" = 1 ] || { log "RELAUNCH PREWARM sweeping (phase $(gate_field phase), $WARM_GATE closed): /health and probe misses are its compiles, not a dead engine -- not counted the normal way until the sweep ends; ${HEALTH_FAILS} /health misses in a row end it (a wedged engine)"; PWHFAILS=0; }
    PWSWEEP=1; BOOTED=1; HFAILS=0; FAILS=0
    if prewarm_sweep_health; then sleep 60; continue; fi
    sleep 30; continue
  fi
  PWSWEEP=0

  hcode=$(health)
  if [ "$hcode" != 200 ]; then
    if [ $((now - LAST_LAUNCH)) -gt "$BOOT_GRACE" ]; then
      elapsed=$((now - LAST_LAUNCH))
      HFAILS=$((HFAILS + 1))
      log "health $hcode (miss $HFAILS/$HEALTH_FAILS, ${elapsed}s since launch)"
      if [ "$HFAILS" -ge "$HEALTH_FAILS" ]; then
        pstart=$(date +%s)
        probe_code; pcode=$PCODE
        pelapsed=$(($(date +%s) - pstart))
        if [ "$pcode" = 200 ]; then
          log "health missed ${HEALTH_FAILS}x but probe answered in ${pelapsed}s -- busy, not dead"
          HFAILS=0
          sleep 30
        else
          bage=$(serve_log_age)
          if [ "$BOOTED" = 0 ]; then
            pg=""; pg_rc=1
          else
            pg=$(progress); pg_rc=$?
          fi
          if [ "$BOOTED" = 0 ] && [ "$elapsed" -ge "$BOOT_MAX" ]; then
            relaunch "boot ceiling: no /health 200 in ${elapsed}s since launch (>= boot_max ${BOOT_MAX}s), serve log age ${bage}s -- a boot this long is looping, not loading"
            sleep 60
          elif boot_progressing "$bage"; then
            log "booting, not dead (${elapsed}s since launch, no /health 200 yet, serve log age ${bage}s < boot_stall ${BOOT_STALL}s)"
            HFAILS=0; FAILS=0
            sleep 30
          elif [ "$pg_rc" -eq 0 ]; then
            read -r prun pqueue plogage <<< "$pg"
            log "busy, not dead (running $prun queue $pqueue, log age ${plogage}s)"
            HFAILS=0; FAILS=0
            sleep 30
          elif [ "$BOOTED" = 0 ]; then
            relaunch "boot stalled: no /health 200 in ${elapsed}s since launch and the serve log silent ${bage}s (>= boot_stall ${BOOT_STALL}s)"
            sleep 60
          else
            relaunch "health missed ${HEALTH_FAILS}x and 1-token probe ${pcode} after ${PROBE_TIMEOUT}s"
            sleep 60
          fi
        fi
      else
        sleep 30
      fi
    else
      sleep 30
    fi
    continue
  fi
  HFAILS=0
  [ "$BOOTED" = 1 ] || { log "engine up: first /health 200, $((now - LAST_LAUNCH))s since launch (or since this watchdog started)"; BOOTED=1; }

  probe_code; code=$PCODE
  if [ "$code" = 200 ]; then
    FAILS=0
  else
    FAILS=$((FAILS + 1))
    log "probe $code (fail $FAILS/$PROBE_FAILS)"
    if [ "$FAILS" -ge "$PROBE_FAILS" ]; then
      pg=$(progress); pg_rc=$?
      if [ "$pg_rc" -eq 0 ]; then
        read -r prun pqueue plogage <<< "$pg"
        log "busy, not dead (running $prun queue $pqueue, log age ${plogage}s)"
        FAILS=0; HFAILS=0
        sleep 30; continue
      fi
      relaunch "1-token probe failed $FAILS times in a row (last $code)"
      sleep 60; continue
    fi
  fi
  sleep 30
done
