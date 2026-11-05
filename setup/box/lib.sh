#!/bin/bash
# Shared preamble for every script in this tool set. Source it first:
#   . "$(dirname "$0")/lib.sh"
# It requires BOX (the box name, e.g. box-a), sources
# boxes/$BOX.env, and defines the handful of helpers every script uses.
# A script never hardcodes an ip, a port, a hostname or a box name — those
# live in the box file, and a fresh box is a new file under boxes/, nothing else.
# Every box_ssh/box_scp is additionally fenced to the box this file names (the target fence
# below): a box file that has gone stale stops the script at its first ssh instead of sending it
# at whatever box the file happens to point at — the box-g incident, 2026-09-13.
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${BOX:?set BOX=<box name> (a file under $TOOLS_DIR/boxes/)}"
BOX_ENV="${BOX_ENV:-$TOOLS_DIR/boxes/$BOX.env}"
[ -r "$BOX_ENV" ] || { echo "lib.sh: no box file $BOX_ENV" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
. "$BOX_ENV"
set +a

# Derived, never set in the box file.
export ENGINE_URL="${ENGINE_URL:-http://127.0.0.1:${ENGINE_PORT}}"
export DOOR_URL="${DOOR_URL:-https://${HOSTNAME}:${APP_PORT_EXT}}"
SSH_KEY_PATH="${SSH_KEY/#\~/$HOME}"
ENV_FILE_PATH="${ENV_FILE/#\~/$HOME}"

note() { printf '[%s %s] %s\n' "$BOX" "$(date -u +%H:%M:%S)" "$*" >&2; }
die()  { note "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------- engines: more than one engine per box
# A box file MAY set ENGINES -- a space-separated list of id:gpus:engine_port:kv_base entries, e.g.
# box-c's 2xTP4 layout `ENGINES="a:0,1,2,3:8002:5557 b:4,5,6,7:8012:5567"`. With it unset (box-a, every
# other box) nothing below changes a single value the lines above derived: no script calls
# engine_select with an id there, and engine_select with none is the legacy one-engine layout.
#   engines_list          print the ids, one per line (nothing on a box without ENGINES)
#   engine_select [<id>]  export the per-engine names (no id: the legacy names, today's paths)
# Per engine <id> (SFX=-<id>): ENGINE_PORT/ENGINE_URL on its port, ENGINE_NCCL_PORT = port + 100,
# KV_PORT = kv_base (sglang's --kv-events-config endpoint), KV_REPLAY_PORT = +1, KV_LOAD_PORT = +2
# (the args files' --load-publish-endpoint tcp://*:5559, rewritten per engine; it was `auto`), ENGINE_HOST = <first label><id>.<rest of HOSTNAME>
# (box-ca.<DOMAIN>), ENGINE_DOOR_URL on it, and every per-engine file with $SFX in its
# name. DOOR_URL keeps naming the box's legacy hostname, which fronts the copies together.
# Refused (die, naming the entry): a malformed entry, a repeated id, a GPU in two entries, any two
# of the five ports per engine equal (across the whole list), an unknown id, and an id on a box
# without ENGINES.
#   engine_kv_ranks <n>   an A/B arm's per-DP-rank KV layout for the selected engine (below)
# A DP-attention arm (--attn-dp-size N, or --dp-size N: N = dp x attn-dp ranks) publishes per rank:
# the fork offsets the events and replay endpoints by the rank (rank r binds port+r) and gives the load
# feed the range base..base+N-1, and refuses a load range overlapping a KV range -- so kv_base..+2
# cannot hold it (KV rank 1 would bind replay rank 0's port). Such an engine takes its ports from its
# DP area instead, ENGINES_DP_BASE (5600) onward, 3 x <its GPU count> ports per engine in ENGINES
# order (box-c: a 5600-5611, b 5612-5623), off every engine's record ports and off each other, so
# either copy -- or both -- may boot an arm while the other serves its record layout. In it
# `engine_kv_ranks N` (relaunch.sh, from the args line) sets KV_PORT = the area's base (events,
# ranks [b, b+N)), KV_REPLAY_PORT = b+N (replay, [b+N, b+2N)), KV_LOAD_PORT = b+2N (load,
# [b+2N, b+3N)), KV_RANKS = N; box-c at N=4, engine a: events 5600-5603, replay 5604-5607, load
# 5608-5611. N = 1 is the record layout untouched; no config of record carries a DP width
# (args/check.sh FORBIDDEN_FLAGS), so the layout is an arm's only. The router's tunnels name the
# record ports, so an arm copy publishes to no router -- it runs out of the pool (box-out).
ENGINES_DP_BASE=5600
ENGINES_LEGACY_PORT="$ENGINE_PORT"
ENGINES_LEGACY_URL="$ENGINE_URL"
ENGINES_IDS=""

# Validate ENGINES and fill ENGINES_IDS (space-separated). An unset or empty ENGINES is the legacy
# layout: ENGINES_IDS stays empty.
engines_check() {
  ENGINES_IDS=""
  [ -n "${ENGINES:-}" ] || return 0
  local re='^([a-z][a-z0-9]*):((0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*):(0|[1-9][0-9]*):(0|[1-9][0-9]*)$'
  local entries e id gpus port kv g p seen_ids=" " seen_gpus=" " seen_ports=" " owner
  read -r -a entries <<< "$ENGINES"
  for e in "${entries[@]}"; do
    [[ "$e" =~ $re ]] || die "ENGINES: malformed entry '$e' in $BOX_ENV (want id:gpu,gpu,...:engine_port:kv_base, e.g. a:0,1,2,3:8002:5557; id lowercase, numbers without leading zeros)"
    id="${BASH_REMATCH[1]}"; gpus="${BASH_REMATCH[2]}"; port="${BASH_REMATCH[6]}"; kv="${BASH_REMATCH[7]}"
    case "$seen_ids" in *" $id "*) die "ENGINES: engine id '$id' appears twice in $BOX_ENV" ;; esac
    seen_ids="$seen_ids$id "
    for g in ${gpus//,/ }; do
      case "$seen_gpus" in
        *" $g="*) owner="${seen_gpus#*" $g="}"; die "ENGINES: GPU $g is in engine '${owner%% *}' and engine '$id' ($BOX_ENV): two engines never share a GPU" ;;
      esac
      seen_gpus="$seen_gpus$g=$id "
    done
    for p in "$port" "$((port + 100))" "$kv" "$((kv + 1))" "$((kv + 2))"; do
      [ "$p" -ge 1 ] && [ "$p" -le 65535 ] || die "ENGINES: engine '$id' derives port $p, outside 1-65535 ($BOX_ENV: engine port $port, nccl +100, kv $kv..$((kv + 2)))"
      case "$seen_ports" in
        *" $p="*) owner="${seen_ports#*" $p="}"; die "ENGINES: port $p is used twice (engine '${owner%% *}' and engine '$id', $BOX_ENV): each engine holds its port, port+100 (nccl) and kv_base..kv_base+2" ;;
      esac
      seen_ports="$seen_ports$p=$id "
    done
    ENGINES_IDS="${ENGINES_IDS:+$ENGINES_IDS }$id"
  done
  return 0
}

engines_list() {
  engines_check
  local id
  for id in $ENGINES_IDS; do printf '%s\n' "$id"; done
}

engine_select() {
  local want="${1:-}" e id gpus port kv found="" dp="$ENGINES_DP_BASE" g
  engines_check
  if [ -z "$want" ]; then
    # The legacy layout: today's names and paths, byte for byte (the TP8 line, box-a, the rollback).
    export ENGINE_ID="" SFX="" ENGINE_GPUS=""
    export ENGINE_PORT="$ENGINES_LEGACY_PORT" ENGINE_URL="$ENGINES_LEGACY_URL"
    export ENGINE_NCCL_PORT=""   # sglang picks its own; relaunch.sh adds no --nccl-port
    export KV_PORT=5557 KV_REPLAY_PORT=5558 KV_LOAD_PORT=5559   # what the args files name
    export KV_RANKS=1 ENGINE_DP_BASE=""   # no DP area: engine_kv_ranks refuses a DP width here
    export ENGINE_HOST="$HOSTNAME" ENGINE_DOOR_URL="$DOOR_URL"
  else
    [ -n "$ENGINES_IDS" ] || die "engine '$want' named, but $BOX_ENV sets no ENGINES: this box runs one legacy engine -- drop the engine id"
    for e in $ENGINES; do
      id="${e%%:*}"
      if [ "$id" != "$want" ]; then
        [ -n "$found" ] && continue
        g="${e#*:}"; g="${g%%:*}"; g="${g//[!,]/}"; dp=$((dp + 3 * (${#g} + 1)))   # an earlier engine's DP area
        continue
      fi
      found=1
      e="${e#*:}"; gpus="${e%%:*}"; e="${e#*:}"; port="${e%%:*}"; kv="${e#*:}"
    done
    [ -n "$found" ] || die "unknown engine '$want' (ENGINES in $BOX_ENV names: $ENGINES_IDS)"
    case "$HOSTNAME" in
      *.*) ;;
      *) die "engine '$want': HOSTNAME=$HOSTNAME has no domain to derive the engine's host name from (<first label>$want.<domain>)" ;;
    esac
    export ENGINE_ID="$want" SFX="-$want" ENGINE_GPUS="$gpus"
    export ENGINE_PORT="$port" ENGINE_URL="http://127.0.0.1:$port"
    export ENGINE_NCCL_PORT="$((port + 100))"
    export KV_PORT="$kv" KV_REPLAY_PORT="$((kv + 1))" KV_LOAD_PORT="$((kv + 2))"
    export KV_RANKS=1 ENGINE_DP_BASE="$dp"
    export ENGINE_HOST="${HOSTNAME%%.*}$want.${HOSTNAME#*.}"
    export ENGINE_DOOR_URL="https://$ENGINE_HOST:$APP_PORT_EXT"
  fi
  export ENGINE_ARGS_FILE="$TOOLS_REMOTE/sglang-args$SFX.txt"
  export ENGINE_ENV_FILE="$TOOLS_REMOTE/sglang-env$SFX.txt"
  export ENGINE_SERVE_LOG="/workspace/sglang-serve$SFX.log"
  export ENGINE_PIDFILE="$TOOLS_REMOTE/engine$SFX.pid"
  export ENGINE_WARMING="$TOOLS_REMOTE/engine$SFX.warming"
  export WATCHDOG_RUN="$TOOLS_REMOTE/watchdog$SFX.run"
  export WATCHDOG_HOLD="$TOOLS_REMOTE/watchdog$SFX.hold"
  export WATCHDOG_LOG="/workspace/$BOX$SFX-watchdog.log"
  export WARM_LOG="/workspace/$BOX$SFX-warm.log"
  export PREWARM_MARK_SFX="$SFX"   # restart.sh appends it to its .prewarmed-<sha>-<venv>-cp<n> marker
  return 0
}

# engine_kv_ranks <n> -- after engine_select <id>: the KV ports for <n> DP ranks (the header above).
# n = 1 leaves the record layout as engine_select set it. Refused (die): n not a positive integer; a
# DP width on the legacy layout (no engine id: its ports are the args file's own, and a TP8 line's
# 5557+r collides with its replay 5558); n above the engine's GPU count (no layout has more ranks than
# GPUs, and the area holds 3 per GPU); a port of the range outside 1-65535 or on any engine's record
# port (port, port+100, kv_base..+2) -- a box file whose layout reaches into 5600 says so here.
engine_kv_ranks() {
  local n="${1:-}" b gpus e p q id lo hi
  case "$n" in ''|*[!0-9]*|0*) die "engine_kv_ranks: '$n' is not a DP rank count (a positive integer)" ;; esac
  [ "$n" = 1 ] && return 0
  [ -n "${ENGINE_ID:-}" ] && [ -n "${ENGINE_DP_BASE:-}" ] || die "a DP width of $n ranks boots per engine only (relaunch.sh <id> on a box with ENGINES): the legacy launch has no per-rank KV layout -- its ranks would bind 5557+r, onto replay 5558"
  gpus="${ENGINE_GPUS//[!,]/}"; gpus=$((${#gpus} + 1))
  [ "$n" -le "$gpus" ] || die "engine $ENGINE_ID: a DP width of $n ranks on $gpus GPUs ($ENGINE_GPUS) -- the per-rank KV area holds 3 ports per GPU"
  b="$ENGINE_DP_BASE"; lo="$b"; hi=$((b + 3 * n - 1))
  [ "$lo" -ge 1 ] && [ "$hi" -le 65535 ] || die "engine $ENGINE_ID: the per-rank KV range $lo-$hi is outside 1-65535"
  for e in $ENGINES; do
    id="${e%%:*}"; e="${e#*:}"; e="${e#*:}"; p="${e%%:*}"; q="${e#*:}"
    for p in "$p" "$((p + 100))" "$q" "$((q + 1))" "$((q + 2))"; do
      [ "$p" -lt "$lo" ] || [ "$p" -gt "$hi" ] || die "engine $ENGINE_ID: its per-rank KV range $lo-$hi ($n ranks) holds port $p, engine '$id''s record port ($BOX_ENV) -- move ENGINES_DP_BASE in lib.sh off the record layout"
    done
  done
  export KV_RANKS="$n" KV_PORT="$b" KV_REPLAY_PORT="$((b + n))" KV_LOAD_PORT="$((b + 2 * n))"
}

# ---------------------------------------------------------------- the box target fence
# box_ssh/box_scp resolve $BOX_IP:$SSH_PORT from the box FILE, so a stale or half-filled file sends
# the call at whatever box it names — exactly how the box-g incident (2026-09-13) did its damage
# AFTER the create step: bringup.sh created box-c, and stage.sh / tls.sh / restart.sh then ran against
# the live box-g whose ip the file still held. So before the first box_ssh/box_scp of this process,
# ask the box itself which box it is — the BOX= line stage.sh writes into its /etc/environment —
# and refuse a POSITIVE MISMATCH, naming what the file says, what the box says, and the two ways
# out. bringup.sh's own guard fences the one orchestrator that creates a box; this one fences every
# tool that reaches a box through this file.
#
# It never refuses a legitimate call. The box does not name itself yet (fresh, pre-stage) -> UNKNOWN,
# allowed and said so once. The probe cannot reach the box at all -> UNREACHABLE, allowed, and the
# ssh/scp being made reports why. Only a box that answers and names a DIFFERENT box stops the script.
# fork_ssh is untouched: the probe rides box_ssh/box_scp alone, never $FORK_HOST or a
# local command. Cost is one extra round trip per PROCESS, never per call: the verdict is cached in
# a variable and — because every `$(box_ssh ...)` runs in a subshell that cannot keep that variable —
# in one file under ${TMPDIR:-/tmp} keyed by a token minted HERE at source time ($$ and $RANDOM are
# the same in every subshell of this run) plus $BOX. One run never reads another run's file, so a
# leftover file can never pass a verdict on a later run. The probe is deliberately not skipped under
# DRY_RUN: a box_ssh that executes at all is a real call, dry run or not.
BOX_TARGET_RUN_ID="$$-$RANDOM-$(date -u +%s)"
export BOX_TARGET_RUN_ID
BOX_TARGET_CACHE="${TMPDIR:-/tmp}/lib-box-target.$BOX_TARGET_RUN_ID.$BOX"
# The session-proxy helper: a python file this run writes for itself in the block before box_ssh,
# named with the same run token so no other run — and no stale file from an earlier one — can ever be
# the ProxyCommand of this run's ssh.
BOX_SSH_PROXY_HELPER="${TMPDIR:-/tmp}/lib-ssh-proxy.$BOX_TARGET_RUN_ID.py"
# The helper's SOURCE is this tree's standing `ssh-box-proxy.py` (#252), a sibling of this file,
# copied verbatim onto the minted name above by box_ssh_proxy_setup below. There is no second copy of
# the tunnel program inside this file: a tree without it gets a note and NO ProxyCommand, exactly as a
# tree without python3 does (#270 retired the embedded fallback). The ProxyCommand always names the
# MINTED file and never the repo file itself -- that operand has to be a path no other run can be
# using: the minted name carries this run's token, and lib-proxy.test.sh pins exactly that shape
# (`$TMPDIR/lib-ssh-proxy.<token>.py`, gone when the process exits).
BOX_SSH_PROXY_PY="${BOX_SSH_PROXY_PY:-$TOOLS_DIR/ssh-box-proxy.py}"
BOX_TARGET_STATE=""     # "" until probed; then IDENTITY | MISMATCH | UNKNOWN | UNREACHABLE
BOX_TARGET_SAYS=""      # the BOX= value the box itself reports ("" when it names itself nowhere)
# Housekeeping only — both files are tiny and no other run reads them, so a lost cleanup is harmless,
# but do not clobber an EXIT trap a script set for itself before sourcing this file.
if [ -z "$(trap -p EXIT)" ]; then
  trap 'rm -f "$BOX_TARGET_CACHE" "$BOX_SSH_PROXY_HELPER" 2>/dev/null || true' EXIT
fi

# One round trip: read the box's own BOX= line. Cached, so the many `$(box_ssh ...)` of one script
# pay for it once; a half-written cache file is ignored and probed again rather than trusted. The
# two allowed-but-unconfirmed states are named HERE, on the fresh probe only, so a script whose
# first box_ssh is inside a command substitution still says it once and not once per call.
box_probe_target() {
  if [ -n "$BOX_TARGET_STATE" ]; then return 0; fi
  if [ -r "$BOX_TARGET_CACHE" ]; then
    BOX_TARGET_STATE=$(head -n 1 "$BOX_TARGET_CACHE")
    BOX_TARGET_SAYS=$(sed -n 2p "$BOX_TARGET_CACHE")
    case "$BOX_TARGET_STATE" in
      IDENTITY|MISMATCH|UNKNOWN|UNREACHABLE) return 0 ;;
    esac
    BOX_TARGET_STATE=""; BOX_TARGET_SAYS=""
  fi
  # A transport failure is the only thing that reaches the first branch: the remote pipeline ends in
  # `tr`, so it exits 0 whether or not /etc/environment carries a BOX= line.
  # The session-proxy argument ($BOX_SSH_PROXY_ARG, set below before any call) belongs on THIS ssh as
  # much as on box_ssh itself: a probe that cannot reach the box reports UNREACHABLE and the very next
  # box_ssh then fails with the sandbox's own message, which reads like a dead box.
  #
  # STDIN BELONGS TO THE CALLER'S OWN ssh, NOT TO THIS PROBE. ssh forwards its stdin to the remote
  # command and reads it to EOF, and the probe ran with the caller's -- so `box_ssh 'cat > /f' < local`
  # and `box_ssh 'bash -s' < script` had their input eaten here before the call they were written for
  # was even built: the redirect target came out empty and the command exited 0 (#262, reproduced
  # 2026-10-17 with a 0-byte file). `</dev/null` is the fix, and it is a REDIRECTION rather than ssh's
  # `-n` on purpose: the probe's argv then stays byte-identical (lib-proxy.test.sh counts argv 1-3, and
  # `-n` would displace the second `-o`), while the probe still gets an EOF at once instead of the
  # caller's bytes. It must never sit on the ssh that box_ssh/box_scp build -- that one is the call the
  # caller's stdin is FOR.
  if ! BOX_TARGET_SAYS=$(ssh ${BOX_SSH_PROXY_ARG:+-o "$BOX_SSH_PROXY_ARG"} \
        -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
        -p "$SSH_PORT" -i "$SSH_KEY_PATH" "root@$BOX_IP" \
        "grep -a '^BOX=' /etc/environment 2>/dev/null | head -1 | cut -d= -f2- | tr -d '\"[:space:]'" </dev/null); then
    BOX_TARGET_SAYS=""
    BOX_TARGET_STATE=UNREACHABLE
    note "target fence: $BOX_IP:$SSH_PORT did not answer the identity probe, so it cannot be confirmed as $BOX — allowing; the ssh/scp being made reports why (an unreachable box is UNKNOWN here, never a mismatch)"
  elif [ -z "$BOX_TARGET_SAYS" ]; then
    BOX_TARGET_STATE=UNKNOWN
    note "target fence: the box at $BOX_IP:$SSH_PORT does not name itself yet (no BOX= line in its /etc/environment — a fresh box, before stage.sh), so it cannot be confirmed as $BOX — allowing"
  elif [ "$BOX_TARGET_SAYS" = "$BOX" ]; then
    BOX_TARGET_STATE=IDENTITY
  else
    BOX_TARGET_STATE=MISMATCH
  fi
  printf '%s\n%s\n' "$BOX_TARGET_STATE" "$BOX_TARGET_SAYS" > "$BOX_TARGET_CACHE" 2>/dev/null || true
  return 0
}

box_assert_target() {
  if [ -z "${BOX_IP:-}" ] || [ -z "${SSH_PORT:-}" ]; then
    die "half-filled box file: $BOX_ENV gives no BOX_IP/SSH_PORT to ssh to (BOX_IP='${BOX_IP:-}' SSH_PORT='${SSH_PORT:-}') — fill both in from the instance's own record (\`hostctl show instance <id> --raw\`); bringup.sh's reconcile step does that for a box it has just created"
  fi
  box_probe_target
  if [ "$BOX_TARGET_STATE" = MISMATCH ]; then
    die "wrong box: $BOX_ENV resolves $BOX to $BOX_IP:$SSH_PORT, but the box answering there says BOX=$BOX_TARGET_SAYS — refusing to send this command to it. Fix one of the two, then re-run:
  (1) $BOX is the box you mean, the file is stale:  hostctl show instance \$INSTANCE_ID --raw   # then set BOX_IP/SSH_PORT/APP_PORT_EXT in $BOX_ENV to that instance's values (bringup.sh's reconcile step does exactly that for a box it has just created)
  (2) the box at that address is the one you mean:  BOX=$BOX_TARGET_SAYS bash <script>.sh   (its own file is boxes/$BOX_TARGET_SAYS.env)"
  fi
}

# ---------------------------------------------------------------- the session proxy
# A sandboxed session has no direct route to a box: a bare
# `ssh -p $SSH_PORT -i $SSH_KEY_PATH root@$BOX_IP` dies at once with "connect to host ... Operation
# not permitted" -- the sandbox refusing the connect, not the box being down, which is why the message
# has to be read before the box is declared dead. The only way out is the session HTTP proxy the
# environment already carries in $https_proxy (`http://srt.<id>:<password>@localhost:<port>`). The
# route that looks like it should work, `nc -X 5 -x localhost:<port>` (what $GIT_SSH_COMMAND uses),
# does not: BSD nc speaks SOCKS without the authentication this proxy requires. What does work is an
# HTTP CONNECT tunnel carrying `Proxy-Authorization: Basic`, which is what the helper below does:
# every ssh/scp this file makes gets `-o ProxyCommand=...` when a proxy variable is present.
#
# Nothing changes in the un-sandboxed case: with no proxy variable set there is no ProxyCommand at
# all and every command line below is what it always was. And when one IS set, the helper still tries
# a direct TCP connection FIRST and tunnels only if that fails -- so a box reached directly costs
# nothing extra and only a blocked call takes the proxy. That ordering is also what keeps this option
# safe to attach to fork_ssh, whose host is reached directly from an operator's shell.
# BOX_SSH_PROXY_TRACE=1 makes the helper say on stderr which route it took.
#
# The URL is a credential, so it never appears on a command line: the ProxyCommand string carries the
# helper's PATH and the variable's NAME (the helper's argv is host, port, var -- it reads the value
# from its own inherited environment), and the path is run through `printf %q` because ssh hands the
# command to /bin/sh (this repo lives under a path with a space in it).
#
# It is a FILE and not a `python3 -c` one-liner for one concrete reason: ssh expands %-tokens anywhere
# in the ProxyCommand string, so a python program inline would have every `%` of its own format
# strings rewritten by ssh. In a file, only the command string's `%h %p` are tokens.
#
# WHAT the ProxyCommand runs is this tree's own `ssh-box-proxy.py` (#252): the same program, and the
# same CONNECT handshake, that an ssh which never sourced this file is told to point at in the
# README's Script reference row, and the same file boxenv.py mints its helper from for the Python
# tools' ssh (boxenv-proxy.test.sh compares the two minted helpers byte for byte) -- so the tool set
# has one tunnel program, in one file, and no copy of its text anywhere else. A copy of THAT file is
# minted into $TMPDIR for this run; if it is missing, the ssh this file makes runs with no
# ProxyCommand and says so once (#270 retired the embedded fallback that used to stand in for it).
#
# Said plainly, because each is a real environment and each degrades to the old behaviour rather than
# failing a script that would otherwise run: this needs python3 on PATH, a readable sibling
# `ssh-box-proxy.py`, $TMPDIR writable, and an http:// (or bare host:port) proxy URL. A socks5h:// URL
# -- the shape ftp_proxy has here -- is refused the same way, since not authenticating to a SOCKS
# proxy is the whole reason this exists.
BOX_SSH_PROXY_VAR=""     # the environment variable the URL came from ("" = no proxy in this session)
BOX_SSH_PROXY_ARG=""     # the value of ssh's -o, i.e. "ProxyCommand=..." ("" = add no argument)

# The first proxy variable that the ENVIRONMENT carries, by NAME only, never by value: the value
# holds the password. printenv and not an indirect `${!v}` expansion on purpose -- the proxy is only
# real if the child processes (ssh, and the helper it starts) can see it, which is exactly what
# printenv asks; and `${!v:-}` on an UNSET name is a fatal "invalid indirect expansion" under bash 5,
# where this file has to run on bash 3.2 as well.
box_ssh_proxy_var() {
  local v
  for v in https_proxy HTTPS_PROXY http_proxy HTTP_PROXY all_proxy ALL_PROXY; do
    if [ -n "$(printenv "$v" 2>/dev/null || true)" ]; then printf '%s\n' "$v"; return 0; fi
  done
  return 1
}

# Runs once, at source time. Every failure here is a NOTE and leaves BOX_SSH_PROXY_ARG empty: a
# session without python3 must still run its script, just without the tunnel, and then a sandboxed
# ssh reports the sandbox's own message -- which this note has already explained.
box_ssh_proxy_setup() {
  local var url
  var="$(box_ssh_proxy_var)" || return 0
  url="$(printenv "$var" 2>/dev/null || true)"
  case "$url" in
    *://*)
      if [ "${url%%://*}" != http ]; then
        note "session proxy: \$$var is ${url%%://*}://, which this file cannot tunnel through (it speaks HTTP CONNECT) -- ssh calls run with no ProxyCommand, so a sandboxed one reports 'Operation not permitted'"
        return 0
      fi
      ;;
    *:*) ;;   # a bare host:port is the other spelling of an http:// proxy
    *) note "session proxy: \$$var names no proxy host -- ssh calls run with no ProxyCommand, so a sandboxed one reports 'Operation not permitted'"; return 0 ;;
  esac
  command -v python3 >/dev/null 2>&1 || {
    note "session proxy: \$$var is set but python3 is not on PATH -- ssh calls run with no ProxyCommand, so a sandboxed one reports 'Operation not permitted'"
    return 0
  }
  if [ ! -s "$BOX_SSH_PROXY_PY" ]; then
    note "session proxy: no readable $BOX_SSH_PROXY_PY -- ssh calls run with no ProxyCommand, so a sandboxed one reports 'Operation not permitted'; the tunnel program is this tree's own ssh-box-proxy.py, and this file carries no copy of it (#270)"
    BOX_SSH_PROXY_HELPER=""
    return 0
  fi
  if ! cat "$BOX_SSH_PROXY_PY" > "$BOX_SSH_PROXY_HELPER" 2>/dev/null; then
    note "session proxy: cannot write $BOX_SSH_PROXY_HELPER -- ssh calls run with no ProxyCommand, so a sandboxed one reports 'Operation not permitted'"
    BOX_SSH_PROXY_HELPER=""
    return 0
  fi
  # shellcheck disable=SC2034  # read by callers, and by a test that asserts which variable was used
  BOX_SSH_PROXY_VAR="$var"
  # `%h %p` are ssh's own tokens, expanded by ssh for the host and port it settled on (HostName and
  # Port from ~/.ssh/config included), which is what the helper must CONNECT to. `%q` on the path only:
  # the var name is from the fixed list above.
  BOX_SSH_PROXY_ARG="ProxyCommand=python3 -I $(printf '%q' "$BOX_SSH_PROXY_HELPER") %h %p $var"
  return 0
}
box_ssh_proxy_setup

# ssh/scp to the box (plain remote commands; never into /root or /). Both are fenced: the first call
# in a process checks that the box at $BOX_IP:$SSH_PORT is the box this file names. All four ssh calls
# below — box_ssh, box_scp, fork_ssh — carry the session-proxy argument when this session
# has one. A caller that builds its own ssh line instead of calling one of them (stage.sh's rsync -e)
# can add the same two words: ${BOX_SSH_PROXY_ARG:+-o "$BOX_SSH_PROXY_ARG"}, the bash-3.2-safe spelling
# of "add these two arguments only if the variable is non-empty" — an unquoted $VAR would split the
# ProxyCommand value into several argv elements and an empty one would be read as ssh's hostname.
box_ssh() { box_assert_target; ssh ${BOX_SSH_PROXY_ARG:+-o "$BOX_SSH_PROXY_ARG"} -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" -i "$SSH_KEY_PATH" "root@$BOX_IP" "$@"; }
box_scp() { box_assert_target; scp ${BOX_SSH_PROXY_ARG:+-o "$BOX_SSH_PROXY_ARG"} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P "$SSH_PORT" -i "$SSH_KEY_PATH" "$@"; }
# ssh to the fork host (the fork host that pushes our SGLang fork).
fork_ssh() { ssh ${BOX_SSH_PROXY_ARG:+-o "$BOX_SSH_PROXY_ARG"} -o BatchMode=yes -o ConnectTimeout=15 "$FORK_HOST" "$@"; }

# A secret from the dev box's env file (KEY=VALUE lines), never echoed by callers.
secret() { grep -a "^$1=" "$ENV_FILE_PATH" | head -1 | cut -d= -f2- | tr -d '"'; }

# --dry-run support: `run <cmd...>` prints instead of executing when DRY_RUN=1.
# The printed line is the same argv the command would get, with every secret VALUE `run` can
# recognize replaced by <redacted:NAME>. A command line is where a secret leaks by construction,
# and a dry run's whole output is a command line: the leak this rule exists for is
# `offer-search.sh create`, whose origin bearer rides INSIDE a larger argument --
# `--env "-p N:N -e VLLM_API_KEY=<key> -e XDG_CACHE_HOME=…"` -- so an unredacted dry pass wrote
# the live key into the operator's terminal and whatever log it was redirected to. That call site
# is not one this file may edit, and there the key is a substring of a bigger argument, so
# redaction is BY VALUE and automatic: never a flag a call site has to remember. Names come from
# the two files this file already knows -- `$ENV_FILE_PATH` (the secrets file, grepped the way
# `secret()` greps it) and the box file -- plus the operator's own exported environment.
#
# A name counts when it is KEY/TOKEN/SECRET/PASSWORD/PASS/CREDENTIAL(S) or ends in `_` + one of
# those. A value counts when it is printable, not starting with `/` or `~` -- that shape is a PATH,
# and the box file's own `SSH_KEY`/`FORK_DEPLOY_KEY` are paths; masking a path would blind the dry
# run without hiding anything -- at least DRY_REDACT_MIN_LEN characters long, and not all spaces. A
# SPACE inside a value is part of it: the file format is KEY=VALUE, so everything after the first
# `=` is the value (`secret()` reads the line the same way) and only a trailing `# comment` is not;
# the whole value is masked, so a space-bearing secret cannot print its own tail. Over-masking is
# the safe direction: an unrelated ambient `*_TOKEN` masked in a dry line costs one re-run with
# DRY_RUN=0, a missed secret costs a leaked bearer. Nothing below runs when DRY_RUN=0 -- a real
# run's argv, behaviour and exit status are exactly as before, and a secret still reaches the real
# command line, which is where hostctl/ssh need it.
#
# The rule's limits, stated rather than left to be discovered:
#   * a value SHORTER than DRY_REDACT_MIN_LEN (8) characters is not masked. The floor is deliberate:
#     a short value is also a common substring (`abc` inside `--label abc-foo`), and masking it
#     would garble the line this redaction exists to keep readable. Nothing this tool set mints is
#     anywhere near that short (`openssl rand -hex 16` -> 32 chars, a Cloudflare DNS token, an
#     `hf_...` token), so the floor costs nothing on this tool set's own secrets;
#   * a value carrying a TAB, CR or LF is not masked: DRY_REDACT_PAIRS holds one value per line, so
#     such a value cannot be represented there, and a half-masked value is worse than a named gap
#     (this is why a whitespace-bearing value is now masked, but a control-character one is refused
#     outright rather than cut at the control character);
#   * a path-shaped value (`/...`, `~/...`) is not masked, for the reason above.
# A secret in any of those three shapes still reaches the real command line under DRY_RUN=0, exactly
# like every other one. The fix for a dry line that leaked one is `DRY_RUN=0`, which never prints it.
DRY_REDACT_MIN_LEN=8
DRY_REDACT_PAIRS=""        # one "NAME value" line per secret found; a value may carry spaces and is masked whole
DRY_REDACT_COLLECTED=""    # "" until the first dry call has scanned the three sources

# KEY / VLLM_API_KEY / CF_DNS_TOKEN / *_PASSWORD / ... — the name shapes that hold secret material.
dry_redact_name_is_secret() {
  case "$1" in *[!A-Za-z0-9_]*) return 1 ;; esac
  case "$1" in
    KEY|TOKEN|SECRET|PASSWORD|PASS|CREDENTIAL|CREDENTIALS) return 0 ;;
    *_KEY|*_TOKEN|*_SECRET|*_PASSWORD|*_PASS|*_CREDENTIAL|*_CREDENTIALS) return 0 ;;
  esac
  return 1
}

# A maskable value: printable, not a path, long enough, not all spaces. An internal SPACE is allowed
# and the value is masked whole (see the limits above); a TAB/CR/LF still disqualifies a value
# because DRY_REDACT_PAIRS carries one value per line and cannot represent one.
dry_redact_value_ok() {
  # The tilde MUST be quoted: an unquoted `~` in a case pattern is tilde-expanded by the shell, so
  # the arm that used to sit here, `~/*`, was `$HOME/*` -- a subset of the `/*` arm beside it. It
  # therefore excluded no literal-tilde value and masked every one, the box file's own `SSH_KEY` and
  # `FORK_DEPLOY_KEY` (`~/.ssh/...`) included, which is the opposite of what the block above says
  # this rule does. `'~'*` matches a literal `~` and anything after it, a bare `~` included (`*`
  # matches the empty remainder), so a tilde-spelled path stays readable on the dry line.
  case "$1" in
    ''|/*|'~'*) return 1 ;;
    *[![:print:]]*) return 1 ;;
  esac
  case "$1" in *[![:space:]]*) ;; *) return 1 ;; esac   # spaces alone are not a secret
  [ "${#1}" -ge "$DRY_REDACT_MIN_LEN" ]
}

# One "NAME value" line per secret-shaped assignment in an env file (KEY=VALUE, optional `export`,
# optional surrounding quotes — stripped the way `secret()` strips them). Never fatal: an absent or
# unreadable file contributes nothing, and the `return 0` keeps a caller's `set -e` intact.
dry_redact_scan_file() {
  local file="$1" line name value
  [ -n "$file" ] && [ -r "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in export\ *) line="${line#export }" ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    name="${line%%=*}"
    value="${line#*=}"
    # Everything after the first `=` is the value, `secret()`'s own reading of the line. The one
    # thing that is not part of it is a trailing `# comment`: a `#` at a whitespace boundary, the
    # same rule lib.sh's box-file writer (bringup.sh's rewrite) and boxenv.py use. Cutting at the
    # first whitespace instead -- what this line used to do -- printed the tail of any secret whose
    # value carried a space. A quoted value keeps any `#` of its own: the closing quote ends it.
    case "$value" in
      \"*) value="${value#\"}"; value="${value%%\"*}" ;;
      *)   value="${value%%[[:space:]]#*}" ;;
    esac
    dry_redact_name_is_secret "$name" || continue
    dry_redact_value_ok "$value" || continue
    printf '%s %s\n' "$name" "$value"
  done < "$file"
  return 0
}

# The operator's own exported environment: a secret the box file never sees (`HOST_API_KEY`, a
# GitHub token). Reading it can only ADD masks, never remove the two files' own.
dry_redact_scan_env() {
  local line name value
  printenv 2>/dev/null | while IFS= read -r line; do
    case "$line" in *=*) ;; *) continue ;; esac
    name="${line%%=*}"
    value="${line#*=}"
    dry_redact_name_is_secret "$name" || continue
    dry_redact_value_ok "$value" || continue
    printf '%s %s\n' "$name" "$value"
  done
  return 0
}

dry_redact_collect() {
  dry_redact_scan_file "${ENV_FILE_PATH:-}"
  dry_redact_scan_file "${BOX_ENV:-}"
  dry_redact_scan_env
  return 0
}

# Mask every known secret in ONE already-%q-quoted DRY line. The substitution runs AFTER the
# quoting, on purpose: a token with no shell-special character survives `%q` literally, so it is
# still a plain substring there, and inserting the marker afterwards prints <redacted:VLLM_API_KEY>
# rather than an escaped marker. The `%q` form of the value is compared too, for a token that does
# carry a character `%q` would escape.
dry_redact_line() {
  local line="$1" name value quoted
  while IFS=' ' read -r name value; do
    [ -n "$value" ] || continue
    line="${line//"$value"/"<redacted:$name>"}"
    quoted="$(printf '%q' "$value")"
    [ "$quoted" = "$value" ] || line="${line//"$quoted"/"<redacted:$name>"}"
  done <<< "$DRY_REDACT_PAIRS"
  printf '%s' "$line"
  return 0
}

run() {
  if [ "${DRY_RUN:-0}" = 1 ]; then
    local line=""
    if [ -z "$DRY_REDACT_COLLECTED" ]; then
      DRY_REDACT_PAIRS="$(dry_redact_collect)"
      DRY_REDACT_COLLECTED=1
    fi
    # Same shape as before this redaction existed: one `DRY: ` per argument, `%q`-quoted, with
    # the masks applied to the assembled line. `%q` never emits a literal newline, so the command
    # substitution round trip is lossless.
    line="$(printf 'DRY: %q ' "$@")"
    printf '%s\n' "$(dry_redact_line "$line")"
  else
    "$@"
  fi
}
