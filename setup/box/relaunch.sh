#!/bin/bash
# SGLang launcher (runs ON the box, staged to $TOOLS_REMOTE/relaunch.sh; always started under
# `nohup setsid` so it survives the caller's exit, and started under `bash`, never sourced).
# Ported from the earlier per-box sglang-relaunch.sh lines, box-parametrized and generalized to the new
# args-are-tuning-flags-only contract: the args file (default $TOOLS_REMOTE/sglang-args.txt)
# carries ONLY tuning flags — this script is the one place that adds --model-path,
# --served-model-name, --host, --port and the API key, so a box's args file never repeats them and
# never needs to name a model, a host or a port.
#
# The full composed line (fixed shape, do not reorder — later flags win on a duplicate):
#   sglang serve <args file> --model-path $MODEL_DIR --served-model-name $SERVED_MODEL
#                --host 127.0.0.1 --port $ENGINE_PORT --config $API_KEY_FILE
# The key never rides the argv: $API_KEY_FILE is a root-only (0600) YAML file holding the one line
# `api-key: "<VLLM_API_KEY>"`, rewritten from /etc/environment on every launch, and sglang's own
# `--config` merges its keys in AHEAD of the command line (ConfigArgumentMerger: CLI > config >
# default), so the engine gets exactly `--api-key <key>` in-process while `pgrep -af`/`ps`/
# /proc/<pid>/cmdline show only the file's path. The YAML key is spelled `api-key`, never `api_key`:
# the merger emits `--<key>` as the file spells it, and `--api_key` is an unrecognized argument.
# Nor does it ride the engine's environ, or a watchdog's (#3003): VLLM_API_KEY is unset twice -- once
# right after lib.sh is sourced, ahead of every watchdog start (the launch's and `--watchdog`'s), and
# again after the sglang-env.txt loop, just before the exec, so a stray key line there cannot put one
# back (the two `unset` lines below).
# 127.0.0.1: the engine is loopback-only on every box; the direct-TLS front door (tls.sh) is what
# terminates on the box's externally mapped port and reverse-proxies to $ENGINE_PORT.
#
# Required override file (this script refuses to boot without it):
#   $TOOLS_REMOTE/sglang-env.txt — KEY=VALUE lines exported before launch (e.g. a rung's
#                                  SGLANG_RAGGED_VERIFY_MODE=compact, or the boot line's own
#                                  SGLANG_ENABLE_REQUEST_DECOMPRESSION=1). `stage.sh` seeds it
#                                  the first time it stages a box from the env of the serve line
#                                  the box file's SGLANG_TREE runs (tools/args/env-box-d-official.txt
#                                  on box-d-up, tools/args/env-box-e.txt on box-e-up,
#                                  tools/args/env-v3.txt on any other tree --
#                                  stage_env_seed_name, which restart.sh/cutover.sh's `default`
#                                  reseed shares), and never again — a hand-tuned live file is
#                                  never clobbered — so in normal operation this file already
#                                  exists before the first relaunch. Required, not optional, so
#                                  a box can never silently drift back to missing the boot
#                                  line's env flags.
# Rollback = restore its .pre-<tag> backup (never delete it outright — a missing file is now a
# hard failure here) and restart. restart.sh and cutover.sh write/back it up; this script never
# touches it.
#
# `relaunch.sh --watchdog` touches the WATCHDOG only, never the engine: stage.sh runs it on the box
# after every stage (see watchdog_refresh below).
#
# Usage: relaunch.sh [--watchdog] [<engine id>]. With no id: the legacy one-engine layout above, byte
# for byte. With an id (a box file's ENGINES, lib.sh's engine_select) it launches that ONE engine of
# the box's per-engine layout (box-c's 2xTP4) from its own files and ports:
#   * $ENGINE_ARGS_FILE / $ENGINE_ENV_FILE (sglang-args-<id>.txt / sglang-env-<id>.txt), never the
#     legacy pair -- that pair is the TP8 line, kept for --single rollback and for the onstart's
#     legacy boot (a reboot starts `relaunch.sh <id>` per engine when the staged box file sets
#     ENGINES and every engine's pair is on the box; the legacy pair otherwise);
#   * CUDA_VISIBLE_DEVICES=$ENGINE_GPUS, exported after the env file's loop so no file line overrides it;
#   * --kv-events-config's tcp://127.0.0.1:5557 / tcp://127.0.0.1:5558 rewritten to $KV_PORT /
#     $KV_REPLAY_PORT and --load-publish-endpoint's tcp://*:5559 to $KV_LOAD_PORT (one args file serves
#     both copies), the host kept. Since #1300 the events and replay are on loopback (the router's ssh
#     tunnel forwards to the box's 127.0.0.1): a 5557/5558 endpoint on any other host is refused; the
#     load feed (5559, counters only) may stay on the wildcard the fork binds it on, or loopback. An
#     args file naming any other tcp:// port is refused: it would collide between the copies;
#   * an A/B arm with a DP width (--attn-dp-size N / --dp-size N; never a config of record, args/check.sh
#     FORBIDDEN_FLAGS): the same three endpoints rewritten to the engine's DP area instead (lib.sh's
#     engine_kv_ranks: events [b, b+N), replay [b+N, b+2N), load [b+2N, b+3N); box-c a from 5600, b from
#     5612), the fork offsetting each rank's port; refused on the legacy launch and with
#     SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION on (sglang #46676, unmerged);
#   * --nccl-port $ENGINE_NCCL_PORT appended after the args file's flags (an args file carrying its own
#     --nccl-port is refused), the composed tail (--model-path ... --config) unchanged, --port
#     $ENGINE_PORT;
#   * refused while any of the engine, NCCL or three kv ports (a DP arm: every rank's) is already
#     listening (`ss -ltn`, or /proc/net/tcp + tcp6 on a box without iproute2);
#   * $ENGINE_PIDFILE written with this script's pid before the exec: under `nohup setsid` that pid
#     leads the engine's process group, so watchdog.sh/restart.sh stop exactly this engine's group;
#   * its own warm gate ($ENGINE_WARMING, log $WARM_LOG) and its own watchdog, `watchdog.sh <id>`,
#     singleton per engine (run record $WATCHDOG_RUN, hold $WATCHDOG_HOLD). The legacy singleton
#     matches only the id-less watchdog, so it never counts, stops or replaces an engine's.
#
# No production traffic at rest on a rented box (#1300, the user's standing rule of 2026-10-29): the provider's
# host has root and its disks are unencrypted, so nothing this script starts may write a prompt or a
# completion to disk. Two parts, both ahead of everything else:
#   * `ulimit -c 0`, the first command, so this shell, the watchdog it forks and the engine it execs
#     all run with RLIMIT_CORE 0 -- the box's core_pattern is
#     /var/lib/hostctl_kaalia/data/core-%e.%p.%h.%t (the HOST's disk), and a crash of an engine with an
#     unlimited core size would dump its memory, prompts and KV included, there;
#   * the deploy gate, rest_gate below: the launch path runs it over the composed args and the engine's
#     whole environment right after the last unset (ahead of the note, the watchdog, the warm gate and
#     the exec) and dies naming each refusal. It refuses any --log-requests* flag, --crash-dump-folder,
#     any --debug-tensor-dump*, --msprobe-dump-config, a --hicache-storage-backend other than the
#     RAM-only `sim` and `shm`, --enable-trace toward a non-loopback --otlp-traces-endpoint (its default,
#     localhost:4317, passes), any OTEL_EXPORTER_OTLP_*ENDPOINT that is not 127.0.0.1/localhost/::1, the
#     fork's request-logging and dump switches in the env (SGLANG_LOG_REQUEST_HEADERS,
#     SGLANG_LOG_PICKLE_IPC_OBJECTS, SGLANG_CUDA_COREDUMP*, CUDA_ENABLE_*COREDUMP*, CUDA_COREDUMP_FILE,
#     DUMPER_ENABLE, SGLANG_KERNEL_API_LOGLEVEL, and any other SGLANG_*DUMP* / SGLANG_*LOG*REQ* set to
#     a true value -- SGLANG_PYSPY_DUMP_BEFORE_CRASH excepted: default on, a py-spy stack without
#     locals), RLIMIT_CORE other than 0, and the engine's sockets off loopback, failing CLOSED: a
#     --kv-events-config the gate cannot read as a flat JSON object (or with a repeated key) is refused,
#     and one with a publisher but no endpoint binds the fork's default tcp://*:5557, refused as a
#     wildcard; a --kv-events-config
#     endpoint or replay_endpoint that is not 127.0.0.1/localhost/::1 (the events carry token ids --
#     kept, the router's parity oracle reads them -- so they travel only over loopback and the router's
#     ssh tunnel, which forwards to the box's 127.0.0.1), a loopback endpoint without "bind":true (a
#     concrete host makes the fork's PUB socket CONNECT, the router would hear nothing), and a
#     --load-publish-endpoint that is `auto` (the fork derives it only from a wildcard KV endpoint) or a
#     concrete non-loopback host. The load feed alone may stay on the wildcard (tcp://*:5559): it
#     carries per-rank load counters, no request content, and a fork tree before a84f5742 (#1321:
#     box-c's 9b262265/bdc3f67f, box-a's 819756d4 and their rollbacks) binds it only on a wildcard host
#     (python/sglang/srt/disaggregation/kv_events.py parse_bindable_tcp) -- a loopback load feed boots
#     only on a tree carrying a84f5742 (box-c/TODO.md). No exception for an A/B arm.
#   `relaunch.sh --rest-gate [--live] [--args <file>]... [--legacy-args <file>]... [--env <file>]...`
#   runs the same gate over files without launching anything and without lib.sh or a box file:
#   args/check.sh runs it over every args and env file of record; `--legacy-args` skips the endpoint
#   rules for a frozen record whose tree cannot take them (the 264ea0f9-era ladder: no "bind" key), which
#   the launch gate still refuses; `--live` adds this shell's RLIMIT_CORE and its environment.
# Not in --rest-gate mode: there `--live` reports the CALLING shell's limit as it stands.
[ "${1:-}" = --rest-gate ] || ulimit -c 0 2>/dev/null || true   # never fatal: the gate's RLIMIT_CORE check refuses

# >>> rest-gate: from here to the `<<< rest-gate` line, build-onstart.sh inlines this block (comment
# lines dropped) into the onstart's first-boot launcher, so that launcher runs this same gate.
# Bash builtins only (#1348): no grep/tr/cut/sort, so a missing or broken tool cannot turn a refusal into
# an empty answer; and a caller refuses the launch on ANY non-zero exit of the gate, not only on its output.
# Truthy as the fork's env readers take it: anything but empty, 0, false, no, off, none or -1 (any case).
rg_on() { case "$1" in ''|0|[Ff][Aa][Ll][Ss][Ee]|[Nn][Oo]|[Oo][Ff][Ff]|[Nn][Oo][Nn][Ee]|-1) return 1 ;; esac; return 0; }
# Loopback endpoint: scheme and path stripped, then 127.0.0.1, localhost or ::1 (bare or [::1]:port).
rg_loopback() {
  local h=${1#*://}; h=${h%%/*}
  case "$h" in
    \[*\]*) h=${h#\[}; h=${h%%\]*} ;;
    *:*:*) ;;
    *) h=${h%%:*} ;;
  esac
  case "$h" in 127.0.0.1|[Ll][Oo][Cc][Aa][Ll][Hh][Oo][Ss][Tt]|::1) return 0 ;; esac
  return 1
}
# Wildcard bind host (the fork's _BIND_WILDCARD_HOSTS): *, 0.0.0.0, ::.
rg_wild() { local h=${1#*://}; h=${h%:*}; case "$h" in '*'|0.0.0.0|::|'[::]') return 0 ;; esac; return 1; }
# rg_kv_pairs <json>: sets RG_KVP to the key=value lines (each after a newline) of a FLAT JSON object
# whose values are plain strings (no escapes, no newline), true/false/null or numbers -- every
# KVEventsConfig field is one -- and RG_KVDUP to the keys it repeats; exit 1 for anything else (nesting,
# an escape, trailing text, not an object at all). The gate fails CLOSED on a 1: it cannot prove where an
# unparsed config binds. Bash 3.2-safe (args/check.sh runs under /bin/bash).
rg_kv_pairs() {
  local s=$1 k v sep n=$'\n'
  RG_KVP=$n; RG_KVDUP=""
  local re_pair='^[[:space:]]*"([A-Za-z0-9_]+)"[[:space:]]*:[[:space:]]*("[^"\\]*"|true|false|null|-?[0-9]+(\.[0-9]+)?)[[:space:]]*(,|\})(.*)$'
  [[ "$s" =~ ^[[:space:]]*\{(.*)$ ]] || return 1
  s=${BASH_REMATCH[1]}
  [[ "$s" =~ ^[[:space:]]*\}[[:space:]]*$ ]] && return 0
  while [[ "$s" =~ $re_pair ]]; do
    k=${BASH_REMATCH[1]}; v=${BASH_REMATCH[2]}; sep=${BASH_REMATCH[4]}; s=${BASH_REMATCH[5]}
    v=${v#\"}; v=${v%\"}
    case "$v" in *"$n"*) return 1 ;; esac
    case "$RG_KVP" in *"$n$k="*) case " $RG_KVDUP" in *" $k "*) ;; *) RG_KVDUP+="$k " ;; esac ;; esac
    RG_KVP+=$k=$v$n
    if [ "$sep" = "}" ]; then [[ "$s" =~ ^[[:space:]]*$ ]]; return; fi
  done
  return 1
}
# rg_kv_get <key>: that key's first value in RG_KVP (exit 1 when absent).
rg_kv_get() { local s=$RG_KVP n=$'\n'; case "$s" in *"$n$1="*) s=${s#*"$n$1="}; printf '%s\n' "${s%%"$n"*}" ;; *) return 1 ;; esac; }
RG_ENDPOINTS=1
# rest_gate <label> <env lines (NAME=VALUE, one per line)> [<arg word>...]: prints one refusal per
# line, nothing when it passes; exit 0, or non-zero when it could not finish (#1348) -- which every caller
# refuses like a refusal line. The caller decides what a refusal does.
rest_gate() {
  local label=$1 envl=$2; shift 2
  local w=("$@") i n=$# a name val trace="" otlp="localhost:4317" line kv="" kvset="" pub="" kvep="" rp="" load="" RG_KVP RG_KVDUP
  for ((i = 0; i < n; i++)); do
    a=${w[i]}; name=${a%%=*}
    if [ "$name" != "$a" ]; then val=${a#*=}; else val=${w[i+1]:-}; fi
    case "$name" in
      --log-requests*) echo "$label: $name -- request logging" ;;
      --crash-dump-folder*) echo "$label: $name -- dumps requests" ;;
      --debug-tensor-dump*) echo "$label: $name -- dumps tensors" ;;
      --msprobe-dump-config*) echo "$label: $name -- dumps tensors" ;;
      --hicache-storage-backend)
        case "$val" in sim|shm) ;; *) echo "$label: $name $val -- only RAM (sim/shm)" ;; esac ;;
      --enable-trace) trace=1 ;;
      --otlp-traces-endpoint) otlp=$val ;;
      --kv-events-config) kv=$val; kvset=1 ;;
      --load-publish-endpoint) load=$val ;;
    esac
  done
  # KV events, failing CLOSED: a --kv-events-config the gate cannot parse is refused; a config with a
  # publisher other than "null" (the fork's default when the key is absent) and no "endpoint" binds the
  # fork's default, tcp://*:5557 -- a wildcard, refused like one written out.
  if [ -n "$RG_ENDPOINTS" ] && [ -n "$kvset" ]; then
    if ! rg_kv_pairs "$kv"; then
      echo "$label: --kv-events-config '$kv' -- not a flat JSON object the gate can read"
    elif [ -n "$RG_KVDUP" ]; then
      echo "$label: --kv-events-config repeats key(s) ${RG_KVDUP% }"
    else
      pub=$(rg_kv_get publisher || echo null)
      if [ "$pub" != null ]; then
        kvep=$(rg_kv_get endpoint || echo 'tcp://*:5557')
        rp=$(rg_kv_get replay_endpoint || true)
        rg_loopback "$kvep" || echo "$label: --kv-events-config endpoint $kvep -- off loopback (none = tcp://*:5557)"
        [ -z "$rp" ] || rg_loopback "$rp" || echo "$label: --kv-events-config replay_endpoint $rp -- off loopback"
        if rg_loopback "$kvep" && [ "$(rg_kv_get bind || true)" != true ]; then
          echo "$label: --kv-events-config endpoint $kvep without \"bind\":true -- it would connect, not bind"
        fi
      fi
    fi
  fi
  if [ -n "$RG_ENDPOINTS" ] && [ -n "$load" ]; then
    if [ "$load" = auto ]; then
      echo "$label: --load-publish-endpoint auto -- needs a wildcard KV endpoint; name tcp://*:5559"
    else
      rg_loopback "$load" || rg_wild "$load" || echo "$label: --load-publish-endpoint $load -- off loopback"
    fi
  fi
  [ -z "$trace" ] || rg_loopback "$otlp" || echo "$label: --enable-trace with --otlp-traces-endpoint $otlp -- off loopback"
  # The last command, so its status is the gate's: every arm below ends 0, and a here-string bash cannot
  # create (bash < 5.1 writes it to $TMPDIR) ends the loop 1 -- the env never read is a failed gate.
  while IFS= read -r line; do
    name=${line%%=*}; [ "$name" != "$line" ] || continue
    val=${line#*=}
    case "$name" in
      OTEL_EXPORTER_OTLP_ENDPOINT|OTEL_EXPORTER_OTLP_*_ENDPOINT)
        [ -z "$val" ] || rg_loopback "$val" || echo "$label: $name=$val -- off loopback" ;;
      SGLANG_LOG_REQUEST_HEADERS|CUDA_COREDUMP_FILE|SGLANG_CUDA_COREDUMP_DIR|SGLANG_KERNEL_API_DUMP_DIR)
        [ -z "$val" ] || echo "$label: $name=$val -- request logging or a dump" ;;
      SGLANG_PYSPY_DUMP_BEFORE_CRASH) ;;
      SGLANG_LOG_PICKLE_IPC_OBJECTS|SGLANG_CUDA_COREDUMP*|CUDA_ENABLE_*COREDUMP*|DUMPER_ENABLE|SGLANG_KERNEL_API_LOGLEVEL|SGLANG_*DUMP*|SGLANG_*LOG*REQ*|SGLANG_*REQ*LOG*)
        ! rg_on "$val" || echo "$label: $name=$val -- request logging or a dump" ;;
    esac
  done <<< "$envl"
}
# The env lines of this shell's exported environment (a newline inside a value folded to a space).
rg_env_now() { local v; for v in $(compgen -e); do printf '%s=%s\n' "$v" "${!v//$'\n'/ }"; done; }
rg_core() { local c; c=$(ulimit -c); [ "$c" = 0 ] || echo "${1:-launch}: RLIMIT_CORE is $c, not 0 -- a crash would write engine memory to the host's disk (ulimit -c 0)"; }
# <<< rest-gate

if [ "${1:-}" = --rest-gate ]; then
  # Fails CLOSED like the launch path (#1348): a file it cannot read, a gate that exits non-zero for any
  # reason, exit 2; builtins only here too (no grep/sed), so a missing tool cannot read as "nothing to
  # refuse".
  shift; RG_LIVE=""; RG_OUT=""; RG_N=0; RG_NL=$'\n'
  [ "${1:-}" != --live ] || { RG_LIVE=1; shift; }
  # rg_file <file>: RG_TXT = its lines that do not start with '#' (args/check.sh's `grep -v '^#'`).
  rg_file() {
    local l; RG_TXT=""
    [ -r "$1" ] || { echo "rest-gate: cannot read $1" >&2; exit 2; }
    { while IFS= read -r l || [ -n "$l" ]; do case "$l" in '#'*) ;; *) RG_TXT+=$l$RG_NL ;; esac; done < "$1"; } \
      || { echo "rest-gate: cannot read $1" >&2; exit 2; }
  }
  # rg_run <label> <command...>: its refusal lines into RG_OUT; a non-zero exit stops here, exit 2.
  rg_run() {
    local o l=$1; shift
    o=$("$@") || { echo "rest-gate: the gate failed on $l (exit $?) -- nothing it checks can be taken as clean" >&2; exit 2; }
    [ -z "$o" ] || RG_OUT+=$o$RG_NL
  }
  while [ $# -gt 0 ]; do
    case "$1:${2:-}" in
      --args:?*)
        rg_file "$2"; eval "A=($RG_TXT)" || { echo "rest-gate: $2 does not parse" >&2; exit 2; }
        rg_run "$2" rest_gate "$2" "" ${A[@]+"${A[@]}"} ;;
      --legacy-args:?*)
        rg_file "$2"; eval "A=($RG_TXT)" || { echo "rest-gate: $2 does not parse" >&2; exit 2; }
        RG_ENDPOINTS=""; rg_run "$2" rest_gate "$2" "" ${A[@]+"${A[@]}"}; RG_ENDPOINTS=1 ;;
      --env:?*)
        rg_file "$2"; rg_run "$2" rest_gate "$2" "$RG_TXT" ;;
      *) echo "usage: relaunch.sh --rest-gate [--live] [--args <file>]... [--legacy-args <file>]... [--env <file>]..." >&2; exit 2 ;;
    esac
    RG_N=$((RG_N + 1)); shift 2
  done
  if [ -n "$RG_LIVE" ]; then RG_E=$(rg_env_now) || { echo "rest-gate: cannot list the environment" >&2; exit 2; }; rg_run environment rest_gate environment "$RG_E"; rg_run RLIMIT_CORE rg_core; fi
  [ "$RG_N" -gt 0 ] || [ -n "$RG_LIVE" ] || { echo "usage: relaunch.sh --rest-gate [--live] [--args <file>]... [--legacy-args <file>]... [--env <file>]..." >&2; exit 2; }
  [ -z "$RG_OUT" ] || { IFS=$RG_NL; set -f; set -- $RG_OUT; printf 'REFUSED (no prod traffic at rest): %s\n' "$@" >&2; exit 1; }
  exit 0
fi

. "$(dirname "$0")/lib.sh"

RL_WATCHDOG_ONLY=""
if [ "${1:-}" = --watchdog ]; then RL_WATCHDOG_ONLY=1; shift; fi
RL_ID="${1:-}"
[ $# -le 1 ] || die "usage: relaunch.sh [--watchdog] [<engine id>]"
[ -z "$RL_ID" ] || engine_select "$RL_ID"

# No secret rides a watchdog's environ (#3003): unset here, ahead of every watchdog start below (the
# launch path's and `--watchdog`'s) -- the key is read from $ENVIRONMENT_FILE, never from this env.
# The launch path unsets again after its sglang-env.txt loop, for the engine's exec.
unset VLLM_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN CF_DNS_TOKEN

# ------------------------------------------------------------------------------------------ watchdog
# Exactly one watchdog.sh stays alive across boots and relaunches; it relaunches this script when the
# server dies or stops answering. A launch only STARTS one when none runs, so a watchdog.sh change
# staged onto a box whose watchdog is up never took effect (the boot grace/stall/ceiling change sat
# staged behind the old daemon): bash keeps reading the inode it opened, and rsync's rename leaves
# that inode to it. So every start writes a run record, $WATCHDOG_RUN -- the pid, its start time
# (`ps -o lstart=`, so a pid reused after a container restart is not mistaken for the one recorded),
# the md5 of the watchdog.sh it was started from, and the md5 of every file that script sources at
# start: the box file (`box=`, $BOX_ENV: its WATCHDOG_BOOT_GRACE/STALL/MAX, prewarm and ENGINES are
# read once, by lib.sh, when the daemon starts) and lib.sh itself (`lib=`, $TOOLS_REMOTE/lib.sh, the
# `. "$(dirname "$0")/lib.sh"` in watchdog.sh: engine_select, the per-engine names, note/die) --
# and `--watchdog` replaces a running watchdog whose script, box file or lib.sh is not the staged one.
# Before #827 the record had no `box=`, so a box-file-only change (#825's WATCHDOG_BOOT_STALL=1800)
# sat behind a watchdog that answered `kept`; before #863 it had no `lib=`, so a lib.sh-only change did.
# Nothing else is sourced: relaunch.sh and caddy-relaunch.sh are run afresh by each relaunch.
if [ -n "$RL_ID" ]; then
  # engine_select set WATCHDOG_RUN, WATCHDOG_HOLD and WATCHDOG_LOG to this engine's.
  WATCHDOG_LOG_FILE="${RELAUNCH_TEST_WATCHDOG_LOG:-$WATCHDOG_LOG}"   # the override: tests only
  WD_HOLD="$WATCHDOG_HOLD"
  WD_PATTERN="$TOOLS_REMOTE/watchdo[g].sh $RL_ID( |\$)"
else
  WATCHDOG_RUN="$TOOLS_REMOTE/watchdog.run"
  WATCHDOG_LOG_FILE="${RELAUNCH_TEST_WATCHDOG_LOG:-/workspace/$BOX-watchdog.log}"   # the override: tests/watchdog.test.sh only
  WD_HOLD="$TOOLS_REMOTE/watchdog.hold"
  # The id-less watchdog only (its argv ends in watchdog.sh, or a --flag follows): an engine's
  # `watchdog.sh <id>` is never this singleton's to count, stop or replace.
  WD_PATTERN="$TOOLS_REMOTE/watchdo[g].sh( --|\$)"
fi

wd_md5() { { md5sum "$1" 2>/dev/null || md5 -q "$1" 2>/dev/null || true; } | awk 'NR == 1 { print $1 }'; }
# wd_pids_all: every process whose command line names the staged watchdog.sh. Self-safe pgrep pattern
# (bracketed character): no caller's own command line ever self-matches.
wd_pids_all() { pgrep -f "$WD_PATTERN" 2>/dev/null || true; }
wd_lstart() { ps -o lstart= -p "$1" 2>/dev/null | sed 's/^ *//; s/ *$//' || true; }
wd_ppid() { ps -o ppid= -p "$1" 2>/dev/null | tr -d ' ' || true; }
# wd_pids: the watchdogs THEMSELVES -- a match whose parent is also a match is a subshell of a
# watchdog, never a second one: bash forks a copy with the same argv for a `( ... ) &`, a pipeline
# inside `$(...)`, and for the instant before its relaunch's `nohup` execs, so counting it would read
# one watchdog as two and replace it on every stage.
wd_pids() {
  local all p pp
  all=" $(wd_pids_all | tr '\n' ' ') "
  for p in $all; do
    pp=$(wd_ppid "$p")
    case "$all" in *" ${pp:-none} "*) ;; *) echo "$p" ;; esac
  done
}
wd_run_get() { sed -n "s/^$1=//p" "$WATCHDOG_RUN" 2>/dev/null | head -1 || true; }

# DOUBLE-FORKED (the subshell exits at once, so the watchdog is re-parented to PID 1): this script
# execs into `sglang serve`, and SGLang's crash handler kills its whole process tree -- a watchdog
# forked directly from here would die with the engine on the first OOM and nothing would relaunch.
# `$!` is the watchdog's own pid: nohup and setsid (never a group leader here, so it does not fork)
# both exec. The record is best effort -- a write that fails never stops the launch; a watchdog with
# no record reads as stale, and the next stage replaces it once.
wd_start() {
  local md5 box lib
  md5=$(wd_md5 "$TOOLS_REMOTE/watchdog.sh")
  box=$(wd_md5 "$BOX_ENV")
  lib=$(wd_md5 "$TOOLS_REMOTE/lib.sh")
  ( nohup setsid bash "$TOOLS_REMOTE/watchdog.sh" ${RL_ID:+"$RL_ID"} >> "$WATCHDOG_LOG_FILE" 2>&1 < /dev/null &
    pid=$!
    { printf 'pid=%s\nlstart=%s\nmd5=%s\nbox=%s\nlib=%s\nstarted=%s\n' "$pid" "$(wd_lstart "$pid")" "$md5" "$box" "$lib" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$WATCHDOG_RUN.tmp.$$" \
        && mv -f "$WATCHDOG_RUN.tmp.$$" "$WATCHDOG_RUN"; } 2>/dev/null || rm -f "$WATCHDOG_RUN.tmp.$$" )
}

# wd_recorded <pid>: the run record names this pid AND its start time.
wd_recorded() { [ "$(wd_run_get pid)" = "$1" ] && [ -n "$(wd_run_get lstart)" ] && [ "$(wd_run_get lstart)" = "$(wd_lstart "$1")" ]; }
# wd_ran_md5 <pid> -> the md5 of the watchdog.sh that pid was started from, or "" when unknown: the
# run record when it names this pid; else the script bash itself holds open on fd 255 (Linux /proc; a
# watchdog started by the onstart, which writes no record); else unknown.
wd_ran_md5() {
  local pid="$1"
  if wd_recorded "$pid"; then wd_run_get md5; return 0; fi
  case "$(readlink "/proc/$pid/fd/255" 2>/dev/null || true)" in
    *watchdog.sh*) wd_md5 "/proc/$pid/fd/255" ;;
  esac
}
# wd_ran_box <pid> -> the md5 of the box file that pid read at start, or "" when unknown: only the run
# record knows it (a box file is sourced and closed, so no fd holds it), so an unrecorded watchdog, or
# one recorded before `box=` existed, reads unknown and is replaced once, by the next stage.
wd_ran_box() { if wd_recorded "$1"; then wd_run_get box; fi; }
# wd_ran_lib <pid> -> the md5 of the lib.sh that pid sourced at start, or "" when unknown: the run
# record only, as for the box file; a record from before `lib=` existed reads unknown, replaced once.
wd_ran_lib() { if wd_recorded "$1"; then wd_run_get lib; fi; }

# watchdog_refresh (`relaunch.sh --watchdog`, run by stage.sh after every stage): replace a running
# watchdog that is not running the staged watchdog.sh on the staged box file and lib.sh, and never
# touch the engine.
#   * none running -> nothing started: a watchdog with no engine would relaunch one, and a stage
#     never starts the engine (relaunch.sh starts the watchdog with the engine).
#   * exactly one, running the staged script's md5 and started on the staged box file's and lib.sh's
#     md5 -> kept, so a re-stage that changed nothing keeps the daemon's state (its boot clock, its
#     miss counters).
#   * otherwise (a different md5 of any of the three, an unknown one, or more than one watchdog; a
#     box-file mismatch reads `<pid>:box:<md5>` in the notes, a lib.sh one `<pid>:lib:<md5>`) -> every
#     one is stopped
#     (TERM, KILL after 10 s) and one fresh watchdog is started on the staged script, recorded --
#     UNLESS $TOOLS_REMOTE/watchdog.hold exists: a restart/cutover/long run holds the box, and a
#     daemon swapped under it would start without that run's HOLD_MAX. Then nothing is replaced and
#     the stage says so; re-stage once the hold clears (the watchdog drops a stale one itself).
# A fresh watchdog starts with LAST_LAUNCH=now and BOOTED=0, so a serving engine is BOOTED at its
# first /health 200, and an engine mid-boot just gets a fresh boot grace. Its environ carries no key
# (#3003): the unset above.
watchdog_refresh() {
  local pids all staged sbox slib n pid ran rbox rlib stale="" alive
  [ -f "$TOOLS_REMOTE/watchdog.sh" ] || { note "watchdog: no $TOOLS_REMOTE/watchdog.sh staged -- nothing to refresh"; return 0; }
  pids=$(wd_pids)
  [ -n "$pids" ] || { note "watchdog: none running -- nothing replaced (relaunch.sh starts one with the engine)"; return 0; }
  staged=$(wd_md5 "$TOOLS_REMOTE/watchdog.sh")
  [ -n "$staged" ] || die "watchdog: cannot md5 $TOOLS_REMOTE/watchdog.sh"
  sbox=$(wd_md5 "$BOX_ENV")
  [ -n "$sbox" ] || die "watchdog: cannot md5 the box file $BOX_ENV"
  slib=$(wd_md5 "$TOOLS_REMOTE/lib.sh")
  [ -n "$slib" ] || die "watchdog: cannot md5 $TOOLS_REMOTE/lib.sh"
  n=$(printf '%s\n' "$pids" | wc -l | tr -d ' ')
  for pid in $pids; do
    ran=$(wd_ran_md5 "$pid")
    if [ "$ran" != "$staged" ]; then stale="$stale $pid:${ran:-unknown}"; continue; fi
    rbox=$(wd_ran_box "$pid")
    [ "$rbox" = "$sbox" ] || stale="$stale $pid:box:${rbox:-unknown}"
    rlib=$(wd_ran_lib "$pid")
    [ "$rlib" = "$slib" ] || stale="$stale $pid:lib:${rlib:-unknown}"
  done
  pids=$(printf '%s' "$pids" | tr '\n' ' ')
  if [ "$n" = 1 ] && [ -z "$stale" ]; then
    note "watchdog: pid $pids runs the staged watchdog.sh (md5 $staged) on the staged box file (md5 $sbox) and lib.sh (md5 $slib) -- kept"; return 0
  fi
  if [ -e "$WD_HOLD" ]; then
    note "watchdog: HELD -- $WD_HOLD exists, so watchdog pid(s) ${pids% } (ran:$stale; $n running; staged md5 $staged, box $sbox, lib $slib) are NOT replaced; re-stage once the hold clears"
    return 0
  fi
  note "watchdog: replacing pid(s) ${pids% } (ran:$stale; $n running) with one on the staged watchdog.sh (md5 $staged), box file $BOX_ENV (md5 $sbox) and lib.sh (md5 $slib) -- the engine is not touched"
  # Every match goes, the watchdogs' own subshells too: one outlives its parent's TERM otherwise.
  all=$(wd_pids_all | tr '\n' ' ')
  # shellcheck disable=SC2086  # a pid list
  kill $pids $all 2>/dev/null || true
  # Gone = no longer matched by the pattern (never `kill -0`: a zombie answers it until PID 1 reaps).
  for _ in $(seq 1 20); do
    alive=$(wd_pids_all | tr '\n' ' ')
    [ -n "$alive" ] || break
    sleep 0.5
  done
  if [ -n "$alive" ]; then
    note "watchdog: pid(s) ${alive% } ignored TERM for 10 s -- KILL"
    # shellcheck disable=SC2086
    kill -KILL $alive 2>/dev/null || true; sleep 1
  fi
  wd_start
  note "watchdog: started pid $(wd_run_get pid) on md5 $(wd_run_get md5), box $(wd_run_get box), lib $(wd_run_get lib) (record $WATCHDOG_RUN)"
}

if [ -n "$RL_WATCHDOG_ONLY" ]; then
  watchdog_refresh
  exit 0
fi

if [ -n "$RL_ID" ]; then
  ARGS_FILE="$ENGINE_ARGS_FILE"; ENV_TXT="$ENGINE_ENV_FILE"
else
  ARGS_FILE="$TOOLS_REMOTE/sglang-args.txt"; ENV_TXT="$TOOLS_REMOTE/sglang-env.txt"
fi
[ -s "$ARGS_FILE" ] || die "no args file (or empty): $ARGS_FILE"
A=$(grep -v '^#' "$ARGS_FILE" || true)   # `|| true`: a comment-only file must reach the die below, not abort here under -e
[ -n "$A" ] || die "$ARGS_FILE has no flags after stripping comments"

ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-/etc/environment}"   # overridable for tests/relaunch-api-key.test.sh only
KEY=$(grep -a '^VLLM_API_KEY=' "$ENVIRONMENT_FILE" | head -1 | cut -d= -f2- | tr -d '"' || true)
[ -n "$KEY" ] || die "no VLLM_API_KEY in $ENVIRONMENT_FILE"

# The key file the composed line's `--config` names (header above). A key carrying any character
# outside this set is refused rather than escaped: the file is a YAML double-quoted scalar, and every
# bearer this tool set mints (`openssl rand -hex 16`) is hex, so no escaping rule is ever needed.
API_KEY_FILE="${API_KEY_FILE:-/etc/sglang-api-key.yaml}"
case "$KEY" in
  *[!A-Za-z0-9._~+/=-]*) die "VLLM_API_KEY in $ENVIRONMENT_FILE carries a character outside [A-Za-z0-9._~+/=-]; refusing to write it into $API_KEY_FILE" ;;
esac
rm -f "$API_KEY_FILE.tmp.$$"
( umask 077; printf 'api-key: "%s"\n' "$KEY" > "$API_KEY_FILE.tmp.$$" ) || die "cannot write $API_KEY_FILE.tmp.$$"
mv -f "$API_KEY_FILE.tmp.$$" "$API_KEY_FILE" || die "cannot move the key file into $API_KEY_FILE"
chmod 600 "$API_KEY_FILE" || die "cannot chmod 600 $API_KEY_FILE"

command -v sglang >/dev/null 2>&1 || export PATH="$VENV/bin:$PATH"
command -v sglang >/dev/null 2>&1 || die "sglang not on PATH ($VENV/bin)"

export XDG_CACHE_HOME="$CACHE" HF_HUB_OFFLINE=1 SGLANG_DISABLE_USAGE_STATS=1 DO_NOT_TRACK=1
export PATH="$VENV/bin:$PATH"

if [ -n "$RL_ID" ]; then
  [ -f "$ENV_TXT" ] || die "engine $RL_ID: no $ENV_TXT -- restart.sh --engine $RL_ID installs it (with sglang-args-$RL_ID.txt); restore its .pre-<tag> backup, or run restart.sh, before relaunching"
fi
[ -f "$ENV_TXT" ] || die "no $TOOLS_REMOTE/sglang-env.txt -- stage.sh seeds it from the env of SGLANG_TREE=${SGLANG_TREE:-<unset>}'s serve line (tools/args/env-box-d-official.txt on box-d-up, tools/args/env-box-e.txt on box-e-up, tools/args/env-v3.txt otherwise) the first time it stages a box; run stage.sh, or restore its .pre-<tag> backup, before relaunching"
while IFS= read -r line; do
  case "$line" in ''|'#'*) ;; *) export "$line" ;; esac
done < "$ENV_TXT"
# An engine's GPUs: exported AFTER the env file's loop, so no line there can move it off its own.
[ -z "$RL_ID" ] || export CUDA_VISIBLE_DEVICES="$ENGINE_GPUS"

# The args file may carry single-quoted JSON; eval re-quotes it. The newline before `)` is the one
# `$(grep ...)` stripped: without it a trailing ` # ...` comment on the last kept line would comment
# out the `)` too (tests/relaunch-args-comments.test.sh).
eval "A=($A"$'\n'")"

# A DP-attention A/B arm: its DP rank count, the fork's num_dp_ranks (--attn-dp-size or
# --attention-data-parallel-size, times --dp-size or --data-parallel-size; the last of each wins, as
# argparse's does). Above 1 each rank publishes on its own KV ports, which kv_base..+2 cannot hold:
# lib.sh's engine_kv_ranks moves the engine onto its DP area (lib.sh "engines"), and refuses the
# legacy launch. No config of record carries a DP width (args/check.sh FORBIDDEN_FLAGS); at 1 nothing
# here changes a value. SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1 with a DP width is refused: the
# coordination fix is upstream sglang #46676, unmerged, and not in the fork.
kv_attn=1 kv_dp=1
for i in "${!A[@]}"; do
  a="${A[$i]}"; v="${A[$((i + 1))]:-}"
  case "$a" in --*=*) v="${a#*=}"; a="${a%%=*}" ;; esac
  case "$a" in
    --attn-dp-size|--attention-data-parallel-size) kv_attn="$v" ;;
    --dp-size|--data-parallel-size) kv_dp="$v" ;;
    *) continue ;;
  esac
  case "$v" in ''|*[!0-9]*|0*) die "$ARGS_FILE: $a '$v' -- not a positive integer" ;; esac
done
kv_ranks=$((kv_attn * kv_dp))
if [ "$kv_ranks" -gt 1 ]; then
  ! rg_on "${SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION:-}" || die "$ARGS_FILE: a DP width of $kv_ranks ranks with SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=$SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION in $ENV_TXT -- never paired (its fix is sglang #46676, unmerged); drop the variable"
  [ -n "$RL_ID" ] || die "$ARGS_FILE: a DP width of $kv_ranks ranks boots per engine only (relaunch.sh <id> on a box with ENGINES) -- the legacy launch has no per-rank KV layout: rank 1 would bind 5558, replay rank 0's port"
  engine_kv_ranks "$kv_ranks"
fi

# An engine of a per-engine box: its kv-event endpoints moved to its own ports, its own NCCL port, and
# no launch onto a port something already holds (a copy that did would bind-fail deep in its boot, or
# worse, publish into the other copy's subscriber).
if [ -n "$RL_ID" ]; then
  kv_re='tcp://([^"'"'"'[:space:],:/]*):([0-9]+)'
  for i in "${!A[@]}"; do
    case "${A[$i]}" in
      --nccl-port|--nccl-port=*) die "engine $RL_ID: $ARGS_FILE carries its own --nccl-port -- relaunch.sh appends --nccl-port $ENGINE_NCCL_PORT per engine; drop it from the args file" ;;
    esac
    w="${A[$i]}"; out=""
    while [[ "$w" =~ $kv_re ]]; do
      m="${BASH_REMATCH[0]}"; host="${BASH_REMATCH[1]}"; port="${BASH_REMATCH[2]}"
      case "$port" in
        5557|5558)
          rg_loopback "tcp://$host:$port" || die "engine $RL_ID: $ARGS_FILE names $m -- KV events and replay carry token ids and bind on loopback only (tcp://127.0.0.1:$port, \"bind\":true; #1300)"
          if [ "$port" = 5557 ]; then port="$KV_PORT"; else port="$KV_REPLAY_PORT"; fi ;;
        5559)
          rg_loopback "tcp://$host:$port" || rg_wild "tcp://$host:$port" || die "engine $RL_ID: $ARGS_FILE names $m -- the load feed binds on loopback or the wildcard only (#1300)"
          port="$KV_LOAD_PORT" ;;
        *) die "engine $RL_ID: $ARGS_FILE names kv port $port ($m) -- only tcp://127.0.0.1:5557 (events), tcp://127.0.0.1:5558 (replay) and tcp://*:5559 (load) are rewritten per engine; any other would collide between the copies" ;;
      esac
      out+="${w%%"$m"*}tcp://$host:$port"; w="${w#*"$m"}"
    done
    A[$i]="$out$w"
  done
  A+=(--nccl-port "$ENGINE_NCCL_PORT")
  # The ports already listening, one decimal port per line: `ss -ltn` when the box has it (iproute2 is
  # not in every image -- a recreated container lacks it), else the kernel's own tables, /proc/net/tcp
  # and tcp6 (st 0A = LISTEN, the local port in hex after the last colon). Neither answering is a
  # refusal, as a failed ss always was. RELAUNCH_PROC_NET moves /proc/net for the tests only.
  proc_net=${RELAUNCH_PROC_NET:-/proc/net}
  if command -v ss > /dev/null 2>&1 && listening=$(ss -ltn 2> /dev/null); then
    listening=$(printf '%s\n' "$listening" | awk 'NR > 1 { n = split($4, a, ":"); print a[n] }')
  elif [ -r "$proc_net/tcp" ]; then
    tables=("$proc_net/tcp"); [ -r "$proc_net/tcp6" ] && tables+=("$proc_net/tcp6")
    listening=""
    for h in $(awk 'FNR > 1 && $4 == "0A" { n = split($2, a, ":"); print a[n] }' "${tables[@]}"); do
      case "$h" in ''|*[!0-9A-Fa-f]*) continue ;; esac
      listening+="$((16#$h))"$'\n'
    done
  elif command -v ss > /dev/null 2>&1; then
    die "engine $RL_ID: ss -ltn failed and $proc_net/tcp is not readable -- cannot check its ports are free"
  else
    die "engine $RL_ID: no ss on PATH (iproute2) and $proc_net/tcp is not readable -- cannot check its ports are free"
  fi
  bound=""
  for v in ENGINE_PORT ENGINE_NCCL_PORT KV_PORT KV_REPLAY_PORT KV_LOAD_PORT; do
    printf '%s\n' "$listening" | grep -qx -- "${!v}" && bound+=" $v=${!v}"
    case "$v" in KV_*) ;; *) continue ;; esac
    for ((r = 1; r < KV_RANKS; r++)); do   # a DP arm: every rank's port of the range
      printf '%s\n' "$listening" | grep -qx -- "$((${!v} + r))" && bound+=" $v+$r=$((${!v} + r))"
    done
  done
  [ -z "$bound" ] || die "engine $RL_ID: port(s) already listening:$bound -- another process (the other copy, a stale engine) holds them; stop it first"
  # This pid, after the exec, is the engine; under `nohup setsid` it leads the engine's process group.
  printf '%s\n' "$$" > "$ENGINE_PIDFILE.tmp.$$" && mv -f "$ENGINE_PIDFILE.tmp.$$" "$ENGINE_PIDFILE"
fi

# No key in the environ the watchdog and the engine inherit (#3003). This script's own caller exports
# VLLM_API_KEY more often than not -- the onstart's container env carries the create-time `-e
# VLLM_API_KEY`, and a restart.sh/cutover.sh ssh session gets /etc/environment through sshd's
# pam_env -- and /proc/<pid>/environ of the engine would hold it for the engine's whole life. Nothing
# below needs it: the engine reads the key from $API_KEY_FILE (written above, before this line), the
# watchdog reads /etc/environment itself, and sglang reads no key variable from its environment.
# Unset AFTER the sglang-env.txt loop, so a stray key line there cannot put one back. HF_TOKEN and
# CF_DNS_TOKEN go with it: the engine serves HF_HUB_OFFLINE=1 from a local path, and no box holds the
# Cloudflare token any more (#1306; this unset only keeps a stray one out of the engine). `KEY` is this
# shell's own, never exported; it goes too.
unset VLLM_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN CF_DNS_TOKEN KEY

# The deploy gate (#1300, the header's "No production traffic at rest"): the composed args, the
# engine's whole environment and this shell's RLIMIT_CORE, ahead of the note, the watchdog, the warm
# gate and the exec. A refusal names each flag/variable and stops here: no engine starts. So does a gate
# that exits non-zero for ANY reason (#1348: an internal error, a crash, a here-string bash could not
# write): its status is checked, never piped away -- a gate that could not finish proved nothing clean.
RG_RC=0; RG_OUT=$(RG_E=$(rg_env_now) && rest_gate "${ARGS_FILE:-args}" "$RG_E" "${A[@]}" && rg_core) || RG_RC=$?
[ "$RG_RC" = 0 ] || RG_OUT="gate exit $RG_RC: it could not finish, nothing is proved clean $RG_OUT"
[ -z "$RG_OUT" ] || die "REFUSED (no prod traffic at rest) -- no engine started:
$RG_OUT"

note "sglang serve box=$BOX tree=$(git -C "$SGLANG_TREE" rev-parse --short=12 HEAD 2>/dev/null || echo n/a) env=$( [ -f "$ENV_TXT" ] && grep -v '^#' "$ENV_TXT" | cut -d= -f1 | tr '\n' ',' || echo none)${RL_ID:+ engine=$RL_ID gpus=$ENGINE_GPUS port=$ENGINE_PORT nccl=$ENGINE_NCCL_PORT kv=$KV_PORT/$KV_REPLAY_PORT/$KV_LOAD_PORT${KV_RANKS:+$( [ "$KV_RANKS" = 1 ] || echo " kv_ranks=$KV_RANKS (each base +0..$((KV_RANKS - 1)))")} pid=$$}"

# Watchdog: started (and recorded) when none runs -- see wd_start above. One already running is left
# alone here: it may be the very watchdog that ran this relaunch. stage.sh's `--watchdog` is what
# replaces one running a superseded script.
if [ -f "$TOOLS_REMOTE/watchdog.sh" ] && [ -z "$(wd_pids)" ]; then
  wd_start
fi

# Warm gate (#379): EVERY engine boot -- this launch path is what the onstart, the watchdog's
# relaunch() and restart.sh/cutover.sh all run -- closes the front door until a short in-process warm
# (warm-short.py: one cold ~60k fill, three ~700-token extends, one image turn) has run on the fresh
# process. The tree marker keeps the compile sweep (prewarm-box-d.py) once per tree; this is not that:
# a new process pays in-memory first-use costs (Triton autotune, cuBLASLt heuristics, pinned staging)
# on its first real requests whatever the disk caches hold. $WARM_GATE present = door CLOSED (tls.sh's
# Caddy @warming matcher answers 503, so the router sees no READY backend); absent = open. It is
# written here naming THIS pid (the engine's, after the exec) with owner=relaunch; warm-short.py,
# double-forked like the watchdog (SGLang's crash handler kills its whole process tree), takes it
# over, waits for the engine's /health 200, warms, and removes it -- or removes it at its timeout
# (WARM_TIMEOUT, default 180 s) with a WARNING. Fail-open: no warm-short.py staged, or a flag that
# cannot be written, leaves the door open; watchdog.sh clears a flag whose owner is gone. The key is
# never on its argv: it reads VLLM_API_KEY from $ENVIRONMENT_FILE, as this script did above.
if [ -n "$RL_ID" ]; then
  WARM_GATE="$ENGINE_WARMING"
  WARM_LOG_FILE="${RELAUNCH_TEST_WARM_LOG:-$WARM_LOG}"
else
  WARM_GATE="$TOOLS_REMOTE/engine.warming"
  WARM_LOG_FILE="${RELAUNCH_TEST_WARM_LOG:-/workspace/$BOX-warm.log}"   # the override: tests/warm-short.test.sh only
fi
if [ -f "$TOOLS_REMOTE/warm-short.py" ] \
   && { printf 'pid=%s\nowner=relaunch\nstarted=%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$WARM_GATE.tmp.$$" \
        && mv -f "$WARM_GATE.tmp.$$" "$WARM_GATE"; } 2>/dev/null; then
  ( nohup setsid python3 "$TOOLS_REMOTE/warm-short.py" --gate "$WARM_GATE" --engine-pid "$$" \
      --base-url "http://127.0.0.1:$ENGINE_PORT" --environment-file "$ENVIRONMENT_FILE" \
      --timeout "${WARM_TIMEOUT:-180}" >> "$WARM_LOG_FILE" 2>&1 < /dev/null & )
  note "warm gate CLOSED ($WARM_GATE): warm-short.py runs after /health, log $WARM_LOG_FILE"
else
  rm -f "$WARM_GATE" "$WARM_GATE.tmp.$$" 2>/dev/null || true
  note "WARNING: warm gate left OPEN -- $TOOLS_REMOTE/warm-short.py not staged or $WARM_GATE not writable; this boot skips the short warm"
fi

# Scheduler CPU pinning (#636), optional: box-file key SCHED_CPU_PIN (the legacy launch) or
# SCHED_CPU_PIN_<id> (an engine of a box's ENGINES -- box-c's two copies share one box's cores, so each
# names its own; the plain key is never read for an engine). One CPU number per TP rank, in rank
# order, space- or comma-separated: SCHED_CPU_PIN="2 4 6 8" pins TP0 to CPU 2, TP1 to 4, TP2 to 6,
# TP3 to 8. Unset = no pinner, no taskset call, the boot exactly as before. Measured on box-a (#385,
# qa/box-a-ship1-20261027/README.md "CPU"): every thread of each scheduler on its own core took C1
# from 6.51-6.96 to 6.00-6.36 ms/step; the main thread alone did nothing, a 4-core set per rank was
# worse -- so it is `taskset -a` (all threads), one core per rank.
# A pinner, double-forked like the warm gate but left in this launch's process group (a stop of the
# engine's group takes a pinner still waiting with it), waits for the engine's /health 200 (every
# SCHED_CPU_PIN_POLL s, default 10; gives up after SCHED_CPU_PIN_TIMEOUT s, default the watchdog's
# boot ceiling WATCHDOG_BOOT_MAX or 5400, or once this pid is gone), then finds the schedulers by
# PROCESS TITLE among THIS engine's descendants -- `sglang::scheduler_TP<n>` (with _DP/_EP/_PP parts
# around it; a bare `sglang::scheduler` is TP0) -- never by pid order, and never box-wide (the other
# copy's schedulers carry the same titles), and runs `taskset -a -pc <core> <pid>` per rank, logging
# rank, pid and core to $SCHED_PIN_LOG_FILE. Never fatal: a bad list, a missing scheduler, a
# failed taskset or no /health is a WARNING line, and the engine serves unpinned. Threads started
# after the pin inherit their creator's single-core mask, so the pin holds for the process's life.
if [ -n "$RL_ID" ]; then SCHED_PIN_VAR="SCHED_CPU_PIN_$RL_ID"; else SCHED_PIN_VAR=SCHED_CPU_PIN; fi
SCHED_PIN="${!SCHED_PIN_VAR:-}"
if [ -n "$RL_ID" ] && [ -z "$SCHED_PIN" ] && [ -n "${SCHED_CPU_PIN:-}" ]; then
  note "scheduler pinning: SCHED_CPU_PIN is the legacy launch's; engine $RL_ID reads $SCHED_PIN_VAR, which is unset -- this boot pins nothing"
fi
if [ -n "$SCHED_PIN" ]; then
  SCHED_PIN_LOG_FILE="${RELAUNCH_TEST_SCHED_PIN_LOG:-/workspace/$BOX${RL_ID:+-$RL_ID}-schedpin.log}"   # the override: tests only
  read -r -a SCHED_PIN_CORES <<< "${SCHED_PIN//,/ }"
  sp_bad=""
  [ "${#SCHED_PIN_CORES[@]}" = 0 ] || for c in "${SCHED_PIN_CORES[@]}"; do case "$c" in *[!0-9]*) sp_bad+=" '$c'" ;; esac; done
  sched_pin_log() { printf '[%s %s] sched pin: %s\n' "$BOX${RL_ID:+-$RL_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
  # sched_pin <engine pid>: the pinner (runs in the background, output to $SCHED_PIN_LOG_FILE).
  sched_pin() {
    local epid="$1" t0 code scheds i c pid line pinned=0 n=${#SCHED_PIN_CORES[@]}
    local poll="${SCHED_CPU_PIN_POLL:-10}" timeout="${SCHED_CPU_PIN_TIMEOUT:-${WATCHDOG_BOOT_MAX:-5400}}"
    sched_pin_log "engine pid $epid: waiting for /health 200 on 127.0.0.1:$ENGINE_PORT, then TP0..TP$((n - 1)) -> CPU ${SCHED_PIN_CORES[*]} ($SCHED_PIN_VAR)"
    t0=$(date +%s)
    while :; do
      code=$(curl -s -A "curl/8.7.1" -m 30 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$ENGINE_PORT/health" 2> /dev/null || true)
      [ "$code" = 200 ] && break
      if ! kill -0 "$epid" 2> /dev/null; then sched_pin_log "done: WARNING engine pid $epid exited before /health 200 -- nothing pinned"; return 0; fi
      if [ $(($(date +%s) - t0)) -ge "$timeout" ]; then sched_pin_log "done: WARNING no /health 200 in ${timeout}s -- nothing pinned"; return 0; fi
      sleep "$poll"
    done
    command -v taskset > /dev/null 2>&1 || { sched_pin_log "done: WARNING no taskset on PATH (util-linux) -- nothing pinned"; return 0; }
    # "<pid> <rank>" for each sglang::scheduler process that descends from the engine's pid.
    scheds=$(ps -e -o pid=,ppid=,args= 2> /dev/null | awk -v root="$epid" '
      { pid[NR] = $1; ppid[NR] = $2; t = $0; sub(/^[ \t]*[0-9]+[ \t]+[0-9]+[ \t]+/, "", t); args[NR] = t }
      END {
        mine[root] = 1
        do { grew = 0; for (i = 1; i <= NR; i++) if ((ppid[i] in mine) && !(pid[i] in mine)) { mine[pid[i]] = 1; grew = 1 } } while (grew)
        for (i = 1; i <= NR; i++) {
          if (!(pid[i] in mine) || pid[i] == root || args[i] !~ /^sglang::scheduler([_ ]|$)/) continue
          split(args[i], w, " "); r = 0
          if (match(w[1], /_TP[0-9]+/)) r = substr(w[1], RSTART + 3, RLENGTH - 3) + 0
          print pid[i], r
        }
      }' || true)
    for i in "${!SCHED_PIN_CORES[@]}"; do
      c="${SCHED_PIN_CORES[$i]}"
      pid=$(printf '%s\n' "$scheds" | awk -v r="$i" '$2 == r { print $1; exit }')
      if [ -z "$pid" ]; then sched_pin_log "WARNING no sglang::scheduler TP$i process under engine pid $epid -- CPU $c not applied"; continue; fi
      if line=$(taskset -a -pc "$c" "$pid" 2>&1); then
        sched_pin_log "TP$i pid $pid -> CPU $c (taskset -a: every thread)"; pinned=$((pinned + 1))
      else
        sched_pin_log "WARNING taskset -a -pc $c $pid (TP$i) failed: $(printf '%s' "$line" | tr '\n' ' ')"
      fi
    done
    printf '%s\n' "$scheds" | awk -v n="$n" 'NF == 2 && $2 >= n { print $1, $2 }' | while read -r pid i; do
      sched_pin_log "WARNING TP$i pid $pid has no CPU in $SCHED_PIN_VAR (${n} listed) -- left unpinned"
    done
    sched_pin_log "done: $pinned of $n scheduler(s) pinned"
  }
  if [ -n "$sp_bad" ] || [ "${#SCHED_PIN_CORES[@]}" = 0 ]; then
    note "WARNING: scheduler pinning skipped -- $SCHED_PIN_VAR='$SCHED_PIN' wants one CPU number per TP rank (space- or comma-separated); not a CPU number:${sp_bad:- none given}"
  else
    ( sched_pin "$$" >> "$SCHED_PIN_LOG_FILE" 2>&1 < /dev/null & )
    note "scheduler pinning: TP0..TP$((${#SCHED_PIN_CORES[@]} - 1)) -> CPU ${SCHED_PIN_CORES[*]} ($SCHED_PIN_VAR) after /health, log $SCHED_PIN_LOG_FILE"
  fi
fi

exec sglang serve "${A[@]}" \
  --model-path "$MODEL_DIR" --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 --port "$ENGINE_PORT" --config "$API_KEY_FILE"
