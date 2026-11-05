#!/bin/bash
# dev box: box-c's engine verb -- cutover.sh minus the gateway steps. box-c is router-fronted: live traffic
# reaches it through the serving router's two backend rows, one per engine copy -- `box-c-a` and `box-c-b` on
# prod, `box-c-a-dev` and `box-c-b-dev` on dev, in pool box-c-prod (dev box-c-dev) beside box-a's row; the old
# `box-c` / `box-c-dev` rows are in no pool, so nothing keyed on them takes a copy out -- and the box's
# own direct router row is HELD disabled and carries none (the serving notes, "The serving route"). So a box-c
# engine change runs this script (`--engine a|b|all`; `--single` is the TP8 rollback) with the
# affected rows out of service, by hand on every router pod (one copy's rows for `--engine a|b`, both
# for `all` or `--single`):
#   - while another pool member serves (box-a, or the other copy), a hard cut and roll: take the rows
#     out (preferred: a per-pod `sbr admin backend maintenance <row> --window
#     3600s --hold-until-recheck` on every router pod, no pod roll; fallback: a pool publish without
#     the rows plus a router restart), restart, put them back -- never a soft or paced drain
#     (07-operations.md "box-c as 2×TP4: a hard cut and roll, not a bracket");
#   - only when no pool member would serve, and only with the user's go-ahead, a per-row maintenance
#     window: BACKEND-MAINT-box-c-a / BACKEND-MAINT-box-c-b (dev BACKEND-MAINT-box-c-a-dev /
#     BACKEND-MAINT-box-c-b-dev; the serving router docs/runbooks/backends.md section 5.5).
#     Our model is offline for it and its requests fail: by the user's no-reroute order nothing
#     covers them -- never open a route to OpenRouter or any non-own provider (the failover
#     routes and the router's or-standby are OFF; no route leaves our own providers).
# Either way the restart silences open first, on every restart -- prod's pair matching
# `backend=~"box-c-a|box-c-b"` AND one matching `backend=~"box-c-a-dev|box-c-b-dev"` in sbr-dev's own
# Alertmanager, which prod's never reaches -- re-armed for the whole engine step.
# At READY come the gates (tools/smoke.py, or 10-updating.md's battery for a tree or checkpoint
# change) with the rows still out, then `sbr admin backend recheck` of each row taken out on every
# pod. The recipe is 07-operations.md "Restart and cutover recipes"; no script here drives the
# router, and this one never calls the gateway, in either direction -- never re-enables
# box-c's held row. It also serves any swap that never touched live traffic (a
# box not yet enrolled, or a bounce out-of-band of the gateway's disable/enable bracket). Same args/env
# install, same self-safe stop, same optional tree checkout, same relaunch and readiness poll as
# cutover.sh.
#
# usage: restart.sh <tag> [<args-file>|-] [<env-file>|-|default] [--tree <ref>]
#   see cutover.sh's header for the full contract of every argument.
# On a box whose box file sets ENGINES (lib.sh's engine_select: box-c's 2xTP4, two engines per box) a
# bare restart is refused; pick the mode:
#   restart.sh --engine <id>|all <tag> [<args-file>|-] [<env-file>|-|default] [--tree <ref>]
#     the named engine(s) of the per-engine layout, each from its own sglang-args-<id>.txt /
#     sglang-env-<id>.txt (installed with .pre-<tag> backups; the legacy pair is never written), stopped
#     by its own process group, launched by `relaunch.sh <id>` into its own log, polled and gated on its
#     own port. `all` launches every engine together. A box still on the one-engine line is switched:
#     that line's watchdog, launcher and engine stopped first, under watchdog.hold. `--engine b <tag>
#     <args> <env>` runs an arm on copy b alone while a serves on. Exit 6/7 name the engine that failed.
#     When the live Caddyfile lacks a per-copy site for any engine of ENGINES (its `https://<copy
#     name>:` line or its port as an upstream -- a door rendered by --single, or a prestage), tls.sh
#     re-renders the per-copy door from the box file's ENGINES after the stop (Caddyfile.pre-<tag> backup
#     named; a failure is a WARNING with the hand command); on a per-copy door nothing re-renders.
#     The block "the per-engine mode" below has the steps.
#   restart.sh --single <tag> ...
#     the one-engine line (sglang-args.txt / sglang-env.txt, the TP8 rollback) exactly as on a box
#     without ENGINES, after each engine's hold is claimed and each engine's watchdog stopped; once the
#     engines are down, each engine's engine-<id>.warming is removed and tls.sh re-renders the one-engine
#     Caddyfile (ENGINES emptied for the render, Caddyfile.pre-<tag> backup named), so no copy's flag
#     or site holds $HOSTNAME at 503 behind the rollback.
#
# Order -- every validation and preflight step before the hold runs only once the ones before it
# passed, so a bad invocation dies with the watchdog unheld and the engine untouched; from the hold on
# the script is `set +e`, and every box step's failure is decided explicitly (die_installing /
# die_held, below):
#   validate every argument with no box contact at all (--tree has a value with no leading -, quote
#   or whitespace; at most 3 positionals; the tag is A-Za-z0-9._+:@=- only, not starting with -; the
#   args file exists and carries a non-# line; the env file exists unless it is - or `default`)
#   -> the box target fence (box_assert_target: explicit when `default` or --tree needs a box
#      check, else at the first box_ssh)
#   -> the read-only box checks (`default`'s seed -- $TOOLS_REMOTE/args/env-box-d-official.txt when
#      SGLANG_TREE is box-d-up, env-box-e.txt when it is box-e-up, else env-v3.txt, or the box file's
#      ENV_SEED when it sets one (TOOLS_ENV_SEED, below) -- is on the box;
#      --tree resolves to a commit, or to exactly one remote-tracking branch, as the checkout will;
#      ssh's 255 is re-sent (box_ssh_retry), then dies in a real run and is a note under DRY_RUN)
#   -> take watchdog.hold as this run's CLAIM, with an owner token (#3316): a FRESH hold that is not
#      this run's is another run's box reservation and is REFUSED, touching nothing -- so a restart
#      cannot stop the engine under somebody else's live read
#   -> args/env install, each backed up first to sglang-args.txt.pre-<tag> / sglang-env.txt.pre-<tag>
#      (a failure dies with the engine untouched: this run's backups restored, the hold released)
#   -> stop: any in-flight relaunch.sh first, then the engine (self-safe pkill patterns), then wait
#      for the busiest GPU to drop under 8,000 MiB used
#   -> optional `git checkout -q <ref> --` on $SGLANG_TREE
#   -> relaunch.sh (nohup+setsid; sent ONCE, never re-sent)
#      From the stop through the relaunch a failure is a FATAL that KEEPS watchdog.hold (die_held):
#      the engine is stopped, or in a state this run cannot see, and a released hold lets the
#      watchdog relaunch into the operator's retry -- two engines on one box, both dead on
#      `Loaded weights leave no GPU memory for the KV cache` (gpu1021, 2026-10-21 01:27Z). It prints
#      the retry -- `BOX_HOLD_OVERRIDE=1 <script> <a NEW tag> - - [--tree <ref>]`, the override being
#      part of the printed line because the hold it just kept is fresh and is not the retry's token --
#      and the exact command that releases the hold by hand.
#   -> the tree read-back: the tree the engine LAUNCHES from
#      (TREE_PROBE_SH, below, run under the installed env file's PYTHONPATH -- tree_probe_env, #806 --
#      so a PYTHONPATH overlay's checkout is the tree; every later `tree=` names it, and SGLANG_TREE
#      beside it when the two differ) -> readiness poll from this run's own RESTART marker in
#      sglang-serve.log (the "fired up and ready to roll" banner, else /health; ceiling 1,800 s; it
#      ends early, NOT READY, when the latest launch's log section -- from the later of the last
#      `===== RESTART ` line-start marker and this run's own marker line -- carries argparse's
#      refusal -- `unrecognized arguments` / `error: argument ` -- because that engine has already
#      exited, or (CRASH_CHECK) `Scheduler hit an exception` (or the tokenizer's / detokenizer's own
#      `TokenizerManager` / `DetokenizerManager hit an exception`), which takes the whole engine down, or a
#      second `server_args=` line, i.e. a second engine booting beside this one, or (VALIDATION_CHECK)
#      a last traceback ending in `ValueError`/`AssertionError` with no `sglang serve` process left --
#      a startup validation hook refusing the line; cutover.sh sends the identical probes)
#   -> release the hold. NOT READY (the ceiling, the refusal, the crash, two engines, a validation
#      hook's refusal) exits 6.
#   Every box step that is safe to send twice -- the preflight reads, the hold's touch and rm -f, the
#   backups, the installs, the stop, the checkout, and the reads the run acts on (the marker line, the
#   launched-tree probe, the args read-back, the sampling gate) -- is re-sent on ssh's own exit 255
#   (the box did not answer), 3 tries in all (box_ssh_retry, below); the relaunch is not. (The poll's
#   own reads need none: a missed one is asked again 5 s later.)
#   -> READY only: the prewarm (#340) -- one box_ssh running the staged prewarm-box-d.py under the box
#      file's VENV python against $ENGINE_URL, its output in /workspace/prewarm-<tag>.log and its
#      SLOW/ERROR lines and `PREWARM DONE` echoed here with the wall time. It pays a new tree's
#      first-time compiles (mHC's mix-stats Triton kernel per m class, the DeepGEMM einsum configs)
#      inside the window instead of on live traffic; near a no-op on a warm cache. A failure (or a
#      PREWARM_TIMEOUT, default 1800 s) WARNs and the run goes on; never a retry (it is not a read).
#      PREWARM=off skips it with a note. It runs under this run's watchdog.hold taken again (#353:
#      the poll released it, and a compile fails /health for 17-34 s), refreshed every 110 s on the
#      box and released after it, on a failure and a timeout too; an INT/TERM/HUP releases it through
#      the trap once the box command returns (bash defers the trap while it waits, up to
#      PREWARM_TIMEOUT), and a SIGKILL leaves it to the watchdog's stale-hold rule,
#      HOLD_MAX (default 900 s).
#      A hold it cannot take skips it with a WARNING. A prewarm that exits 0 writes the box marker
#      /root/.cache/sglang/.prewarmed-<tree sha>-<VENV path cksum>-cp<chunk>; with that marker there,
#      the next restart skips it (`prewarm skipped: marker ...`) unless PREWARM=force. A marker the
#      box cannot write is said so (`prewarm marker NOT written (...)`): the next restart prewarms.
#   -> READY only: the sampling gate (below, SAMPLING_GATE_PY), read-only on the box. It PASSES only
#      when an omitted sampler resolves to SGLang's OpenAI defaults (temperature 1.0, top_p 1.0,
#      top_k -1, min_p 0.0, repetition_penalty 1.0) -- what DeepSeek's card recommends, and what box-c
#      serves -- with neither the installed args file nor the live engine's cmdline carrying
#      --sampling-defaults and /server_info's preferred_sampling_params empty. PASS exits 0; FAIL (or
#      a check that could not run: UNVERIFIED) exits 7 with the engine UP and serving -- the router
#      window is still open, so the operator rolls back per the header below or fixes forward.
#      SAMPLING_GATE=off skips it with a note (a box that serves a checkpoint whose own
#      generation_config.json is meant to apply); box-c's recipe never sets it.
#
# The hold is the box's CLAIM, not an idle flag (#3316): on gpu1018, 2026-10-21 23:14:53Z, a restart
# took the box under a live HalluLens read 6 min after the read's kit-wdhold keeper (hold.sh) had
# written the hold with its owner token -- the bare `touch` refreshed that keeper's hold, the stop
# killed the engine under 64 in-flight items, and the release `rm -f`d the keeper's hold on the way
# out. So the take below reads the hold first: one that is FRESH by the watchdog's own rule (mtime,
# stale after HOLD_MAX, read from the running daemon's environ so a raised window counts) and is not
# this run's token is refused, and a refusal touches nothing. The take is atomic (`set -C`, as
# hold.sh writes its own), the token (`restart.sh <tag> pid<N> <utc>`) makes a 255 re-send find its
# own hold and proceed, the readiness poll re-asserts the hold every 110 s (its 1,800 s ceiling
# outlives HOLD_MAX, so a slow boot would otherwise hand the box to the watchdog mid-poll), and the
# release removes the hold only while it is still this run's -- a peer that took over deliberately
# keeps its own. BOX_HOLD_OVERRIDE=1 is the deliberate override, for a hold the operator KNOWS is
# dead: the retry after a die_held FATAL (which keeps its hold), or the watchdog's own loop-guard
# `touch` (watchdog.sh, "RELAUNCH LOOP: no backup -- holding"). An override whose hold names a
# restart.sh or cutover.sh run this host cannot prove dead is refused before the take, nothing
# touched (override_owner_live, failing closed: a hidden first run is not a dead one, #460). The
# override writes this run's own token, so its own post-READY prewarm re-takes the hold as its own.
# cutover.sh takes the same claim, its box-side text byte for byte (#3361), with the same live-owner
# check, and both scripts' pre-stop FATAL (die_installing) releases the hold only while it is still
# this run's, like the run's own end.
#
# tests/cutover-preflight.test.sh pins the order up to the hold, for this script and cutover.sh;
# tests/restart-sampling-gate.test.sh runs the sampling gate against a stub /server_info;
# tests/prewarm-plan.test.sh pins the prewarm's M plan, its post-READY call, its hold and its marker,
# here and in cutover.sh;
# tests/restart-fatal-hold.test.sh runs the kept hold, the 255 re-sends and CRASH_CHECK, for this
# script and cutover.sh, and watchdog.sh's hold re-check;
# tests/restart-hold-override.test.sh runs BOX_HOLD_OVERRIDE=1 against a live, an unprovable and a
# dead holder, for this script and cutover.sh.
#
# Rollback, with the same rows out again (a fresh drain or window): cp sglang-args.txt.pre-<tag>
# sglang-args.txt (and sglang-env.txt.pre-<tag> if one exists; on an ENGINES box each copy's
# sglang-args-<id>.txt / sglang-env-<id>.txt pair), then restart.sh [--engine <id>|all] <tag>-back - -
# [--tree <previous ref>].
# That restores args, env and tree only. A change that also flipped the checkpoint (SERVE_BUILD/
# MODEL_DIR in boxes/$BOX.env) has its model-flip half undone first and separately, because no
# .pre-<tag> file carries it and relaunch.sh reads MODEL_DIR live from the staged box file on every
# boot: revert the box file's two lines, re-run stage.sh (exit 0), and confirm on the box that
# boxes/$BOX.env's MODEL_DIR and serve-model.env both name the previous path -- all BEFORE the
# rollback boot, or it boots the new checkpoint on the old line.
unset ENV_SEED   # the env seed override is the box file's line alone (stage_env_seed_name below)
. "$(dirname "$0")/lib.sh"
set +e   # many steps here (pkill, a missing backup, a slow poll) fail routinely by design

USAGE="usage: restart.sh <tag> [<args-file>|-] [<env-file>|-|default] [--tree <ref>]"
USAGE_ENGINES="on a box file with ENGINES: restart.sh --engine <id>|all <tag> [<args-file>|-] [<env-file>|-|default] [--tree <ref>], or restart.sh --single <tag> ... for the one-engine line"
TREE=""
ENG_SEL=""
SINGLE=""
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tree)
      { [ $# -ge 2 ] && [ -n "$2" ]; } || die "--tree needs a ref -- $USAGE"
      TREE=$2; shift 2 ;;
    --engine)
      { [ $# -ge 2 ] && [ -n "$2" ]; } || die "--engine needs an engine id or all -- $USAGE_ENGINES"
      ENG_SEL=$2; shift 2 ;;
    --single) SINGLE=1; shift ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
[ "${#POSITIONAL[@]}" -ge 1 ] || die "$USAGE"
[ "${#POSITIONAL[@]}" -le 3 ] || die "too many arguments (${#POSITIONAL[@]} positional, at most 3: tag, args file, env file) -- $USAGE"
case "$TREE" in
  -*|*\'*|*[[:space:]]*) die "bad --tree ref '$TREE': it rides single-quoted remote commands, so no leading -, quote or whitespace -- $USAGE" ;;
esac
set -- "${POSITIONAL[@]}"
TAG=$1
ARGS_FILE=${2:-}
ENV_FILE_ARG=${3:-}

TOOLS_ARGS="$TOOLS_REMOTE/sglang-args.txt"
TOOLS_ENV="$TOOLS_REMOTE/sglang-env.txt"
# The env-file argument's `default` word reseeds TOOLS_ENV from the box's staged copy of the env of the
# serve line this box file's tree runs (the `cp -f` below), so whatever that file holds is what the box
# runs after a by-the-book `default` restart. The seed is the box file's ENV_SEED=<env-*.txt> when it
# sets one (box-a: env-box-a.txt on the box-e-up tree box-c also runs; stage.sh's comment on its copy has the
# rules), else picked by SGLANG_TREE's name, by the same function stage.sh's first-boot seed uses (tests/restart-env-seed.test.sh holds the three copies --
# stage.sh, this file, cutover.sh -- byte for byte equal):
#   box-d-up  -> args/env-box-d-official.txt (18 keys), box-c's live env and v5-b200-box-d.args' pair:
#               env-v3-official.txt byte for byte plus the box-d delta (args/check.sh), whose
#               SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python the box-d tree needs to start at all --
#               SGLANG_SWA_BRANCH_REBUILD=true refuses upstream's rust radix core -- and whose
#               SGLANG_FORWARD_UNKNOWN_TOOLS=1 box-c serves. A box-d box never gets env-v3.txt here: it
#               does not start on it (ISSUES.md, the `default` env row);
#   box-e-up  -> args/env-box-e.txt (18 keys), v6-b200-box-e.args' pair on the upstream-merged box-e tree:
#               env-box-d-official.txt minus SGLANG_OPT_DSV4_C4_INDEXER_LOGITS_BUDGET_BYTES (a dead knob
#               there) plus SGLANG_OPT_DSV41_BF16_INDEX_LOGITS=0, which holds that tree's index logits at
#               box-d's fp32 (args/check.sh's box-e pin). Before this case a box-e-up box fell to env-v3.txt,
#               which does not start an upstream-based tree (the box-d-up row above), and a box-e-up box
#               never gets env-box-d-official.txt either: it would serve bf16 index logits;
#   anything else (the image's /sgl-workspace/sglang) -> args/env-v3.txt, v4-b200.args' pair: the old
#               line's env, 16 lines (md5 7d92bc7580d52d3e5f538851da0093a6, byte-identical to
#               args/env-v3-official.txt, the env the #1688 official-line cutover installed) -- its 13
#               gates (12 SGLANG_* plus PYTHONFAULTHANDLER=1) and the official thinking default,
#               SGLANG_DEFAULT_THINKING=1 + SGLANG_DSV41_REASONING_EFFORT=high. Not a stub: while it held
#               only SGLANG_ENABLE_REQUEST_DECOMPRESSION=1, every `default` restart silently dropped 13
#               live gates (ledger #229; the args file got the same treatment in #204), and without the
#               last two lines a `default` restart drops the thinking default (a no-effort request stops
#               thinking). Its first 14 lines (md5 558c41e618c06cd816d7b059dfd8fb4b) are the strict line's
#               env, the rollback content of sglang-env.txt.pre-e2e-official-thinkfix.
# The name read is THIS run's box file's SGLANG_TREE (lib.sh), the tree --tree checks out in and the
# one stage.sh seeded by. stage.sh syncs tools/args/ to the box, which is how the seed gets there; the
# preflight below checks it is there before the hold.
stage_env_seed_name() {
  if [ -n "${2:-}" ]; then
    case "$2" in
      */*|*[!A-Za-z0-9._-]*) ;;
      env-*.txt) printf '%s' "$2"; return 0 ;;
    esac
    printf '%s\n' "ENV_SEED='$2' in the box file is not an env file name under tools/args/ (env-<name>.txt, no path) -- e.g. ENV_SEED=env-box-a.txt" >&2
    return 1
  fi
  case "${1##*/}" in
    box-d-up) printf '%s' env-box-d-official.txt ;;
    box-e-up) printf '%s' env-box-e.txt ;;
    *)       printf '%s' env-v3.txt ;;
  esac
}
TOOLS_ENV_SEED="$TOOLS_REMOTE/args/$(stage_env_seed_name "${SGLANG_TREE%/}" "${ENV_SEED:-}")" \
  || die "ENV_SEED in $BOX_ENV holds '${ENV_SEED:-}', not an args/env-*.txt file name -- nothing done"
HOLD="$TOOLS_REMOTE/watchdog.hold"

# ---- the mode: a box file's ENGINES (#415) -------------------------------------------------------
# A box with ENGINES runs one engine per entry (lib.sh's engine_select), so a bare restart is
# ambiguous there and refused: `--engine <id>|all` restarts engine(s) of the per-engine layout (the
# block before the hold below), `--single` the one-engine line from the legacy sglang-args.txt /
# sglang-env.txt -- the TP8 rollback -- stopping every engine's watchdog first (the stop takes the
# engines). A box without ENGINES runs exactly as it always has; `--single` there changes nothing.
E_IDS=""
if [ -n "$ENG_SEL" ] && [ -n "$SINGLE" ]; then
  die "--engine and --single are two modes, pick one -- $USAGE_ENGINES"
fi
if [ -n "$ENG_SEL" ]; then
  if [ "$ENG_SEL" = all ]; then
    [ -n "${ENGINES:-}" ] || die "--engine all: $BOX_ENV has no ENGINES -- this box runs one engine; restart.sh <tag> ... restarts it"
    for e in $ENGINES; do E_IDS="$E_IDS ${e%%:*}"; done
  else
    E_IDS=" $ENG_SEL"
  fi
  E_IDS=${E_IDS# }
  for id in $E_IDS; do ( engine_select "$id" ) || exit 1; done   # an unknown id, or no ENGINES: dies here, nothing touched
elif [ -n "${ENGINES:-}" ]; then
  [ -n "$SINGLE" ] || die "$BOX runs per-engine (ENGINES=\"$ENGINES\" in $BOX_ENV): pass --engine <id>|all to restart engines, or --single to go back to the one-engine line (sglang-args.txt, the TP8 rollback) -- nothing touched. $USAGE_ENGINES"
  for e in $ENGINES; do E_IDS="$E_IDS ${e%%:*}"; done
  E_IDS=${E_IDS# }
fi

# ---- re-sending a step the box did not answer (#2870) -------------------------------------------
# box_ssh_retry [--stdin <text>] <remote command>: box_ssh, re-sent when ssh ITSELF failed -- exit
# 255, the box did not answer ('Connection timed out during banner exchange' on gpu1021's checkout,
# 2026-10-21 01:26Z, which killed a restart with its engine already stopped) -- up to BOX_SSH_TRIES
# (3) sends in all, BOX_SSH_BACKOFF (3) s x the try number apart. Any other exit returns at once: a
# remote "no" (git's 128, `[ -f ]`'s 1) is an answer, not a dropped link. Prints the LAST send's
# stdout only. ONLY for a remote command that is safe to send twice -- a read, the hold's touch and
# rm -f, a cp of fixed content, the stop's kill-and-wait (killing what is already dead is a no-op),
# the checkout of one ref (no fetch between sends, so the same commit) -- and NEVER the relaunch: a
# second send of that boots a second engine. box_scp_retry re-sends a copy of one local file to one
# path on any failure (the same bytes to the same place, whatever went wrong). restart.sh and
# cutover.sh carry this block -- from here through die_held below -- byte for byte
# (tests/restart-fatal-hold.test.sh compares the two); fatal_hint, after it, is each script's own.
box_ssh_retry() {
  local stdin="" has_stdin=0 out rc i=1 tries=${BOX_SSH_TRIES:-3} wait
  if [ "${1:-}" = --stdin ]; then stdin=$2; has_stdin=1; shift 2; fi
  while :; do
    if [ "$has_stdin" = 1 ]; then out=$(box_ssh "$@" <<<"$stdin"); rc=$?; else out=$(box_ssh "$@"); rc=$?; fi
    { [ "$rc" -eq 255 ] && [ "$i" -lt "$tries" ]; } || break
    wait=$(( ${BOX_SSH_BACKOFF:-3} * i )); i=$((i + 1))
    note "ssh exit 255 (the box did not answer) -- re-sending this step (safe to send twice), try $i/$tries in ${wait}s"
    sleep "$wait"
  done
  [ -z "$out" ] || printf '%s\n' "$out"
  return "$rc"
}
box_scp_retry() {
  local rc i=1 tries=${BOX_SSH_TRIES:-3} wait
  while :; do
    box_scp "$@"; rc=$?
    { [ "$rc" -ne 0 ] && [ "$i" -lt "$tries" ]; } || break
    wait=$(( ${BOX_SSH_BACKOFF:-3} * i )); i=$((i + 1))
    note "scp exit $rc -- re-sending the same file, try $i/$tries in ${wait}s"
    sleep "$wait"
  done
  return "$rc"
}

# What a FATAL after the hold leaves, and how the operator gets out of it. BACKED_UP lists the box
# files this run has backed up to <file>.pre-<tag> (and then overwritten, or was about to).
BACKED_UP=""
hold_release_cmd() {   # the exact command, run on this machine, that releases the hold by hand
  local env="BOX=$BOX"
  [ "$BOX_ENV" = "$TOOLS_DIR/boxes/$BOX.env" ] || env="$env BOX_ENV=\"$BOX_ENV\""
  printf '%s' "$env bash -c '. \"$TOOLS_DIR/lib.sh\" && box_ssh \"rm -f $HOLD\"'   (on the box itself: rm -f $HOLD)"
}
# die_installing <what>: a FATAL BEFORE the stop -- the engine still runs the line it had. Put back
# what this run overwrote (its own .pre-<tag> backups), release THIS run's hold so the watchdog guards
# that engine again, and exit 1. The release is token-guarded, the same one the run's own end uses
# (#3316, #3361): an unconditional `rm -f` here would drop a hold a peer took over since this run's
# claim, and hand the box to the watchdog in the middle of that peer's restart.
die_installing() {
  local f rel
  note "FATAL: $1 -- the engine was NOT touched: it still runs the line it had"
  for f in $BACKED_UP; do
    if box_ssh_retry "cp -f '$f.pre-$TAG' '$f'" >/dev/null; then
      note "restored $f from $f.pre-$TAG"
    else
      note "could NOT restore $f from $f.pre-$TAG -- put it back by hand before anything relaunches the engine (the watchdog's next relaunch reads it)"
    fi
  done
  rel=$(box_ssh_retry "o=\$(head -1 '$HOLD' 2>/dev/null); if [ \"\$o\" = '$CLAIM_TOKEN' ]; then rm -f '$HOLD'; echo RELEASED; else echo \"NOT-OURS \${o:-<none>}\"; fi")
  case "$rel" in
    RELEASED*) note "released watchdog.hold: the engine is up, and the watchdog guards it again" ;;
    NOT-OURS*) note "watchdog.hold is no longer this run's ($rel): left in place, its holder releases it -- the engine is up and this run touched nothing" ;;
    *)         note "could NOT release watchdog.hold (the box did not answer): the watchdog drops it once it is HOLD_MAX (default 900 s) old, or release it by hand: $(hold_release_cmd)" ;;
  esac
  fatal_hint
  exit 1
}
# die_held <what>: a FATAL from the stop on (the stop, the checkout, the relaunch's own send). The
# engine is stopped, or in a state this run could not see, so the hold is KEPT -- never released
# with no engine: on gpu1021 (2026-10-21 01:26-01:27Z) the checkout's ssh timed out, the FATAL
# released the hold, the watchdog relaunched into the operator's retry 12 s later, and the two engines
# booted side by side and both died on `Loaded weights leave no GPU memory for the KV cache`. It
# re-touches the hold once (best effort), so the watchdog's HOLD_MAX (default 900 s) counts from
# now, and prints the retry -- carrying BOX_HOLD_OVERRIDE=1, since the hold it meets is fresh and
# is not the retry's token -- and the release command. Exit 1.
die_held() {
  local f pre="" fresh retry="BOX_HOLD_OVERRIDE=1 ${0##*/} <a NEW tag> - -${TREE:+ --tree $TREE}"
  note "FATAL: $1"
  if box_ssh "touch '$HOLD'" >/dev/null 2>&1; then
    fresh="re-touched just now: the watchdog sleeps through it for HOLD_MAX (default 900 s) from here, then drops it and relaunches whatever is installed"
  else
    fresh="could not be re-touched (the box did not answer): the watchdog drops it once it is HOLD_MAX (default 900 s) old, then relaunches whatever is installed"
  fi
  note "watchdog.hold KEPT ($HOLD; $fresh). The engine is stopped, or in a state this run could not see; releasing the hold now would let the watchdog relaunch into your retry -- two engines on one box, KV-cache OOM (gpu1021, 2026-10-21)."
  for f in $BACKED_UP; do pre="$pre $f.pre-$TAG"; done
  [ -z "$pre" ] || note "this run's installs are in place (rollback copies:$pre) -- the retry passes - for them, and a NEW tag so it never overwrites those copies"
  note "retry: $retry -- it re-takes the hold, its stop step clears whatever did start, and it releases the hold itself"
  note "or, to hand the box back to the watchdog with no retry: $(hold_release_cmd)"
  fatal_hint
  exit 1
}
# The FATALs' closing line, the one part of this block the two scripts do not share: restart.sh makes
# no gateway call, so it has nothing to add (cutover.sh's names the row its disable left).
fatal_hint() { :; }

# The post-READY sampling gate (#2134), run ON the box as `python3 - <engine url> <args file>` with
# this text on stdin (two more optional argv, the environ file holding VLLM_API_KEY and the /proc
# root, exist for tests/restart-sampling-gate.test.sh only). Read-only: three reads, one verdict.
#   1. the installed args file's flag lines and every live `sglang serve` cmdline (with a 5th argument,
#      a port: only the `sglang serve --port <port>` one -- --engine mode, one engine of several) carry no
#      --sampling-defaults (box-c serves with the tree's own default; a named mode is a change to it);
#   2. /server_info (else the deprecated /get_server_info) -- preferred_sampling_params is empty
#      (null, {} or ""), so the engine merges no sampler of its own into requests;
#   3. sampling_defaults resolves to the OpenAI defaults: 'openai' does by definition; 'model' (the
#      box-d-up and 264ea0f9 trees' default) reads <model_path>/generation_config.json, and every
#      sampler that file sets (temperature, top_p, top_k, min_p, repetition_penalty -- the five
#      ModelConfig.get_default_sampling_params() lifts) must equal the OpenAI default, so a file that
#      is absent (the 1017 checkpoint has none) or sets only those values passes. A missing
#      sampling_defaults key, another mode, or an unreadable file/endpoint is a FAIL.
# It prints `SAMPLING PASS: ...` and exits 0, or `SAMPLING FAIL: ...` and exits 1.
IFS= read -r -d '' SAMPLING_GATE_PY <<'PY'   # read's exit 1 at EOF is expected (set +e)
import json, os, sys, urllib.request, urllib.error
url, args_file = sys.argv[1].rstrip("/"), sys.argv[2]
environ = sys.argv[3] if len(sys.argv) > 3 else "/etc/environment"
proc = sys.argv[4] if len(sys.argv) > 4 else "/proc"
port = sys.argv[5] if len(sys.argv) > 5 else ""   # --engine mode: only the engine on this --port
OPENAI = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}
fails = []
try:
    flags = [l for l in open(args_file) if not l.startswith("#")]
    if any("--sampling-defaults" in l for l in flags):
        fails.append("%s carries --sampling-defaults" % args_file)
except OSError as e:
    fails.append("cannot read %s: %s" % (args_file, e))
pids = []
for d in (os.listdir(proc) if os.path.isdir(proc) else []):
    if not d.isdigit():
        continue
    try:
        argv = [a.decode("utf-8", "replace") for a in open(os.path.join(proc, d, "cmdline"), "rb").read().split(b"\0") if a]
    except OSError:
        continue
    if "sglang serve" not in " ".join(argv):
        continue
    if port and not any(a == "--port=" + port or (a == "--port" and i + 1 < len(argv) and argv[i + 1] == port) for i, a in enumerate(argv)):
        continue
    pids.append(d)
    if any(a.startswith("--sampling-defaults") for a in argv):
        fails.append("engine pid %s's cmdline carries --sampling-defaults" % d)
if not pids:
    fails.append("no live `sglang serve` process under %s to read a cmdline from" % proc)
key = ""
try:
    for l in open(environ):
        if l.startswith("VLLM_API_KEY="):
            key = l.split("=", 1)[1].strip().strip('"')
            break
except OSError:
    pass
info, src = None, None
for path in ("/server_info", "/get_server_info"):
    req = urllib.request.Request(url + path, headers={"Authorization": "Bearer " + key} if key else {})
    try:
        info, src = json.loads(urllib.request.urlopen(req, timeout=20).read().decode()), path
        break
    except urllib.error.HTTPError as e:
        if e.code != 404:
            fails.append("GET %s: HTTP %s" % (path, e.code))
            break
    except Exception as e:
        fails.append("GET %s: %s" % (path, e))
        break
if info is None and not any(f.startswith("GET ") for f in fails):
    fails.append("neither /server_info nor /get_server_info answered")
summary = ""
if isinstance(info, dict):
    sd = info.get("sampling_defaults", "<absent>")
    psp = info.get("preferred_sampling_params")
    mp = info.get("model_path") or ""
    if isinstance(psp, str) and psp.strip():
        try:
            psp = json.loads(psp)
        except ValueError:
            pass
    if psp not in (None, {}, ""):
        fails.append("preferred_sampling_params is set: %s" % json.dumps(psp))
    resolved, why = {}, ""
    if sd == "openai":
        why = "sampling_defaults=openai"
    elif sd == "model":
        gc = os.path.join(mp, "generation_config.json")
        if not mp:
            fails.append("sampling_defaults=model but %s names no model_path" % src)
        elif os.path.exists(gc):
            try:
                cfg = json.load(open(gc))
                resolved = dict((k, cfg[k]) for k in OPENAI if cfg.get(k) is not None)
                why = "sampling_defaults=model, %s sets %s" % (gc, json.dumps(resolved, sort_keys=True))
            except (OSError, ValueError) as e:
                fails.append("cannot read %s: %s" % (gc, e))
        else:
            why = "sampling_defaults=model, no generation_config.json in %s" % mp
    else:
        fails.append("sampling_defaults is %r in %s (want 'model' or 'openai')" % (sd, src))
    eff = dict(OPENAI, **resolved)
    bad = dict((k, v) for k, v in sorted(eff.items()) if not isinstance(v, (int, float)) or float(v) != float(OPENAI[k]))
    if bad:
        fails.append("omitted samplers resolve to %s, not the OpenAI defaults %s" % (json.dumps(bad, sort_keys=True), json.dumps(dict((k, OPENAI[k]) for k in bad))))
    summary = "%s -> temperature %s, top_p %s; preferred_sampling_params %s; %s model_path=%s version=%s" % (
        why or "sampling_defaults=%r" % sd, eff.get("temperature"), eff.get("top_p"), json.dumps(psp),
        src, mp, info.get("version"))
elif info is not None:
    fails.append("%s is not a JSON object" % src)
if fails:
    print("SAMPLING FAIL: " + "; ".join(fails) + ("" if not summary else " [" + summary + "]"))
    sys.exit(1)
print("SAMPLING PASS: %s; no --sampling-defaults in %s or the engine cmdline (pid %s)" % (summary, args_file, ",".join(pids)))
PY

# The launched-tree probe (#2389), run ON the box as `bash -s -- <staged box file> <SGLANG_TREE>
# <VENV>` with this text on stdin, the last two the values THIS run's box file names. Read-only.
# The tree the engine runs is not necessarily `$SGLANG_TREE` as this machine's box file names it:
# relaunch.sh reads VENV and SGLANG_TREE from the box's STAGED copy ($TOOLS_REMOTE/boxes/$BOX.env),
# and `sglang` is whatever $VENV/bin's python imports. A campaign line flip that edits only the
# staged copy (VENV=/opt/sglang-up, SGLANG_TREE=/workspace/box-d-up on gpu1018/gpu1018b, with the
# tracked file still on /sgl-workspace/sglang) made every box-d-line READY line print the image tree's
# 264ea0f9774b while the engine ran 7b0a2e8d93. So: source the staged copy as lib.sh does (its two
# values win over the passed ones), ask $VENV/bin/python3 where `sglang` resolves (find_spec: no
# package import), and name that file's git toplevel -- else, for a non-git install, the staged
# SGLANG_TREE. Prints LAUNCH_TREE=<dir>, LAUNCH_SHA=<12 hex or n/a>, LAUNCH_HOW=import|boxfile and
# SAME=yes|no (the launched dir is, physically, the SGLANG_TREE this run's box file names).
# cutover.sh carries this text and the read-back below byte for byte (tests/restart-launched-tree.test.sh).
IFS= read -r -d '' TREE_PROBE_SH <<'SH'   # read's exit 1 at EOF is expected (set +e)
f=$1 t=$2 v=$3 t0=$2
if [ -r "$f" ]; then
  bt=$(set -a; . "$f" >/dev/null 2>&1; printf '%s' "${SGLANG_TREE:-}")
  bv=$(set -a; . "$f" >/dev/null 2>&1; printf '%s' "${VENV:-}")
  [ -n "$bt" ] && t=$bt
  [ -n "$bv" ] && v=$bv
fi
py="$v/bin/python3"; [ -x "$py" ] || py=$(command -v python3)
o=""; [ -n "$py" ] && o=$("$py" -c 'import importlib.util as u, os
s = u.find_spec("sglang")
print(os.path.realpath(s.origin) if s and s.origin else "")' 2>/dev/null)
d=""; how=import
[ -n "$o" ] && d=$(git -C "$(dirname "$o")" rev-parse --show-toplevel 2>/dev/null)
[ -n "$d" ] || { d=$t; how=boxfile; }
s=$(git -C "$d" rev-parse --short=12 HEAD 2>/dev/null || echo n/a)
same=no; [ "$(cd "$d" 2>/dev/null && pwd -P)" = "$(cd "$t0" 2>/dev/null && pwd -P)" ] && same=yes
printf 'LAUNCH_TREE=%s\nLAUNCH_SHA=%s\nLAUNCH_HOW=%s\nSAME=%s\n' "$d" "$s" "$how" "$same"
SH

# The engine's env, ahead of the probe (#806). relaunch.sh exports every line of the INSTALLED env
# file (sglang-env.txt, or an engine's sglang-env-<id>.txt) before the exec, so a PYTHONPATH there
# puts its first entry holding sglang/ ahead of SGLANG_TREE's editable install in $VENV -- box-c's
# copies served the trim tree that way until the #750 roll (env-box-e-w2.txt:
# PYTHONPATH=/workspace/box-e-trim/python over /workspace/box-e-up), its second per-copy rollback still
# does, and the cached-refs roll launched both copies with PYTHONPATH=/workspace/box-c-mmrefs/python (bdc3f67f
# over box-e-up at 5f117c89; copy a still, until #1320) and #1170 relaunched copy b with
# PYTHONPATH=/workspace/box-c-prodnext4/python (prod 9b262265) until their next --tree restart; the env of
# record, env-box-e-w4.txt on a plain checkout of the box file's FORK_TREE (prod) in box-e-up, carries no
# PYTHONPATH, so the probe answers
# SGLANG_TREE itself. Without this the probe's find_spec
# answered SGLANG_TREE, and READY's tree= (and every rollback anchor taken from it) named
# /workspace/box-e-up's sha while both copies ran the overlay's. So this text, bound by tree_probe_env to
# the installed env file and to the env seed (read only when the installed file is missing), exports
# that file's PYTHONPATH as relaunch.sh does (its last `PYTHONPATH=` line, verbatim; a file with none
# leaves the box shell's as it is) and the probe's find_spec then names the overlay's git checkout --
# the run-l3a-slice.sh rule (#776): the first PYTHONPATH entry holding sglang/ is the tree, else
# SGLANG_TREE; an entry under no git checkout leaves SGLANG_TREE (the probe's boxfile route). Prints
# LAUNCH_ENV=<file read or empty>, LAUNCH_PYTHONPATH=<value or empty> and BASE_SHA=<SGLANG_TREE's 12
# hex or n/a> ahead of the probe's lines. TREE_PROBE_SH itself stays cutover.sh's text byte for byte
# (tests/restart-launched-tree.test.sh), and cutover.sh carries this text and tree_probe_env byte for
# byte too (tests/restart-engines.test.sh).
IFS= read -r -d '' TREE_PROBE_ENV_SH <<'SH'   # read's exit 1 at EOF is expected (set +e)
pf=""; for c in "$pe1" "$pe2"; do [ -n "$c" ] && [ -r "$c" ] && { pf=$c; break; }; done
pp=""
if [ -n "$pf" ] && grep -q '^PYTHONPATH=' "$pf" 2>/dev/null; then
  pp=$(sed -n 's/^PYTHONPATH=//p' "$pf" | tail -1)
  export PYTHONPATH="$pp"
fi
b=$(git -C "$2" rev-parse --short=12 HEAD 2>/dev/null || echo n/a)
printf 'LAUNCH_ENV=%s\nLAUNCH_PYTHONPATH=%s\nBASE_SHA=%s\n' "$pf" "$pp" "$b"
SH
# tree_probe_env <installed env file> <env seed>: TREE_PROBE_RUN = the env text above bound to the two
# files, then TREE_PROBE_SH -- sent as `bash -s -- <staged box file> <SGLANG_TREE> <VENV>`, as before.
tree_probe_env() { printf -v TREE_PROBE_RUN "pe1='%s' pe2='%s'\n%s%s" "$1" "$2" "$TREE_PROBE_ENV_SH" "$TREE_PROBE_SH"; }

# ---- validation BEFORE the watchdog hold (cutover.sh's own preflight, minus the gateway) --------
# Every argument is checked before the first mutation: the tag's shape and the args/env files on
# THIS machine (no box contact at all), then the read-only box-side checks (the `default` seed, the
# --tree ref). A bad invocation therefore dies with the watchdog unheld and the engine untouched.
# (Until #1420 these checks sat after the hold, and each die had to remove it again.)
case "$TAG" in
  ''|-*|*[!A-Za-z0-9._+:@=-]*) die "bad tag '$TAG': letters, digits and . _ + : @ = - only, not starting with - -- $USAGE" ;;
esac
if [ -n "$ARGS_FILE" ] && [ "$ARGS_FILE" != "-" ]; then
  [ -f "$ARGS_FILE" ] || die "no such args file: $ARGS_FILE (nothing touched: the watchdog and the engine are as they were)"
  grep -v '^#' "$ARGS_FILE" | grep -q -- . || die "args file carries no flags: $ARGS_FILE (nothing touched)"
fi
if [ -n "$ENV_FILE_ARG" ] && [ "$ENV_FILE_ARG" != "-" ] && [ "$ENV_FILE_ARG" != default ]; then
  [ -f "$ENV_FILE_ARG" ] || die "no such env file: $ENV_FILE_ARG (nothing touched: the watchdog and the engine are as they were)"
fi

# The read-only box-side checks. The fence is asserted explicitly first (box_ssh would assert it
# anyway; this names the step) so a MISMATCH refuses before any check is even sent. A remote "no"
# dies; ssh's 255 (the box did not answer) dies in a real run and is a note under DRY_RUN.
if [ "$ENV_FILE_ARG" = default ] || [ -n "$TREE" ]; then
  box_assert_target
fi
preflight_remote() {   # <what> <remote command>
  local rc=0
  # stderr kept: ssh's own reason, and the fence's notes. A real run re-sends a 255 (a read);
  # DRY_RUN sends once and notes it.
  if [ "${DRY_RUN:-0}" = 1 ]; then box_ssh "$2" >/dev/null || rc=$?; else box_ssh_retry "$2" >/dev/null || rc=$?; fi
  [ "$rc" -eq 0 ] && return 0
  if [ "$rc" -eq 255 ] && [ "${DRY_RUN:-0}" = 1 ]; then
    note "DRY_RUN: could not check $1 (ssh exit 255, the box did not answer) -- a real run dies here"
    return 0
  fi
  [ "$rc" -eq 255 ] && die "could not check $1: the box did not answer (ssh exit 255) after ${BOX_SSH_TRIES:-3} tries -- nothing touched"
  return 1
}
if [ "$ENV_FILE_ARG" = default ]; then
  preflight_remote "the env seed $TOOLS_ENV_SEED" "[ -f '$TOOLS_ENV_SEED' ]" \
    || die "no $TOOLS_ENV_SEED on the box to reseed from -- stage.sh syncs tools/args/ to the box; run it first (nothing touched)"
fi
# The --tree check accepts exactly what the `git checkout -q '<ref>' --` below accepts: `<ref>^{commit}`
# resolving, or else exactly ONE remote-tracking branch refs/remotes/<remote>/<ref> (checkout's guess,
# which creates a local branch tracking it -- a <branch> the tree only fetched as fork/<branch>);
# several matches are refused, naming them, as checkout refuses them. cutover.sh's comment on its own
# copy of this block has the long version (glob characters, slash-bearing remote names, the quoting).
TREE_CHECK="r='$TREE'; git -C '$SGLANG_TREE' rev-parse -q --verify '$TREE^{commit}' >/dev/null && exit 0"
case "$TREE" in
  *'*'*|*'?'*|*'['*) TREE_CHECK="$TREE_CHECK; exit 1" ;;
  *) TREE_CHECK="$TREE_CHECK; m=\$(git -C '$SGLANG_TREE' for-each-ref --format='%(refname)' 'refs/remotes/*/$TREE' 2>/dev/null | tr '\n' ' '); set -- \$m; [ \$# -eq 1 ] && { echo \"--tree preflight: '\$r' is no local branch, tag or commit; git checkout will create it as a local branch tracking \$1\" >&2; exit 0; }; [ \$# -gt 1 ] && echo \"--tree preflight: '\$r' matches \$# remote-tracking branches (\$m) -- git checkout refuses that as ambiguous; name one as --tree <remote>/\$r\" >&2; exit 1" ;;
esac
if [ -n "$TREE" ]; then
  preflight_remote "--tree $TREE on $SGLANG_TREE" "$TREE_CHECK" \
    || die "--tree $TREE does not resolve to a commit in $SGLANG_TREE on the box (\`git rev-parse --verify '$TREE^{commit}'\` failed), nor to exactly one remote-tracking branch refs/remotes/<remote>/$TREE -- fetch it first (stage.sh's fork fetch); nothing touched"
fi

# ---- the box claim: watchdog.hold with an owner token (#3316) ------------------------------------
# The hold was an idle flag to this script: #3299 `touch`ed #3293's live hold at gpu1018 23:14:53Z
# without reading it, refreshed a reservation it did not own, stopped the engine under 64 in-flight
# items, and released the keeper's hold on the way out. So the take reads the hold first and refuses
# a fresh one that is not this run's (the header's "The hold is the box's CLAIM"). Fresh follows
# watchdog.sh's hold_fresh exactly -- mtime, stale after HOLD_MAX, taken from the RUNNING daemon's
# environ when it names one (docs option 2 raises it to 43200 s) else this run's HOLD_MAX else 900 --
# so a hold the daemon would honour is never called stale here. The create is atomic (`set -C`, the
# kit hold.sh's own protocol), and the token makes the send idempotent: a 255 re-send finds its own
# hold and proceeds instead of refusing itself.
CLAIM_TOKEN="restart.sh $TAG pid$$ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
CLAIM_OVERRIDE=${BOX_HOLD_OVERRIDE:-0}
case "$CLAIM_OVERRIDE" in 1) ;; *) CLAIM_OVERRIDE=0 ;; esac
CLAIM_MAX=${HOLD_MAX:-}
case "$CLAIM_MAX" in ''|*[!0-9]*) CLAIM_MAX="" ;; esac
# The box-side take, read like SAMPLING_GATE_PY and TREE_PROBE_SH below (never `$(cat <<...)`: bash
# 3.2 mis-parses a here-document inside command substitution); __HOLD__ / __TOKEN__ / __OVERRIDE__ /
# __MAX__ are filled on the next four lines.
IFS= read -r -d '' CLAIM_TXT <<'CLAIM_SH'   # read's exit 1 at EOF is expected (set +e)
H='__HOLD__'; T='__TOKEN__'; O=__OVERRIDE__; M='__MAX__'
o=$(head -1 "$H" 2>/dev/null)
if [ "$o" = "$T" ]; then echo "TOOK $T (already this run's)"; exit 0; fi
if [ -e "$H" ]; then
  m=$(stat -c %Y "$H" 2>/dev/null); ok=$?
  w=$(pgrep -f 'watchdo[g].sh' 2>/dev/null | head -1)
  max=""
  [ -n "$w" ] && max=$(tr '\0' '\n' < "/proc/$w/environ" 2>/dev/null | sed -n 's/^HOLD_MAX=//p' | head -1)
  case "$max" in ''|*[!0-9]*) max="$M" ;; esac
  case "$max" in ''|*[!0-9]*) max=900 ;; esac
  if [ "$ok" -ne 0 ]; then age=unknown; else age=$(( $(date +%s) - m )); fi
  own="${o:-<none>}"
  if [ "$O" = 1 ]; then
    echo "OVERRIDE age=$age max=${max}s owner=$own"
  elif [ "$age" = unknown ] || [ "$age" -le "$max" ]; then
    echo "HELD age=$age max=${max}s owner=$own"; exit 3
  else
    echo "STALE age=$age max=${max}s owner=$own"
  fi
  rm -f "$H" || { echo "COULD-NOT-CLEAR owner=$own"; exit 6; }
fi
( set -C; printf '%s\n' "$T" > "$H" ) 2>/dev/null || { echo "RACE owner=$(head -1 "$H" 2>/dev/null)"; exit 5; }
touch '__HOLD__' || exit 7   # spelled out, never "$H": the tests pin this literal as the hold step
echo "TOOK $T"
CLAIM_SH
CLAIM_CMD=${CLAIM_TXT//__HOLD__/$HOLD}
CLAIM_CMD=${CLAIM_CMD//__TOKEN__/$CLAIM_TOKEN}
CLAIM_CMD=${CLAIM_CMD//__OVERRIDE__/$CLAIM_OVERRIDE}
CLAIM_CMD=${CLAIM_CMD//__MAX__/$CLAIM_MAX}
# override_owner_live <hold>: BOX_HOLD_OVERRIDE=1 is for a holder the operator KNOWS is dead, so before
# the take it reads the hold's owner line and, when that names a restart.sh or cutover.sh run
# (`<script> <tag> pid<N> <utc>`) that THIS host cannot prove gone, sets OVR_LIVE / OVR_PID and
# returns 0 -- the caller refuses with nothing touched. It fails CLOSED: two answers alone admit the
# override, `ps` naming that pid as some other process (a reused pid) or `kill -0` answering "No such
# process". Any other `ps` outcome -- refused, missing, an error exit, an empty answer, or exit 1 for
# a pid `kill -0` still finds -- leaves `kill -0` to decide, and a pid alive, alive and not ours to
# signal, or not provably gone counts as live, said so, with the hand release as the way on for a pid
# that was merely reused. box-a's C1 settle (#405, A2, 2026-10-31): a Cosmos restart hid the first run
# (pid4804) without killing it; the override rerun took its hold and restarted the engine under it,
# the first run's poll then saw READY, retook the hold the rerun had just released, and swept its
# prewarm on the rerun's engine while the rerun's own prewarm met that hold (`HELD age=1 ...
# pid4804`) and was skipped -- two restart.sh runs on one box. A holder elsewhere (another host, a
# kit keeper, the watchdog's loop-guard touch) is not seen here: the override goes on as before. A
# dry run and an answer-less box skip the check (the take decides). cutover.sh carries this function
# and override_refusal byte for byte (tests/restart-hold-override.test.sh compares them).
override_owner_live() {
  local o pid scr tag args psrc k krc
  OVR_LIVE=""; OVR_PID=""
  [ "$CLAIM_OVERRIDE" = 1 ] && [ "${DRY_RUN:-0}" != 1 ] || return 1
  o=$(box_ssh_retry "head -1 '$1' 2>/dev/null" 2>/dev/null | tr -d '\r' | grep -E '^(restart|cutover)\.sh [^ ]+ pid[0-9]+ ' | tail -1)
  [ -n "$o" ] || return 1
  scr=${o%% *}; tag=${o#* }; tag=${tag%% *}
  pid=$(printf '%s\n' "$o" | sed -nE 's/^[a-z]+\.sh [^ ]+ pid([0-9]+) .*/\1/p')
  [ -n "$pid" ] && [ "$pid" != "$$" ] || return 1
  args=$(ps -p "$pid" -o args= 2>/dev/null); psrc=$?
  if [ "$psrc" -eq 0 ] && [ -n "$args" ]; then   # ps named the pid: the holder still, or a reused pid
    case "$args" in
      *"$scr $tag "*|*"$scr $tag"|*"$scr "*" $tag "*|*"$scr "*" $tag") OVR_PID=$pid; OVR_LIVE="pid $pid is still running on this host ($args), owner line '$o'"; return 0 ;;
    esac
    return 1
  fi
  k=$(LC_ALL=C kill -0 "$pid" 2>&1); krc=$?
  if [ "$krc" -eq 0 ]; then k="is alive"
  else
    case "$k" in
      *"o such process"*) return 1 ;;   # the one proof of a dead holder
      *"ot permitted"*) k="is alive" ;;
      *) k="is not provably gone (kill -0: ${k:-exit $krc})" ;;
    esac
  fi
  OVR_PID=$pid
  OVR_LIVE="pid $pid $k on this host, owner line '$o' (its argv is unreadable here: ps exit $psrc${args:+, '$args'}; if that pid is not the run -- a reused pid -- release the hold by hand, on the box: rm -f '$1', and re-run WITHOUT the override)"
  return 0
}
override_refusal() {   # override_refusal <what is refused> <hold>: the die text for a live holder
  printf '%s' "REFUSING BOX_HOLD_OVERRIDE=1 for $1: ${2##*/}'s owner is NOT dead -- $OVR_LIVE. The override is for a holder you know is gone; this one is a live run that will go on polling, retake the hold and prewarm on the engine beside yours (box-a C1 settle A2, #405). Nothing was touched: wait for it to finish, or stop it first (kill $OVR_PID), then re-run with the override"
}

# ---- the front door's re-render: tls.sh, for --single and for --engine ------------------------------
# A rollback or a switch changes which engines serve, and Caddy's sites must follow: --single needs the
# one-engine door, and --engine on a box whose live Caddyfile was rendered for one engine (after a
# --single) needs the per-copy door back, or the copy names get no site and $HOSTNAME keeps forwarding
# to the one-engine port. caddy_render_cmd / caddy_hand_cmd <ENGINES line> [<mktemp prefix>]: tls.sh
# against a copy of the staged box file with <ENGINES line> appended -- remote shell text, `ENGINES=`
# for --single, `"ENGINES='<this run's ENGINES>'"` for --engine -- since lib.sh sources the box file
# (an ENGINES in the environment would not reach the render), with a newline first so a box file with
# no trailing newline cannot fuse the line onto its last one; after a Caddyfile.pre-<tag> backup,
# never overwritten (a re-send cannot back up its own render). Caddy's --watch loads the new file.
CADDY_BAK="$TOOLS_REMOTE/Caddyfile.pre-$TAG"
caddy_hand_cmd() {
  printf '%s' "cp -n '$TOOLS_REMOTE/Caddyfile' '$CADDY_BAK'; f=\$(mktemp); { cat '$TOOLS_REMOTE/boxes/$BOX.env'; echo; echo $1; } > \"\$f\"; BOX='$BOX' BOX_ENV=\"\$f\" TLS_PRESTAGE= bash '$TOOLS_REMOTE/tls.sh'"
}
caddy_render_cmd() {
  printf '%s' "f=\$(mktemp /tmp/$2-$TAG.XXXXXX) || { echo TLS_RC=mktemp; exit 0; }; { cat '$TOOLS_REMOTE/boxes/$BOX.env' && echo && echo $1; } > \"\$f\" || { rm -f \"\$f\"; echo TLS_RC=boxfile; exit 0; }; if [ -e '$CADDY_BAK' ]; then echo CADDY_BACKUP='$CADDY_BAK (kept from an earlier send)'; elif [ -f '$TOOLS_REMOTE/Caddyfile' ]; then cp -p '$TOOLS_REMOTE/Caddyfile' '$CADDY_BAK' && echo CADDY_BACKUP='$CADDY_BAK' || { rm -f \"\$f\"; echo CADDY_BACKUP=FAILED; echo TLS_RC=backup; exit 0; }; else echo CADDY_BACKUP=none; fi; BOX='$BOX' BOX_ENV=\"\$f\" TLS_PRESTAGE= bash '$TOOLS_REMOTE/tls.sh' > \"\$f.out\" 2>&1; rc=\$?; tail -n 4 \"\$f.out\" | sed 's/^/tls.sh: /'; rm -f \"\$f\" \"\$f.out\"; echo TLS_RC=\$rc"
}
# caddy_render <mode> <ENGINES line> <mktemp prefix> <layout> <how> <rendered for> <else stays on>:
# send the render, print the backup's name and tls.sh's tail. A failure is a WARNING with the hand
# command, never fatal: the engine(s) still come back.
caddy_render() {
  local mode=$1 line=$2 pfx=$3 layout=$4 how=$5 for=$6 stays=$7 cmd out trc said
  cmd=$(caddy_render_cmd "$line" "$pfx")
  note "$mode: re-rendering Caddy for the $layout layout (tls.sh, $how); Caddyfile backup: $CADDY_BAK"
  if [ "${DRY_RUN:-0}" = 1 ]; then run box_ssh_retry "$cmd"; return 0; fi
  out=$(box_ssh_retry "$cmd" | tr -d '\r')
  trc=$(printf '%s\n' "$out" | sed -n 's/^TLS_RC=//p' | tail -1)
  said=$(printf '%s\n' "$out" | sed -n 's/^CADDY_BACKUP=//p' | tail -1)
  printf '%s\n' "$out" | grep -a '^tls.sh: ' | while IFS= read -r l; do note "$l"; done
  case "$said" in
    none) note "Caddyfile backup: none (no $TOOLS_REMOTE/Caddyfile on the box to back up)" ;;
    ""|FAILED) note "Caddyfile backup: NOT taken (${said:-no answer})" ;;
    *) note "Caddyfile backup: $said" ;;
  esac
  if [ "$trc" = 0 ]; then
    note "$mode: Caddy re-rendered for $for (tls.sh exit 0; Caddy's --watch loads it) -- roll back with: cp '$CADDY_BAK' '$TOOLS_REMOTE/Caddyfile'"
  else
    note "WARNING: $mode: the $layout Caddyfile was NOT rendered (tls.sh: ${trc:-no answer from the box}) -- $HOSTNAME may stay on $stays; on the box: $(caddy_hand_cmd "$line")"
  fi
}
# ==== the per-engine mode: --engine <id>|all (#415) ===============================================
# A box file's ENGINES (lib.sh's engine_select; box-c's 2xTP4) runs one engine per entry, each from its
# own sglang-args-<id>.txt / sglang-env-<id>.txt on its own GPUs and ports, each guarded by its own
# `watchdog.sh <id>` under its own hold watchdog-<id>.hold. `--engine <id>|all` restarts the named
# engine(s): the one-engine flow's steps, each scoped to them, never the legacy pair -- sglang-args.txt
# and sglang-env.txt stay the TP8 line `--single` goes back to and a reboot reads. In order:
#   preflight  read-only: whether the one-engine line still runs (its watchdog, its launcher, or an
#              `sglang serve` without --nccl-port: relaunch.sh <id> always passes one, and refuses an
#              args file that carries its own), and each engine's installed files where `-` keeps them
#   hold       each named engine's hold -- and watchdog.hold while the one-engine line runs -- claimed
#              by CLAIM_TXT as the one-engine flow claims its own; one refusal releases the others
#   install    sglang-args-<id>.txt / sglang-env-<id>.txt, each backed up to .pre-<tag> first (a file
#              not there yet is new: a failure removes it, nothing to restore)
#   switch     only while the one-engine line runs: its watchdog and launcher, then its engine (it
#              holds every GPU, so nothing per-engine runs beside it)
#   stop       per engine: its launcher (`relaunch.sh <id>`), then its process group from
#              engine-<id>.pid -- or, that file stale, the engine on its port -- TERM then KILL, then
#              its own GPUs under 8,000 MiB. Never a box-wide pkill: an engine not named keeps serving
#   checkout   --tree, once (one tree serves every engine: an engine not named runs on from what it
#              loaded, and its next relaunch boots the new checkout)
#   launch     ONE send, never re-sent: every named engine's log gets its own `===== RESTART <tag>`
#              marker and its `nohup setsid relaunch.sh <id>`, all together (the hand-set
#              --mem-fraction-static fits both copies booting at once)
#   poll       every named engine each round: its banner or /health on its port, and its own log's
#              fail-fast probes (E_DEAD_CHECK, E_CRASH_CHECK -- a second server_args= in ITS log --,
#              E_VALIDATION_CHECK, which reads its process group and its port, not the box's)
#   after      per READY engine: its warm gate, its prewarm against its URL (marker suffixed -<id>,
#              log prewarm-<tag>-<id>.log, under its own hold), the sampling gate on its port. Exit 0
#              when every engine passed; 6 when any is NOT READY; else 7 when any sampling gate failed
#              -- the last line names each engine's verdict.
#   marker     The per-engine marker is keyed by the engine's BOOT as well (#1256): it holds
#              `boot=<pid>@<start>` (the pid in engine-<id>.pid, its start time from /proc/<pid>/stat,
#              boxout-gate's boot id), and a prewarm is skipped only when that line names the boot
#              running now. A new engine process -- this run's own launch, a watchdog relaunch, a
#              container restart -- always sweeps: the sweep's in-process compiles die with the
#              process (#1255), whatever the disk caches hold. The one-engine block below keys its
#              marker the same way (#1271), off the line's `sglang serve` leader.
# tests/restart-engines.test.sh runs it end to end on a shimmed box, and holds E_*_CHECK equal to the
# one-engine DEAD_CHECK / CRASH_CHECK / VALIDATION_CHECK below with the legacy log put back.
eget() { local v="E_$1_$2"; printf '%s' "${!v:-}"; }   # eget <NAME> <id>
eset() { printf -v "E_$1_$2" '%s' "$3"; }              # eset <NAME> <id> <value>
# The engine's boot id on the box (#1256): with pf=<engine-<id>.pid> set, b=<pid>@<start time in
# clock ticks, /proc/<pid>/stat field 22> -- boxout-gate's boot id -- or empty (no pid file, a
# non-numeric one, no such process). The per-engine prewarm marker carries it (`boot=` line). The
# one-engine prewarm block's PREWARM_BOOT_SH is this text from `b=""` on, its pid from pgrep
# (tests/prewarm-plan.test.sh E holds the two equal).
E_BOOT_SH='p=$(head -1 "$pf" 2>/dev/null); b=""; case "$p" in ""|*[!0-9]*) ;; *) st=$(cat "/proc/$p/stat" 2>/dev/null) && { r="${st##*) }"; set -- $r; [ -n "${20:-}" ] && b="$p@${20}"; } ;; esac'

# e_dead_check / e_crash_check <log> <marker line>: DEAD_CHECK / CRASH_CHECK below, on one engine's
# log. e_validation_check <log> <marker line> <port> <pid file>: VALIDATION_CHECK's log scan, behind
# "no engine left" read as THAT engine's: no process in its pid file's group, and nothing on its port.
e_dead_check() {
  printf '%s' "n=\$(grep -an '^===== RESTART ' $1 2>/dev/null | tail -1 | cut -d: -f1); [ \"\${n:-0}\" -gt $2 ] || n=$2; tail -n +\$n $1 2>/dev/null | grep -aE -m1 'unrecognized arguments|error: argument '"
}
e_crash_check() {
  printf '%s' "n=\$(grep -an '^===== RESTART ' $1 2>/dev/null | tail -1 | cut -d: -f1); [ \"\${n:-0}\" -gt $2 ] || n=$2; tail -n +\$n $1 2>/dev/null | awk '/(Scheduler|TokenizerManager|DetokenizerManager) hit an exception/ && x == \"\" { x = \$0; next } x != \"\" && y == \"\" && /^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)(: |\$)/ { y = \$0 } /server_args=/ { c++ } END { if (x != \"\") print \"SCHEDULER: \" x (y == \"\" ? \"\" : \" ... \" y); else if (c > 1) print \"ENGINES: \" c }'"
}
e_validation_check() {
  printf '%s' "p=\$(head -1 '$4' 2>/dev/null); case \"\$p\" in ''|*[!0-9]*) ;; *) g=\$(ps -A -o pgid= 2>/dev/null) || exit 0; printf '%s\\n' \"\$g\" | awk -v g=\"\$p\" '\$1 == g { f = 1 } END { exit !f }' && exit 0 ;; esac; pgrep -f 'sglang serv[e].*--port $3( |\$)' >/dev/null 2>&1; a=\$?; [ \$a -eq 1 ] || exit 0; n=\$(grep -an '^===== RESTART ' $1 2>/dev/null | tail -1 | cut -d: -f1); [ \"\${n:-0}\" -gt $2 ] || n=$2; tail -n +\$n $1 2>/dev/null | awk '/Traceback \\(most recent call last\\):/ { v = \"\"; t = 1; next } t && /^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)(: |\$)/ { v = \$0; t = 0 } END { if (v ~ /^(ValueError|AssertionError)(: |\$)/) print \"VALIDATION: \" v }'"
}

# The box-side helpers the switch and the stop share (prepended to those two sends):
#   e_tree <pid>                   the pid and every descendant
#   e_sig_proc <sig> <pid>         its process group when it leads one, else it and its descendants
#   e_group <pid file> <ERE>       the pid file's process group while it is still that engine's: its
#                                  leader, alive, matches the engine's `sglang serve ... --port` ERE; a
#                                  dead leader's surviving members (sglang::scheduler_TP*) are sglang's
#   e_sig_engine <sig> <pid file> <ERE>   that group -- or, the pid file stale, e_sig_proc on each
#                                  process the port-scoped ERE matches
#   e_legacy                       the one-engine line's `sglang serve` pids: no --nccl-port in argv
# (The ERE text in a send never matches the send's own shell: `serv[e]` is not `serve`.)
IFS= read -r -d '' E_STOP_FNS <<'SH'   # read's exit 1 at EOF is expected (set +e)
e_tree() { ps -A -o pid=,ppid= 2>/dev/null | awk -v r="$1" '{ k[$2] = k[$2] " " $1 } END { q = r; o = r; while (q != "") { n = split(q, a, " "); q = ""; for (i = 1; i <= n; i++) { m = split(k[a[i]], c, " "); for (j = 1; j <= m; j++) { o = o " " c[j]; q = q " " c[j] } } } print o }'; }
e_sig_proc() { if [ "$(ps -o pgid= -p "$2" 2>/dev/null | tr -d ' ')" = "$2" ]; then kill -"$1" -- -"$2" 2>/dev/null; else kill -"$1" $(e_tree "$2") 2>/dev/null; fi; }
e_group() {
  p=$(head -1 "$1" 2>/dev/null); case "$p" in ''|*[!0-9]*) return 1 ;; esac
  rows=$(ps -A -o pid=,pgid=,args= 2>/dev/null | awk -v g="$p" '$2 == g'); [ -n "$rows" ] || return 1
  lead=$(printf '%s\n' "$rows" | awk -v g="$p" '$1 == g')
  if [ -n "$lead" ]; then printf '%s' "$lead" | grep -qE -- "$2" || return 1
  else printf '%s\n' "$rows" | grep -q sglang || return 1; fi
  echo "$p"
}
e_sig_engine() {
  if g=$(e_group "$2" "$3"); then kill -"$1" -- -"$g" 2>/dev/null
  else for q in $(pgrep -f "$3" 2>/dev/null); do e_sig_proc "$1" "$q"; done; fi
}
e_legacy() { for q in $(pgrep -f 'sglang serv[e]' 2>/dev/null); do tr '\0' ' ' < "/proc/$q/cmdline" 2>/dev/null | grep -q -- '--nccl-port ' || echo "$q"; done; }
SH

if [ -n "$ENG_SEL" ]; then
  # Every named engine's names, from engine_select in a subshell (the run's own globals stay put).
  for id in $E_IDS; do
    eval "$(engine_select "$id" && printf 'eset ARGS %s %q; eset ENV %s %q; eset HOLD %s %q; eset LOG %s %q; eset URL %s %q; eset PORT %s %q; eset GPUS %s %q; eset WARMING %s %q; eset WARMLOG %s %q; eset PID %s %q; eset SFX %s %q\n' \
      "$id" "$ENGINE_ARGS_FILE" "$id" "$ENGINE_ENV_FILE" "$id" "$WATCHDOG_HOLD" "$id" "$ENGINE_SERVE_LOG" "$id" "$ENGINE_URL" \
      "$id" "$ENGINE_PORT" "$id" "$ENGINE_GPUS" "$id" "$ENGINE_WARMING" "$id" "$WARM_LOG" "$id" "$ENGINE_PIDFILE" "$id" "$SFX")"
    [ -n "$(eget PORT "$id")" ] || die "engine $id: engine_select named no port -- nothing touched"
  done
  e_pat() { printf '%s' "sglang serv[e].*--port $(eget PORT "$1")( |\$)"; }
  e_claim_cmd() {   # e_claim_cmd <hold> <override> <max>: CLAIM_TXT for one hold
    local c=${CLAIM_TXT//__HOLD__/$1}
    c=${c//__TOKEN__/$CLAIM_TOKEN}; c=${c//__OVERRIDE__/$2}; c=${c//__MAX__/$3}
    printf '%s' "$c"
  }
  e_release() {   # e_release <hold> <why>: token-guarded, as the one-engine release
    local r
    r=$(run box_ssh_retry "o=\$(head -1 '$1' 2>/dev/null); if [ \"\$o\" = '$CLAIM_TOKEN' ]; then rm -f '$1'; echo RELEASED; else echo \"NOT-OURS \${o:-<none>}\"; fi") \
      || { note "could NOT release ${1##*/} ($2; the box did not answer): its watchdog drops it once it is HOLD_MAX (default 900 s) old, or by hand on the box: rm -f '$1'"; return 0; }
    case "$r" in
      RELEASED*) note "released ${1##*/} ($2)" ;;
      NOT-OURS*) note "${1##*/} is no longer this run's ($r): left in place, the holder releases it" ;;
    esac
  }
  E_TAKEN=""
  e_release_all() { local h; for h in $E_TAKEN; do e_release "$h" "$1"; done; E_TAKEN=""; }

  # ---- preflight: the one-engine line, and the files `-` keeps -----------------------------------
  E_PRE="echo LEGACY_WD=\$(pgrep -f '$TOOLS_REMOTE/watchdo[g].sh( --|\$)' 2>/dev/null | tr '\n' ' '); echo LEGACY_RELAUNCH=\$(pgrep -f '$TOOLS_REMOTE/relaunc[h].sh\$' 2>/dev/null | tr '\n' ' '); echo LEGACY_ENGINE=\$(e_legacy | tr '\n' ' ')"
  for id in $E_IDS; do
    E_PRE="$E_PRE; [ -f '$(eget ARGS "$id")' ] && echo HAS_ARGS_$id; [ -f '$(eget ENV "$id")' ] && echo HAS_ENV_$id"
  done
  # The live Caddyfile's layout: a per-copy site for EVERY engine of ENGINES (not only the named ones:
  # the door is box-wide) is its `https://<copy name>:` site block (tls.sh's tls_site, closed by a
  # `}` at column 0) holding both its own warm gate (`try_files /engine-<id>.warming`) and its port as
  # an upstream field (`127.0.0.1:<port>` on the reverse_proxy line). A copy missing any of them is a
  # door rendered for one engine -- by --single (ENGINES emptied: no copy site), or TLS_PRESTAGE (each
  # copy site routed to the legacy port behind the legacy engine.warming; on box-c copy a's port IS the
  # legacy one, so only the gate tells that site apart) -- and this run re-renders it after the stop.
  E_CADDY_ALL=""
  E_PRE="$E_PRE; c='$TOOLS_REMOTE/Caddyfile'; [ -f \"\$c\" ] && echo HAS_CADDYFILE"
  for e in $ENGINES; do
    id=${e%%:*}
    E_CADDY_ALL="$E_CADDY_ALL $id"
    eval "$(engine_select "$id" && printf 'eset HOST %s %q; eset CPORT %s %q; eset CWARM %s %q\n' "$id" "$ENGINE_HOST" "$id" "$ENGINE_PORT" "$id" "${ENGINE_WARMING##*/}")"
    E_PRE="$E_PRE; awk -v h='https://$(eget HOST "$id"):' -v p='127.0.0.1:$(eget CPORT "$id")' -v w='try_files /$(eget CWARM "$id")' 'index(\$0, h) == 1 { on = 1; next } on && /^}/ { on = 0 } on && index(\$0, w) { fw = 1 } on { for (i = 1; i <= NF; i++) if (\$i == p) fp = 1 } END { exit !(fw && fp) }' \"\$c\" 2>/dev/null && echo CADDY_SITE_$id"
  done
  E_PRE_OUT=$(box_ssh_retry "$E_STOP_FNS
$E_PRE; true"); rc=$?
  E_LEGACY=""
  if [ "$rc" -ne 0 ]; then
    [ "${DRY_RUN:-0}" = 1 ] || die "could not read the engines' state on the box (ssh exit $rc) -- nothing touched"
    note "DRY_RUN: could not read the engines' state (ssh exit $rc) -- showing the switch from the one-engine line too; door check skipped (the Caddyfile was not read: shown as a per-copy door, nothing re-rendered); a real run dies here"
    E_LEGACY="unknown"
    for id in $E_IDS; do E_PRE_OUT="$E_PRE_OUT
HAS_ARGS_$id
HAS_ENV_$id"; done
    E_PRE_OUT="$E_PRE_OUT
HAS_CADDYFILE"
    for id in $E_CADDY_ALL; do E_PRE_OUT="$E_PRE_OUT
CADDY_SITE_$id"; done
  else
    E_LEGACY=$(printf '%s\n' "$E_PRE_OUT" | sed -n 's/^LEGACY_[A-Z]*=//p' | tr -s ' \n' '  ' | sed 's/^ *//; s/ *$//')
  fi
  E_CADDY=""   # the engines of ENGINES the live Caddyfile has no per-copy site for
  if printf '%s\n' "$E_PRE_OUT" | grep -qx HAS_CADDYFILE; then
    for id in $E_CADDY_ALL; do printf '%s\n' "$E_PRE_OUT" | grep -qx "CADDY_SITE_$id" || E_CADDY="$E_CADDY $id"; done
    E_CADDY=${E_CADDY# }
    [ -z "$E_CADDY" ] || note "the live Caddyfile has no per-copy site for: $E_CADDY (rendered for one engine -- --single, or a TLS_PRESTAGE render): this run re-renders the per-copy door after the stop"
  else
    note "no $TOOLS_REMOTE/Caddyfile on the box: no front door to re-render (tls.sh sets one up)"
  fi
  for id in $E_IDS; do
    if [ -z "$ARGS_FILE" ] || [ "$ARGS_FILE" = - ]; then
      printf '%s\n' "$E_PRE_OUT" | grep -qx "HAS_ARGS_$id" \
        || die "engine $id has no $(eget ARGS "$id") on the box to keep (\`-\`): pass the args file its first per-engine restart installs -- nothing touched"
    fi
    if [ -z "$ENV_FILE_ARG" ] || [ "$ENV_FILE_ARG" = - ]; then
      printf '%s\n' "$E_PRE_OUT" | grep -qx "HAS_ENV_$id" \
        || die "engine $id has no $(eget ENV "$id") on the box to keep (\`-\`): pass an env file, or \`default\` -- nothing touched"
    fi
  done
  E_HOLDS=""
  if [ -n "$E_LEGACY" ]; then
    note "the one-engine line still runs (pids: $E_LEGACY): this run switches $BOX to per-engine -- that line's watchdog, launcher and engine are stopped under watchdog.hold"
    E_HOLDS="$HOLD"
  fi
  for id in $E_IDS; do E_HOLDS="$E_HOLDS $(eget HOLD "$id")"; done

  # ---- hold: each a claim, as the one-engine flow takes its own; one refusal releases the rest ----
  e_die_installing() {   # before the stop: no engine ran on what this run installed
    local f
    note "FATAL: $1 -- no engine was touched: each still runs the line it had"
    for f in $BACKED_UP; do
      if box_ssh_retry "cp -f '$f.pre-$TAG' '$f'" >/dev/null; then note "restored $f from $f.pre-$TAG"
      else note "could NOT restore $f from $f.pre-$TAG -- put it back by hand before anything relaunches that engine"; fi
    done
    for f in $E_NEW; do
      if box_ssh_retry "rm -f '$f'" >/dev/null; then note "removed $f (new in this run)"
      else note "could NOT remove $f (new in this run) -- remove it by hand"; fi
    done
    e_release_all "install failed"
    fatal_hint
    exit 1
  }
  e_die_held() {   # from the stop on: every hold KEPT, as die_held keeps the one-engine one
    local h
    note "FATAL: $1"
    for h in $E_TAKEN; do box_ssh "touch '$h'" >/dev/null 2>&1 || note "could not re-touch $h (the box did not answer)"; done
    note "KEPT (re-touched just now; each engine's watchdog sleeps through its hold for HOLD_MAX, default 900 s, then relaunches whatever is installed):$E_TAKEN -- an engine is stopped or in a state this run could not see; releasing now would let a watchdog relaunch into your retry"
    [ -z "$BACKED_UP$E_NEW" ] || note "this run's installs are in place:$BACKED_UP$E_NEW -- the retry passes - - for them"
    note "retry: BOX_HOLD_OVERRIDE=1 ${0##*/} --engine $ENG_SEL <a NEW tag> - -${TREE:+ --tree $TREE}"
    note "or, to hand the engines back to their watchdogs with no retry, on the box: rm -f$E_TAKEN"
    fatal_hint
    exit 1
  }
  for h in $E_HOLDS; do
    note "hold ${h##*/} (a fresh hold is another run's claim: read, never just touched)"
    if [ "${DRY_RUN:-0}" = 1 ]; then
      run box_ssh_retry "$(e_claim_cmd "$h" "$CLAIM_OVERRIDE" "$CLAIM_MAX")"
      E_TAKEN="$E_TAKEN $h"
      continue
    fi
    if override_owner_live "$h"; then e_release_all "refused"; die "$(override_refusal "engine(s) $E_IDS on $BOX" "$h")"; fi
    E_OUT=$(box_ssh_retry "$(e_claim_cmd "$h" "$CLAIM_OVERRIDE" "$CLAIM_MAX")"); rc=$?
    case "$rc" in
      0)
        E_TAKEN="$E_TAKEN $h"
        case "$E_OUT" in
          OVERRIDE*) note "BOX_HOLD_OVERRIDE=1: taking ${h##*/} from a hold this run was told is dead -- $E_OUT" ;;
          STALE*)    note "${h##*/} was stale, its watchdog's own rule would drop it too -- $E_OUT" ;;
          *)         note "hold taken: ${h##*/} ${E_OUT:-$CLAIM_TOKEN}" ;;
        esac ;;
      7) E_TAKEN="$E_TAKEN $h"; e_release_all "refused"; die "${h##*/} is THIS run's ($E_OUT) but could not be touched -- no engine step ran" ;;
      3|5) e_release_all "refused"; die "REFUSING to take engine(s) $E_IDS on $BOX: ${h##*/} is a live claim by another run -- $E_OUT. Nothing was touched; wait for its holder, or BOX_HOLD_OVERRIDE=1 only for a hold you know is dead" ;;
      6) e_release_all "refused"; die "could not clear a STALE ${h##*/} on $BOX ($E_OUT) -- nothing touched" ;;
      255) e_release_all "refused"; die "could not take ${h##*/} (ssh exit 255, the box did not answer) -- nothing touched" ;;
      *) e_release_all "refused"; die "could not take ${h##*/} (the box answered '$E_OUT', ssh exit $rc) -- nothing touched" ;;
    esac
  done

  # ---- install: each engine's own pair, the legacy pair never --------------------------------------
  E_NEW=""
  e_install() {   # e_install <local file|__SEED__> <box file> <what>
    local r rc
    if [ "${DRY_RUN:-0}" = 1 ]; then
      run box_ssh_retry "if [ -f '$2' ]; then cp -f '$2' '$2.pre-$TAG' && echo BACKED || echo FAILED; else echo NEW; fi"
      r=BACKED; rc=0
    else
      r=$(box_ssh_retry "if [ -f '$2' ]; then cp -f '$2' '$2.pre-$TAG' && echo BACKED || echo FAILED; else echo NEW; fi"); rc=$?
    fi
    case "$r" in
      *FAILED*) e_die_installing "could not back up $2 to $2.pre-$TAG" ;;
      *NEW*) E_NEW="$E_NEW $2" ;;
      *) [ "$rc" -eq 0 ] || e_die_installing "could not back up $2 (exit $rc)"; BACKED_UP="$BACKED_UP $2" ;;
    esac
    if [ "$1" = __SEED__ ]; then
      run box_ssh_retry "cp -f '$TOOLS_ENV_SEED' '$2'" || e_die_installing "could not seed $2 from $TOOLS_ENV_SEED"
    else
      run box_scp_retry "$1" "root@$BOX_IP:$2" || e_die_installing "could not install $3 $1 as $2"
    fi
    note "installed $3 -> $2"
  }
  for id in $E_IDS; do
    if [ -n "$ARGS_FILE" ] && [ "$ARGS_FILE" != - ]; then e_install "$ARGS_FILE" "$(eget ARGS "$id")" "engine $id args"
    else note "engine $id: keeping its installed $(eget ARGS "$id")"; fi
    case "$ENV_FILE_ARG" in
      ''|-) note "engine $id: keeping its installed $(eget ENV "$id")" ;;
      default) e_install __SEED__ "$(eget ENV "$id")" "engine $id env (seed ${TOOLS_ENV_SEED##*/})" ;;
      *) e_install "$ENV_FILE_ARG" "$(eget ENV "$id")" "engine $id env" ;;
    esac
  done

  # ---- switch from the one-engine line -------------------------------------------------------------
  if [ -n "$E_LEGACY" ]; then
    note "switch: stopping the one-engine line's watchdog, launcher and engine ($TAG)"
    run box_ssh_retry "$E_STOP_FNS
pkill -TERM -f '$TOOLS_REMOTE/watchdo[g].sh( --|\$)' 2>/dev/null; pkill -TERM -f '$TOOLS_REMOTE/relaunc[h].sh\$' 2>/dev/null; sleep 1
for q in \$(e_legacy); do e_sig_proc TERM \$q; done; sleep 5; for q in \$(e_legacy); do e_sig_proc KILL \$q; done
echo \"one-engine line left: \$(pgrep -f '$TOOLS_REMOTE/watchdo[g].sh( --|\$)' 2>/dev/null | wc -l) watchdog, \$(e_legacy | wc -l) engine\"" \
      || e_die_held "the switch from the one-engine line did not complete (exit $?)"
  fi

  # ---- stop: each named engine, scoped --------------------------------------------------------------
  note "stopping engine(s) $E_IDS ($TAG): launcher, process group, GPUs"
  E_STOP=""
  for id in $E_IDS; do
    E_STOP="$E_STOP
pkill -TERM -f '$TOOLS_REMOTE/relaunc[h].sh $id( |\$)' 2>/dev/null; e_sig_engine TERM '$(eget PID "$id")' '$(e_pat "$id")'"
  done
  E_STOP="$E_STOP
sleep 5"
  for id in $E_IDS; do
    E_STOP="$E_STOP
e_sig_engine KILL '$(eget PID "$id")' '$(e_pat "$id")'"
  done
  for id in $E_IDS; do
    E_STOP="$E_STOP
for i in \$(seq 1 60); do u=\$(nvidia-smi --id=$(eget GPUS "$id") --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1); [ \"\${u:-0}\" -lt 8000 ] && break; sleep 2; done; echo \"gpu mem after stop (engine $id, GPUs $(eget GPUS "$id")): \${u:-?} MiB\""
  done
  run box_ssh_retry "$E_STOP_FNS$E_STOP" || e_die_held "the stop of engine(s) $E_IDS did not complete (exit $?)"

  if [ -n "$TREE" ]; then
    note "git checkout -q $TREE -- in $SGLANG_TREE (the tree every engine boots from)"
    run box_ssh_retry "git -C '$SGLANG_TREE' checkout -q '$TREE' --" \
      || e_die_held "git checkout -q $TREE -- in $SGLANG_TREE failed (exit $?)"
  fi

  # ---- the front door: back to the per-copy layout, when the live one was rendered for one engine ----
  # After a --single rollback (or a prestage) the copy names have no site and $HOSTNAME forwards to the
  # one-engine port, so the engines below would serve behind a door that never reaches them. caddy_render
  # re-renders it from the box file plus this run's ENGINES, after a Caddyfile.pre-<tag> backup whose
  # name is printed; a failure is a WARNING with the hand command, never fatal.
  if [ -n "$E_CADDY" ]; then
    caddy_render "--engine $ENG_SEL" "\"ENGINES='$ENGINES'\"" engines per-copy "the box file's ENGINES=\"$ENGINES\"" "the per-copy layout" "the one-engine site (no site for $E_CADDY)"
  fi

  # ---- launch: one send, every named engine ---------------------------------------------------------
  note "relaunch.sh $E_IDS ($TAG)"
  E_LAUNCH=""
  for id in $E_IDS; do
    lg=$(eget LOG "$id")
    # `ulimit -c 0` (#1300): RLIMIT_CORE 0 for the launch, so a crash never dumps engine memory onto the host's disk (relaunch.sh sets it again and its gate refuses any other value).
    E_LAUNCH="$E_LAUNCH( umask 077; : >> $lg ); echo \"===== RESTART $TAG engine $id \$(date -u +%Y-%m-%dT%H:%M:%SZ) =====\" >> $lg; ulimit -c 0; nohup setsid bash '$TOOLS_REMOTE/relaunch.sh' $id >> $lg 2>&1 < /dev/null & "
  done
  run box_ssh "${E_LAUNCH}sleep 3; echo launched $E_IDS" \
    || e_die_held "the launch of engine(s) $E_IDS failed (exit $?): whether any relaunch.sh started is unknown -- the retry's stop clears whatever did"
  T0=$(date +%s)
  for id in $E_IDS; do
    m=$(box_ssh_retry "grep -an '===== RESTART $TAG ' $(eget LOG "$id") 2>/dev/null | tail -1 | cut -d: -f1")
    case "$m" in ''|*[!0-9]*) m=1 ;; esac
    eset MARK "$id" "$m"
  done
  if [ "${DRY_RUN:-0}" = 1 ]; then
    note "DRY_RUN: skipping the tree/readiness read-back (no live box to query)"
    e_release_all "DRY_RUN"
    note "DRY_RUN: done (engines $E_IDS)"
    exit 0
  fi
  SHA=$(box_ssh "git -C '$SGLANG_TREE' log -1 --format='%h %s'" 2>/dev/null)
  # Each engine's tree, read under ITS installed env (tree_probe_env, #806): a copy whose
  # sglang-env-<id>.txt sets a PYTHONPATH overlay runs that checkout, not SGLANG_TREE. The label is
  # `tree=<dir>@<sha>` when that is SGLANG_TREE, else `tree=<dir>@<sha> ([PYTHONPATH overlay over ]
  # SGLANG_TREE=<dir> at <its sha>)` -- the leading <dir>@<sha> is always the tree the copy runs.
  for id in $E_IDS; do
    tree_probe_env "$(eget ENV "$id")" "$TOOLS_ENV_SEED"
    E_PROBE=$(box_ssh_retry --stdin "$TREE_PROBE_RUN" "bash -s -- '$TOOLS_REMOTE/boxes/$BOX.env' '$SGLANG_TREE' '$VENV'" 2>/dev/null | tr -d '\r')
    L_TREE=$(printf '%s\n' "$E_PROBE" | sed -n 's/^LAUNCH_TREE=//p' | head -1)
    L_SHA=$(printf '%s\n' "$E_PROBE" | sed -n 's/^LAUNCH_SHA=//p' | head -1)
    L_SAME=$(printf '%s\n' "$E_PROBE" | sed -n 's/^SAME=//p' | head -1)
    L_PP=$(printf '%s\n' "$E_PROBE" | sed -n 's/^LAUNCH_PYTHONPATH=//p' | head -1)
    L_ENVF=$(printf '%s\n' "$E_PROBE" | sed -n 's/^LAUNCH_ENV=//p' | head -1)
    L_BASE=$(printf '%s\n' "$E_PROBE" | sed -n 's/^BASE_SHA=//p' | head -1)
    n="tree=${L_TREE:-$SGLANG_TREE}@${L_SHA:-${SHA:-n/a}}"
    if [ -n "$L_TREE" ] && [ -n "$L_SHA" ] && [ "$L_SAME" = no ]; then
      n="$n (${L_PP:+PYTHONPATH overlay over }SGLANG_TREE=$SGLANG_TREE at ${L_BASE:-n/a})"
      [ -n "$TREE" ] && note "WARNING: engine $id: --tree $TREE was checked out in SGLANG_TREE=$SGLANG_TREE, not in $L_TREE, the tree this engine runs -- the checkout did not change what it serves"
    fi
    eset TREE "$id" "$n"; eset LSHA "$id" "$L_SHA"
    note "engine $id launched from $n${L_PP:+ (PYTHONPATH=$L_PP in $L_ENVF, as relaunch.sh exports it)}"
  done

  # ---- poll: every pending engine each round -------------------------------------------------------
  E_PENDING=" $E_IDS "
  E_REFRESHED=$T0
  LAST_PRINT=$T0
  while :; do
    for id in $E_PENDING; do
      lg=$(eget LOG "$id"); mk=$(eget MARK "$id")
      ok=0
      [ -n "$(box_ssh "tail -n +$mk $lg 2>/dev/null | grep -a 'fired up and ready to roll' | tail -1")" ] && ok=1
      if [ "$ok" -ne 1 ]; then
        [ "$(box_ssh "curl -s -m 10 -o /dev/null -w '%{http_code}' '$(eget URL "$id")/health'" 2>/dev/null)" = 200 ] && ok=1
      fi
      if [ "$ok" = 1 ]; then
        eset VERDICT "$id" READY; eset UP "$id" $(( $(date +%s) - T0 )); E_PENDING=${E_PENDING/ $id / }
        continue
      fi
      v=$(box_ssh "$(e_dead_check "$lg" "$mk")" 2>/dev/null)
      [ -z "$v" ] || v="DEAD: $v"
      if [ -z "$v" ]; then
        v=$(box_ssh "$(e_crash_check "$lg" "$mk")" 2>/dev/null)
        case "$v" in SCHEDULER:*|ENGINES:*) ;; *) v="" ;; esac
      fi
      if [ -z "$v" ]; then
        v=$(box_ssh "$(e_validation_check "$lg" "$mk" "$(eget PORT "$id")" "$(eget PID "$id")")" 2>/dev/null)
        case "$v" in VALIDATION:*) ;; *) v="" ;; esac
      fi
      if [ -n "$v" ]; then
        eset VERDICT "$id" NOT-READY; eset WHY "$id" "$v"; eset UP "$id" $(( $(date +%s) - T0 ))
        E_PENDING=${E_PENDING/ $id / }
        continue
      fi
      w=$(box_ssh "tail -n +$mk $lg 2>/dev/null | grep -aE 'Traceback|OutOfMemory|Xid|CUDA error' | head -1")
      if [ -n "$w" ] && [ "$w" != "$(eget WARN "$id")" ]; then
        note "WARNING (engine $id, this run's log slice): $w"
        eset WARN "$id" "$w"
      fi
    done
    [ -n "${E_PENDING// /}" ] || break
    NOW=$(date +%s)
    if [ $(( NOW - E_REFRESHED )) -ge 110 ]; then
      for h in $E_TAKEN; do
        box_ssh "o=\$(head -1 '$h' 2>/dev/null); [ \"\$o\" = '$CLAIM_TOKEN' ] && touch '$h'" >/dev/null 2>&1
      done
      E_REFRESHED=$NOW
    fi
    if [ $(( NOW - LAST_PRINT )) -ge 60 ]; then
      note "still waiting on engine(s)$E_PENDING($(( NOW - T0 ))s elapsed)"
      LAST_PRINT=$NOW
    fi
    if [ $(( NOW - T0 )) -ge 1800 ]; then
      for id in $E_PENDING; do eset VERDICT "$id" NOT-READY; eset WHY "$id" "no banner or /health 200 in 1800 s"; eset UP "$id" $(( NOW - T0 )); done
      break
    fi
    sleep 5
  done
  note "release holds (readiness poll over)"
  e_release_all "readiness poll over"

  # ---- after: per engine ---------------------------------------------------------------------------
  E_SUMMARY=""; E_DOWN=""; E_SAMPLING=""
  for id in $E_IDS; do
    lg=$(eget LOG "$id"); url=$(eget URL "$id"); hh=$(eget HOLD "$id")
    E_ARGS_LINE=$(box_ssh_retry "grep -v '^#' '$(eget ARGS "$id")' 2>/dev/null")
    if [ "$(eget VERDICT "$id")" != READY ]; then
      why=$(eget WHY "$id")
      note "NOT READY: engine $id after $(eget UP "$id")s (tag=$TAG, $(eget TREE "$id")) -- ${why:-no verdict}; its log on the box: $lg (this launch's section from line $(eget MARK "$id")); args: $E_ARGS_LINE"
      case "$why" in
        ENGINES:*) note "engine $id: TWO ENGINES in its log since this run's marker -- another launch (its watchdog, or a second restart) booted beside it on its GPUs. Re-run with a new tag: the stop clears that engine's launcher and group first." ;;
      esac
      E_SUMMARY="$E_SUMMARY $id=NOT-READY"; E_DOWN="$E_DOWN $id"
      continue
    fi
    note "READY: engine $id ($TAG) up in $(eget UP "$id")s on $url, $(eget TREE "$id"), args: $E_ARGS_LINE"
    wg=$(eget WARMING "$id"); wl=$(eget WARMLOG "$id")
    WARM_RAW=$(box_ssh "g='$wg'; for i in \$(seq 1 ${WARM_GATE_WAIT:-240}); do [ -e \"\$g\" ] || { echo \"OPEN after \${i}s\"; tail -n 8 '$wl' 2>/dev/null; exit 0; }; sleep 1; done; echo CLOSED; tail -n 8 '$wl' 2>/dev/null" 2>&1 | tr -d '\r')
    # The the provider ssh banner lands ahead of the box's answer: read from its first OPEN/CLOSED line on.
    WARM_OUT=$(printf '%s\n' "$WARM_RAW" | sed -nE '/^(OPEN after [0-9]+s|CLOSED)$/,$p')
    case "$WARM_OUT" in
      OPEN*)   note "engine $id: warm gate: ${WARM_OUT%%$'\n'*} (warm-short.py log $wl)" ;;
      CLOSED*) note "WARNING: engine $id: warm gate still CLOSED after ${WARM_GATE_WAIT:-240}s -- its door answers 503 until warm-short.py finishes or times out (log $wl)" ;;
      *)       note "WARNING: engine $id: warm gate state unknown -- the box answered: ${WARM_RAW:-nothing}" ;;
    esac
    if [ "${PREWARM:-on}" = off ]; then
      note "engine $id: prewarm SKIPPED (PREWARM=off): first-time kernel compiles fall on live traffic"
    else
      pw_log="/workspace/prewarm-$TAG$(eget SFX "$id").log"
      pw_sha=$(eget LSHA "$id"); pw_sha=${pw_sha:-$SHA}; pw_sha=${pw_sha%% *}
      pw_cp=$(printf '%s\n' "$E_ARGS_LINE" | sed -n 's/.*--chunked-prefill-size[= ][= ]*\([0-9][0-9]*\).*/\1/p' | head -1)
      pw_mark=""
      case "$pw_sha" in
        ''|*[!0-9a-f]*) note "engine $id: prewarm marker: none -- the launched tree's sha is '${pw_sha:-<none>}'" ;;
        *) pw_mark="${PREWARM_MARKER_DIR:-/root/.cache/sglang}/.prewarmed-$pw_sha-$(printf '%s' "${VENV:-}" | cksum | awk '{print $1}')-cp${pw_cp:-auto}$(eget SFX "$id")" ;;
      esac
      pw_pf=$(eget PID "$id")
      if [ -n "$pw_mark" ] && [ "${PREWARM:-on}" != force ] \
         && box_ssh_retry "pf='$pw_pf'; $E_BOOT_SH; [ -n \"\$b\" ] && grep -qx \"boot=\$b\" '$pw_mark'" >/dev/null 2>&1; then
        note "engine $id: prewarm skipped: marker $pw_mark names the boot running now (PREWARM=force runs it anyway)"
      else
        pw_hold=$(box_ssh_retry "$(e_claim_cmd "$hh" 0 "${CLAIM_MAX:-}")" 2>&1); rc=$?
        if [ "$rc" -eq 7 ]; then E_TAKEN=" $hh"; e_release_all "the take could not touch it"; fi
        [ "$rc" -eq 0 ] || hh=""
        if [ -z "$hh" ]; then
          note "WARNING: engine $id: prewarm NOT run: its hold could not be taken again for it (exit $rc: ${pw_hold:-no output}) -- the engine is up; rerun it by hand under its hold: box_ssh \"cd $TOOLS_REMOTE && BOX=$BOX ${VENV:-/opt/sglang}/bin/python3 prewarm-box-d.py --base-url $url\" (log $pw_log)"
        else
          E_TAKEN=" $hh"
          trap 'e_release_all "interrupted"; exit 130' INT
          trap 'e_release_all "terminated"; exit 143' TERM
          trap 'e_release_all "hung up"; exit 129' HUP
          note "engine $id: prewarm under ${hh##*/} ($pw_hold); prewarm-box-d.py --base-url $url (box log $pw_log)"
          pw_t0=$(date +%s)
          E_PW_OUT=$(trap - INT TERM HUP; box_ssh "cd '$TOOLS_REMOTE' || exit 9; ( while sleep ${PREWARM_HOLD_EVERY:-110}; do kill -0 \$\$ 2>/dev/null || exit 0; o=\$(head -1 '$hh' 2>/dev/null); [ \"\$o\" = '$CLAIM_TOKEN' ] && touch '$hh'; done ) >/dev/null 2>&1 & k=\$!; py='${VENV:-/nonexistent}/bin/python3'; [ -x \"\$py\" ] || py=python3; pf='$pw_pf'; $E_BOOT_SH; b0=\$b; BOX='$BOX' timeout ${PREWARM_TIMEOUT:-1800} \"\$py\" prewarm-box-d.py --base-url '$url' > '$pw_log' 2>&1; rc=\$?; { kill \$k; wait \$k; } 2>/dev/null; m='$pw_mark'; $E_BOOT_SH; if [ \$rc -eq 0 ] && [ -n \"\$m\" ]; then if [ -z \"\$b\" ] || [ \"\$b\" != \"\$b0\" ]; then echo \"WARNING: prewarm marker NOT written (\$m): the engine's boot was '\${b0:-none}' before the sweep and '\${b:-none}' after\"; elif e=\$( { mkdir -p \"\${m%/*}\" && printf 'boot=%s\\n%s %s\\n' \"\$b\" '$TAG' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \"\$m\"; } 2>&1 ); then echo \"prewarm marker written: \$m (boot \$b)\"; else echo \"WARNING: prewarm marker NOT written (\$m: \${e:-no error text})\"; fi; fi; grep -aE ' SLOW|ERROR' '$pw_log' | head -20; grep -a '^PREWARM DONE' '$pw_log' | tail -1; exit \$rc" 2>&1)
          rc=$?
          e_release_all "prewarm exit $rc"
          trap - INT TERM HUP
          [ -n "$E_PW_OUT" ] && note "$E_PW_OUT"
          if [ "$rc" -eq 0 ]; then note "engine $id: prewarm done in $(( $(date +%s) - pw_t0 ))s"
          else note "WARNING: engine $id: prewarm exit $rc after $(( $(date +%s) - pw_t0 ))s, no marker written -- the engine is up; rerun it by hand under its hold (log $pw_log)"; fi
        fi
      fi
    fi
    if [ "${SAMPLING_GATE:-on}" = off ]; then
      note "engine $id: sampling gate SKIPPED (SAMPLING_GATE=off)"
      E_SUMMARY="$E_SUMMARY $id=READY"
      continue
    fi
    note "engine $id: sampling gate: /server_info + $(eget ARGS "$id") + the cmdline of the engine on port $(eget PORT "$id")"
    E_SG_OUT=$(box_ssh_retry --stdin "$SAMPLING_GATE_PY" "python3 - '$url' '$(eget ARGS "$id")' /etc/environment /proc '$(eget PORT "$id")'"); rc=$?
    [ -n "$E_SG_OUT" ] && note "$E_SG_OUT"
    if [ "$rc" -eq 0 ] && printf '%s\n' "$E_SG_OUT" | grep -q '^SAMPLING PASS'; then
      E_SUMMARY="$E_SUMMARY $id=READY"
    else
      printf '%s\n' "$E_SG_OUT" | grep -q '^SAMPLING FAIL' \
        || note "engine $id: SAMPLING UNVERIFIED: the gate did not run to a verdict (exit $rc) -- treat as a failed gate"
      note "engine $id: SAMPLING GATE FAILED: it is UP and serving but omitted samplers do not resolve to temperature 1.0 / top_p 1.0 as pinned"
      E_SUMMARY="$E_SUMMARY $id=SAMPLING-FAIL"; E_SAMPLING="$E_SAMPLING $id"
    fi
  done
  if [ -n "$E_DOWN" ]; then
    note "engines ($TAG):$E_SUMMARY -- NOT READY:$E_DOWN. Exit 6."
    exit 6
  fi
  if [ -n "$E_SAMPLING" ]; then
    note "engines ($TAG):$E_SUMMARY -- sampling gate failed:$E_SAMPLING. Exit 7."
    exit 7
  fi
  note "engines ($TAG):$E_SUMMARY"
  exit 0
fi

note "hold watchdog (a fresh watchdog.hold is another run's claim: read, never just touched)"
if [ "${DRY_RUN:-0}" = 1 ]; then
  run box_ssh_retry "$CLAIM_CMD"
else
  if override_owner_live "$HOLD"; then die "$(override_refusal "$BOX" "$HOLD")"; fi
  CLAIM_OUT=$(box_ssh_retry "$CLAIM_CMD"); CLAIM_RC=$?
  case "$CLAIM_RC" in
    0)
      case "$CLAIM_OUT" in
        OVERRIDE*) note "BOX_HOLD_OVERRIDE=1: taking the box from a hold this run was told is dead -- $CLAIM_OUT" ;;
        STALE*)    note "the hold was stale, the watchdog's own rule would drop it too -- $CLAIM_OUT" ;;
        *)         note "hold taken: ${CLAIM_OUT:-$CLAIM_TOKEN}" ;;
      esac ;;
    255)
      die "could not take watchdog.hold (ssh exit 255, the box did not answer) -- nothing else touched: the engine runs as it was. If the take landed before the link dropped, the watchdog drops it once it is HOLD_MAX (default 900 s) old, or release it by hand: $(hold_release_cmd)" ;;
    3|5)
      die "REFUSING to take $BOX: watchdog.hold is a live claim by another run -- $CLAIM_OUT. A fresh hold is a box reservation, not an idle flag: on gpu1018 (2026-10-21 23:14:53Z) a restart touched a live read's hold, stopped the engine under 64 in-flight items and removed the read's hold (#3299 under #3293). Nothing was touched here: the engine, args, env and watchdog are as they were. Wait for the holder to release it -- a live read refreshes its hold, so the mtime stays young -- and re-run. Only for a holder you KNOW is gone (a restart that died with its hold kept: die_held's retry; or the watchdog's own loop-guard hold) take it deliberately: BOX_HOLD_OVERRIDE=1, which says so in this run's log. To hand an idle box back to the watchdog instead: $(hold_release_cmd)" ;;
    6)
      die "could not clear a STALE watchdog.hold on $BOX ($CLAIM_OUT) -- nothing else touched: the engine runs as it was" ;;
    7)
      die "watchdog.hold is THIS run's ($CLAIM_OUT) but could not be touched -- no engine step ran; release it by hand: $(hold_release_cmd) -- else the watchdog drops it once it is HOLD_MAX (default 900 s) old" ;;
    *)
      die "could not take watchdog.hold (the box answered '$CLAIM_OUT', ssh exit $CLAIM_RC) -- nothing else touched: the engine runs as it was" ;;
  esac
fi

# Each install: the backup first (its failure: nothing changed yet), then the copy (its failure: the
# backup is put back). Either dies through die_installing, with the engine untouched.
if [ -n "$ARGS_FILE" ] && [ "$ARGS_FILE" != "-" ]; then
  note "installing args from $ARGS_FILE (backup: sglang-args.txt.pre-$TAG)"
  run box_ssh_retry "cp -f '$TOOLS_ARGS' '$TOOLS_ARGS.pre-$TAG' 2>/dev/null; true" \
    || die_installing "could not back up $TOOLS_ARGS (ssh exit 255, the box did not answer)"
  BACKED_UP="$BACKED_UP $TOOLS_ARGS"
  run box_scp_retry "$ARGS_FILE" "root@$BOX_IP:$TOOLS_ARGS" \
    || die_installing "could not install $ARGS_FILE as $TOOLS_ARGS (scp failed ${BOX_SSH_TRIES:-3} times)"
fi

if [ -n "$ENV_FILE_ARG" ] && [ "$ENV_FILE_ARG" != "-" ]; then
  run box_ssh_retry "cp -f '$TOOLS_ENV' '$TOOLS_ENV.pre-$TAG' 2>/dev/null; true" \
    || die_installing "could not back up $TOOLS_ENV (ssh exit 255, the box did not answer)"
  BACKED_UP="$BACKED_UP $TOOLS_ENV"
  if [ "$ENV_FILE_ARG" = default ]; then
    note "reseeding sglang-env.txt from $TOOLS_ENV_SEED (default: ${ENV_SEED:+ENV_SEED in the box file, over }the env of the serve line SGLANG_TREE=$SGLANG_TREE runs -- see the seed variable's comment; its presence was checked before the hold) -- relaunch.sh now refuses to boot without one, so this can no longer just delete it"
    run box_ssh_retry "cp -f '$TOOLS_ENV_SEED' '$TOOLS_ENV'" \
      || die_installing "could not reseed $TOOLS_ENV from $TOOLS_ENV_SEED"
  else
    note "installing env from $ENV_FILE_ARG (backup: sglang-env.txt.pre-$TAG)"
    run box_scp_retry "$ENV_FILE_ARG" "root@$BOX_IP:$TOOLS_ENV" \
      || die_installing "could not install $ENV_FILE_ARG as $TOOLS_ENV (scp failed ${BOX_SSH_TRIES:-3} times)"
  fi
fi

# An in-flight relaunch.sh is killed first: watchdog.sh decides to relaunch the moment the
# old engine is gone, which can be a second before this run touched the hold, and its
# launcher would then race the relaunch below (two engines loading weights on one box,
# 03:27 UTC 2026-09-17 - the survivor sized its pool against a half-occupied GPU).
if [ -n "$SINGLE" ] && [ -n "$E_IDS" ]; then
  # --single on a per-engine box: each engine's hold claimed (a live one is another run's: refused),
  # each engine's watchdog and launcher stopped, the holds released; the stop below then takes every
  # engine with the box-wide pkill the one-engine line always used.
  note "--single: back to the one-engine line -- engine(s) $E_IDS: holds claimed, watchdogs stopped; the stop below takes the engines"
  S_TAKEN=""
  s_release() {
    local h
    for h in $S_TAKEN; do
      run box_ssh_retry "o=\$(head -1 '$h' 2>/dev/null); [ \"\$o\" = '$CLAIM_TOKEN' ] && rm -f '$h'; true" >/dev/null \
        || note "could NOT release ${h##*/} (the box did not answer): rm -f '$h' on the box"
    done
  }
  for id in $E_IDS; do
    h=$(engine_select "$id" && printf '%s' "$WATCHDOG_HOLD")
    S_CMD=${CLAIM_TXT//__HOLD__/$h}
    S_CMD=${S_CMD//__TOKEN__/$CLAIM_TOKEN}
    S_CMD=${S_CMD//__OVERRIDE__/$CLAIM_OVERRIDE}
    S_CMD=${S_CMD//__MAX__/$CLAIM_MAX}
    if [ "${DRY_RUN:-0}" = 1 ]; then run box_ssh_retry "$S_CMD"; S_TAKEN="$S_TAKEN $h"; continue; fi
    if override_owner_live "$h"; then s_release; die_installing "--single: $(override_refusal "engine $id on $BOX" "$h")"; fi
    S_OUT=$(box_ssh_retry "$S_CMD"); rc=$?
    case "$rc" in
      0|7) S_TAKEN="$S_TAKEN $h"; note "hold taken: ${h##*/} ${S_OUT:-$CLAIM_TOKEN}" ;;
      *) s_release; die_installing "--single: ${h##*/} is not this run's to take (exit $rc: ${S_OUT:-no output}) -- another run holds engine $id" ;;
    esac
  done
  run box_ssh_retry "pkill -TERM -f '$TOOLS_REMOTE/watchdo[g].sh [a-z][a-z0-9]*( |\$)' 2>/dev/null; pkill -TERM -f '$TOOLS_REMOTE/relaunc[h].sh [a-z][a-z0-9]*( |\$)' 2>/dev/null; sleep 1; echo \"engine watchdogs left: \$(pgrep -f '$TOOLS_REMOTE/watchdo[g].sh [a-z][a-z0-9]*( |\$)' 2>/dev/null | wc -l)\"" \
    || { s_release; die_held "--single: could not stop the engines' watchdogs (exit $?)"; }
  s_release
fi
note "stopping engine ($TAG)"
# Its last command is an echo, so a nonzero exit is ssh's own 255: re-sent, then die_held (the stop
# may have run before the link dropped, so the engine's state is unknown -- the hold stays).
run box_ssh_retry "pkill -TERM -f '$TOOLS_REMOTE/relaunc[h].sh' 2>/dev/null; pkill -TERM -f 'sglang serv[e]' 2>/dev/null; pkill -TERM -f 'sglang:[:]' 2>/dev/null; sleep 5; pkill -KILL -f 'sglang serv[e]' 2>/dev/null; pkill -KILL -f 'sglang:[:]' 2>/dev/null; sleep 2; for i in \$(seq 1 60); do used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ -n \"\$used\" ] && [ \"\$used\" -lt 8000 ] && break; sleep 2; done; echo \"gpu mem after kill: \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')\"" \
  || die_held "the engine stop got no answer from the box (ssh exit 255, ${BOX_SSH_TRIES:-3} tries) -- it may have run in part"

if [ -n "$SINGLE" ] && [ -n "$E_IDS" ]; then
  # --single's front door, once the engines are down: a leftover engine-<id>.warming (a copy stopped
  # mid-warm) and the per-copy Caddy sites would hold $HOSTNAME at 503 behind the one engine about to
  # boot, and nothing else on this path re-runs tls.sh. So each engine's warm flag is removed (the
  # legacy engine.warming stays relaunch.sh's: it gates the one engine below), and tls.sh re-renders
  # the one-engine Caddyfile -- the same tls.sh the 2-engine cutover runs, by caddy_render with
  # ENGINES emptied for the render. A failure here is a WARNING with the hand command: the engine
  # still comes back.
  S_FLAGS=""
  for id in $E_IDS; do S_FLAGS="$S_FLAGS '$(engine_select "$id" && printf '%s' "$ENGINE_WARMING")'"; done
  note "--single: clearing the engines' warm flags:$S_FLAGS (the legacy engine.warming stays relaunch.sh's)"
  run box_ssh_retry "rm -f$S_FLAGS; echo \"engine warm flags left: \$(ls -d$S_FLAGS 2>/dev/null | wc -l)\"" \
    || note "WARNING: --single: could not clear the engines' warm flags (the box did not answer): rm -f$S_FLAGS on the box, or $HOSTNAME stays at 503"
  caddy_render --single ENGINES= single one-engine "ENGINES emptied for the render" "the one engine" "the per-copy sites"
fi

if [ -n "$TREE" ]; then
  note "checkout: $TREE"
  # `--`: <ref> is a ref, never a pathspec (see cutover.sh's checkout step). A 255 is re-sent (one
  # ref, no fetch between sends: the same commit); git's own refusal is not.
  CO_RC=0
  run box_ssh_retry "git -C '$SGLANG_TREE' checkout -q '$TREE' --" || CO_RC=$?
  if [ "$CO_RC" -ne 0 ]; then
    if [ "$CO_RC" -eq 255 ]; then CO_WHY="ssh exit 255, the box did not answer, ${BOX_SSH_TRIES:-3} tries"; else CO_WHY="git exit $CO_RC"; fi
    die_held "git checkout $TREE failed ($CO_WHY) -- the engine is already stopped"
  fi
fi

note "restart $TAG"
# Sent ONCE, never re-sent: a second send boots a second engine. Its last command is an echo, so a
# nonzero exit is ssh's 255 -- whether relaunch.sh started is then unknown, and die_held keeps the hold.
# The serve log carries the key (the engine's server_args= line): it is CREATED 0600 (#3003), in a
# subshell so the umask never reaches relaunch.sh or the engine; `>>` keeps an existing log's mode.
LAUNCH_RC=0
# `ulimit -c 0` (#1300): RLIMIT_CORE 0 for the launch, so a crash never dumps engine memory onto the host's disk (relaunch.sh sets it again and its gate refuses any other value).
# The legacy engine.pid (#932): relaunch.sh writes the pidfile only for `relaunch.sh <id>`, so a bare
# launch left the previous boot's pid there and boxout-gate.sh read a dead boot (GATE NOBOOT). `$!` is
# the same pid relaunch.sh <id> writes: nohup and setsid exec (a background child of a non-interactive
# shell leads no group, so setsid never forks), and relaunch.sh execs `sglang serve`
# (tests/restart-engine-pid.test.sh runs this line against a shim box).
run box_ssh "( umask 077; : >> /workspace/sglang-serve.log ); echo '===== RESTART $TAG $(date -u) =====' >> /workspace/sglang-serve.log; ulimit -c 0; nohup setsid bash '$TOOLS_REMOTE/relaunch.sh' >> /workspace/sglang-serve.log 2>&1 < /dev/null & printf '%s\n' \$! > '$TOOLS_REMOTE/engine.pid.tmp' && mv -f '$TOOLS_REMOTE/engine.pid.tmp' '$TOOLS_REMOTE/engine.pid'; sleep 3; echo launched pid \$(pgrep -f 'sglang serv[e]' | tr '\n' ' ')" || LAUNCH_RC=$?
[ "$LAUNCH_RC" -eq 0 ] \
  || die_held "the relaunch got no answer from the box (ssh exit $LAUNCH_RC): relaunch.sh may or may not have started, and it is never re-sent (a second send boots a second engine)"
T0=$(date +%s)
# The hold stays until the readiness poll ends (see cutover.sh: an early release let the
# 30 s watchdog poll count a loading engine as three health misses and kill it).

MARKER_LINE=$(box_ssh_retry "grep -an '===== RESTART $TAG ' /workspace/sglang-serve.log 2>/dev/null | tail -1 | cut -d: -f1")
MARKER_LINE=${MARKER_LINE:-1}
note "this run's marker: sglang-serve.log line $MARKER_LINE (readiness poll never looks before it)"

if [ "${DRY_RUN:-0}" = 1 ]; then
  note "DRY_RUN: skipping the tree/readiness read-back (no live box to query)"
  run box_ssh_retry "rm -f '$HOLD'"
  exit 0
fi

# The probe below runs under the installed sglang-env.txt's PYTHONPATH (tree_probe_env, #806): the
# block from here through its `fi` stays cutover.sh's byte for byte, so the env text rides in
# TREE_PROBE_SH itself for this run, and the overlay is named on its own note after the block.
tree_probe_env "$TOOLS_ENV" "$TOOLS_ENV_SEED"; TREE_PROBE_SH=$TREE_PROBE_RUN
SHA=$(box_ssh "git -C '$SGLANG_TREE' rev-parse --short=12 HEAD 2>/dev/null || echo n/a")
# Every later `tree=$SHA` names the tree the engine LAUNCHES from (TREE_PROBE_SH above): the plain
# sha when that is this box file's SGLANG_TREE (or the probe answered nothing usable), else
# `<launched sha> (<launched dir>; SGLANG_TREE=<dir> at <its sha>)`.
LAUNCH=$(box_ssh_retry --stdin "$TREE_PROBE_SH" "bash -s -- '$TOOLS_REMOTE/boxes/$BOX.env' '$SGLANG_TREE' '${VENV:-}'" 2>/dev/null | tr -d '\r')
L_TREE=$(printf '%s\n' "$LAUNCH" | sed -n 's/^LAUNCH_TREE=//p' | head -1)
L_SHA=$(printf '%s\n' "$LAUNCH" | sed -n 's/^LAUNCH_SHA=//p' | head -1)
L_HOW=$(printf '%s\n' "$LAUNCH" | sed -n 's/^LAUNCH_HOW=//p' | head -1)
L_SAME=$(printf '%s\n' "$LAUNCH" | sed -n 's/^SAME=//p' | head -1)
if [ -n "$L_TREE" ] && [ -n "$L_SHA" ] && [ "$L_SAME" = no ]; then
  case "$L_HOW" in
    import) L_WHY="the tree the box's VENV imports sglang from" ;;
    *) L_WHY="the SGLANG_TREE the box's staged boxes/$BOX.env names, which relaunch.sh reads" ;;
  esac
  note "tree now at $L_SHA in $L_TREE ($L_WHY) -- NOT this box file's SGLANG_TREE=$SGLANG_TREE, which is at $SHA; READY's tree= names the launched one"
  [ -n "$TREE" ] && note "WARNING: --tree $TREE was checked out in SGLANG_TREE=$SGLANG_TREE, not in $L_TREE, the tree this engine runs -- the checkout did not change what it serves"
  SHA="$L_SHA ($L_TREE; SGLANG_TREE=$SGLANG_TREE at $SHA)"
else
  note "tree now at $SHA"
fi
L_PP=$(printf '%s\n' "$LAUNCH" | sed -n 's/^LAUNCH_PYTHONPATH=//p' | head -1)
L_ENVF=$(printf '%s\n' "$LAUNCH" | sed -n 's/^LAUNCH_ENV=//p' | head -1)
[ -n "$L_PP" ] && note "the tree above was read under PYTHONPATH=$L_PP from $L_ENVF, as relaunch.sh exports it to the engine"

note "waiting for readiness (fired up and ready to roll / health) from this run's marker (line $MARKER_LINE), progress every 60s, ceiling 1800s, failing fast on argparse's refusal, a scheduler exception, a second engine or a validation hook's refusal (its traceback, no engine process left) in the latest launch's log section"
# The fail-fast probe: argparse's refusal (`unrecognized arguments`, `error: argument `) in the LATEST
# launch's section of the serve log -- cutover.sh's DEAD_CHECK word for word, with bringup.sh's
# wait_engine_ready (#1588) patterns. `sglang serve` prints it and exits at once (relaunch.sh execs
# into it, and the watchdog is held), so nothing will ever answer /health: on gpu1018 (2026-10-18)
# such an engine sat out the whole ceiling. The slice starts at the LATER of the last line-start
# `===== RESTART ` marker and this run's own MARKER_LINE (MARKER_LINE alone when the log has no
# line-start marker): the last line-start marker can be an EARLIER launch's -- this run's marker
# landed mid-line after engine output with no trailing newline, so `^` misses it while MARKER_LINE's
# unanchored grep finds it -- and that launch's refusal must never condemn this run. A refusal
# before either line never misfires. tests/restart-fail-fast.test.sh runs both scripts.
DEAD_CHECK="n=\$(grep -an '^===== RESTART ' /workspace/sglang-serve.log 2>/dev/null | tail -1 | cut -d: -f1); [ \"\${n:-0}\" -gt $MARKER_LINE ] || n=$MARKER_LINE; tail -n +\$n /workspace/sglang-serve.log 2>/dev/null | grep -aE -m1 'unrecognized arguments|error: argument '"
# The crash probe (#2870), over the SAME slice, sent when DEAD_CHECK finds nothing. It prints
#   SCHEDULER: <the first `Scheduler hit an exception` line> [... <the first Error/Exception line
#              after it, the traceback's verdict>] -- sglang's scheduler logs that, then SIGQUITs the
#              server (managers/scheduler.py), so the whole engine is going down and nothing will
#              answer /health; `TokenizerManager hit an exception` / `DetokenizerManager hit an
#              exception` match too (the same print_exception_wrapper then kills the process tree: #833's
#              MERGE arm, 2026-11-04 12:25:02Z, an AttributeError on the first prefill output, and the
#              poll sat 1,800 s on an engine gone at 12:26:08Z);
#   ENGINES: <n> -- else, when the slice holds n > 1 `server_args=` lines: each `sglang serve` logs
#              it ONCE per launch (entrypoints/engine.py, before its subprocesses start), so a second
#              one means a second engine booting on the same GPUs (gpu1021, 2026-10-21 01:27:22Z: a
#              watchdog relaunch beside this run's; both died on `Loaded weights leave no GPU memory
#              for the KV cache`, and the poll sat out its ceiling).
# Either ends the poll NOT READY (exit 6). cutover.sh sends it word for word.
CRASH_CHECK="n=\$(grep -an '^===== RESTART ' /workspace/sglang-serve.log 2>/dev/null | tail -1 | cut -d: -f1); [ \"\${n:-0}\" -gt $MARKER_LINE ] || n=$MARKER_LINE; tail -n +\$n /workspace/sglang-serve.log 2>/dev/null | awk '/(Scheduler|TokenizerManager|DetokenizerManager) hit an exception/ && x == \"\" { x = \$0; next } x != \"\" && y == \"\" && /^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)(: |\$)/ { y = \$0 } /server_args=/ { c++ } END { if (x != \"\") print \"SCHEDULER: \" x (y == \"\" ? \"\" : \" ... \" y); else if (c > 1) print \"ENGINES: \" c }'"
# The validation probe (#305), over the SAME slice, sent when neither probe above fired. It prints
#   VALIDATION: <the traceback's verdict line> -- when the slice's LAST `Traceback (most recent call
#              last):` ends in `ValueError` or `AssertionError` (its first `<Name>Error:` line, the one
#              CRASH_CHECK reads too) AND no `sglang serve` / `sglang::` process is left on the box.
#              That is a startup check refusing the line -- box-d-up's arg_groups validation hooks
#              (`validate_short_prefill_protection` refused the R1 ladder's 2048 rungs, 2026-10-24) or
#              a ServerArgs assertion -- raised in the main process before any scheduler starts, so
#              neither argparse's refusal nor `Scheduler hit an exception` is in the log, the engine
#              is gone, and the poll used to sit out its 1,800 s ceiling on it (#296 stopped it by hand,
#              twice). The process check keeps a traceback the engine survives (a logged, handled
#              ValueError) from ending a live boot; pgrep's exit 1 (no match) is the only "gone", so a
#              pgrep that cannot answer never fires it. Ends the poll NOT READY (exit 6), the hold
#              released like every NOT READY. cutover.sh sends it word for word.
#              tests/restart-validation-fail-fast.test.sh runs it against both scripts.
VALIDATION_CHECK="pgrep -f 'sglang serv[e]' >/dev/null 2>&1; a=\$?; pgrep -f 'sglang:[:]' >/dev/null 2>&1; b=\$?; [ \$a -eq 1 ] && [ \$b -eq 1 ] || exit 0; n=\$(grep -an '^===== RESTART ' /workspace/sglang-serve.log 2>/dev/null | tail -1 | cut -d: -f1); [ \"\${n:-0}\" -gt $MARKER_LINE ] || n=$MARKER_LINE; tail -n +\$n /workspace/sglang-serve.log 2>/dev/null | awk '/Traceback \(most recent call last\):/ { v = \"\"; t = 1; next } t && /^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)(: |\$)/ { v = \$0; t = 0 } END { if (v ~ /^(ValueError|AssertionError)(: |\$)/) print \"VALIDATION: \" v }'"
LAST_PRINT=$T0
LAST_HOLD_REFRESH=$T0
LAST_WARN=""
READY=0
DEAD=""
CRASH=""
VALIDATION=""
while :; do
  banner=$(box_ssh "tail -n +$MARKER_LINE /workspace/sglang-serve.log 2>/dev/null | grep -a 'fired up and ready to roll' | tail -1")
  if [ -n "$banner" ]; then READY=1; fi
  if [ "$READY" -ne 1 ]; then
    code=$(box_ssh "curl -s -m 10 -o /dev/null -w '%{http_code}' '$ENGINE_URL/health'" 2>/dev/null)
    [ "$code" = 200 ] && READY=1
  fi
  [ "$READY" = 1 ] && break
  DEAD=$(box_ssh "$DEAD_CHECK" 2>/dev/null)
  [ -n "$DEAD" ] && break
  CRASH=$(box_ssh "$CRASH_CHECK" 2>/dev/null)
  case "$CRASH" in SCHEDULER:*|ENGINES:*) break ;; *) CRASH="" ;; esac
  VALIDATION=$(box_ssh "$VALIDATION_CHECK" 2>/dev/null)
  case "$VALIDATION" in VALIDATION:*) break ;; *) VALIDATION="" ;; esac
  WARN=$(box_ssh "tail -n +$MARKER_LINE /workspace/sglang-serve.log 2>/dev/null | grep -aE 'Traceback|OutOfMemory|Xid|CUDA error' | head -1")
  if [ -n "$WARN" ] && [ "$WARN" != "$LAST_WARN" ]; then
    note "WARNING (from this run's log slice): $WARN"
    LAST_WARN=$WARN
  fi
  NOW=$(date +%s)
  # Re-assert this run's claim every 110 s, the kit keeper's own cadence (#3316): the watchdog's
  # staleness rule is mtime + HOLD_MAX (900 s by default) and this poll's ceiling is 1,800 s, so a
  # slow boot would otherwise let the hold go stale mid-poll and hand the box to the watchdog under
  # this very restart (a relaunch beside this one's, KV-cache OOM on gpu1021). Only while the hold is
  # still ours: a peer's deliberate takeover is neither re-created nor clobbered.
  if [ $(( NOW - LAST_HOLD_REFRESH )) -ge 110 ]; then
    box_ssh "o=\$(head -1 '$HOLD' 2>/dev/null); [ \"\$o\" = '$CLAIM_TOKEN' ] && touch '$HOLD'" >/dev/null 2>&1
    LAST_HOLD_REFRESH=$NOW
  fi
  if [ $(( NOW - LAST_PRINT )) -ge 60 ]; then
    note "still waiting (${NOW}-${T0}=$(( NOW - T0 ))s elapsed)"
    LAST_PRINT=$NOW
  fi
  [ $(( NOW - T0 )) -ge 1800 ] && break
  sleep 5
done
ELAPSED=$(( $(date +%s) - T0 ))
ARGS_LINE=$(box_ssh_retry "grep -v '^#' '$TOOLS_ARGS' 2>/dev/null")
note "release hold (readiness poll over)"
# Only while it is still THIS run's claim (#3316): a peer that took the box over with
# BOX_HOLD_OVERRIDE must keep its hold -- an unconditional rm here would hand the box back to the
# watchdog in the middle of that peer's restart (two engines, gpu1021).
RELEASE_OUT=$(run box_ssh_retry "o=\$(head -1 '$HOLD' 2>/dev/null); if [ \"\$o\" = '$CLAIM_TOKEN' ]; then rm -f '$HOLD'; echo RELEASED; else echo \"NOT-OURS \${o:-<none>}\"; fi") \
  || note "could NOT release watchdog.hold (the box did not answer): the watchdog drops it once it is HOLD_MAX (default 900 s) old, or release it by hand: $(hold_release_cmd)"
case "$RELEASE_OUT" in
  NOT-OURS*) note "watchdog.hold is no longer this run's ($RELEASE_OUT): left in place, the holder releases it" ;;
esac

if [ "$READY" = 1 ]; then
  note "READY: $TAG up in ${ELAPSED}s, tree=$SHA, args: $ARGS_LINE"
  # The short warm gate (#379): relaunch.sh closed the front door ($TOOLS_REMOTE/engine.warming) at
  # this boot, and warm-short.py opens it after one cold ~60k fill, three ~700-token extends and one
  # image turn on the fresh process. That runs on EVERY boot, watchdog relaunches included, so this
  # run never sends the warm itself (no double run). It is waited for here so the prewarm sweep below
  # never overlaps it and the router recheck after READY finds the door open; a gate still closed
  # after WARM_GATE_WAIT s (default 240) only WARNs -- warm-short.py opens it itself at its timeout.
  WARM_RAW=$(box_ssh "g='$TOOLS_REMOTE/engine.warming'; for i in \$(seq 1 ${WARM_GATE_WAIT:-240}); do [ -e \"\$g\" ] || { echo \"OPEN after \${i}s\"; tail -n 8 '/workspace/$BOX-warm.log' 2>/dev/null; exit 0; }; sleep 1; done; echo CLOSED; tail -n 8 '/workspace/$BOX-warm.log' 2>/dev/null" 2>&1 | tr -d '\r')
  # The the provider ssh banner lands ahead of the box's answer: read from its first OPEN/CLOSED line on.
  WARM_OUT=$(printf '%s\n' "$WARM_RAW" | sed -nE '/^(OPEN after [0-9]+s|CLOSED)$/,$p')
  case "$WARM_OUT" in
    OPEN*)   note "warm gate: ${WARM_OUT%%$'\n'*} (warm-short.py log /workspace/$BOX-warm.log)" ;;
    CLOSED*) note "WARNING: warm gate still CLOSED after ${WARM_GATE_WAIT:-240}s -- the door answers 503 until warm-short.py finishes or times out (log /workspace/$BOX-warm.log)" ;;
    *)       note "WARNING: warm gate state unknown -- the box answered: ${WARM_RAW:-nothing}" ;;
  esac
  printf '%s\n' "$WARM_OUT" | sed -n '2,$p' | while IFS= read -r l; do note "  warm: $l"; done
  # The post-READY prewarm (#340), cutover.sh's word for word: prewarm-box-d.py against the loopback
  # engine while the router window is still open, so a new tree's first-time compiles (mHC's
  # mix-stats Triton kernel per m class, 17-34 s a rank -- box-a's 35 s prod stall, 2026-10-27; the
  # DeepGEMM einsum configs) are paid here, not by the first user. A warm cache makes it near a
  # no-op. A failure only WARNs: the engine is up either way. PREWARM=off skips it. It runs under
  # this run's watchdog.hold again and leaves a marker that skips it next time (#353, below).
  if [ "${PREWARM:-on}" = off ]; then
    note "prewarm SKIPPED (PREWARM=off): first-time kernel compiles fall on live traffic"
  else
    PREWARM_LOG="/workspace/prewarm-$TAG.log"
    # The warm-tree marker (#353): <dir>/.prewarmed-<launched tree sha>-<cksum of the VENV path>-cp<the
    # args line's --chunked-prefill-size, else auto>, written on the box only by a prewarm that exited
    # 0. With it present the prewarm is skipped; PREWARM=force runs it anyway. No sha, no marker. A
    # write the box refuses (mkdir or the redirect) is said so, with its error, and the run goes on:
    # `prewarm marker NOT written (<marker>: <why>); the next restart will prewarm again`.
    # The marker is keyed by the engine's BOOT as well (#1256's per-engine rule, #1271): it holds
    # `boot=<pid>@<start>` -- the one-engine line's `sglang serve` leader (the oldest such process;
    # this line launches no pid file) and its start time from /proc/<pid>/stat, restart.sh's
    # E_BOOT_SH from the pid on -- and a prewarm is skipped only when that line names the boot
    # running now. A new engine process (this run's launch, a watchdog relaunch, a container
    # restart) always sweeps: the sweep's in-process compiles die with the process. A boot that
    # cannot be read, or moved during the sweep, writes no marker, said so.
    PREWARM_BOOT_SH='p=$(pgrep -o -f "sglang serv[e]" 2>/dev/null); b=""; case "$p" in ""|*[!0-9]*) ;; *) st=$(cat "/proc/$p/stat" 2>/dev/null) && { r="${st##*) }"; set -- $r; [ -n "${20:-}" ] && b="$p@${20}"; } ;; esac'
    PREWARM_SHA=${L_SHA:-$SHA}; PREWARM_SHA=${PREWARM_SHA%% *}
    PREWARM_CP=$(printf '%s\n' "$ARGS_LINE" | sed -n 's/.*--chunked-prefill-size[= ][= ]*\([0-9][0-9]*\).*/\1/p' | head -1)
    PREWARM_MARK=""
    case "$PREWARM_SHA" in
      ''|*[!0-9a-f]*) note "prewarm marker: none -- the launched tree's sha is '${PREWARM_SHA:-<none>}', so this prewarm is never skipped and writes no marker" ;;
      *) PREWARM_MARK="${PREWARM_MARKER_DIR:-/root/.cache/sglang}/.prewarmed-$PREWARM_SHA-$(printf '%s' "${VENV:-}" | cksum | awk '{print $1}')-cp${PREWARM_CP:-auto}" ;;
    esac
    # The prewarm's own hold (#353): the readiness poll released this run's watchdog.hold above, and a
    # cold tree's compiles fail the watchdog's /health probe for 17-34 s at a time, so the same claim
    # (CLAIM_TXT, this run's token, never an override) is taken again for the prewarm, kept fresh on
    # the box every 110 s while it runs, and released after it -- done, failed or timed out. An
    # INT/TERM/HUP releases it through the trap, but only once the box command returns: bash defers a
    # trap while it waits on the $( ) below, so a TERM sent to this script waits out the prewarm (up to
    # PREWARM_TIMEOUT, 1800 s) before the release and the exit 143. A SIGKILL runs no trap and leaves
    # the hold to the watchdog's stale-hold rule (dropped HOLD_MAX (default 900 s) after the box's
    # last touch).
    # A hold that cannot be taken (a peer's, or no answer) skips it.
    prewarm_release() {   # $1: why; token-guarded like the release above, once per take
      [ "${PREWARM_HELD:-0}" = 1 ] || return 0
      PREWARM_HELD=0
      trap - INT TERM HUP
      PREWARM_REL=$(box_ssh_retry "o=\$(head -1 '$HOLD' 2>/dev/null); if [ \"\$o\" = '$CLAIM_TOKEN' ]; then rm -f '$HOLD'; echo RELEASED; else echo \"NOT-OURS \${o:-<none>}\"; fi" 2>/dev/null)
      case "$PREWARM_REL" in
        RELEASED*) note "prewarm: watchdog.hold released ($1)" ;;
        NOT-OURS*) note "prewarm: watchdog.hold is no longer this run's ($PREWARM_REL): left in place, the holder releases it" ;;
        *) note "could NOT release watchdog.hold after the prewarm ($1; the box did not answer): the watchdog drops it once it is HOLD_MAX (default 900 s) old, or release it by hand: $(hold_release_cmd)" ;;
      esac
    }
    if [ -n "$PREWARM_MARK" ] && [ "${PREWARM:-on}" != force ] \
       && box_ssh_retry "$PREWARM_BOOT_SH; [ -n \"\$b\" ] && grep -qx \"boot=\$b\" '$PREWARM_MARK'" >/dev/null 2>&1; then
      note "prewarm skipped: marker $PREWARM_MARK on the box names the boot running now (this tree, venv, chunk and engine process were prewarmed already; PREWARM=force runs it anyway)"
    else
      [ "${PREWARM:-on}" = force ] && note "prewarm: PREWARM=force -- running it whatever marker the box holds"
      PREWARM_CLAIM=${CLAIM_TXT//__HOLD__/$HOLD}
      PREWARM_CLAIM=${PREWARM_CLAIM//__TOKEN__/$CLAIM_TOKEN}
      PREWARM_CLAIM=${PREWARM_CLAIM//__OVERRIDE__/0}
      PREWARM_CLAIM=${PREWARM_CLAIM//__MAX__/${CLAIM_MAX:-}}
      PREWARM_HOLD_OUT=$(box_ssh_retry "$PREWARM_CLAIM" 2>&1); PREWARM_HOLD_RC=$?
      if [ "$PREWARM_HOLD_RC" -ne 0 ]; then
        [ "$PREWARM_HOLD_RC" -eq 7 ] && { PREWARM_HELD=1; prewarm_release "the take could not touch it"; }
        note "WARNING: prewarm NOT run: watchdog.hold could not be taken again for it (exit $PREWARM_HOLD_RC$([ "$PREWARM_HOLD_RC" -eq 255 ] && echo ', ssh: the box did not answer'): ${PREWARM_HOLD_OUT:-no output}) -- a prewarm with the watchdog live can fail its /health probe through a compile and get a healthy engine relaunched. The engine is up; once the box is this run's to touch, rerun it by hand under a hold before the recheck: box_ssh \"cd $TOOLS_REMOTE && BOX=$BOX ${VENV:-/opt/sglang}/bin/python3 prewarm-box-d.py --base-url $ENGINE_URL\" (log $PREWARM_LOG)"
      else
        PREWARM_HELD=1
        trap 'prewarm_release "interrupted"; exit 130' INT
        trap 'prewarm_release "terminated"; exit 143' TERM
        trap 'prewarm_release "hung up"; exit 129' HUP
        note "prewarm: watchdog.hold taken for it ($PREWARM_HOLD_OUT); prewarm-box-d.py --base-url $ENGINE_URL (box log $PREWARM_LOG)"
        PREWARM_T0=$(date +%s)
        PREWARM_OUT=$(trap - INT TERM HUP; box_ssh "cd '$TOOLS_REMOTE' || exit 9; ( while sleep ${PREWARM_HOLD_EVERY:-110}; do kill -0 \$\$ 2>/dev/null || exit 0; o=\$(head -1 '$HOLD' 2>/dev/null); [ \"\$o\" = '$CLAIM_TOKEN' ] && touch '$HOLD'; done ) >/dev/null 2>&1 & k=\$!; py='${VENV:-/nonexistent}/bin/python3'; [ -x \"\$py\" ] || py=python3; $PREWARM_BOOT_SH; b0=\$b; BOX='$BOX' timeout ${PREWARM_TIMEOUT:-1800} \"\$py\" prewarm-box-d.py --base-url '$ENGINE_URL' > '$PREWARM_LOG' 2>&1; rc=\$?; { kill \$k; wait \$k; } 2>/dev/null; m='$PREWARM_MARK'; $PREWARM_BOOT_SH; if [ \$rc -eq 0 ] && [ -n \"\$m\" ]; then if [ -z \"\$b\" ] || [ \"\$b\" != \"\$b0\" ]; then echo \"WARNING: prewarm marker NOT written (\$m): the engine's boot was '\${b0:-none}' before the sweep and '\${b:-none}' after; the next restart will prewarm again\"; elif e=\$( { mkdir -p \"\${m%/*}\" && printf 'boot=%s\\n%s %s\\n' \"\$b\" '$TAG' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \"\$m\"; } 2>&1 ); then echo \"prewarm marker written: \$m (boot \$b)\"; else echo \"WARNING: prewarm marker NOT written (\$m: \${e:-no error text}); the next restart will prewarm again\"; fi; fi; grep -aE ' SLOW|ERROR' '$PREWARM_LOG' | head -20; grep -a '^PREWARM DONE' '$PREWARM_LOG' | tail -1; exit \$rc" 2>&1)
        PREWARM_RC=$?
        prewarm_release "prewarm exit $PREWARM_RC"
        [ -n "$PREWARM_OUT" ] && note "$PREWARM_OUT"
        if [ "$PREWARM_RC" -eq 0 ]; then
          note "prewarm done in $(( $(date +%s) - PREWARM_T0 ))s"
        else
          note "WARNING: prewarm exit $PREWARM_RC$([ "$PREWARM_RC" -eq 255 ] && echo ', ssh: the box did not answer')$([ "$PREWARM_RC" -eq 124 ] && echo ", timed out after ${PREWARM_TIMEOUT:-1800}s") after $(( $(date +%s) - PREWARM_T0 ))s, no marker written -- the engine is up; rerun it by hand under a hold before the recheck: box_ssh \"cd $TOOLS_REMOTE && BOX=$BOX ${VENV:-/opt/sglang}/bin/python3 prewarm-box-d.py --base-url $ENGINE_URL\" (log $PREWARM_LOG)"
        fi
      fi
    fi
  fi
  if [ "${SAMPLING_GATE:-on}" = off ]; then
    note "sampling gate SKIPPED (SAMPLING_GATE=off): /server_info's sampling defaults were not checked"
    exit 0
  fi
  note "sampling gate: /server_info (sampling_defaults, preferred_sampling_params) + the args file + the engine cmdline"
  SAMPLING_OUT=$(box_ssh_retry --stdin "$SAMPLING_GATE_PY" "python3 - '$ENGINE_URL' '$TOOLS_ARGS'")
  SAMPLING_RC=$?
  [ -n "$SAMPLING_OUT" ] && note "$SAMPLING_OUT"
  if [ "$SAMPLING_RC" -eq 0 ] && printf '%s\n' "$SAMPLING_OUT" | grep -q '^SAMPLING PASS'; then
    exit 0
  fi
  if ! printf '%s\n' "$SAMPLING_OUT" | grep -q '^SAMPLING FAIL'; then
    note "SAMPLING UNVERIFIED: the gate did not run to a verdict (exit $SAMPLING_RC$([ "$SAMPLING_RC" -eq 255 ] && echo ', ssh: the box did not answer')) -- treat as a failed gate"
  fi
  note "SAMPLING GATE FAILED: $TAG is UP and serving (tree=$SHA) but omitted samplers do not resolve to temperature 1.0 / top_p 1.0 as pinned -- the router window is still open: roll back per this script's header, or fix forward (remove --sampling-defaults / preferred_sampling_params, or the generation_config.json that sets them) and restart again. Exit 7."
  exit 7
else
  note "NOT READY after ${ELAPSED}s (tag=$TAG, tree=$SHA) -- check /workspace/sglang-serve.log on the box"
  if [ -n "$DEAD" ]; then
    note "the engine cannot start -- /workspace/sglang-serve.log (this launch's section) says: $DEAD -- argparse refused the args, so they do not fit the tree it is running (tree=$SHA; args: $ARGS_LINE); roll back per this script's header. Not waiting out the 1800s ceiling."
  fi
  case "$CRASH" in
    SCHEDULER:*) note "the engine died -- /workspace/sglang-serve.log (this launch's section) says: ${CRASH#SCHEDULER: } -- sglang's scheduler logs that and takes the whole engine down, so nothing will answer /health; read the traceback there, then fix forward or roll back per this script's header (tree=$SHA; args: $ARGS_LINE). Not waiting out the 1800s ceiling." ;;
    ENGINES:*) note "TWO ENGINES: /workspace/sglang-serve.log carries ${CRASH#ENGINES: } \`server_args=\` lines since this run's marker (line $MARKER_LINE) -- one per \`sglang serve\` launch, so another launch (a watchdog relaunch, or a second restart) booted beside this one on the same GPUs, and each sizes its KV pool against the other's weights (gpu1021, 2026-10-21: both died on \`Loaded weights leave no GPU memory for the KV cache\`). Re-run restart.sh with a new tag: its stop step clears every engine and launcher before its one relaunch. Not waiting out the 1800s ceiling." ;;
  esac
  if [ -n "$VALIDATION" ]; then
    note "the engine cannot start -- /workspace/sglang-serve.log (this launch's section) ends in a startup traceback, ${VALIDATION#VALIDATION: }, and no \`sglang serve\` process is left on the box: a startup check (the tree's validation hooks, or a ServerArgs assertion) refused these args before any weight loaded, so nothing will answer /health (tree=$SHA; args: $ARGS_LINE). Fix the line -- args/check-import.sh runs the tree's validation hooks against it before a boot -- or roll back per this script's header. Not waiting out the 1800s ceiling."
  fi
  exit 6
fi
