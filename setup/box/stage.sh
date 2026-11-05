#!/bin/bash
# dev box: stage the whole box-parametrized tool set (this directory: lib.sh, boxes/, every
# *.sh/*.py, args/, dsml/, gatekit/ -- the box-out gate's greedy/image kit, boxout-gate.sh's default
# $TOOLS_REMOTE/gatekit) onto $TOOLS_REMOTE on the box (never into /root or /), with an md5 check per
# file, then sglang-env.txt (seeded the first time only, from the env of the serve line SGLANG_TREE
# runs: args/env-box-d-official.txt on the box-d-up tree, args/env-box-e.txt on box-e-up, args/env-v3.txt
# otherwise), serve-model.env
# (always rewritten from this box file's MODEL_DIR/SERVED_MODEL), the fork deploy key, caddy.env
# (the door's source-IP allowlist from this box file's TLS_ALLOW -- never written without one, and
# never a token: the Cloudflare DNS token stays off every rented box, #1306), the cert-push key's
# forced-command line (CERT_PUSH_PUBKEY, "The cert-push key" below), the
# side paths $SGLANG_TREE / $VENV (BUILT when absent -- a fresh container on the box-d layout -- by the
# box-d pre-stage recipe, verified when present; "the side paths" below), and
# a `fork` remote + fetch (NO checkout) from $FORK_PRIVATE_REPO on $SGLANG_TREE -- $FORK_BRANCH alone --,
# the engine's runtime gate files (TOOL_STRICT / SBR_* box-file keys -> tool-strict.on / sbr-*.gate,
# created or removed to match; "the gate files" below),
# and relaunch.staged.sh (the onstart's restart backup, re-copied from the relaunch.sh just staged),
# the hang watcher's script at the path the box file's HANGWATCH_CMD names (installed from the staged
# hangwatch.py when the box has none, compared and never replaced when it has one; "The hang
# watcher's script" below),
# and -- last, on a box file whose CADDY_EXPECT_SHA256 is not the stock build -- the door's Caddy binary
# from this Mac's artifact store onto a box that has none ("The door's Caddy binary" below).
# Never restarts the engine or Caddy, never touches the router. The one process it may restart is the WATCHDOG: one
# running a watchdog.sh other than the one just staged is replaced (`relaunch.sh --watchdog`, below).
#
# tests/ is EXCLUDED from the rsync and from the md5 manifest below, both sides in step. Those
# scripts are dev-box tests (each copies the tool set into its own `mktemp -d` -- see README
# "Regression tests"), the box never runs a test, and pushing the stubs and fixtures to
# $TOOLS_REMOTE would only md5-check them over ssh again on every stage. A box staged before this
# exclusion keeps its old copy: rsync never deletes, and nothing on the box reads it.
#
# Idempotent and safe to re-run: rsync only ever pushes files that exist in this local tree and
# never deletes anything on the far side, and sglang-args.txt / sglang-env.txt (the live,
# possibly hand-tuned serve line and its env override) are explicitly excluded on both sides —
# they are runtime state the box owns, never part of this repo's tree, so a re-stage never
# clobbers a tuned live line. sglang-env.txt gets one exception, seeded (never overwritten) from
# the tree's env (tools/args/env-box-d-official.txt, env-box-e.txt or env-v3.txt) when the box has none yet — see
# "sglang-env.txt" below — because
# relaunch.sh now refuses to boot without one; a box that already has a file, tuned or not, is
# always left untouched.
#
# The identification write (BOX= into /etc/environment + /etc/profile.d/box.sh) is GUARDED. It is
# the step that would otherwise let a wrong ssh TARGET become a wrong box IDENTITY: on-box scripts
# read $BOX, so overwriting it silently renames the whole box. Before writing, stage.sh reads the
# target's own existing BOX= line (/etc/environment, else /etc/profile.d/box.sh) and REFUSES when
# the target already names a DIFFERENT box than this run's $BOX, naming both, with nothing written.
# A re-stage of the same box and a fresh box with no line at all proceed exactly as before.
# BOX_FORCE=1 is the escape hatch for a genuine rename.
#
# The rsync is fenced explicitly, not incidentally. It builds its own ssh from $BOX_IP (rsync's -e
# takes a command, so it cannot call box_ssh), which puts it outside lib.sh's target fence; the
# explicit box_assert_target immediately above it is what refuses a mismatched box file before any
# byte moves, rather than relying on the fenced `run box_ssh` two lines earlier running first.
#
# usage: stage.sh   (everything comes from $BOX / boxes/$BOX.env; no positional args)
unset ENV_SEED   # the env seed override is the box file's line alone (stage_env_seed_name below)
. "$(dirname "$0")/lib.sh"

# This ssh line carries the session-proxy option lib.sh's box_ssh/box_scp do
# ($BOX_SSH_PROXY_ARG, set at source time): rsync's -e is a command string, so this is the one ssh in
# the tool set that cannot be box_ssh, and without the option a sandboxed run reaches the target fence
# (box_assert_target above, which now DOES go through the tunnel) and then fails at the byte move with
# "connect to host ... Operation not permitted" -- a transport refusal that reads like a dead box.
#
# The inner quotes are LITERAL (\" inside this double-quoted assignment), not shell grouping, on
# purpose: rsync hands the -e string to its own tokenizer, and the ProxyCommand value holds spaces
# ("ProxyCommand=python3 -I <helper path> %h %p https_proxy"). Quoted, it arrives at ssh as ONE argv
# element; unquoted, it arrives as six ("-o", "ProxyCommand=python3", "-I", path, "%h", "%p", var) and
# ssh reads the first as the hostname. With no proxy variable set the option is empty and this line is
# byte-for-byte what it always was.
SSH_RSH="ssh -p $SSH_PORT -i $SSH_KEY_PATH -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20${BOX_SSH_PROXY_ARG:+ -o \"$BOX_SSH_PROXY_ARG\"}"

# The repo the fork remote below fetches from is FORK_PRIVATE_REPO, never FORK_REPO (#609; the long
# version of why stands at that step). Checked HERE, before any byte moves: a box file with no such
# line is a half-filled file, and the run would otherwise push the whole tool set, md5 it, and only
# then die at the fetch -- the same "refuse before bytes move" ordering box_assert_target below
# follows. The name check also keeps `set -u` from answering with a bare "unbound variable".
[ -n "${FORK_PRIVATE_REPO:-}" ] || die "no FORK_PRIVATE_REPO in $BOX_ENV -- the fork-remote step needs the private mirror's ssh clone URL (e.g. <SGLANG_FORK_URL>); FORK_REPO is the PUBLIC upstream slug fork-sync.sh builds its upstream remote and gh lookups from, and is never the box's fetch source (#609)"

# The branch the fork remote fetches, checked beside the fetch source and for the same reason: a box
# file without it is half-filled, and the run would otherwise push and md5 the whole tool set before
# the fetch died on an empty refspec.
[ -n "${FORK_BRANCH:-}" ] || die "no FORK_BRANCH in $BOX_ENV -- the branch this box's fork remote fetches; every file under boxes/ carries it"

# The door's allowlist the caddy.env step writes: the BOX FILE's TLS_ALLOW line only (lib.sh sourced
# it, so $TLS_ALLOW holds its value when the line exists; a value merely exported in this shell is not
# the box file's and is dropped -- the caddy.env step says why). Shape-checked here, before any byte
# moves, because tls.sh pastes it into the Caddyfile's remote_ip matcher and caddy.env is sourced.
BOX_TLS_ALLOW=""
if grep -Eq '^[[:space:]]*TLS_ALLOW=' "$BOX_ENV"; then BOX_TLS_ALLOW="${TLS_ALLOW:-}"; fi
for cidr in $BOX_TLS_ALLOW; do
  [[ "$cidr" =~ ^[0-9A-Fa-f:.]+(/[0-9]+)?$ ]] || die "TLS_ALLOW in $BOX_ENV holds '$cidr', not an IP or CIDR -- space-separated CIDRs only (e.g. TLS_ALLOW=\"<ALLOW_CIDR> 127.0.0.1/32\")"
done
# Every box file names a door (HOSTNAME), so a box file with no allowlist refuses the WHOLE stage
# here, before any byte moves -- not just the caddy.env write, which would leave a stage that
# "succeeded" with the box's door config stale or missing. TLS_ALLOW_ANY=1 (box file or environment)
# stages an open door on purpose.
if [ -z "$BOX_TLS_ALLOW" ] && [ "${TLS_ALLOW_ANY:-0}" != 1 ]; then
  die "REFUSED caddy.env (no TLS_ALLOW in $BOX_ENV: the door's source-IP allowlist is not written open. Add TLS_ALLOW=\"<cidr> <cidr> ...\" to the box file and re-stage, or set TLS_ALLOW_ANY=1 to stage an open door on purpose${TLS_ALLOW:+; the TLS_ALLOW exported in this shell is ignored -- it would not survive the next stage}). Nothing was staged; the box's caddy.env is left as it was, and tls.sh refuses to render a Caddyfile without an allowlist."
fi

# The Cloudflare DNS token never reaches a rented box (#1306, the user's rule of 2026-10-29: the provider's
# host has root and the token can rewrite <DOMAIN>). Certificates are issued off the box
# by the operator's own issuer and pushed in as files; nothing staged carries the token. The tool set this stage
# rsyncs holds every box file and every args/env file, so an assignment of CF_DNS_TOKEN in any of them
# (a box file, an args/*.txt env file, a *.env beside them) refuses the WHOLE stage here, before any
# byte moves, naming the file. A comment that merely mentions the name is not an assignment.
CF_HITS=$(grep -rlE '^[[:space:]]*(export[[:space:]]+)?CF_DNS_TOKEN=' --include='*.env' --include='*.txt' \
  --exclude-dir=tests "$TOOLS_DIR" 2>/dev/null || true)
[ -z "$CF_HITS" ] || die "REFUSED: a CF_DNS_TOKEN assignment in the tool set this stage would ship -- $(echo $CF_HITS). The Cloudflare token never sits on a rented box (#1306): the door serves certificate files the operator pushes from off the box. Delete the line; nothing was staged"

# The door's Caddy binary (caddy-h2/README.md "A fresh box: where the binary comes from"). tls.sh holds
# $TOOLS_REMOTE/caddy to CADDY_EXPECT_SHA256 (box file or environment, else the stock sha) and, with no
# binary on disk, downloads only the STOCK build -- so a box pinned to anything else (the patched
# h2-window build) stops at tls.sh (exit 6) on a fresh box unless something delivers that binary. The
# caddy step at the END of this stage does (stage_caddy below): on a non-stock pin it copies
# $CADDY_ARTIFACT -- this Mac's artifact store, our internal artifact policy; never a download --
# onto a box that has NO binary there. The stock sha is read off tls.sh's own default, so the two never
# disagree about what "stock" means; the pin's shape is checked here, before any byte moves.
CADDY_STOCK=${CADDY_STOCK_SHA256:-$(sed -n 's/^CADDY_STOCK_SHA256="\${CADDY_STOCK_SHA256:-\([0-9a-f]\{64\}\)}".*$/\1/p' "$TOOLS_DIR/tls.sh")}
[[ "$CADDY_STOCK" =~ ^[0-9a-f]{64}$ ]] || die "could not read the stock caddy sha256 off $TOOLS_DIR/tls.sh's CADDY_STOCK_SHA256 default (got '$CADDY_STOCK') -- the caddy step compares the box file's pin with it"
CADDY_PIN=${CADDY_EXPECT_SHA256:-$CADDY_STOCK}
[[ "$CADDY_PIN" =~ ^[0-9a-f]{64}$ ]] || die "CADDY_EXPECT_SHA256 (from $BOX_ENV or the environment) holds '$CADDY_PIN', not a sha256 (64 lowercase hex) -- tls.sh holds the box's caddy binary to it"
CADDY_ARTIFACT=${CADDY_ARTIFACT:-$HOME/workspace/backups/caddy-h2/caddy-h2win-2.11.7}

# sha256 of a file on THIS Mac (shasum ships with macOS; sha256sum where it exists).
stage_local_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | awk '{ print $1 }'
}

# Why $CADDY_ARTIFACT cannot be delivered as the pin -- empty when it can: it exists, the .sha256 beside
# it records the pin, and its own bytes hash to the pin.
stage_caddy_artifact_refusal() {
  local rec got
  if [ ! -f "$CADDY_ARTIFACT" ]; then printf 'there is no artifact at %s' "$CADDY_ARTIFACT"; return 0; fi
  if [ ! -f "$CADDY_ARTIFACT.sha256" ]; then printf 'there is no %s.sha256 beside the artifact' "$CADDY_ARTIFACT"; return 0; fi
  rec=$(awk 'NR == 1 { print $1 }' "$CADDY_ARTIFACT.sha256")
  if [ "$rec" != "$CADDY_PIN" ]; then printf '%s.sha256 records %s, not the pin' "$CADDY_ARTIFACT" "${rec:-nothing}"; return 0; fi
  got=$(stage_local_sha256 "$CADDY_ARTIFACT")
  [ "$got" = "$CADDY_PIN" ] || printf '%s hashes to %s, not the pin its .sha256 records (the file changed after that was written)' "$CADDY_ARTIFACT" "${got:-nothing}"
  return 0
}

# The caddy step. A stock pin: nothing (tls.sh's own stock download serves a fresh box, as always). A
# non-stock pin: read $TOOLS_REMOTE/caddy's sha256 on the box -- the pin already there is left as it is;
# a DIFFERENT binary there is left as it is too, with a WARNING (it may be the running door: stage never
# writes over a binary and never restarts Caddy -- replacing it is the deploy's hard cut, caddy-h2/README.md
# "Swap it in", and tls.sh refuses it until then); NO binary (a fresh box) gets $CADDY_ARTIFACT, checked
# on this Mac first (stage_caddy_artifact_refusal: a refusal stops the stage naming caddy-h2/build.sh as
# the fallback), copied beside the final name, re-hashed ON THE BOX, then renamed into place with
# caddy.sha256 written beside it (tls.sh's record). It runs last, so a refusal leaves every other part of
# the stage done -- caddy-h2/build.sh included, for the on-box fallback -- and still exits non-zero, which
# stops bringup.sh before its tls.sh step.
stage_caddy() {
  local remote="$TOOLS_REMOTE/caddy" got why
  if [ "$CADDY_PIN" = "$CADDY_STOCK" ]; then
    note "caddy binary: stock pin -- nothing to deliver (tls.sh downloads the stock build onto a box with none)"
    return 0
  fi
  why=$(stage_caddy_artifact_refusal)
  if [ "${DRY_RUN:-0}" = 1 ]; then
    note "DRY_RUN: skipping the caddy binary check -- a live run copies $CADDY_ARTIFACT to $remote (pin $CADDY_PIN) only when the box has no binary there${why:+; it would REFUSE that copy: $why}"
    return 0
  fi
  got=$(box_ssh "if [ -e '$remote' ]; then sha256sum '$remote' | cut -d' ' -f1; else echo absent; fi" </dev/null) \
    || die "could not read $remote's sha256 on $BOX -- the rest of the stage is done; re-stage once ssh answers"
  case "$got" in
    "$CADDY_PIN")
      note "caddy binary: $remote already hashes to the pin $CADDY_PIN -- left as it is"
      return 0 ;;
    absent) ;;
    *)
      note "WARNING: caddy binary: $remote hashes to ${got:-<unreadable>}, not the pin $CADDY_PIN ($BOX_ENV) -- left as it is: stage writes the binary only onto a box with none, never over one that may be serving the door (the swap is the deploy's hard cut, caddy-h2/README.md \"Swap it in\"), and tls.sh refuses it (exit 6) until that lands"
      return 0 ;;
  esac
  [ -z "$why" ] || die "REFUSED caddy binary: CADDY_EXPECT_SHA256=$CADDY_PIN ($BOX_ENV) is not the stock build and $remote is absent on $BOX, but $why -- nothing was copied, the rest of the stage is done, and tls.sh refuses this box (exit 6) until the binary is there. Restore the artifact (<HOME>/workspace/backups/caddy-h2/, its .sha256 beside it; CADDY_ARTIFACT names another path) and re-stage, or fall back to building it ON THE BOX with tools/caddy-h2/build.sh: BUILD_DIR=/opt/caddy-h2 bash $TOOLS_REMOTE/caddy-h2/build.sh, then cp /opt/caddy-h2/caddy-h2win $remote once its sha256 is the pin, and re-stage (caddy-h2/README.md)"
  note "caddy binary: $remote absent on $BOX -- copying $CADDY_ARTIFACT (sha256 $CADDY_PIN, the pin) from this Mac"
  if ! box_scp -O "$CADDY_ARTIFACT" "root@$BOX_IP:$remote.stage-tmp"; then
    # A copy cut short can leave part of the file behind: remove it (best effort -- the ssh that failed
    # the copy may fail this too, and the next stage's copy overwrites it either way).
    box_ssh "rm -f '$remote.stage-tmp'" </dev/null || true
    die "could not copy $CADDY_ARTIFACT to $BOX:$remote.stage-tmp (the partial copy removed if ssh answered) -- $remote is still absent; re-stage"
  fi
  box_ssh bash -s -- "$remote" "$CADDY_PIN" <<'CADDYBIN' || die "caddy binary: the copy on $BOX did not land as the pin -- $remote is still absent (see above); re-stage, or build it on the box with tools/caddy-h2/build.sh (caddy-h2/README.md)"
set -u
BIN=$1 PIN=$2
TMP="$BIN.stage-tmp"
got=$(sha256sum "$TMP" 2>/dev/null | cut -d' ' -f1)
if [ "$got" != "$PIN" ]; then
  rm -f "$TMP"
  echo "REFUSED: the copy on the box hashes to ${got:-<unreadable>}, not the pin $PIN -- removed; $BIN left absent" >&2
  exit 1
fi
chmod 755 "$TMP"
if [ -e "$BIN" ]; then
  rm -f "$TMP"
  echo "caddy binary: $BIN appeared during the copy -- left as it is, the copy removed (stage never writes over a binary on the box)"
  exit 0
fi
mv "$TMP" "$BIN"
printf '%s  %s\n' "$PIN" "$BIN" > "$BIN.sha256"
echo "OK  $BIN = $PIN (re-hashed on the box; $BIN.sha256 written)"
CADDYBIN
}

# The engine's runtime gates, from the box file. The fork's SGLANG_SBR_*_GATE_FILE env keys (every
# args/env-*.txt names all four, 04-engine.md "Environment") switch a feature on only while the named
# file EXISTS, rechecked at most once a second -- so a box whose gate file was never created runs with
# the feature off, silently. box-a's first boot (2026-10-26) ran tool-strict level 0 for exactly that
# reason: the README's step-5 `touch` was a hand step and nobody took it. Each key below, set in the box
# file, is materialized by the gate-file step on every stage: 1 creates the file, 0 removes it. A key the
# box file does not set leaves that file as it is (the hand state stands). Only the BOX FILE's line counts
# (as for TLS_ALLOW: a value exported in this shell is dropped), and a value other than 0/1 refuses the
# whole stage here, before any byte moves.
stage_gate_file() {
  case "$1" in
    TOOL_STRICT)          printf '%s' tool-strict.on ;;
    SBR_SCHEMA_MEMO)      printf '%s' sbr-schema-memo.gate ;;
    SBR_UNFINISHED_REUSE) printf '%s' sbr-unfinished-reuse.gate ;;
    SBR_PHASE_TIMING)     printf '%s' sbr-phase-timing.gate ;;
  esac
}
GATE_OPS=()
for gk in TOOL_STRICT SBR_SCHEMA_MEMO SBR_UNFINISHED_REUSE SBR_PHASE_TIMING; do
  grep -Eq "^[[:space:]]*$gk=" "$BOX_ENV" || continue
  gv="${!gk:-}"
  case "$gv" in 0|1) ;; *) die "$gk in $BOX_ENV holds '$gv', not 0 or 1 -- 1 creates $TOOLS_REMOTE/$(stage_gate_file "$gk") on the box (the gate open), 0 removes it" ;; esac
  # <0|1>:<the sglang-env.txt key that names the file>:<the file's default path>
  GATE_OPS+=("$gv:SGLANG_SBR_${gk#SBR_}_GATE_FILE:$TOOLS_REMOTE/$(stage_gate_file "$gk")")
done

# The box file's ENGINES (box-c's 2xTP4 layout, lib.sh): a malformed list refuses the whole stage here,
# before any byte moves. With ENGINES unset every step below is the one-engine stage it always was;
# with it, the per-engine files are named once, from lib.sh's engine_select: each engine's serve log
# (created 0600 with the legacy one), its sglang-env-<id>.txt (seeded like sglang-env.txt), and its
# watchdog (refreshed only when it runs). caddy.env and serve-model.env stay shared by every engine.
engines_check
STAGE_ENGINE_LOGS=""
STAGE_ENGINE_ENVS=()
for id in $ENGINES_IDS; do
  engine_select "$id"
  STAGE_ENGINE_LOGS="${STAGE_ENGINE_LOGS:+$STAGE_ENGINE_LOGS }$ENGINE_SERVE_LOG"
  STAGE_ENGINE_ENVS+=("$ENGINE_ENV_FILE")
done
engine_select

note "tool set -> $BOX_IP:$TOOLS_REMOTE"
run box_ssh "mkdir -p '$TOOLS_REMOTE/args' '$TOOLS_REMOTE/boxes'"

# rsync takes its remote shell as a COMMAND (-e), so it can never go through box_ssh/box_scp and is
# NOT covered by lib.sh's target fence: $SSH_RSH above is built from $BOX_IP by hand. Left implicit,
# the transport that carries this whole tool set -- the box file and its door config included -- is protected
# only by the fact that the fenced `run box_ssh` above happens to run first; reorder the two and the
# tool set lands on whatever box $BOX_IP names, unfenced. So assert the target HERE, where $SSH_RSH is
# aimed, making the invariant the code enforces rather than the position of two lines. It is the same
# verdict box_ssh would give, from the same cached probe (lib.sh), so it does not contradict the
# remote-side BOX= check below -- both refuse a positive mismatch. Skipped under DRY_RUN, which runs no
# rsync and by design makes no ssh at all (the reason the BOX= check below is skipped too).
if [ "${DRY_RUN:-0}" != 1 ]; then
  box_assert_target
else
  note "DRY_RUN: skipping the target fence (no rsync is executed, so no byte can move; a live run asserts $BOX_IP here)"
fi
# Every pattern here has a counterpart in the md5 manifest's find below -- tests/, .gitignore and
# __pycache__/ included: the box never runs a test and has no git checkout here, and a find that
# still listed a file the push never created would report MISMATCH for it (a missing remote file
# md5s to the empty string) and die before the box is staged. The two lists are ONE fence, pattern
# for pattern: each --exclude here is spent as TWO predicates there -- `! -name '<pat>'` for a file
# of that name, and `! -path '*/<pat>/*'` for anything UNDER A DIRECTORY of that name. The second is
# the one that is easy to lose, and losing it is the nested-__pycache__ failure over again: rsync
# drops a matching directory whole, while `-name` alone still lets `find -type f` walk inside and
# list its children -- an x.hold/, x.pyc/, x.pre-y/, .gitignore/ or sglang-args.txt/ directory
# holding any file would MISMATCH on that file and die, which is every one of these patterns and not
# just the three *.globs. A pattern written with a trailing / (tests/, __pycache__/) matches
# DIRECTORIES ONLY -- rsync would still push a FILE of that name -- so it gets its -path clause with
# no -name: that is the whole difference between those two and the rest. `watchdog.hold` is the one
# pattern here with no clause of its own, only because the `*.hold` glob on the same line already
# covers it. `gates/` is the box's own record dir (boxout-gate.sh writes $TOOLS_REMOTE/gates/
# <engine>-<sha12>.PASS there, and window-keep.sh's stop/recheck trust it): a gates/ directory on the
# Mac never reaches the box, so a PASS can only come from a gate run ON the box. Change one list,
# change the other in the same commit.
run rsync -az -e "$SSH_RSH" \
  --exclude 'onstart.rendered.sh' \
  --exclude 'tests/' --exclude 'gates/' \
  --exclude 'sglang-args.txt' --exclude 'sglang-env.txt' \
  --exclude '*.pre-*' --exclude '*.hold' --exclude 'watchdog.hold' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.gitignore' \
  "$TOOLS_DIR/" "root@$BOX_IP:$TOOLS_REMOTE/"

if [ "${DRY_RUN:-0}" != 1 ]; then
  note "md5 manifest"
  MISMATCH=0; CHECKED=0
  # ONE ssh for the whole manifest, never one per file. The per-file shape opened ~110 connections a
  # stage, and under set -e any single one failing at the transport -- gpu1018's first stage (#2582):
  # `Connection timed out during banner exchange`, exit 255, after the rsync had already landed --
  # killed the whole stage with nothing wrong on the box. So: (1) the find below lists the files and
  # md5s each one LOCALLY; (2) one box_ssh md5sums every listed path on the box, in list order, one
  # line per path (empty for a missing file, which md5s to nothing exactly as before), closed by an
  # end marker; (3) the two lists are compared here, file by file, into the same `OK  <rel>` /
  # `MISMATCH <rel> (<local> vs <remote>)` lines and the same die. The remote pass only reads, so an
  # ssh exit 255 (ssh's own transport failure; a remote command never answers 255 here) is retried,
  # three attempts in all; any other failure, an answer that is not exactly one line per file plus
  # the marker, or the third 255 dies naming it -- a manifest that was not read is never "OK".
  #
  # The box_ssh reads /dev/null, never the stage's stdin: ssh without -n reads its stdin to EOF to
  # forward it to the remote command, which is how a bare box_ssh inside the old find-fed loop ate
  # every path after the first (qa/box-c-0022, 2026-09-30: only `OK  restart.sh`). The paths travel in
  # the command itself, each single-quoted (an embedded ' spelled '\''), so they arrive as data the
  # remote shell never splits or globs. sshd hands that command to the remote shell as ONE argument,
  # and Linux caps one argument at 128 KiB (MAX_ARG_STRLEN): today's ~110 paths are ~6 KB, and a tool
  # set that ever outgrows the cap dies here saying so, before the ssh, rather than as an E2BIG.
  #
  # The find at the loop's foot is the fence's other half, in the --exclude list's own order: the
  # names first, then anything UNDER a directory of one of those names (the subtree rsync drops
  # whole), then the two trailing-/ patterns that are directories only. See the rsync above.
  MD5_REL=(); MD5_LOCAL=(); MD5_PATHS=""; MD5_SQ="'\\''"
  while IFS= read -r -d '' f; do
    rel=${f#"$TOOLS_DIR"/}
    MD5_REL[CHECKED]=$rel
    MD5_LOCAL[CHECKED]=$(md5 -q "$f" 2>/dev/null || md5sum "$f" | cut -d' ' -f1)
    q=$TOOLS_REMOTE/$rel
    q=${q//\'/$MD5_SQ}
    MD5_PATHS="$MD5_PATHS '$q'"
    CHECKED=$((CHECKED + 1))
  done < <(find "$TOOLS_DIR" -type f \
      ! -name 'onstart.rendered.sh' ! -name 'sglang-args.txt' ! -name 'sglang-env.txt' \
      ! -name '*.pre-*' ! -name '*.hold' ! -name '*.pyc' ! -name '.gitignore' \
      ! -path '*/onstart.rendered.sh/*' ! -path '*/sglang-args.txt/*' ! -path '*/sglang-env.txt/*' \
      ! -path '*/*.pre-*/*' ! -path '*/*.hold/*' ! -path '*/*.pyc/*' ! -path '*/.gitignore/*' \
      ! -path '*/tests/*' ! -path '*/gates/*' ! -path '*/__pycache__/*' -print0)
  [ "$CHECKED" -gt 0 ] || die "md5 manifest: the find over $TOOLS_DIR listed no file at all -- nothing to check, so nothing was verified"
  MD5_END=md5-manifest-end
  MD5_CMD="for f in$MD5_PATHS; do printf '%s\n' \"\$(md5sum \"\$f\" 2>/dev/null | cut -d' ' -f1)\"; done; echo $MD5_END"
  [ "${#MD5_CMD}" -le 131072 ] || die "md5 manifest: the one-ssh md5 command for $CHECKED files is ${#MD5_CMD} bytes, past Linux's 128 KiB cap on a single argument (the remote shell's -c) -- split the remote pass into batches before staging a tool set this large"
  MD5_TRY=1
  while :; do
    MD5_RC=0
    MD5_OUT=$(box_ssh "$MD5_CMD" </dev/null) || MD5_RC=$?
    { [ "$MD5_RC" = 255 ] && [ "$MD5_TRY" -lt 3 ]; } || break
    note "md5 manifest: ssh exit 255 on attempt $MD5_TRY of 3 (a transport failure, e.g. 'Connection timed out during banner exchange') -- retrying the read-only md5 pass in $((MD5_TRY * 5))s"
    sleep $((MD5_TRY * 5))
    MD5_TRY=$((MD5_TRY + 1))
  done
  [ "$MD5_RC" = 0 ] || die "md5 manifest: the remote md5 pass over $CHECKED files failed (ssh exit $MD5_RC, attempt $MD5_TRY of 3) -- no staged file was verified; the rsync above already landed, so re-run stage.sh (idempotent) before any restart"
  MD5_REMOTE=(); MD5_N=0
  while IFS= read -r r; do MD5_REMOTE[MD5_N]=$r; MD5_N=$((MD5_N + 1)); done <<<"$MD5_OUT"
  [ "$MD5_N" = $((CHECKED + 1)) ] && [ "${MD5_REMOTE[CHECKED]}" = "$MD5_END" ] \
    || die "md5 manifest: the remote md5 pass answered $MD5_N line(s) for $CHECKED files plus its '$MD5_END' marker (want $((CHECKED + 1)), the marker last) -- the answer cannot be matched to the list, so no staged file was verified; re-run stage.sh"
  i=0
  while [ "$i" -lt "$CHECKED" ]; do
    rel=${MD5_REL[i]}; l=${MD5_LOCAL[i]}; r=${MD5_REMOTE[i]}
    if [ "$l" = "$r" ]; then echo "OK  $rel"; else echo "MISMATCH $rel ($l vs $r)"; MISMATCH=1; fi
    i=$((i + 1))
  done
  note "md5 manifest: $CHECKED files checked"
  [ "$MISMATCH" = 0 ] || die "one or more staged files failed the md5 check"
else
  note "DRY_RUN: skipping md5 manifest"
fi

run box_ssh "chmod +x '$TOOLS_REMOTE'/*.sh '$TOOLS_REMOTE'/*.py 2>/dev/null; ls -la '$TOOLS_REMOTE' '$TOOLS_REMOTE/args'"

# relaunch.staged.sh is the restart backup of the staged launcher. The box's onstart has a staged-launcher
# arm: on a container restart it keeps a relaunch.sh that is already this lib.sh launcher, and restores
# relaunch.staged.sh over relaunch.sh when it finds the onstart's inline first-boot launcher there instead.
# The backup is not in this tree (rsync never deletes, so it survives every stage), so it is re-taken here
# from the relaunch.sh just staged -- else a restart after staging a newer launcher could restore the older
# copy. Written whenever relaunch.sh is; never restarts anything.
note "relaunch.staged.sh <- relaunch.sh (the onstart's restart backup of the staged launcher, kept in step with this stage)"
run box_ssh "cp -p '$TOOLS_REMOTE/relaunch.sh' '$TOOLS_REMOTE/relaunch.staged.sh' && sha256sum '$TOOLS_REMOTE/relaunch.sh' '$TOOLS_REMOTE/relaunch.staged.sh'"

note "BOX=$BOX -> /etc/environment and /etc/profile.d/box.sh (so a staged script on the box finds boxes/\$BOX.env without being told); refused if the target already names a different box (BOX_FORCE=1 to rename deliberately); /etc/environment and /workspace/sglang-serve.log (the two files that hold VLLM_API_KEY) set to 0600"
# Read what the target already calls itself BEFORE writing. Skipped under DRY_RUN, which makes no
# ssh at all and so has nothing to protect.
if [ "${DRY_RUN:-0}" = 1 ]; then
  note "DRY_RUN: skipping the target-identity check (nothing is written)"
else
  # The probe is deliberately the only ssh made before the check: if it fails we die rather than
  # fall through to the write not knowing which box we are about to rename. The quote-strip is run
  # REMOTELY, so the assignment's status is ssh's own status -- this does not lean on `pipefail`
  # to notice an unreachable box.
  REMOTE_BOX=$(box_ssh "b=\$(sed -n 's/^BOX=//p' /etc/environment 2>/dev/null | head -1); [ -n \"\$b\" ] || b=\$(sed -n 's/^export BOX=//p' /etc/profile.d/box.sh 2>/dev/null | head -1); printf '%s' \"\$b\" | tr -d '\"'") \
    || die "could not read the existing BOX= line off $BOX_IP:$SSH_PORT (/etc/environment, /etc/profile.d/box.sh) -- refusing to write BOX=$BOX blind; check the box is reachable and is the one you meant, then re-run"
  if [ -z "$REMOTE_BOX" ]; then
    note "target carries no BOX= line yet (fresh box) -- writing BOX=$BOX"
  elif [ "$REMOTE_BOX" = "$BOX" ]; then
    note "target already calls itself BOX=$REMOTE_BOX -- re-staging the same box, rewriting its own line"
  elif [ "${BOX_FORCE:-0}" != 1 ]; then
    die "wrong box: $BOX_IP:$SSH_PORT already calls itself BOX=$REMOTE_BOX, but this run is BOX=$BOX -- refusing to overwrite the target's identity (every on-box script reads \$BOX: restart.sh, tls.sh, watchdog.sh, and stage.sh's own re-runs). If $BOX really is this box, re-run with BOX_FORCE=1 to rename it deliberately."
  else
    note "BOX_FORCE=1: rewriting the target's identity BOX=$REMOTE_BOX -> BOX=$BOX"
  fi
fi
# The same write makes the box's two KEY HOLDERS root-only (#2994). /etc/environment carries
# VLLM_API_KEY and is created 0644 at container create; the serve log carries it too, inside the
# engine's `server_args=` line, and is created 0644 by whichever launch first appends to it (the
# onstart's first boot, restart.sh, cutover.sh). Every reader of either is root (relaunch.sh,
# watchdog.sh, restart.sh, tls.sh --check, boxenv.py and the probes, sshd's pam_env in root's own
# session), so 0600 costs nothing. `sed -i` keeps the mode that stands (GNU sed 4.9), so the chmod is
# what moves it, on every stage. The serve log is CREATED empty here when absent, under umask 077, so
# a box staged before its first launch still gets a 0600 log: every launcher appends with `>>`, which
# keeps the mode of a file that exists. `/etc/profile.d/box.sh` holds BOX= alone and stays 0644 (a
# login shell reads it). `set -e`: a chmod that fails fails the stage, never a silent world-readable
# key. tests/stage-keyholders.test.sh runs this line against a fake box.
run box_ssh "set -e; umask 077; (grep -q '^BOX=' /etc/environment 2>/dev/null && sed -i \"s/^BOX=.*/BOX=$BOX/\" /etc/environment) || echo 'BOX=$BOX' >> /etc/environment; chmod 600 /etc/environment; printf 'export BOX=%s\n' '$BOX' > /etc/profile.d/box.sh; chmod 644 /etc/profile.d/box.sh; if [ -d /workspace ]; then : >> /workspace/sglang-serve.log; chmod 600 /workspace/sglang-serve.log; fi"
# Each engine's own serve log (ENGINES: /workspace/sglang-serve-<id>.log) carries the same serve line
# and the same key, so it is created and chmodded the same way, beside the legacy one -- a separate
# step so the line above stays the one-engine box's, byte for byte. tests/stage-engines.test.sh runs it.
if [ -n "$STAGE_ENGINE_LOGS" ]; then
  run box_ssh "set -e; umask 077; if [ -d /workspace ]; then for f in $STAGE_ENGINE_LOGS; do : >> \"\$f\"; chmod 600 \"\$f\"; done; fi"
fi

# The watchdog runs the script it was started from (bash keeps the inode it opened; rsync's rename
# leaves it that inode), so a watchdog.sh change staged above never took effect on a box whose
# watchdog was up. The just-staged relaunch.sh's `--watchdog` mode compares the running watchdog's
# recorded md5 with the staged file's and replaces a stale one -- never the engine, never a box
# with no watchdog running, never while watchdog.hold stands (relaunch.sh's watchdog_refresh has the
# rules). After the BOX write, so the box's own BOX= agrees; BOX is also passed explicitly. The
# command names relaunch.sh only, never the watchdog's file name, so its pgrep never matches the
# ssh shell running it.
#
# With ENGINES each engine has its own watchdog (`watchdog.sh <id>`, beside or instead of the legacy
# `watchdog.sh`), and each RUNNING one is refreshed with its own id; a watchdog that is not running
# -- legacy or engine -- is never started from here, so the window's pre-switch sync (the TP8 engine
# and its legacy watchdog still up) never spawns an engine b watchdog. The probe is pgrep on the
# watchdog's own argv: the legacy one ENDS at watchdog.sh, an engine's at `watchdog.sh <id>`. The
# remote body arrives on stdin, so the ssh command line names no watchdog file at all.
if [ -z "$ENGINES_IDS" ]; then
  note "watchdog: replace a running watchdog whose script is not the staged watchdog.sh (relaunch.sh --watchdog; the engine is never touched)"
  run box_ssh "BOX='$BOX' bash '$TOOLS_REMOTE/relaunch.sh' --watchdog"
else
  note "watchdog: each RUNNING watchdog (the legacy one, engines $ENGINES_IDS) on a script that is not the staged watchdog.sh is replaced under its own id (relaunch.sh --watchdog [<id>]); none is started, the engines are never touched"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    run box_ssh bash -s -- "$BOX" "$TOOLS_REMOTE" $ENGINES_IDS
  else
    box_ssh bash -s -- "$BOX" "$TOOLS_REMOTE" $ENGINES_IDS <<'WATCHDOGS'
set -uo pipefail
export BOX="$1"
T="$2"
shift 2
rc=0
if pgrep -f "$T/watchdo[g]\.sh\$" > /dev/null; then
  bash "$T/relaunch.sh" --watchdog || rc=1
else
  echo "SKIP watchdog (legacy): none running -- never started from a stage"
fi
for id in "$@"; do
  if pgrep -f "$T/watchdo[g]\.sh $id\$" > /dev/null; then
    bash "$T/relaunch.sh" --watchdog "$id" || rc=1
  else
    echo "SKIP watchdog $id: none running -- never started from a stage"
  fi
done
exit $rc
WATCHDOGS
  fi
fi

# The hang watcher's script (autopool B1). The box file's HANGWATCH_CMD names the script the onstart
# starts on every boot (`python3 <script> --service ... --out <dir>`, build-onstart.sh); the canonical
# copy is this tree's hangwatch.py, staged above at $TOOLS_REMOTE/hangwatch.py and md5-checked with the
# rest. When the knob names another path -- box-a/box-b run /workspace/mk29/harness/hangwatch.py, where
# it was first placed by hand -- the staged copy is installed THERE on a box that has none (a rebuilt
# box would otherwise boot to HANGWATCH-MISSING). A copy already there is compared, never replaced:
# the same bytes are reported, other bytes are a WARNING (the running watcher started from that file;
# taking the canonical copy is a hand step the WARNING spells out). Nothing is started or stopped:
# a running watcher keeps the code it loaded, and the onstart starts the script on the next boot.
# stage_hw_script <HANGWATCH_CMD>: the script it runs (empty for no knob); rc 1 for a malformed knob.
stage_hw_script() {
  [ -n "${1:-}" ] || return 0
  [[ "$1" =~ ^python3\ (/[A-Za-z0-9._/-]+\.py)(\ |$) ]] || return 1
  printf '%s' "${BASH_REMATCH[1]}"
}
if ! HW_SCRIPT=$(stage_hw_script "${HANGWATCH_CMD:-}"); then
  note "WARNING hangwatch: HANGWATCH_CMD in $BOX_ENV is not 'python3 /abs/path.py ...' (build-onstart.sh refuses it too) -- no script installed"
elif [ -z "$HW_SCRIPT" ]; then
  note "hangwatch: no HANGWATCH_CMD in $BOX_ENV -- hangwatch.py is staged at $TOOLS_REMOTE with the tool set, installed nowhere else"
elif [ "$HW_SCRIPT" = "$TOOLS_REMOTE/hangwatch.py" ]; then
  note "hangwatch: HANGWATCH_CMD runs the staged copy itself ($HW_SCRIPT)"
elif [ "${HW_SCRIPT##*/}" != hangwatch.py ]; then
  note "hangwatch: HANGWATCH_CMD runs $HW_SCRIPT, not a hangwatch.py -- this tool set ships no such script; left to the box"
else
  note "hangwatch: $HW_SCRIPT (HANGWATCH_CMD's script) <- $TOOLS_REMOTE/hangwatch.py when the box has none; compared when present, never replaced; no watcher started or stopped"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    run box_ssh bash -s -- "$TOOLS_REMOTE/hangwatch.py" "$HW_SCRIPT"
  else
    box_ssh bash -s -- "$TOOLS_REMOTE/hangwatch.py" "$HW_SCRIPT" <<'HANGWATCHPY' || die "hangwatch: could not install $HW_SCRIPT from the staged $TOOLS_REMOTE/hangwatch.py -- the rest of the tool set is staged; re-run stage.sh"
set -uo pipefail
src="$1" dst="$2"
[ -f "$src" ] || { echo "hangwatch: no staged $src on the box -- nothing to install" >&2; exit 1; }
if [ -e "$dst" ] || [ -L "$dst" ]; then
  if [ -f "$dst" ] && [ ! -L "$dst" ] && cmp -s "$src" "$dst"; then
    echo "hangwatch: $dst is the staged copy byte for byte -- kept"
  else
    echo "WARNING hangwatch: $dst is not the staged $src -- left as it is (a running watcher started from it). To take the canonical copy: cp -p $src $dst (a running watcher keeps the code it loaded until its next start)"
  fi
  exit 0
fi
mkdir -p "$(dirname "$dst")" || exit 1
if cp -p "$src" "$dst.stage-tmp" && cmp -s "$src" "$dst.stage-tmp" && mv -f "$dst.stage-tmp" "$dst"; then
  echo "hangwatch: installed $dst from the staged $src"
else
  rm -f "$dst.stage-tmp"
  echo "hangwatch: could not install $dst" >&2
  exit 1
fi
HANGWATCHPY
  fi
fi

# The seed is the env half of the serve line this box file's tree runs (#2661), picked by SGLANG_TREE's
# name -- the tree name args files pin with `# TREE: <name>` (args/check-import.sh "TREE-PINNED FILES",
# whose default tree name is the checkout's basename):
#   box-d-up  -> env-box-d-official.txt, v5-b200-box-d.args' pair. env-v3.txt does NOT start this tree:
#               it carries SGLANG_SWA_BRANCH_REBUILD=true without SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND
#               =python, and the swa rebuild refuses upstream's rust radix core, so a re-create seeded
#               from it never came up;
#   box-e-up  -> env-box-e.txt, v6-b200-box-e.args' pair: env-box-d-official.txt minus
#               SGLANG_OPT_DSV4_C4_INDEXER_LOGITS_BUDGET_BYTES plus SGLANG_OPT_DSV41_BF16_INDEX_LOGITS=0
#               (args/check.sh's box-e pin). Neither env-v3.txt (the box-d-up reason) nor
#               env-box-d-official.txt (bf16 index logits on that tree) is its env;
#   anything else (the image's /sgl-workspace/sglang) -> env-v3.txt, v4-b200.args' pair, as before.
# A box file's ENV_SEED=<env-*.txt> wins over the tree name: two boxes on one tree can serve different
# envs (box-a runs box-e-up like box-c, with env-box-a.txt = env-box-e.txt + the engram host table, its tuning
# round; args/check.sh's box-a pin). It is a bare file name under args/ -- a path, or anything not
# env-*.txt, stops the stage at this step, before the seed -- and only the BOX FILE's line counts: the scripts unset
# an ENV_SEED exported in the calling shell before lib.sh sources the box file. box-c's box file sets it
# too (env-box-e-w1.txt over box-e-up's env-box-e.txt); a box file without the key keeps the tree-name pick.
# Only the FIRST stage of a box seeds; a box that has a sglang-env.txt keeps it. restart.sh and
# cutover.sh carry this function byte for byte for their `default` env reseed, so a box's first-boot
# env and its `default` restarts are one file (tests/restart-env-seed.test.sh S1 holds the three copies
# equal): a change to it is made in all three in the same change.
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
ENV_SEED_NAME=$(stage_env_seed_name "${SGLANG_TREE%/}" "${ENV_SEED:-}") \
  || die "ENV_SEED in $BOX_ENV holds '${ENV_SEED:-}', not an args/env-*.txt file name -- the stage stops before the env seed"
note "sglang-env.txt (seeds $TOOLS_DIR/args/$ENV_SEED_NAME -- ${ENV_SEED:+ENV_SEED in the box file, over }the env of the serve line SGLANG_TREE=$SGLANG_TREE runs -- the first time this box is staged, since relaunch.sh now refuses to boot without a sglang-env.txt; a box that already has one, hand-tuned or not, is never touched)"
ENV_SEED_PATH="$TOOLS_DIR/args/$ENV_SEED_NAME"
[ -f "$ENV_SEED_PATH" ] || die "no $ENV_SEED_PATH -- the sglang-env.txt seed file is missing from this tree"
if [ "${DRY_RUN:-0}" = 1 ]; then
  note "DRY_RUN: skipping the sglang-env.txt presence check and seed (nothing is written)"
else
  if box_ssh "[ -s '$TOOLS_REMOTE/sglang-env.txt' ]"; then
    echo "SKIP sglang-env.txt (already present on the box -- not overwriting a possibly hand-tuned live override)"
  else
    box_scp "$ENV_SEED_PATH" "root@$BOX_IP:$TOOLS_REMOTE/sglang-env.txt"
    echo "OK  sglang-env.txt (seeded from $(basename "$ENV_SEED_PATH"))"
  fi
fi
# ENGINES: each listed engine's own env file (relaunch.sh <id> reads sglang-env-<id>.txt) is seeded the
# same way, from the same file, under the same never-overwrite rule. Only the ids ENGINES lists: an
# sglang-env-<id>.txt for any other id is never created (nor touched, if it is there).
for envf in ${STAGE_ENGINE_ENVS[@]+"${STAGE_ENGINE_ENVS[@]}"}; do
  if [ "${DRY_RUN:-0}" = 1 ]; then
    note "DRY_RUN: skipping the ${envf##*/} presence check and seed (would seed $ENV_SEED_NAME when absent)"
  elif box_ssh "[ -s '$envf' ]"; then
    echo "SKIP ${envf##*/} (already present on the box -- not overwriting a possibly hand-tuned live override)"
  else
    box_scp "$ENV_SEED_PATH" "root@$BOX_IP:$envf"
    echo "OK  ${envf##*/} (seeded from $(basename "$ENV_SEED_PATH"))"
  fi
done

note "the gate files (box-file keys TOOL_STRICT SBR_SCHEMA_MEMO SBR_UNFINISHED_REUSE SBR_PHASE_TIMING: 1 creates, 0 removes, unset leaves the file as it is)"
# After the sglang-env.txt step on purpose: the remote side reads the PATH each gate lives at from the
# box's own sglang-env.txt (the file the engine reads it from), falling back to the default under
# $TOOLS_REMOTE only when that file names none -- and says so, since the engine then reads no gate at
# all. A flip reaches a RUNNING engine within a second (no restart): a stage whose box file matches the
# box's state changes nothing and says "already". Every sglang-env*.txt on the box is read -- the legacy
# sglang-env.txt and, with ENGINES, each sglang-env-<id>.txt -- and each gate is applied at the path
# each file names (the same path twice says "already" the second time); with more than one file, each
# line names the file it followed.
if [ "${#GATE_OPS[@]}" -eq 0 ]; then
  echo "SKIP gate files ($BOX_ENV sets none of the four keys -- every gate file is left as it is)"
elif [ "${DRY_RUN:-0}" = 1 ]; then
  run box_ssh bash -s -- "$TOOLS_REMOTE" "${GATE_OPS[@]}"
else
  box_ssh bash -s -- "$TOOLS_REMOTE" "${GATE_OPS[@]}" <<'GATES'
set -uo pipefail
ENVFS=("$1/sglang-env.txt")
for f in "$1"/sglang-env-*.txt; do [ -e "$f" ] && ENVFS+=("$f"); done
shift
rc=0
for op in "$@"; do
 for ENVF in "${ENVFS[@]}"; do
  want=${op%%:*}; rest=${op#*:}; key=${rest%%:*}; path=${rest#*:}
  named=$(sed -n "s/^$key=//p" "$ENVF" 2>/dev/null | tail -n 1)
  said=""
  if [ -z "$named" ]; then
    said=" -- WARNING: $ENVF names no $key, so the engine reads no gate here"
  elif [ "$named" != "$path" ]; then
    path=$named; said=" (the path $ENVF names)"
  elif [ "${#ENVFS[@]}" -gt 1 ]; then
    said=" (as ${ENVF##*/} names it)"
  fi
  if [ "$want" = 1 ]; then
    if [ -e "$path" ]; then echo "OK  gate on: $path (already present)$said"
    elif : > "$path"; then echo "OK  gate on: $path CREATED -- a running engine opens it within a second$said"
    else echo "FAILED gate on: cannot create $path"; rc=1; fi
  else
    if [ ! -e "$path" ]; then echo "OK  gate off: $path (already absent)$said"
    elif rm -f "$path"; then echo "OK  gate off: $path REMOVED -- a running engine closes it within a second$said"
    else echo "FAILED gate off: cannot remove $path"; rc=1; fi
  fi
 done
done
exit $rc
GATES
fi

note "serve-model.env (always rewritten from this box file's current MODEL_DIR/SERVED_MODEL -- a box-identity fact, not runtime-tunable state like sglang-args.txt/sglang-env.txt, so unlike those this one IS clobbered on every stage)"
if [ "${DRY_RUN:-0}" = 1 ]; then
  note "DRY_RUN: skipping the serve-model.env write (nothing is written)"
else
  TMP_SM="$(mktemp "${TMPDIR:-/tmp}/serve-model-env.XXXXXX")"
  printf 'MODEL_DIR=%s\nSERVED_MODEL=%s\n' "$MODEL_DIR" "$SERVED_MODEL" > "$TMP_SM"
  box_scp "$TMP_SM" "root@$BOX_IP:$TOOLS_REMOTE/serve-model.env"
  rm -f "$TMP_SM"
  echo "OK  serve-model.env (MODEL_DIR=$MODEL_DIR SERVED_MODEL=$SERVED_MODEL) -- only the onstart's inline first-boot launcher reads this (tools/relaunch.sh already gets these values live from boxes/$BOX.env via lib.sh, every run); protects a container restart that re-runs the create-frozen onstart from resurrecting a stale render. Does not itself restart anything."
fi

note "caddy.env"
# The door's source-IP allowlist comes from the BOX FILE's TLS_ALLOW line and nowhere else. It used to
# ride along only when TLS_ALLOW happened to be exported in the shell that ran this script, so every
# stage from a plain shell rewrote caddy.env without it and the next tls.sh rendered a Caddyfile open
# to the whole internet (found standing up box-a, 2026-10-26). A value exported in the shell but absent
# from the box file is ignored on purpose: it would not survive the next stage. caddy.env is never
# written open: with no allowlist in the box file the WHOLE stage was refused up top, before any byte
# moved, leaving the box's caddy.env as it was -- tls.sh refuses to render a Caddyfile without an
# allowlist too, so nothing opens the door behind this -- unless TLS_ALLOW_ANY=1 (box file or
# environment) says an open door is meant. It carries NO token (#1306): it used to hold Caddy's
# DNS-01 CF_DNS_TOKEN; certificates are issued off the box by the operator and pushed as files,
# and since this step rewrites caddy.env whole on every stage, the first stage after #1306 also takes
# the token out of a box's old caddy.env (tls.sh refuses one that still has it).
TMP="$(mktemp "${TMPDIR:-/tmp}/caddy-env.XXXXXX")"
umask 077
printf '# written by stage.sh: the door allowlist only -- never a token (#1306)\n' > "$TMP"
# Single-quoted: tls.sh SOURCES caddy.env, and an unquoted space-separated list would run its second
# CIDR as a command. Safe because every word was checked above to hold only [0-9A-Fa-f:./].
[ -n "$BOX_TLS_ALLOW" ] && printf "TLS_ALLOW='%s'\n" "$BOX_TLS_ALLOW" >> "$TMP"
run box_scp "$TMP" "root@$BOX_IP:$TOOLS_REMOTE/caddy.env"
rm -f "$TMP"
run box_ssh "chmod 600 '$TOOLS_REMOTE/caddy.env'"
if [ -n "$BOX_TLS_ALLOW" ]; then
  echo "OK  caddy.env (TLS_ALLOW='$BOX_TLS_ALLOW' from $BOX_ENV staged; no token)"
else
  echo "OK  caddy.env (NO allowlist -- TLS_ALLOW_ANY=1 stages an open door; no token)"
fi

note "cert-push key"
# The cert-push key (#1306): the operator's issuer pushes the door's certificate with a key the box
# accepts ONLY as the forced command cert-install.sh -- no shell, no forwarding, no pty. Its PUBLIC
# half is the box file's CERT_PUSH_PUBKEY (config, not a secret); this step keeps exactly one line for
# it in /root/.ssh/authorized_keys, replacing any earlier cert-install line (a rotated key), and
# removes them all when the box file names none. Re-run on every stage because a container re-create
# can rewrite authorized_keys. Pushed over box_ssh from the operator's machine, it needs none.
CERT_PUSH_PUBKEY="${CERT_PUSH_PUBKEY:-}"
if [ -n "$CERT_PUSH_PUBKEY" ]; then
  [[ "$CERT_PUSH_PUBKEY" =~ ^(ssh-ed25519|ecdsa-sha2-nistp256|ssh-rsa)\ [A-Za-z0-9+/=]+(\ [A-Za-z0-9@._-]+)?$ ]] \
    || die "CERT_PUSH_PUBKEY in $BOX_ENV is not one public key line (<type> <base64> [comment])"
  CERT_LINE="command=\"bash $TOOLS_REMOTE/cert-install.sh\",restrict $CERT_PUSH_PUBKEY"
else
  CERT_LINE=""
fi
# grep exits 1 when it keeps no line (an empty file) and 2 on an error: only 0/1 go on to the mv, so a
# failed read or write never replaces authorized_keys (the Mac's own key lives there too).
run box_ssh "f=/root/.ssh/authorized_keys; mkdir -p /root/.ssh && touch \$f && { grep -vF 'cert-install.sh\",restrict ' \$f > \$f.cert-tmp; [ \$? -le 1 ]; } ${CERT_LINE:+&& printf '%s\\n' '$CERT_LINE' >> \$f.cert-tmp }&& chmod 600 \$f.cert-tmp && mv -f \$f.cert-tmp \$f"
if [ -n "$CERT_LINE" ]; then
  echo "OK  cert-push key (forced command: bash $TOOLS_REMOTE/cert-install.sh)"
else
  echo "OK  cert-push key (none in $BOX_ENV -- any earlier cert-install line removed; the certificate is pushed from the operator's machine over box_ssh)"
fi

note "fork deploy key"
# This key is registered (read-only) on the private mirror named by FORK_PRIVATE_REPO, and so is the
# repo the box's `fork` remote must point at -- never FORK_REPO (#609). Since #592 a box file's
# FORK_REPO is the PUBLIC upstream slug (sgl-project/sglang) that fork-sync.sh builds its `upstream`
# remote and its `gh pr view --repo` lookups from, while the private mirror this deploy key opens is
# FORK_PRIVATE_REPO: the same ssh value mirror-and-branch.sh pushes to, and the one 10-updating.md
# already documents this step as fetching from. The two are not interchangeable: a slug is not even a
# fetchable URL, so `git remote add fork sgl-project/sglang` would register a relative local path and
# the `git fetch` below would die, taking this script -- and every step bringup.sh runs after it --
# down on any clone that has no `fork` remote yet (a fresh box), which is exactly what the post-#592
# judge saw at this line. Its absence from the box file is checked far above, before any byte moves.
[ -f "$FORK_DEPLOY_KEY" ] || die "no fork deploy key at $FORK_DEPLOY_KEY -- place the read-only GitHub deploy key there before staging"
DEPLOY_KEY_NAME=$(basename "$FORK_DEPLOY_KEY")
run box_ssh "mkdir -p /root/.ssh && chmod 700 /root/.ssh"
run box_scp "$FORK_DEPLOY_KEY" "root@$BOX_IP:/root/.ssh/$DEPLOY_KEY_NAME"
run box_ssh "chmod 600 /root/.ssh/$DEPLOY_KEY_NAME"
echo "OK  $DEPLOY_KEY_NAME (600)"

# ------------------------------------------------------------------------------- the side paths (#2661)
# SGLANG_TREE and VENV are where relaunch.sh serves from: the checkout restart.sh --tree checks the
# served commit out in, and the venv whose bin/ relaunch.sh puts first on PATH. On the box-d layout both
# are SIDE PATHS built on the container -- /workspace/box-d-up and /opt/sglang-up -- that the pinned image
# (dev-dsv41) does not carry, so a fresh container has neither, and the fork step below died at its
# first line (`cd "$SGLANG_TREE"`, set -e) on every re-create: the box rented, the deploy key staged,
# nothing after it (README.md "The box-d layout on a re-create", ISSUES.md "Pending cleanup").
#
# So this step BUILDS a side path that is ABSENT, by the recipe that built box-c's, gpu1018's and
# gpu1018b's (plans/box-d-cutover.md §6 step 1; the release-gate reports' "Venv
# recipe"), and only VERIFIES one that exists. Building rather than refusing with a pointer is the
# change that lets a re-create finish unattended: a refusal still leaves bringup.sh dead at this step
# with a billed box and the recipe to run by hand, which is the gap itself.
#   * the checkout, when SGLANG_TREE is absent: a --depth 200 single-branch clone of $FORK_BRANCH from
#     $FORK_PRIVATE_REPO, remote named `fork`, through the deploy key just staged, into a sibling
#     `<SGLANG_TREE>.stage-clone` that is renamed into place only once it is DETACHED at $FORK_TREE (the
#     branch tip when FORK_TREE is unset) -- detached because the fork step below force-moves
#     $FORK_BRANCH in this repo and git refuses to move a checked-out branch. A clone that died half
#     way never sits at SGLANG_TREE; the next stage removes the leftover `.stage-clone` and clones again.
#     Shallow on purpose: it is the shape the fork step's `--depth 200` re-fetch keeps, and FORK_TREE
#     must be inside that window anyway (the fork step's own comment) -- a pin outside it is refused
#     here, naming the window, with nothing left at SGLANG_TREE.
#   * the venv, when VENV is absent: `uv venv <VENV> --python 3.12`, `uv pip install --no-deps` of
#     the tree's freeze pins (stage_freeze_name below: args/image-box-e.freeze.txt for box-e-up's
#     /opt/sglang-box-e, args/image-up.freeze.txt for any other tree, box-d-up's /opt/sglang-up
#     included; the staged copy just md5-checked; minus its comments, the
#     `@ file://` lines and the `-e git+` sglang line, with the recipe's index flags), then
#     `uv pip install --no-deps [--no-build-isolation] -e <SGLANG_TREE>/python` -- WITHOUT build isolation
#     when the freeze pins setuptools-rust (image-box-e.freeze.txt's build-time pins, beside the venv's own
#     torch 2.14.1+cu130 and setuptools/wheel), the way box-c's box-e-up (2026-10-26) and box-a's 4943d6ed90
#     build (the .so files box-c serves, #96) were made: from upstream 3a3a8611 on, the radix-tree crate
#     (mem_cache) links libtorch through tch at build time against the ACTIVE torch, and an isolated build
#     would fetch a second, PyPI torch==2.14.1 and link against that instead (#122). A freeze with no
#     setuptools-rust pin (image-up.freeze.txt) keeps uv's isolated build, which resolves its own; then
#     `uv pip freeze` against the pins, where
#     only the skipped lines may differ, and the venv's `sglang` must resolve inside SGLANG_TREE/python,
#     and every Rust extension module the tree builds (stage_rust_modules below: box-e-up's five) must
#     import. The tree's setup.py builds its Rust extensions only when `setuptools_rust` imports at build
#     time and SKIPS them silently otherwise -- box-c, 2026-10-26: a `--no-build-isolation` install of
#     box-e-up into a venv without setuptools-rust 'succeeded' in 1.25 s as sglang 0.0.0 with no .so at
#     all -- so a clean uv exit proves nothing, and a venv missing one is refused, its marker kept. With
#     modules to build, cargo must be on PATH (else $HOME/.cargo/bin, prepended) before anything is built.
#     uv is the image's /opt/sglang/bin/uv (else uv on PATH); every uv call runs with XDG_CACHE_HOME
#     unset and UV_CACHE_DIR=/root/.cache/uv-<venv name>, a cache of the build's own, never the box's
#     CACHE (the box's /etc/environment sets XDG_CACHE_HOME to it). About 8 minutes, 7 of them the
#     rust/sglang-grpc build, in the foreground like the graft step's graft-apply.sh. A marker
#     `<VENV>.stage-building` is written before `uv venv` and removed only after every check passed: a
#     VENV beside that marker is this step's own interrupted (or refused) build, and the next stage
#     removes it and builds again; a VENV without one is never deleted.
#   * an EXISTING path is never rebuilt, moved or deleted. A SGLANG_TREE that is not a checkout of its
#     own, or a VENV with no bin/python3, is refused (neither can serve, and neither is this step's to
#     remove); a VENV whose sglang resolves somewhere other than SGLANG_TREE/python, or that cannot
#     import one of those Rust modules (one import, bounded at 300 s, GPUs hidden), is a WARNING, never
#     a refusal -- a restage of a serving box must not die on a read it cannot be sure of.
#   * the image's own paths (/sgl-workspace/sglang, /opt/sglang) are never built: absent, the container
#     is not the pinned CONTAINER_IMAGE, and this step refuses.
# Nothing here restarts anything. The image's own tree and venv are untouched.
note "side paths: SGLANG_TREE=$SGLANG_TREE, VENV=$VENV (built when absent, verified when present)"
# The pins a side venv is built to, by SGLANG_TREE's name like the env seed above: box-e-up's venv
# carries the upstream-merge deltas (image-box-e.freeze.txt's header), so building it to
# image-up.freeze.txt would leave the box-e tree without the wheels it imports.
stage_freeze_name() {
  case "${1##*/}" in
    box-e-up) printf '%s' image-box-e.freeze.txt ;;
    *)       printf '%s' image-up.freeze.txt ;;
  esac
}
FREEZE_NAME=$(stage_freeze_name "${SGLANG_TREE%/}")
# The Rust extension modules a side venv's editable sglang must import, comma-separated, by
# SGLANG_TREE's name like the pins: one per crate under the tree's rust/ declaring
# [package.metadata.sglang] python-module (the five .so of box-c's good box-e build, 2026-10-26; the same
# five crates declare one at upstream d72664d903cb and at box-c's served prod cc31e16a557c). Non-empty turns
# the check on, and the step below then checks the modules the STAGED tree's rust/ declares (what its
# setup.py builds), naming any difference from this list, so a crate added or dropped by a later
# FORK_TREE is neither missed nor demanded. Empty for any other tree: box-d-up's /opt/sglang-up venvs
# were built and verified by hand, and its Rust set is not recorded in this tree, so no check is made
# up for it here.
stage_rust_modules() {
  case "${1##*/}" in
    box-e-up) printf '%s' sglang.srt.mem_cache.rust_tree_core.mem_cache,sglang.srt.rust_extensions._bench,sglang.srt.rust_extensions._grpc,sglang.srt.rust_extensions._multimodal,sglang.srt.rust_extensions._server ;;
    *)       printf '%s' '' ;;
  esac
}
RUST_MODS=$(stage_rust_modules "${SGLANG_TREE%/}")
FREEZE_REMOTE="$TOOLS_REMOTE/args/$FREEZE_NAME"
[ -f "$TOOLS_DIR/args/$FREEZE_NAME" ] || die "no $TOOLS_DIR/args/$FREEZE_NAME -- the pins a side venv on SGLANG_TREE=$SGLANG_TREE is built to are missing from this tree"
if [ "${DRY_RUN:-0}" = 1 ]; then
  run box_ssh bash -s -- "$SGLANG_TREE" "$VENV" "$FORK_PRIVATE_REPO" "/root/.ssh/$DEPLOY_KEY_NAME" "$FORK_BRANCH" "${FORK_TREE:-}" "$FREEZE_REMOTE" ${RUST_MODS:+"$RUST_MODS"}
else
  # The module list rides as an 8th operand only when there is one, so a tree with none sends the
  # seven operands it always did.
  box_ssh bash -s -- "$SGLANG_TREE" "$VENV" "$FORK_PRIVATE_REPO" "/root/.ssh/$DEPLOY_KEY_NAME" "$FORK_BRANCH" "${FORK_TREE:-}" "$FREEZE_REMOTE" ${RUST_MODS:+"$RUST_MODS"} <<'SIDEPATHS'
set -euo pipefail
SGL=$1; VENV=$2; REPO=$3; KEY=$4; BRANCH=$5; TREE=$6; FREEZE=$7; RUST=${8:-}
IMAGE_TREE=/sgl-workspace/sglang; IMAGE_VENV=/opt/sglang
RECIPE='README.md "The box-d layout on a re-create" (plans/box-d-cutover.md section 6 step 1)'
fail() { echo "side paths: REFUSED: $*" >&2; exit 1; }
df -h "$(dirname "$SGL")" "$(dirname "$VENV")" 2>/dev/null || true

# ---- the checkout
if [ -e "$SGL" ]; then
  top=$(git -C "$SGL" rev-parse --show-toplevel 2>/dev/null || true)
  [ -n "$top" ] && [ "$(cd "$top" && pwd -P)" = "$(cd "$SGL" && pwd -P)" ] \
    || fail "$SGL exists but is not a git checkout of its own -- it cannot be served from, and it is not this step's to delete: move it aside by hand (stage.sh then clones it), or build it by $RECIPE"
  echo "side path: $SGL is a checkout (HEAD $(git -C "$SGL" rev-parse --short=12 HEAD 2>/dev/null || echo '<unborn>')) -- left as it is"
else
  [ "$SGL" != "$IMAGE_TREE" ] || fail "the image's own tree $IMAGE_TREE is missing -- this container is not the pinned CONTAINER_IMAGE, and only a side path is ever built here"
  CLONE="$SGL.stage-clone"
  WANT=${TREE:-fork/$BRANCH}
  SSHCMD="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
  rm -rf "$CLONE"
  mkdir -p "$(dirname "$SGL")"
  echo "side path: $SGL is absent -- cloning $BRANCH from $REPO (--depth 200, remote fork) into $CLONE"
  GIT_SSH_COMMAND="$SSHCMD" git clone -q --depth 200 --single-branch --branch "$BRANCH" --origin fork --no-checkout "$REPO" "$CLONE"
  git -C "$CLONE" config core.sshCommand "$SSHCMD"
  git -C "$CLONE" rev-parse --verify -q "$WANT^{commit}" >/dev/null \
    || fail "$WANT is not a commit in the newest 200 commits of $BRANCH -- FORK_TREE must sit inside the window the fork step's --depth 200 re-fetch keeps (its comment in stage.sh); nothing was put at $SGL (the partial clone $CLONE is removed by the next stage)"
  git -C "$CLONE" checkout -q --detach "$WANT"
  mv "$CLONE" "$SGL"
  echo "side path: $SGL cloned, detached at $(git -C "$SGL" rev-parse --short=12 HEAD) ($WANT)"
fi

# ---- the venv
BUILDING="$VENV.stage-building"
sglang_origin() {
  "$VENV/bin/python3" -c 'import importlib.util as u; s = u.find_spec("sglang"); print((s.origin or "") if s else "")' 2>/dev/null || true
}
# The Rust modules checked: with $RUST (stage.sh's list) non-empty, the ones $SGL's rust/ declares --
# every Cargo.toml under it (target/ aside) with a `python-module = "<module>"` line in its
# [package.metadata.sglang] table, which is what setup.py discovers and builds (the root workspace and
# its extension-manifests, sglang-radix-tree's own workspace among them) -- else $RUST itself when the
# tree declares none. Each difference from $RUST is printed once, below.
rust_declared() {
  [ -d "$SGL/rust" ] || return 0
  find "$SGL/rust" -name target -prune -o -name Cargo.toml -type f -print 2>/dev/null | LC_ALL=C sort | while IFS= read -r f; do
    awk '/^[[:space:]]*\[/ { sec = $0; sub(/[[:space:]]*(#.*)?$/, "", sec); sub(/^[[:space:]]*/, "", sec) }
         sec == "[package.metadata.sglang]" && /^[[:space:]]*python-module[[:space:]]*=[[:space:]]*"/ {
           v = $0; sub(/^[^"]*"/, "", v); sub(/".*$/, "", v); if (v != "") print v }' "$f"
  done | LC_ALL=C sort -u | paste -sd, -
}
# in_list <module> <comma-list>
in_list() { case ",$2," in *",$1,"*) return 0 ;; esac; return 1; }
RUST_CHECK=""
if [ -n "$RUST" ]; then
  DECLARED=$(rust_declared)
  if [ -z "$DECLARED" ]; then
    RUST_CHECK=$RUST
    echo "WARNING: side path: $SGL/rust declares no [package.metadata.sglang] python-module -- checking stage.sh's list ($RUST)"
  else
    RUST_CHECK=$DECLARED
    extra=""; gone=""
    for m in ${DECLARED//,/ }; do in_list "$m" "$RUST" || extra="$extra${extra:+, }$m"; done
    for m in ${RUST//,/ }; do in_list "$m" "$DECLARED" || gone="$gone${gone:+, }$m"; done
    [ -z "$extra" ] || echo "side path: $SGL declares Rust extension(s) stage.sh's stage_rust_modules does not list: $extra -- checked too"
    [ -z "$gone" ] || echo "WARNING: side path: stage.sh's stage_rust_modules lists Rust extension(s) $SGL does not declare: $gone -- not checked (setup.py does not build them); update the list"
  fi
fi
# One import check is bounded: it imports sglang (and torch, for the radix-tree module) on a box that
# may be serving, so it gets STAGE_RUST_IMPORT_TIMEOUT_S (300 s; a test knob, never set on a box)
# under coreutils' timeout, and no stdin.
RUST_TIMEOUT_S=${STAGE_RUST_IMPORT_TIMEOUT_S:-300}
RUST_TIMEOUT=()
command -v timeout >/dev/null 2>&1 && RUST_TIMEOUT=(timeout "$RUST_TIMEOUT_S")
# rust_missing: the modules of $RUST_CHECK that $VENV's python3 cannot import, each with its error,
# `; `-joined; empty when every one imports (or there is none to check). A real import, GPUs hidden: a
# .so on disk that does not load is as missing as one never built.
rust_missing() {
  local out
  [ -n "$RUST_CHECK" ] || return 0
  out=$(CUDA_VISIBLE_DEVICES= ${RUST_TIMEOUT[@]+"${RUST_TIMEOUT[@]}"} "$VENV/bin/python3" -c '
import importlib, sys
bad = []
for m in sys.argv[1].split(","):
    try:
        importlib.import_module(m)
    except BaseException as e:
        bad.append("%s (%s: %s)" % (m, type(e).__name__, (str(e).splitlines() or [""])[0]))
print("RUST-MISSING:" + "; ".join(bad))
' "$RUST_CHECK" </dev/null 2>/dev/null | grep '^RUST-MISSING:' | tail -n 1) || true
  case "$out" in
    RUST-MISSING:*) printf '%s' "${out#RUST-MISSING:}" ;;
    *) printf '%s' "$RUST_CHECK (the import check itself did not finish in $VENV/bin/python3${RUST_TIMEOUT[0]:+ within ${RUST_TIMEOUT_S}s})" ;;
  esac
}
RUST_N=$(printf '%s\n' "$RUST_CHECK" | awk -F, '{ print (NF ? NF : 0) }')
RUST_WHY="setup.py builds them only when setuptools_rust imports at build time and SKIPS them silently otherwise (a --no-build-isolation install into a venv without setuptools-rust/setuptools-scm 'succeeds' as sglang 0.0.0 with no .so), or SGLANG_BUILD_RUST_EXTS filtered them out"
if [ -e "$VENV" ] && [ ! -e "$BUILDING" ]; then
  [ -x "$VENV/bin/python3" ] \
    || fail "$VENV exists but has no bin/python3 -- not a venv relaunch.sh can serve from, and not this step's to delete (no $BUILDING marker beside it): move it aside by hand (stage.sh then builds it), or build it by $RECIPE"
  origin=$(sglang_origin)
  case "$origin" in
    "$SGL"/python/*) echo "side path: $VENV is a venv whose sglang is $SGL's ($origin) -- left as it is" ;;
    *) echo "WARNING: side path: $VENV's sglang resolves to '${origin:-<nothing>}', not inside $SGL/python -- left as it is; relaunch.sh serves whatever that venv imports, so check it (or move it aside and re-stage to have it rebuilt)" ;;
  esac
  if [ -n "$RUST_CHECK" ]; then
    missing=$(rust_missing)
    if [ -n "$missing" ]; then
      echo "WARNING: side path: $VENV cannot import Rust extension(s): $missing -- left as it is; $RUST_WHY. Move it aside and re-stage to have it rebuilt (box-c/qa/box-e-venv/README.md \"Build-time pins\")"
    else
      echo "side path: $VENV imports its $RUST_N Rust extensions"
    fi
  fi
else
  [ "$VENV" != "$IMAGE_VENV" ] || fail "the image's own venv $IMAGE_VENV is missing -- this container is not the pinned CONTAINER_IMAGE, and only a side path is ever built here"
  if [ -e "$VENV" ]; then
    echo "side path: $VENV is an interrupted build of this step ($BUILDING is beside it) -- removing it and building again"
    rm -rf "$VENV"
  fi
  UV=/opt/sglang/bin/uv
  [ -x "$UV" ] || UV=$(command -v uv 2>/dev/null || true)
  [ -n "$UV" ] || fail "no uv to build $VENV with (/opt/sglang/bin/uv, or uv on PATH) -- this container is not the pinned CONTAINER_IMAGE"
  [ -s "$FREEZE" ] || fail "no $FREEZE on the box -- the rsync above stages args/; re-run stage.sh"
  [ -d "$SGL/python" ] || fail "$SGL has no python/ to install editable -- $SGL is not an sglang checkout"
  if [ -n "$RUST_CHECK" ]; then
    # setup.py runs cargo (metadata, then the build) for the Rust extensions; a non-interactive ssh shell
    # does not source ~/.cargo/env, so rustup's bin dir is put on PATH here when cargo is not already.
    command -v cargo >/dev/null 2>&1 || [ ! -x "$HOME/.cargo/bin/cargo" ] || export PATH="$HOME/.cargo/bin:$PATH"
    command -v cargo >/dev/null 2>&1 \
      || fail "no cargo on PATH (nor $HOME/.cargo/bin/cargo) -- $SGL's editable install builds Rust extensions ($RUST_CHECK) with it; install the Rust toolchain (rustup) on the box and re-stage. Nothing was built"
  fi
  mkdir -p "$(dirname "$VENV")"
  printf 'stage.sh is building %s (started %s); a VENV beside this file is an unfinished build, removed and rebuilt by the next stage\n' "$VENV" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BUILDING"
  unset XDG_CACHE_HOME
  export UV_CACHE_DIR="/root/.cache/uv-$(basename "$VENV")"
  INDEX=(--extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://flashinfer.ai/whl/cu130
         --extra-index-url https://flashinfer.ai/whl --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match)
  REQ=$(mktemp "${TMPDIR:-/tmp}/sidepaths-req.XXXXXX")
  GOT=$(mktemp "${TMPDIR:-/tmp}/sidepaths-got.XXXXXX")
  trap 'rm -f "$REQ" "$GOT"' EXIT
  grep -vE '^[[:space:]]*(#|$)' "$FREEZE" | grep -vF '@ file://' | grep -vE '^-e[[:space:]]' > "$REQ" || true
  [ -s "$REQ" ] || fail "$FREEZE holds no pin once its skipped lines are dropped"
  echo "side path: $VENV is absent -- building it: uv venv --python 3.12, $(wc -l < "$REQ" | tr -d ' ') pins --no-deps from $FREEZE, then -e $SGL/python (UV_CACHE_DIR=$UV_CACHE_DIR)"
  "$UV" venv "$VENV" --python 3.12
  "$UV" pip install --python "$VENV/bin/python3" --no-deps "${INDEX[@]}" -r "$REQ"
  BUILD_ISO=()
  if grep -qiE '^setuptools[-_.]rust==' "$REQ"; then BUILD_ISO=(--no-build-isolation); fi
  ISO_SAID="with the build isolation of uv"; [ ${#BUILD_ISO[@]} = 0 ] || ISO_SAID="${BUILD_ISO[*]}"
  echo "side path: editable install $ISO_SAID ($(grep -ciE '^setuptools[-_.]rust==' "$REQ" || true) setuptools-rust pin in $FREEZE)"
  "$UV" pip install --python "$VENV/bin/python3" --no-deps ${BUILD_ISO[@]+"${BUILD_ISO[@]}"} -e "$SGL/python"
  # The freeze check: the venv holds exactly the pins, by normalised name (PEP 503: lower case, runs of
  # -_. as one -), once the editable sglang and any `@ file://` line are set aside on both sides.
  norm() { awk -F'==' 'NF >= 2 { n = tolower($1); gsub(/[-_.]+/, "-", n); sub(/^[^=]*==/, ""); print n "==" $0 }' | LC_ALL=C sort; }
  "$UV" pip freeze --python "$VENV/bin/python3" | grep -vF '@ file://' | grep -vE '^-e[[:space:]]' | norm > "$GOT" || true
  if ! DIFF=$(norm < "$REQ" | diff - "$GOT"); then
    fail "$VENV does not freeze to $FREEZE's pins (< the pin, > the venv; first lines): $(printf '%s\n' "$DIFF" | grep -E '^[<>]' | head -n 12 | tr '\n' ' ')-- $BUILDING stays, so the next stage removes this venv and builds it again"
  fi
  origin=$(sglang_origin)
  case "$origin" in
    "$SGL"/python/*) ;;
    *) fail "$VENV's sglang resolves to '${origin:-<nothing>}' after the editable install, not inside $SGL/python -- $BUILDING stays, so the next stage builds it again" ;;
  esac
  RUST_SAID=""
  if [ -n "$RUST_CHECK" ]; then
    missing=$(rust_missing)
    [ -z "$missing" ] \
      || fail "$VENV's editable sglang cannot import Rust extension(s): $missing -- $RUST_WHY; $BUILDING stays, so the next stage builds it again"
    RUST_SAID=", its $RUST_N Rust extensions import"
  fi
  rm -f "$BUILDING"
  echo "side path: $VENV built -- $(wc -l < "$REQ" | tr -d ' ') pins frozen as pinned, sglang editable from $SGL/python ($origin)$RUST_SAID"
fi
SIDEPATHS
fi

note "fork remote + fetch (no checkout) from $FORK_PRIVATE_REPO: $FORK_BRANCH"
if [ "${DRY_RUN:-0}" = 1 ]; then
  run box_ssh bash -s -- "$SGLANG_TREE" "$FORK_PRIVATE_REPO" "/root/.ssh/$DEPLOY_KEY_NAME" "$FORK_BRANCH"
else
  box_ssh bash -s -- "$SGLANG_TREE" "$FORK_PRIVATE_REPO" "/root/.ssh/$DEPLOY_KEY_NAME" "$FORK_BRANCH" <<'REMOTE'
set -euo pipefail
SGL=$1; REPO=$2; KEY=$3; BRANCH=$4
cd "$SGL"
# The remote's URL is MADE to match $REPO, not merely added when it is absent. `git remote add`
# alone would leave a box that this very step broke before the FORK_PRIVATE_REPO change (#609)
# holding its bad url -- a fresh box staged then got `git remote add fork sgl-project/sglang`, which
# succeeds while the fetch below it fails, so the remote exists with a slug url that every later
# stage would then short-circuit past (`get-url fork` succeeds -> no add) and fail on forever. A
# remote that already carries the right url is untouched by the compare, so a normal re-stage
# changes nothing here and only prints the line below; a remote pointing somewhere else is reported
# before it is repointed, since this remote is what cutover.sh checks the served tree out of.
if git remote get-url fork >/dev/null 2>&1; then
  FORK_URL=$(git remote get-url fork)
  if [ "$FORK_URL" != "$REPO" ]; then
    echo "fork remote: $FORK_URL -> $REPO"
    git remote set-url fork "$REPO"
  fi
else
  git remote add fork "$REPO"
fi
echo "fork remote = $(git remote get-url fork)"
# repo-local core.sshCommand, not GIT_SSH_COMMAND: no per-remote sshCommand config exists, so
# this is set once here and every later fetch/checkout in this repo needs no env var at all.
git config core.sshCommand "ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
# The fetch names its DESTINATION: a single-branch clone's `fork` remote maps only its own branch,
# so a bare branch name fills FETCH_HEAD alone and the `git branch -f` below dies 128; `+src:dst`
# lands fork/$BRANCH whatever the remote's config says, and writes no config.
#
# It is shallow only where the clone already is. `--depth` on a FULL clone does not just limit what
# arrives: git records the new boundary in .git/shallow and the clone BECOMES shallow, cutting every
# ref's history at the window (`git log`, `merge-base`, `--contains` and a later full fetch all see
# the cut). So a full clone gets a plain fetch -- it already holds nearly every object, and nothing
# is truncated -- and `--depth 200` is spent only on a clone `git rev-parse --is-shallow-repository`
# reports shallow (a git too old to know the flag echoes it back, which is not `true`, and gets the
# plain fetch: the safe direction).
FORK_REFSPEC="+refs/heads/$BRANCH:refs/remotes/fork/$BRANCH"
if [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = true ]; then
  echo "fork fetch: shallow clone -> --depth 200"
  git fetch --depth 200 fork "$FORK_REFSPEC"
else
  echo "fork fetch: full clone -> full history"
  git fetch fork "$FORK_REFSPEC"
fi
git branch -f "$BRANCH" "fork/$BRANCH"
git branch --set-upstream-to="fork/$BRANCH" "$BRANCH" >/dev/null 2>&1 || true
# On a SHALLOW clone the window is part of what a pin can name: `--depth 200` lands $BRANCH's
# newest ~200 commits and nothing older. The box file's FORK_TREE must be one of them -- or
# bringup.sh's fork checkout (`git -C $SGLANG_TREE rev-parse --verify '<sha>^{commit}'`) reads it as
# a commit that does not exist when the mirror has it. Keep the pin inside the window, or raise the
# depth above, the fetch every stage re-applies: a one-off `git fetch --depth <n>` of that refspec on
# the box does make a bare sha resolve (the object lands; the next stage truncates the ref's
# ancestry, not the object db -- measured, 300-commit fixture, 2026-10-17) but not a history-walking
# spelling of it (`fork/$BRANCH~<n>`, `--contains`, a branch cut from it), and a `gc` can reap an
# object nothing reaches. A full clone has no window: every commit on $BRANCH resolves.
HEAD_SHA=$(git rev-parse --short=12 HEAD)
FORK_SHA=$(git rev-parse --short=12 "fork/$BRANCH")
echo "image tree HEAD = $HEAD_SHA"
echo "fork/$BRANCH     = $FORK_SHA"
REMOTE
fi

# The door's Caddy binary, LAST (stage_caddy above says why): a refusal here leaves the rest staged and
# exits non-zero, so bringup.sh stops before its tls.sh step.
note "caddy binary -> $BOX_IP:$TOOLS_REMOTE/caddy (pin $CADDY_PIN)"
stage_caddy

note "stage complete -- nothing checked out in an existing tree (a side path built just now was cloned detached at ${FORK_TREE:-fork/$FORK_BRANCH}), the engine not restarted (a watchdog on a superseded script replaced -- see the watchdog line above), sglang-args.txt untouched; sglang-env.txt untouched if present, seeded from $ENV_SEED_NAME if this box had none; serve-model.env rewritten from this box file's MODEL_DIR/SERVED_MODEL"
