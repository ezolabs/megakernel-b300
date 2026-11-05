#!/bin/bash
# dev box: the orchestrator for a fresh box, start to a smoke-tested engine.
#
#   preflight (hostctl key, ssh key, ENV_FILE keys present)
#     -> select an offer (offer-search.sh search --pick-best, or --offer <id> to skip searching)
#     -> build-onstart.sh <args-file> (renders the onstart fragment for THIS boot's args file)
#     -> offer-search.sh create <offer-id> (ONSTART_CMD=<the rendered fragment>), whose stdout is
#        the create output followed by the new instance id on its LAST line
#     -> attach ${SSH_KEY}.pub to THAT instance: `hostctl attach ssh <id> <pub>`, addressed by
#        the id create just printed and never by $INSTANCE_ID (the stale value the reconcile step
#        below exists for). $SSH_KEY is the box file's own key and the only one a sandboxed worker can
#        read and use at all (BatchMode, no ssh-agent, no keychain), and a create is NOT guaranteed to
#        have carried it: a re-create re-runs the instance's create-frozen onstart, rendered before
#        SSH_KEY was repointed, and the passphrase-protected the provider ACCOUNT key that onstart injects
#        instead is no key a BatchMode ssh can use -- `Permission denied (publickey)`, reproduced
#        2026-10-17, and every step after the ssh wait below then fails before it reaches the box. A
#        failure here is a loud WARNING, never a die: the box is already rented (a die would send the
#        operator to re-run a create that rents a SECOND box), and the ssh wait below re-runs this
#        same attach the moment the box refuses the key, then diagnoses the refusal if it persists.
#     -> reconcile: read the NEW instance's own ip/ports back off the provider API (never off the box
#        file) and write them into boxes/$BOX.env (BOX_IP/SSH_PORT/APP_PORT_EXT/INSTANCE_ID), or
#        die naming the exact lines to paste. This is the guard the box-g incident (2026-09-13)
#        needed: create prints the instance it just rented, but lib.sh's box_ssh resolves
#        BOX_IP/SSH_PORT from the box FILE, which still described the PREVIOUS box -- so every step
#        after create (wait for ssh, stage.sh, the engine gates, restart.sh) silently ran against
#        a live production box. A warning does not halt; this step, and the box_ssh guard below, do.
#     -> wait for ssh
#     -> /root/.hf.env: HF_TOKEN piped on stdin into a 0600 file the onstart's download of the PUBLIC
#        upstream checkpoint reads (the token's only use: nothing of ours is ever fetched from HF)
#        (never argv, never rendered, never printed)
#     -> stage.sh (tool set; the side paths SGLANG_TREE/VENV built when absent -- a fresh container on
#        the box-d layout -- and verified when present; sglang-env.txt seeded from the tree's env,
#        env-box-d-official.txt on box-d-up, env-box-e.txt on box-e-up, env-v3.txt on any other tree;
#        fork remote + fetch, no checkout of an existing tree; last, on a box file whose
#        CADDY_EXPECT_SHA256 is not the stock build, the patched Caddy binary copied from this Mac's
#        artifact store -- CADDY_ARTIFACT, <HOME>/workspace/backups/caddy-h2/caddy-h2win-2.11.7 + .sha256 --
#        onto the fresh box, which has none; a refusal stops the run here, naming caddy-h2/build.sh)
#     -> download gate (read the LAST download marker in /workspace/download.log -- only a
#        COSMOS_DOWNLOAD_COMPLETE that no later DOWNLOAD-INCOMPLETE line supersedes counts, because
#        the onstart appends the success marker BEFORE its own completeness check and appends the
#        failure line immediately after it whenever that check finds the checkpoint incomplete)
#     -> graft (optional): when the box file's GRAFT_SRC is non-empty, copy the kit's
#        apply_graft.py out of this checkout onto the box and check it by sha256, then copy the
#        graft's three files (edited_tensors.safetensors, applier.json, manifest.json) from GRAFT_SRC
#        -- the graft's backup directory on THIS Mac, <HOME>/workspace/backups/grafts/<repo>@<branch>/ --
#        into the box file's GRAFT_DIR over box_scp, re-hash all three ON THE BOX and refuse on any
#        mismatch: edited_tensors.safetensors against the box file's GRAFT_SHA256, the other two
#        against the Mac copy's own bytes. Our grafts never come from Hugging Face
#        (our internal artifact policy: boxes receive our files over ssh from the Mac); a box
#        file that still carries the retired GRAFT_REPO/GRAFT_REV keys is refused in preflight,
#        then run tools/graft-apply.sh with STOCK_DIR/GRAFT_DIR/MODEL_DIR/APPLY_GRAFT_PY from the box
#        file, rebuilding MODEL_DIR from the stock checkpoint plus the graft. The pair is checked in
#        PREFLIGHT, before anything is rented (the "graft step's build pair" section below): the graft
#        about to be applied and the MODEL_DIR it would rebuild must name the same build, or this run
#        would put one build's bytes under another build's path and the box would serve them under that
#        name -- box-c's box file did exactly that (GRAFT_DIR is attempt 2's graft, MODEL_DIR named v3)
#        and graft-apply.sh's own "non-empty MODEL_DIR with no scratch marker" refusal cannot see it on
#        a create, where MODEL_DIR does not exist yet. The stage is what makes the
#        step work at all: stage.sh stages this tool set (tools/) and nothing from the grafting kit,
#        so before this the box had no apply_graft.py and graft-apply.sh died at its own
#        APPLY_GRAFT_PY check -- the automated graft could never run. Skipped entirely when
#        GRAFT_SRC is empty (serve STOCK_DIR unmodified). The engine boot gates right after this
#        wait for sglang to come up serving the now-ready MODEL_DIR -- no separate restart is
#        triggered here, the box's own watchdog already retries relaunch until MODEL_DIR exists.
#     -> engine boot gates (/health, /get_server_info backend check)
#     -> fork checkout (restart.sh fork - - --tree fork/$FORK_BRANCH; verifies fork/$FORK_BRANCH
#        exists on the box first, dies pointing at stage.sh if not; restart.sh polls readiness
#        itself, so this waits for a ready engine before continuing; on a box file with ENGINES,
#        `restart.sh --engine all fork <--engine-args> default --tree <FORK_TREE>` and the gates per copy)
#     -> smoke.py (on the box; once per copy with ENGINES)
#
# Every one of these after stage.sh is a SIBLING script from another item in this same tool set
# (P3 smoke.py, P4 offer-search.sh/build-onstart.sh) -- bringup.sh calls each by its documented path/convention and
# refuses clearly if one is not yet built, rather than guessing at a different tool. Supports
# DRY_RUN=1 (or --dry-run) throughout (via lib.sh's `run`) for a full dry pass with nothing
# touched: the offer search and the onstart render only print under DRY_RUN too, and a missing
# HF_TOKEN in ENV_FILE is a warning rather than a hard failure (a real create would still need
# it -- this just lets a dry pass run to the end without a secrets file on hand).
#
# The instance this run created is authoritative for the whole run: its id (from create's own
# output) is resolved to ip/ports through `hostctl show instance <id> --raw` -- never through the
# box file, which is exactly what went stale -- written into boxes/$BOX.env, read back and
# verified, and every box_ssh this script makes afterwards is fenced to it (the guard below
# shadows lib.sh's own box_ssh for this script only). Set BOX_WRITE=0 to make a mismatch a hard
# `die` naming the exact lines to paste instead of writing the box file (a read-only checkout, or
# an operator who wants to eyeball the change first); a write that fails, or whose result does not
# read back as exactly the resolved values, dies the same way. Under DRY_RUN nothing is created,
# nothing is written and nothing is fenced.
#
# usage: bringup.sh <args-file> [--offer <offer-id> | --instance <instance-id>]
#                   [--engine-args <per-copy args-file>] [--dry-run]
#   <args-file>   the tools/args/*.args line to boot with: rendered into the onstart via
#                 build-onstart.sh <args-file>, and it becomes sglang-args.txt on the box.
#   --offer <id>  use this the provider offer id instead of searching for one. Without it, bringup.sh
#                 runs `offer-search.sh search --pick-best` and takes the cheapest verified
#                 rentable offer of this box's own GPU shape (boxes/$BOX.env's GPU_NAME x
#                 NUM_GPUS, NVLink required -- offer-search.sh drops bw_nvlink<=0/absent).
#   --instance <id>  RESUME a bring-up whose create already rented the box (#1704): skip the offer
#                 select, the onstart render and the create entirely, take <id> as this run's
#                 instance, and start at the attach step + the reconcile; every later step runs
#                 unchanged. This is the only way back after a die past the create: a plain re-run
#                 always creates, i.e. rents a SECOND box -- so every die after a create (or after a
#                 resume) prints the exact resume line (`RESUME: ... --instance <id>`), never
#                 "re-run". Refused together with --offer, refused for box-c's instance <INSTANCE_ID>, and
#                 refused for an instance whose own record (`hostctl show instance <id> --raw`) carries
#                 a label other than this box file's (INSTANCE_LABEL, else ${BOX}-dsv41-<lowercased
#                 GPU_NAME> -- offer-search.sh's offer_label rule) or a gpu_name / num_gpus other
#                 than its GPU_NAME / NUM_GPUS. Under DRY_RUN the label check is printed, not
#                 made. Pinned by tests/bringup-resume.test.sh.
#                 Whatever INSTANCE_ID boxes/box-c.env names is refused too, with ONE
#                 exception (#2661): a box-c RE-CREATE (BOX=box-c) this Mac's bringup.sh
#                 rented and reconciled into that file, and never brought to its end -- its open
#                 bring-up record ($BRINGUP_STATE_DIR/<BOX>-<id>.open, default ~/.cache/box-c-bringup;
#                 opened at the create, RECONCILED= appended by the reconcile, removed at the end of the
#                 sequence) names this box, this box file and that id, and is younger than
#                 BRINGUP_RECREATE_MAX_AGE_S (default 172800 = 48 h). Such a resume is then refused
#                 after the ssh wait if an engine already answers /health on the box (it may be taking
#                 box-c's traffic). Pinned by tests/bringup-recreate.test.sh.
#                 Any resume whose instance has an open record (every box-c re-create resume; another
#                 box's when this Mac rented it) is refused unless <args-file> is the one the create
#                 booted with: the record's ARGS_FILE (absolute, symlinks resolved) and, when it carries
#                 one, ARGS_SHA256 (its bytes at the create) -- a resume never re-renders the onstart,
#                 so another line would be reported while the box boots the create's (#2667, pinned by
#                 tests/restart-env-seed.test.sh section R).
# The reconcile's wait for the instance's ip/ports: an instance whose actual_status is still `loading`
# (image pull/build) or not yet set (`created`, null -- being scheduled) is coming up, not failing, so it
# is waited on for up to BRINGUP_INSTANCE_WAIT_S (default 2700 = 45 min), with its status_msg's last line
# printed every BRINGUP_INSTANCE_NOTE_S (default 30); once it is past loading (or its record is not
# readable at all) it gets BRINGUP_PORTS_GRACE_S (default 300) to publish ip/ssh/door ports before the
# run dies. Polled every BRINGUP_INSTANCE_POLL_S (default 5).
#   --dry-run     same as DRY_RUN=1 (nothing touched -- every step below just prints what it
#                 would have run).
#   --engine-args <file>
#                 the per-copy args file of a box whose file sets ENGINES (box-c's 2xTP4:
#                 args/v11-b200-box-e-tp4.args), REQUIRED there and refused on a box without ENGINES,
#                 both in preflight before anything is rented. <args-file> stays the one-engine line
#                 the create's onstart boots (the onstart knows one engine); the fork checkout then runs
#                 `restart.sh --engine all fork <file> default --tree <FORK_TREE>` -- every copy
#                 installed from <file> as sglang-args-<id>.txt and from the box's env seed (`default`:
#                 ENV_SEED, the seed stage.sh wrote each sglang-env-<id>.txt from) -- which moves the
#                 box off the onstart's one-engine line onto the copies. The boot gates and smoke.py
#                 then run per copy, on each copy's own port, log and args (lib.sh's engine_select
#                 <id>; smoke.py with ENGINE=<id>), and the resume refusal reads every copy's /health
#                 as well as the legacy port's. restart.sh renders the per-copy door from the box's
#                 existing Caddyfile. Only smoke.py runs per copy. Without ENGINES
#                 the run is unchanged: a bare
#                 `restart.sh fork - - --tree <FORK_TREE>` and the one-engine gates. Pinned by
#                 tests/bringup.test.sh.
# The graft step's own env knobs -- the inputs this script reads that the usage line above does not
# cover (everything else it reads comes from boxes/$BOX.env or lib.sh): GRAFT_SRC (box file; the
# graft's directory on THIS Mac, `~` allowed -- <HOME>/workspace/backups/grafts/<repo>@<branch>/ -- holding
# edited_tensors.safetensors + applier.json + manifest.json; empty or unset = no graft step at all,
# serve STOCK_DIR unmodified), GRAFT_SHA256 (box file; the sha256 of GRAFT_SRC's
# edited_tensors.safetensors -- required with GRAFT_SRC, checked against the Mac copy and against the
# copy's own manifest.json in preflight and against the box's copy after the transfer; the source
# section below), GRAFT_DIR (box file; where the three files are copied on the box, and what
# graft-apply.sh is run with as GRAFT_DIR), APPLY_GRAFT_PY (box file; the
# box-side path this step stages the checkout's applier to, sha256-checked, and passes in
# graft-apply.sh's argv), and APPLY_GRAFT_PY_SRC (the env override naming the local applier that gets
# staged to APPLY_GRAFT_PY -- default grafting-kit/tools/apply_graft.py two directories up from
# this tool directory, the path the "no apply_graft.py at ..." die prints; set it when this checkout's kit
# lives somewhere else), and GRAFT_BUILD_MISMATCH_OK=1 (the explicit override for a box file whose
# GRAFT_DIR and MODEL_DIR name two different builds -- preflight refuses that pair otherwise, so a run
# that means the pair exactly as it stands sets this), and GRAFT_VERIFY_ALL (box file, 0 or 1, default 0;
# passed to graft-apply.sh as VERIFY_ALL: 1 checks EVERY stock shard against applier.json and every
# output shard of MODEL_DIR against manifest.json's out_model sha256s, not just the rewritten ones --
# the whole-repack check a production rebuild runs; preflight refuses any other value).
# Captured BEFORE lib.sh is sourced: lib.sh derives DOOR_URL from the box file's APP_PORT_EXT
# unless the caller's own environment already exported one, and the reconcile step below has to
# be able to tell those two apart when it corrects APP_PORT_EXT to the new instance's port (see
# the DOOR_URL re-derivation there).
DOOR_URL_FROM_ENV=${DOOR_URL:-}
. "$(dirname "$0")/lib.sh"

OFFER_ID=""
RESUME_INSTANCE=""
ENGINE_ARGS=""
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --offer) OFFER_ID=$2; shift 2 ;;
    --instance)
      [ $# -ge 2 ] && [ -n "$2" ] || die "--instance needs an instance id (the one the earlier run's create rented)"
      RESUME_INSTANCE=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --engine-args)
      [ $# -ge 2 ] && [ -n "$2" ] || die "--engine-args needs the per-copy args file (e.g. tools/args/v11-b200-box-e-tp4.args)"
      ENGINE_ARGS=$2; shift 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- ${POSITIONAL[@]+"${POSITIONAL[@]}"}

[ $# -ge 1 ] || die "usage: bringup.sh <args-file> [--offer <offer-id> | --instance <instance-id>] [--engine-args <per-copy args-file>] [--dry-run]"
ARGS_FILE=$1

# ------------------------------------------------------------ the open bring-up record (#2661)
# One file per instance this Mac's bringup.sh rented (or took over with --instance) and has not yet
# brought to the end of its sequence: $BRINGUP_STATE_DIR/<BOX>-<id>.open (default ~/.cache/box-c-bringup,
# outside every checkout, so stage.sh never ships it and git never sees it). Opened the moment a real
# run owns an instance, appended to when the reconcile writes that instance into the box file
# (RECONCILED=), and removed at "== bring-up sequence done". Nothing is written under DRY_RUN.
#
# Its one reader is the box-c refusal below. A box-c RE-CREATE (BOX=box-c, a new box for box-c)
# rents an instance whose id the reconcile then writes into boxes/box-c.env -- and the refusal
# turned away every id that file names, so a re-create that died after its reconcile (stage.sh, the
# download gate, the engine gates) could not be resumed at all (ISSUES.md "Pending cleanup"). The record
# is what tells that id apart from a STALE one -- box-c's live instance, an id written by hand, a re-create
# finished long ago: the file names the id BECAUSE this Mac's reconcile wrote it there, for this box and
# this box file, within BRINGUP_RECREATE_MAX_AGE_S (default 172800 = 48 h; a re-create's checkpoint
# download is hours, not days), and that bring-up never finished. Losing the record (a wiped cache, a
# run on another Mac) fails safe: the resume is refused as box-c, naming what is missing. A record that
# cannot be written is a WARNING, never a die -- the box is already rented.
BRINGUP_STATE_DIR="${BRINGUP_STATE_DIR:-$HOME/.cache/box-c-bringup}"
open_record() { printf '%s/%s-%s.open' "$BRINGUP_STATE_DIR" "$BOX" "$1"; }
# open_record_get <file> <key> -- the last value of KEY= in the record ("" when absent).
open_record_get() { sed -n "s/^$2=//p" "$1" 2>/dev/null | tail -1 || true; }
# The record pins the args file the create booted with (#2667): ARGS_FILE= is its absolute, symlink-
# resolved path and ARGS_SHA256= the sha256 of its bytes, both taken when the record is opened. A resume
# never re-renders the onstart or reinstalls sglang-args.txt -- the box boots whatever the create's
# onstart wrote -- so a resume handed ANOTHER args file (or this one, edited since) would report and
# hint one serve line while the box runs another. record_args_why_not below refuses that.
args_abs_path() { python3 -I -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1" 2>/dev/null || printf '%s' "$1"; }
args_sha256() { python3 -I -c 'import hashlib, sys; sys.stdout.write(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1" 2>/dev/null || true; }
# record_args_why_not <record> -- empty when this run's ARGS_FILE is the record's (same resolved path,
# and the same bytes when the record carries ARGS_SHA256; a record opened before #2667 has no hash and
# is compared by path alone), else the reason it is not, one clause.
record_args_why_not() {
  local f="$1" rec_path rec_sha cur_path cur_sha
  rec_path=$(open_record_get "$f" ARGS_FILE)
  [ -n "$rec_path" ] || { printf 'the record %s names no ARGS_FILE, so it cannot say which args file the create booted with' "$f"; return 0; }
  cur_path=$(args_abs_path "$ARGS_FILE")
  [ "$(args_abs_path "$rec_path")" = "$cur_path" ] \
    || { printf 'the create booted with args file %s (the record %s), and this resume names %s' "$rec_path" "$f" "$cur_path"; return 0; }
  rec_sha=$(open_record_get "$f" ARGS_SHA256)
  [ -n "$rec_sha" ] || return 0
  cur_sha=$(args_sha256 "$ARGS_FILE")
  [ "$cur_sha" = "$rec_sha" ] \
    || printf 'args file %s has changed since the create booted with it (sha256 %s then, %s now; the record %s)' "$cur_path" "$rec_sha" "${cur_sha:-<unreadable>}" "$f"
  return 0
}
open_record_warn() {
  note "WARNING: could not write the open bring-up record $1 -- this run goes on, but if it stops after the reconcile on a box-c re-create, \`--instance\` will refuse the new id as box-c (boxes/box-c.env will name it) and the rest runs by hand (README.md \"The box-d layout on a re-create\"); set BRINGUP_STATE_DIR to a writable directory for the next run"
}
# open_record_open <id> -- start the record (a fresh run's create, or a first --instance takeover), or
# note a resume in the one that is already there.
open_record_open() {
  local f now how
  f=$(open_record "$1"); now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [ -f "$f" ]; then
    printf 'RESUMED=%s\n' "$now" >> "$f" 2>/dev/null || open_record_warn "$f"
    return 0
  fi
  if [ -n "$RESUME_INSTANCE" ]; then how="--instance $RESUME_INSTANCE (a takeover: no record of its create)"; else how="create from offer ${OFFER_ID:-<none>}"; fi
  { mkdir -p "$BRINGUP_STATE_DIR" \
    && printf 'BOX=%s\nINSTANCE=%s\nBOX_ENV=%s\nARGS_FILE=%s\nARGS_SHA256=%s\nOPENED=%s\nOPENED_EPOCH=%s\nOPENED_BY=%s\n' \
         "$BOX" "$1" "$BOX_ENV" "$(args_abs_path "$ARGS_FILE")" "$(args_sha256 "$ARGS_FILE")" "$now" "$(date +%s)" "$how" > "$f.tmp.$$" \
    && mv -f "$f.tmp.$$" "$f"; } 2>/dev/null || { rm -f "$f.tmp.$$" 2>/dev/null || true; open_record_warn "$f"; return 0; }
  note "open bring-up record: $f (removed when this bring-up reaches its end; it is what lets a box-c re-create be resumed with --instance after the reconcile writes the id into boxes/box-c.env)"
}
# open_record_add <id> <KEY> <value> -- one line appended to an existing record.
open_record_add() {
  local f; f=$(open_record "$1")
  [ -f "$f" ] || return 0
  printf '%s=%s\n' "$2" "$3" >> "$f" 2>/dev/null || open_record_warn "$f"
}
# box_c_recreate_why_not <id> -- empty when <id> is an open box-c re-create of THIS Mac's that the resume
# may take over, else the reason it is not, one clause.
box_c_recreate_why_not() {
  local id="$1" f age max="${BRINGUP_RECREATE_MAX_AGE_S:-172800}" opened
  f=$(open_record "$id")
  case "$max" in ''|*[!0-9]*) printf 'BRINGUP_RECREATE_MAX_AGE_S=%s is not whole seconds' "$max"; return 0 ;; esac
  [ "$BOX" = box-c ] || { printf 'this run is BOX=%s, not a re-create of box-c (BOX=box-c)' "$BOX"; return 0; }
  [ -f "$f" ] || { printf 'no open bring-up record %s -- no bringup.sh on this Mac rented %s and reconciled it into the box file, or that bring-up already reached its end' "$f" "$id"; return 0; }
  [ "$(open_record_get "$f" BOX)" = "$BOX" ] && [ "$(open_record_get "$f" INSTANCE)" = "$id" ] \
    || { printf 'the record %s is not BOX=%s INSTANCE=%s' "$f" "$BOX" "$id"; return 0; }
  [ "$(open_record_get "$f" BOX_ENV)" = "$BOX_ENV" ] \
    || { printf 'the record %s was written for box file %s, not this run'"'"'s %s' "$f" "$(open_record_get "$f" BOX_ENV)" "$BOX_ENV"; return 0; }
  [ -n "$(open_record_get "$f" RECONCILED)" ] \
    || { printf 'the record %s carries no RECONCILED line -- bringup.sh'"'"'s reconcile never wrote %s into %s, so whatever wrote it there was not this bring-up' "$f" "$id" "$BOX_ENV"; return 0; }
  opened=$(open_record_get "$f" OPENED_EPOCH)
  case "$opened" in ''|*[!0-9]*) printf 'the record %s has no readable OPENED_EPOCH' "$f"; return 0 ;; esac
  age=$(( $(date +%s) - opened ))
  [ "$age" -le "$max" ] || { printf 'the record %s was opened %ss ago, past BRINGUP_RECREATE_MAX_AGE_S=%s -- a re-create that old is not resumed blind' "$f" "$age" "$max"; return 0; }
  return 0
}

# --instance's own refusals, all made here, before preflight reads anything (#1704). Plain argument
# checks, so DRY_RUN does not soften them: a dry pass that accepts box-c's id is the one that teaches an
# operator the wrong command. box-c's live instance (<INSTANCE_ID>) is refused outright, always. An id the box-c
# box file names is refused unless it is an OPEN box-c RE-CREATE (the record above, #2661): then this run
# resumes it as any other box's bring-up resumes, plus one guard after the ssh wait -- an instance whose
# engine already answers /health is refused there, because the run would re-run stage.sh and restart.sh's
# relaunch on an engine that may already take box-c's traffic.
BOX_C_INSTANCE="<INSTANCE_ID>"   # box-c (boxes/box-c.env) -- production, never resumed, attached to or reconciled by this script
RESUME_BOX_C__RECREATE=0
if [ -n "$RESUME_INSTANCE" ]; then
  case "$RESUME_INSTANCE" in
    *[!0-9]*) die "--instance '$RESUME_INSTANCE' is not a provider instance id (digits only)" ;;
  esac
  [ -z "$OFFER_ID" ] || die "--instance $RESUME_INSTANCE and --offer $OFFER_ID together: --instance resumes a box an earlier create ALREADY rented and never creates, --offer picks the offer a NEW create rents -- pass one, never both (a resume that also created would rent a SECOND box)"
  BOX_C_FILE_INSTANCE=$(sed -n 's/^INSTANCE_ID=\([0-9][0-9]*\).*/\1/p' "$TOOLS_DIR/boxes/box-c.env" 2>/dev/null | tail -1 || true)
  if [ "$RESUME_INSTANCE" = "$BOX_C_INSTANCE" ]; then
    die "--instance $RESUME_INSTANCE is box-c (production, boxes/box-c.env) -- refusing: bringup.sh would attach a key to it, rewrite a box file with its address and run stage.sh/restart.sh against it. A bring-up never resumes box-c's live instance; box-c engine changes go through 07-operations.md's restart recipe"
  fi
  if [ -n "$BOX_C_FILE_INSTANCE" ] && [ "$RESUME_INSTANCE" = "$BOX_C_FILE_INSTANCE" ]; then
    WHY_NOT=$(box_c_recreate_why_not "$RESUME_INSTANCE")
    [ -z "$WHY_NOT" ] || die "--instance $RESUME_INSTANCE is box-c (boxes/box-c.env names it) and not an open box-c re-create this Mac can resume: $WHY_NOT -- refusing: bringup.sh would attach a key to it, rewrite a box file with its address and run stage.sh/restart.sh against it. A re-create that cannot be resumed is finished by hand (README.md \"The box-d layout on a re-create\"); box-c engine changes go through 07-operations.md's restart recipe"
    RESUME_BOX_C__RECREATE=1
    note "--instance $RESUME_INSTANCE: boxes/box-c.env names it because this Mac's bringup.sh re-created box-c on it and its reconcile wrote it there ($(open_record "$RESUME_INSTANCE"), opened $(open_record_get "$(open_record "$RESUME_INSTANCE")" OPENED)) and that bring-up never reached its end -- resuming the re-create; an engine already answering /health on it is refused after the ssh wait"
  fi
  # Any resume whose instance has an open record -- a box-c re-create always does (above), another box's
  # when this Mac's bringup.sh rented it -- continues the create that record names, so it must name the
  # create's args file, unedited (#2667; the args_abs_path comment above). A takeover with no record has
  # nothing to compare and is not checked.
  if [ -f "$(open_record "$RESUME_INSTANCE")" ]; then
    WHY_NOT=$(record_args_why_not "$(open_record "$RESUME_INSTANCE")")
    [ -z "$WHY_NOT" ] || die "--instance $RESUME_INSTANCE resumes a bring-up whose create rendered its onstart from another serve line: $WHY_NOT -- refusing: a resume never re-renders the onstart or reinstalls sglang-args.txt, so the box boots the create's line while this run would report this one. Resume with the create's args file as it was (the RESUME line the earlier run printed names it); a different line goes on afterwards, through restart.sh inside the box's own change recipe"
    note "--instance $RESUME_INSTANCE: args file $(args_abs_path "$ARGS_FILE") is the one the create booted with ($(open_record "$RESUME_INSTANCE"))"
  fi
fi

need() { [ -x "$1" ] || [ -f "$1" ] || die "not built yet: $1 (see the item that owns it)"; }

# ------------------------------------------------------------- the run's own box target
# create's result, held for the rest of this run: empty until a real create lands, and empty from
# start to finish under DRY_RUN, where nothing is rented and so nothing is fenced.
NEW_INSTANCE=""; NEW_BOX_IP=""; NEW_SSH_PORT=""; NEW_APP_PORT_EXT=""

# The attach's own outcome, declared here with the rest of the run's state because the closing summary
# reads it (#391): ATTACHED turns 1 the moment a `hostctl attach ssh` this run made succeeds, and is
# never reset -- the call is additive, so a later failed attempt does not take away a key an earlier
# one added; ATTACH_RC is the exit status of the LAST attempt (-1 = none made: DRY_RUN, or a run that
# stopped before the attach step). The attach step and the ssh wait's one re-attach are the two sites
# that write them, and neither is fatal on its own.
ATTACHED=0; ATTACH_RC=-1

# ------------------------------------------------------------------ the way back after a die (#1704)
# Once a create has rented a box (or --instance has taken one over), the instance is billed and a plain
# re-run of this script creates again -- a SECOND box. resume_cmd prints the one command that continues
# THIS bring-up instead: the same args file, the same box (BOX_ENV spelled out when it is not the
# default boxes/$BOX.env) and --instance <id> in place of any --offer.
# `resume_cmd <id>` names another id (the create-failure dies, where the operator still has to find
# it in `hostctl show instances`).
resume_cmd() {
  local id="${1:-$NEW_INSTANCE}" pre="" post=""
  [ "$BOX_ENV" = "$TOOLS_DIR/boxes/$BOX.env" ] || pre="BOX_ENV=$(printf '%q' "$BOX_ENV") "
  [ -z "$ENGINE_ARGS" ] || post=" --engine-args $(printf '%q' "$ENGINE_ARGS")"
  printf 'BOX=%s %sbash %q %q --instance %s%s' "$BOX" "$pre" "$0" "$ARGS_FILE" "$id" "$post"
}
# resume_hint: the RESUME line, printed once, on any failing exit after the run owns an instance -- every
# `die` below (die is redefined here to call it) and, through the EXIT trap, a sibling script that fails
# under lib.sh's `set -e` (stage.sh, restart.sh ...) without reaching a die of this script's own.
# Nothing under DRY_RUN, where nothing was rented.
RESUME_HINTED=0
resume_hint() {
  [ -n "$NEW_INSTANCE" ] && [ "${DRY_RUN:-0}" != 1 ] && [ "$RESUME_HINTED" = 0 ] || return 0
  RESUME_HINTED=1
  note "RESUME: instance $NEW_INSTANCE is rented and billed -- continue this bring-up with:  $(resume_cmd)   (never a plain re-run: without --instance bringup.sh creates again and rents a SECOND box)"
}
die() { note "FATAL: $*"; resume_hint; exit 1; }
# lib.sh installs its own cleanup as the EXIT trap only when the script has none; this one replaces it,
# so it carries the same cleanup.
trap 'rc=$?; [ "$rc" = 0 ] || resume_hint; rm -f "$BOX_TARGET_CACHE" "$BOX_SSH_PROXY_HELPER" 2>/dev/null || true' EXIT

# The container port the provider's `--ssh` (which offer-search.sh create passes) maps to the host port the
# box file calls SSH_PORT. It is sshd's own port inside the image -- the same in every image this
# tool set boots, and not a box parameter: the box file's SSH_PORT is the HOST side, which is the
# number the reconcile step resolves from the instance's own record.
export SSH_CONTAINER_PORT=22

# This run has just created an instance, so it knows exactly which box it is allowed to touch.
# assert_box_target is that check, named so it can also be called where a failure has to be LOUD:
# the ssh-wait loop below keeps every box_ssh's stderr in a file instead of on the terminal (which is
# what lets its last error be quoted), so it calls this first, and a wrong target then dies with its
# message visible instead of being filed away as one more connection error.
assert_box_target() {
  [ -n "$NEW_BOX_IP" ] || return 0     # nothing created this run: nothing to fence (DRY_RUN)
  if [ "$BOX_IP" != "$NEW_BOX_IP" ] || [ "$SSH_PORT" != "$NEW_SSH_PORT" ]; then
    die "wrong box: this run owns instance $NEW_INSTANCE at $NEW_BOX_IP:$NEW_SSH_PORT, but $BOX_ENV still resolves to $BOX_IP:$SSH_PORT -- refusing to run anything against it; set BOX_IP=$NEW_BOX_IP SSH_PORT=$NEW_SSH_PORT APP_PORT_EXT=$NEW_APP_PORT_EXT INSTANCE_ID=$NEW_INSTANCE in $BOX_ENV, then resume with: $(resume_cmd)"
  fi
}

# box_ssh, redefined AFTER lib.sh's own on purpose: in bash a later function definition shadows an
# earlier one for every call below it, so this wrapper fences every ssh this script makes without
# lib.sh -- shared by the whole tool set, most of which creates nothing and has no instance to
# check a target against -- having to learn a bringup-specific rule. This is the guard the incident
# needed: bringup created box-c and then ran stage.sh / restart.sh against box-g, because
# box_ssh read the box file's stale BOX_IP/SSH_PORT.
#
# The override adds the fence; it must not DROP anything lib.sh's own box_ssh carries, and what it
# would otherwise drop is the transport. ${BOX_SSH_PROXY_ARG:+-o "$BOX_SSH_PROXY_ARG"} is lib.sh's
# session-proxy argument, set at source time above (before this definition) and carried by lib.sh's
# box_ssh/box_scp/fork_ssh: in a sandboxed session the only route to the box is through the
# session HTTP proxy, and a bare ssh here fails at once with "connect to host ... Operation not
# permitted". This function replaces lib.sh's box_ssh wholesale for the whole run, so without the
# option the ssh-wait loop below can never succeed -- it would spin to its 1800s ceiling and then die
# "ssh never came up after 1800s" on a box that is up and serving. The spelling is lib.sh's own
# (bash-3.2-safe: the two words `-o` and the ProxyCommand value are added only when the variable is
# non-empty; quoting the inner expansion keeps the space-bearing value one argv element).
box_ssh() {
  assert_box_target
  ssh ${BOX_SSH_PROXY_ARG:+-o "$BOX_SSH_PROXY_ARG"} -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" -i "$SSH_KEY_PATH" "root@$BOX_IP" "$@"
}

# ------------------------------------------------------- the ssh key the box file names (ledger #314)
# `hostctl attach ssh` is the provider's instance-scoped key attach (POST /api/v0/instances/<id>/ssh): it adds
# a key to an instance that already exists -- a RUNNING one included, with its GPUs and an in-flight
# checkpoint download untouched -- which is what makes it the answer here rather than an
# authorized_keys edit over ssh (the box may be unreachable, which is the failure this fixes) or a
# create-time `--ssh` (this script's create is offer-search.sh's, and the key it injects is the
# account's). It is addressed by the caller's own instance id, never by $INSTANCE_ID: attaching a
# key is a write to a live instance, and $INSTANCE_ID is the box file's value, which may describe
# the previous box -- the same rule the reconcile step below follows.
#
# The key is ${SSH_KEY_PATH}.pub, passed as a PATH: the CLI reads a public-key file itself (its own
# get_ssh_key helper tests `os.path.exists(argstr)` and reads the file when it is there, then refuses
# anything that is not a public key), and its usage epilog lists BOTH spellings -- `hostctl attach ssh
# 12371 $(cat ~/.ssh/id_rsa.pub)` and `hostctl attach ssh 12371 ~/.ssh/id_rsa.pub`. So this function
# used to be wrong twice over (#391): it sent the CONTENTS while the comment here claimed the CLI
# takes content and "never" a path, which is not true. The path is the form to use: it is what #314's
# fix spelled and what boxes/box-c.env:7 records for the LIVE attach
# (`hostctl attach ssh $INSTANCE_ID ~/.ssh/<SSH_KEY>.pub`), and it is what the failure
# remedies below can hand an operator verbatim -- no `"$(cat ...)"` nesting for a shell to re-split.
# It is not a weaker check: a missing, private or non-key file is refused by get_ssh_key before any
# API call, not attached. (07-operations.md's remedy spells the content form; both are accepted, and
# this function passes the path.) Prints the CLI's own output for the caller to warn with; prints no
# key material -- a public key is not a secret, and the path is not the key.
# This spelling is pinned by tests/bringup-ssh-key-attach.test.sh (case A's argv pair, B's and C's
# paste-ready lines): a re-spelling here has to move those needles with it.
attach_ssh_key() {
  hostctl attach ssh "$1" "${SSH_KEY_PATH}.pub" 2>&1
}

# ------------------------------------------------------------ the graft step's build pair (ledger #314)
# The graft step's paths come out of the box file: STOCK_DIR + GRAFT_DIR are what it applies, MODEL_DIR
# is what it rebuilds -- so MODEL_DIR comes out holding GRAFT_DIR's graft applied to STOCK_DIR. The box
# file names both ends with the build in the directory's own name: <MODEL_PATH>/
# graft-1019-c20 (c20's graft, the fallback behind box-c's STOCK rollback target; box-c serves the L3a
# repack) against repack-1019-c20 (the repack built from it);
# attempt 2's graft-1017-l25 against repack-1017-l25; attempt 1's graft-1017
# against repack-1017. Two paths naming the SAME suffix are a coherent pair; two naming different ones
# are two different builds.
#
# That is the check below, and it runs in PREFLIGHT: before an offer is searched or a box rented, this
# run is told which build it is about to build, and a pair that names two builds stops there. The
# failure it was written for was box-c's own box file at attempt 2's cutover -- GRAFT_DIR was attempt 2's
# graft (graft-1017-l25) while MODEL_DIR still named v3 (repack-woB) -- where
# a create would have rebuilt the v3 path from attempt 2's graft and the box would then have served l25
# bytes under the v3 name, silently. The same shape recurs on any serving flip that moves MODEL_DIR
# alone: a swap's fallback that points MODEL_DIR back at the build serving before the swap while
# GRAFT_DIR still names the swapped-in graft is refused here on a create until GRAFT_DIR (and
# GRAFT_SRC/GRAFT_SHA256) name that build's graft too. graft-apply.sh's own guard (a non-empty MODEL_DIR carrying no .graft-apply-scratch.json
# marker is refused) cannot catch that on a create: MODEL_DIR does not exist yet, so there is nothing
# for it to refuse -- and on a box whose MODEL_DIR is a previous graft-apply output (the cutover
# repair), FORCE=1 legitimately reuses it, so that guard is not a pairing check at all.
#
# What this check cannot see, said plainly: it compares the two NAMES the box file gives, so a MODEL_DIR
# built from another graft and then renamed reads as coherent here. Which BYTES land in GRAFT_DIR is the
# source check's business (check_graft_source below: GRAFT_SHA256 against the Mac copy and its own
# manifest.json), not this one's.
graft_pair_arm() {
  local base="${1%/}"
  base="${base##*/}"
  case "$base" in
    graft-*)  printf '%s' "${base#graft-}" ;;
    repack-*) printf '%s' "${base#repack-}" ;;
  esac
  return 0
}

# check_graft_build_pair -- no arguments: it reads the same box-file values the graft step's
# GRAFT_DIR_REMOTE resolves, including that step's own fallback for GRAFT_DIR (the generic box layout,
# spelled again inside the function so this can run in preflight, long before the graft section is
# reached; if that fallback ever moves, this is the second place to change).
check_graft_build_pair() {
  local graft_dir="${GRAFT_DIR:-<MODEL_PATH>/graft}" model_dir="${MODEL_DIR:-}"
  local graft_arm model_arm want_model
  [ -n "${GRAFT_SRC:-}" ] || return 0     # no graft step at all: nothing is paired
  [ -n "$model_dir" ] || return 0         # an absent MODEL_DIR is build-onstart.sh's own die, and it runs before any create
  graft_arm=$(graft_pair_arm "$graft_dir")
  model_arm=$(graft_pair_arm "$model_dir")
  if [ -z "$graft_arm" ] || [ -z "$model_arm" ]; then
    note "graft pair: GRAFT_DIR=$graft_dir / MODEL_DIR=$model_dir -- one of the two does not spell its build in its name (graft-<build> / repack-<build>), so this run cannot tell whether the graft about to be applied is the one $model_dir was built from. Applying it anyway: graft-apply.sh rebuilds MODEL_DIR from STOCK_DIR + GRAFT_DIR, so what lands there is whatever graft GRAFT_DIR holds applied to whatever stock STOCK_DIR holds -- if those are not the pair you mean, stop and fix the box file's names first (graft-apply.sh's header: a graft applies only to the stock it was baked against)."
    return 0
  fi
  if [ "$graft_arm" = "$model_arm" ]; then
    note "graft pair OK: GRAFT_DIR=$graft_dir (build $graft_arm) will rebuild MODEL_DIR=$model_dir (build $model_arm) from STOCK_DIR=${STOCK_DIR:-$model_dir}; MODEL_DIR is what the box then serves"
    return 0
  fi
  want_model="$(dirname "$model_dir")/repack-$graft_arm"
  if [ "${GRAFT_BUILD_MISMATCH_OK:-0}" = 1 ]; then
    note "WARNING: GRAFT_BUILD_MISMATCH_OK=1 -- applying GRAFT_DIR=$graft_dir (build $graft_arm) into MODEL_DIR=$model_dir, which names build $model_arm: MODEL_DIR is rebuilt from STOCK_DIR + GRAFT_DIR, so the box will serve $graft_arm's bytes at the $model_arm path and under $model_arm's name"
    return 0
  fi
  if [ "${DRY_RUN:-0}" = 1 ]; then
    note "WARNING: $BOX_ENV's GRAFT_DIR and MODEL_DIR name two different builds -- GRAFT_DIR=$graft_dir (build $graft_arm), MODEL_DIR=$model_dir (build $model_arm) -- DRY_RUN: continuing anyway (a real run dies here: it would rebuild MODEL_DIR from GRAFT_DIR's graft, leaving $graft_arm's bytes at the $model_arm path for the box to serve; set GRAFT_BUILD_MISMATCH_OK=1 to mean that pair deliberately, or flip the box file's SERVE_BUILD/MODEL_DIR pair to $graft_arm)"
    return 0
  fi
  die "$BOX_ENV's GRAFT_DIR and MODEL_DIR name two different builds -- refusing to rent a box that would build one and serve it as the other:
  GRAFT_DIR=$graft_dir   (build $graft_arm)   <- what GRAFT_SRC is copied into and graft-apply.sh would apply
  MODEL_DIR=$model_dir   (build $model_arm)   <- the path it would rebuild, and the one the box would serve
  STOCK_DIR=${STOCK_DIR:-$model_dir}
graft-apply.sh applies GRAFT_DIR onto STOCK_DIR into MODEL_DIR, so this create would leave $graft_arm's bytes at $model_dir and the engine serving them as $model_arm. Pick one:
  (1) build the graft GRAFT_DIR names -- flip the box file's serving pair to it (SERVE_BUILD and MODEL_DIR, the two lines edited together; MODEL_DIR is spelled out because the readers that do not source the box file cannot expand a reference), then re-run:
        SERVE_BUILD=$graft_arm
        MODEL_DIR=$want_model
  (2) rebuild $model_dir instead -- point GRAFT_DIR (and GRAFT_SRC/GRAFT_SHA256, the Mac copy that gets copied into it) at the graft that $model_arm was baked from, then re-run;
  (3) mean the pair exactly as it stands -- re-run with GRAFT_BUILD_MISMATCH_OK=1."
}

# ------------------------------------------------------- the retired Hugging Face graft keys (#2430)
# Our grafts never come from Hugging Face: our internal artifact policy -- campaign and box-c
# boxes receive our files over ssh from the Mac, never by download from a third party. The graft step
# used to `hf download` a private repo named by GRAFT_REPO at GRAFT_REV; it now copies the Mac's backup
# (GRAFT_SRC, pinned by GRAFT_SHA256). There is no alias for the old keys: a box file that still
# carries either line -- even an empty one -- is a box file nobody has rewritten for the new step, and
# reading it as "no graft" or guessing a Mac path from a repo slug would build something other than
# what its author meant. So it is refused here, in PREFLIGHT, under DRY_RUN too (a dry pass that reads
# green on a box file a real run cannot use is the worse failure), naming every offending line.
check_retired_graft_keys() {
  local hits
  hits=$(grep -nE '^[[:space:]]*(export[[:space:]]+)?GRAFT_(REPO|REV)=' "$BOX_ENV" 2>/dev/null | cut -c1-160 || true)
  [ -z "$hits" ] && return 0
  die "$BOX_ENV still carries the retired GRAFT_REPO/GRAFT_REV keys -- refusing: our grafts are never fetched from Hugging Face (our internal artifact policy), the graft step copies the Mac's backup over ssh instead, and these keys have no meaning left. The lines:
$hits
Replace them with the Mac source and its pin, keeping GRAFT_DIR (the box path) as it is:
  GRAFT_SRC="<HOME>/workspace/backups/grafts/<repo>@<branch>"      # the graft's directory on this Mac: edited_tensors.safetensors + applier.json + manifest.json
  GRAFT_SHA256="<sha256" of that directory's edited_tensors.safetensors>
-- or, for a box that serves STOCK_DIR unmodified, delete both lines (and leave GRAFT_SRC unset)."
}

# ------------------------------------------------------------ the graft's Mac source and its pin (#2430)
# The graft step copies GRAFT_SRC -- the graft's backup directory on THIS Mac,
# <HOME>/workspace/backups/grafts/<repo>@<branch>/ -- onto the box over box_scp, and GRAFT_SHA256 says which
# bytes that copy must be. The pin is what the old revision pin (#470) bought, now without a third
# party: these three filenames have already carried several different bakes (attempt 1's
# edited_tensors.safetensors was replaced in place by attempt 2's; c20's sits beside both), and
# apply_graft.py verifies a graft only against the manifest that ARRIVES WITH IT, so a wrong graft
# directory applies cleanly and the box serves it under the GRAFT_DIR/MODEL_DIR names it landed on.
# Checked HERE, before an offer is searched or anything rented:
#   * GRAFT_SHA256 is a 64-hex sha256 (it is required: an unpinned Mac copy is refused, not warned);
#   * GRAFT_SRC is a directory on this Mac holding the three files;
#   * its edited_tensors.safetensors hashes to GRAFT_SHA256;
#   * its manifest.json names that same graft (graft.edited_tensors_sha256 == GRAFT_SHA256), which is
#     what ties the manifest -- and the applier it was written with -- to the pin.
# The Mac copy's applier.json and manifest.json shas are taken here too (GRAFT_SRC_SHA_*), and the graft
# step compares the box's copies against these exact values after the transfer, so a Mac file that moves
# between this preflight and that step is a mismatch, never a silent new graft. Under DRY_RUN a failed
# check warns instead of dying, like every other preflight check.
GRAFT_SRC_PATH=""; GRAFT_SRC_SHA_EDITED=""; GRAFT_SRC_SHA_APPLIER=""; GRAFT_SRC_SHA_MANIFEST=""
graft_source_fail() {
  if [ "${DRY_RUN:-0}" = 1 ]; then
    note "WARNING: $* -- DRY_RUN: continuing anyway (a real run dies here, before anything is rented)"
    return 0
  fi
  die "$*"
}
check_graft_source() {
  local out
  if [ -z "${GRAFT_SRC:-}" ]; then
    [ -z "${GRAFT_SHA256:-}" ] || graft_source_fail "$BOX_ENV sets GRAFT_SHA256=$GRAFT_SHA256 but no GRAFT_SRC -- a pin with no Mac source to copy from: set GRAFT_SRC=<the graft's directory on this Mac>, or delete GRAFT_SHA256 for a box that serves STOCK_DIR unmodified"
    return 0
  fi
  GRAFT_SRC_PATH="${GRAFT_SRC/#\~/$HOME}"
  case "${GRAFT_SHA256:-}" in
    '')
      graft_source_fail "$BOX_ENV's GRAFT_SRC=$GRAFT_SRC names no GRAFT_SHA256 -- the graft's bytes must be pinned: GRAFT_SHA256=<sha256 of $GRAFT_SRC_PATH/edited_tensors.safetensors> in $BOX_ENV"
      return 0 ;;
    *[!0-9a-f]*)
      graft_source_fail "GRAFT_SHA256='$GRAFT_SHA256' in $BOX_ENV is not a lowercase hex sha256 -- it is the sha256 of the graft's edited_tensors.safetensors, 64 hex digits"
      return 0 ;;
  esac
  [ "${#GRAFT_SHA256}" = 64 ] || { graft_source_fail "GRAFT_SHA256='$GRAFT_SHA256' in $BOX_ENV is ${#GRAFT_SHA256} hex digits, not 64 -- the full sha256 of the graft's edited_tensors.safetensors, never an abbreviation"; return 0; }
  if [ ! -d "$GRAFT_SRC_PATH" ]; then
    graft_source_fail "GRAFT_SRC=$GRAFT_SRC in $BOX_ENV is not a directory on this Mac ($GRAFT_SRC_PATH) -- the graft step copies the graft from there; the Mac's graft backups are <HOME>/workspace/backups/grafts/<repo>@<branch>/"
    return 0
  fi
  # One python3 pass: each file's presence and sha256, plus the manifest's own graft sha. Printed as
  # `key value` lines; a missing file prints `missing <name>`.
  out=$(python3 -c '
import hashlib, json, os, sys
d = sys.argv[1]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
for key, name in (("edited", "edited_tensors.safetensors"), ("applier", "applier.json"), ("manifest", "manifest.json")):
    p = os.path.join(d, name)
    print(("%s %s" % (key, sha(p))) if os.path.isfile(p) else ("missing %s" % name))
try:
    m = json.load(open(os.path.join(d, "manifest.json")))
    print("manifest_graft %s" % ((m.get("graft") or {}).get("edited_tensors_sha256") or "-"))
except Exception as e:
    print("manifest_graft unreadable:%s" % type(e).__name__)
' "$GRAFT_SRC_PATH" 2>&1) || { graft_source_fail "could not hash the graft at $GRAFT_SRC_PATH: $out"; return 0; }
  local missing manifest_graft
  missing=$(printf '%s\n' "$out" | sed -n 's/^missing //p' | tr '\n' ' ')
  missing=${missing% }
  if [ -n "$missing" ]; then
    graft_source_fail "GRAFT_SRC=$GRAFT_SRC_PATH lacks $missing -- a graft is exactly edited_tensors.safetensors + applier.json + manifest.json (apply_graft.py's load_graft reads those three), and the step copies all three"
    return 0
  fi
  GRAFT_SRC_SHA_EDITED=$(printf '%s\n' "$out" | sed -n 's/^edited //p')
  GRAFT_SRC_SHA_APPLIER=$(printf '%s\n' "$out" | sed -n 's/^applier //p')
  GRAFT_SRC_SHA_MANIFEST=$(printf '%s\n' "$out" | sed -n 's/^manifest //p')
  manifest_graft=$(printf '%s\n' "$out" | sed -n 's/^manifest_graft //p')
  if [ "$GRAFT_SRC_SHA_EDITED" != "$GRAFT_SHA256" ]; then
    graft_source_fail "$GRAFT_SRC_PATH/edited_tensors.safetensors hashes to $GRAFT_SRC_SHA_EDITED, not the box file's GRAFT_SHA256=$GRAFT_SHA256 -- refusing to copy a graft that is not the one $BOX_ENV pins; point GRAFT_SRC at the directory that holds it (<HOME>/workspace/backups/grafts/GRAFTS.tsv lists each backup's sha256), or fix the pin"
    return 0
  fi
  if [ "$manifest_graft" != "$GRAFT_SHA256" ]; then
    graft_source_fail "$GRAFT_SRC_PATH/manifest.json names graft.edited_tensors_sha256=$manifest_graft, not GRAFT_SHA256=$GRAFT_SHA256 -- that manifest (and the applier.json beside it) describes another bake, so the three files are not one graft; refusing to copy them"
    return 0
  fi
  note "graft source OK: GRAFT_SRC=$GRAFT_SRC_PATH -- edited_tensors.safetensors sha256 $GRAFT_SHA256 == GRAFT_SHA256 == its manifest.json's graft.edited_tensors_sha256; applier.json $GRAFT_SRC_SHA_APPLIER, manifest.json $GRAFT_SRC_SHA_MANIFEST. The graft step copies these three into ${GRAFT_DIR:-<MODEL_PATH>/graft} on $BOX over ssh and re-hashes them there"
}

note "== preflight"
# The retired Hugging Face graft keys first: a box file still written for the old download step is
# refused before anything else is read, dry run or not (#2430; the function's own comment above).
check_retired_graft_keys
command -v hostctl >/dev/null 2>&1 || die "hostctl CLI not on PATH"
run hostctl show user >/dev/null || die "hostctl CLI not authenticated (hostctl set api-key ...)"
[ -f "$SSH_KEY_PATH" ] || die "no ssh private key at $SSH_KEY_PATH"
[ -f "${SSH_KEY_PATH}.pub" ] || die "no ssh public key at ${SSH_KEY_PATH}.pub"
[ -f "$ENV_FILE_PATH" ] || die "no env file at $ENV_FILE_PATH"
for k in VLLM_API_KEY; do
  v=$(secret "$k" || true)   # `|| true`: a missing key must reach the die below, not abort here under -e/pipefail
  [ -n "$v" ] || die "$ENV_FILE_PATH has no $k= line"
done
HF_TOKEN_VALUE=$(secret HF_TOKEN || true)
if [ -z "$HF_TOKEN_VALUE" ]; then
  if [ "${DRY_RUN:-0}" = 1 ]; then
    note "WARNING: $ENV_FILE_PATH has no HF_TOKEN= line -- DRY_RUN: continuing anyway (a real run dies here: it writes the token into the new box's /root/.hf.env for the onstart's download of the PUBLIC upstream checkpoint -- the token's only use; our graft comes from this Mac, never from Hugging Face -- and build-onstart.sh never renders it)"
  else
    die "$ENV_FILE_PATH has no HF_TOKEN= line"
  fi
fi
[ -f "$ARGS_FILE" ] || die "no such args file: $ARGS_FILE"
# A box file with ENGINES (box-c's 2xTP4) relaunches every copy at the fork checkout, and a brand-new box
# has no sglang-args-<id>.txt to keep: the per-copy args file is named up front, refused HERE, before
# anything is rented. On a box without ENGINES it would be silently unused, so it is refused there too.
engines_check
if [ -n "$ENGINES_IDS" ]; then
  [ -n "$ENGINE_ARGS" ] || die "$BOX_ENV sets ENGINES=\"$ENGINES\" (one engine per copy: $ENGINES_IDS) -- pass --engine-args <per-copy args file> (box-c: tools/args/v11-b200-box-e-tp4.args): the fork checkout installs it on every copy (restart.sh --engine all), since a new box has no sglang-args-<id>.txt to keep. <args-file> ($ARGS_FILE) stays the one-engine line the onstart boots"
  [ -f "$ENGINE_ARGS" ] || die "no such per-copy args file: $ENGINE_ARGS (--engine-args)"
else
  [ -z "$ENGINE_ARGS" ] || die "--engine-args $ENGINE_ARGS: $BOX_ENV sets no ENGINES -- this box runs one engine, on <args-file> ($ARGS_FILE); drop --engine-args"
fi
# The graft step's build pair, checked HERE so a box file whose GRAFT_DIR and MODEL_DIR name two
# different builds is refused before an offer is searched or anything is rented: a create then is told
# which build it is about to build, instead of silently landing one build's graft in another's path
# (#314). Under DRY_RUN it is a warning, like every other preflight check.
check_graft_build_pair
# ... and the graft's Mac source and pin beside it (#2430): the pair above says WHICH build this run
# means, GRAFT_SHA256 says which BYTES of it land in GRAFT_DIR -- a wrong directory on the Mac and the
# pair is still coherent while the graft is not the one either name means. Also checked before anything
# is rented (the Mac copy is hashed here, once), and also only a warning under DRY_RUN.
check_graft_source
case "${GRAFT_VERIFY_ALL:-0}" in
  0|1) ;;
  *) die "GRAFT_VERIFY_ALL='$GRAFT_VERIFY_ALL' in $BOX_ENV -- 0 (graft-apply.sh verifies the shards it rewrites) or 1 (every stock and output shard against the graft's applier.json/manifest.json)" ;;
esac
note "preflight OK"

note "== select an offer"
OFFER_SEARCH="$TOOLS_DIR/offer-search.sh"
need "$OFFER_SEARCH"
if [ -n "$RESUME_INSTANCE" ]; then
  note "SKIP -- --instance $RESUME_INSTANCE: resuming a box an earlier create already rented, so no offer is selected and nothing is created"
elif [ -z "$OFFER_ID" ]; then
  OFFER_ID=$(run bash "$OFFER_SEARCH" search --pick-best)
  if [ "${DRY_RUN:-0}" != 1 ] && [ -z "$OFFER_ID" ]; then
    die "offer-search.sh search --pick-best returned no offer id (no matching offers?) -- pass --offer <id> yourself"
  fi
  note "picked offer ${OFFER_ID:-<none -- DRY_RUN>} (cheapest verified, rentable offer of this box's own GPU shape via offer-search.sh; pass --offer <id> to pick one yourself)"
else
  note "using offer $OFFER_ID (from --offer)"
fi

note "== build-onstart.sh (render the onstart for $ARGS_FILE)"
BUILD_ONSTART="$TOOLS_DIR/build-onstart.sh"
if [ -n "$RESUME_INSTANCE" ]; then
  # The onstart is create-frozen: the instance being resumed booted with the fragment its own create
  # carried, and a render here would reach nothing.
  ONSTART_CMD=""
  note "SKIP -- --instance $RESUME_INSTANCE: the instance already boots the onstart its own create rendered"
else
  need "$BUILD_ONSTART"
  export HF_TOKEN="$HF_TOKEN_VALUE"
  ONSTART_CMD=$(run bash "$BUILD_ONSTART" "$ARGS_FILE")
fi

# check_resume_instance <id> -- the instance an --instance run is about to take over must be THIS box
# file's: its own record's label is the label offer-search.sh's create gives this box (INSTANCE_LABEL, else
# ${BOX}-dsv41-<lowercased GPU_NAME>, offer_label's rule, with offer-search.sh's B300 fallback for an
# unset GPU_NAME), and its gpu_name (case, spaces and underscores ignored: the box file's search
# spelling H200_NVL is the provider's "H200 NVL") / num_gpus are the box file's shape. Anything else is another
# box's instance -- possibly a live one -- and the steps after this would attach a key to it, write its
# address into this box file and run stage.sh/restart.sh on it. A record that cannot be read refuses
# too: an unverifiable instance is not resumed.
check_resume_instance() {
  local id="$1" raw verdict want_label want_gpu
  want_gpu="${GPU_NAME:-B300}"
  if [ -n "${INSTANCE_LABEL:-}" ]; then
    want_label="$INSTANCE_LABEL"
  else
    want_label="$BOX-dsv41-$(printf '%s' "$want_gpu" | tr '[:upper:]' '[:lower:]')"
  fi
  raw=$(hostctl show instance "$id" --raw 2>&1) || die "--instance $id: \`hostctl show instance $id --raw\` failed ($(printf '%s' "$raw" | tr '\n' ' ' | cut -c1-300)) -- refusing to resume an instance whose record cannot be read"
  verdict=$(printf '%s' "$raw" | WANT_ID="$id" WANT_LABEL="$want_label" WANT_GPU="$want_gpu" WANT_NUM="${NUM_GPUS:-}" python3 -c '
import json, os, sys
try:
    d = json.loads(sys.stdin.read(), strict=False)
except ValueError:
    print("its record is not JSON"); sys.exit(0)
if isinstance(d, list) and len(d) == 1:
    d = d[0]
if not isinstance(d, dict):
    print("its record is not one instance"); sys.exit(0)
bad = []
rid = d.get("id")
if rid not in (None, "") and str(rid) != os.environ["WANT_ID"]:
    bad.append("the record is instance %s, not %s" % (rid, os.environ["WANT_ID"]))
label = d.get("label") or ""
if label != os.environ["WANT_LABEL"]:
    bad.append("its label is %r, this box file creates %r" % (label, os.environ["WANT_LABEL"]))
# The box file spells the GPU as hostctl search does (H200_NVL); the record says "H200 NVL".
norm = lambda s: "".join(str(s).replace("_", " ").split()).lower()
gpu = d.get("gpu_name") or ""
if norm(gpu) != norm(os.environ["WANT_GPU"]):
    bad.append("its gpu_name is %r, this box file rents %r" % (gpu, os.environ["WANT_GPU"]))
num, want_num = d.get("num_gpus"), os.environ.get("WANT_NUM", "")
if want_num and num not in (None, "") and str(num) != want_num:
    bad.append("it has %s GPUs, this box file rents %s" % (num, want_num))
print("; ".join(bad) if bad else "OK")
' 2>&1 || true)
  [ "$verdict" = OK ] || die "--instance $id is not this box's instance -- $verdict -- refusing to resume it: bringup.sh would attach ${SSH_KEY_PATH}.pub to it, write its address into $BOX_ENV and run stage.sh/restart.sh against it. Check the id in \`hostctl show instances\` (the instance an earlier run of this box file created carries label $want_label)"
  note "instance $id is this box's: label $want_label, gpu ${want_gpu}${NUM_GPUS:+ x $NUM_GPUS} (read off \`hostctl show instance $id --raw\`)"
}

note "== offer-search.sh create"
if [ -n "$RESUME_INSTANCE" ]; then
  if [ "${DRY_RUN:-0}" = 1 ]; then
    NEW_INSTANCE=""
    note "DRY_RUN: --instance $RESUME_INSTANCE -- no create; a real run checks \`hostctl show instance $RESUME_INSTANCE --raw\` (label must be ${INSTANCE_LABEL:-$BOX-dsv41-$(printf '%s' "${GPU_NAME:-B300}" | tr '[:upper:]' '[:lower:]')}, gpu_name ${GPU_NAME:-B300}${NUM_GPUS:+ x $NUM_GPUS}), then attaches ${SSH_KEY_PATH}.pub to it, waits for its ip/ports (through any image load) and reconciles $BOX_ENV with them, then runs every step below against it"
  else
    check_resume_instance "$RESUME_INSTANCE"
    NEW_INSTANCE=$RESUME_INSTANCE
    note "resuming instance $NEW_INSTANCE (--instance; nothing created) -- resolving its ip/ports from the provider API, never from $BOX_ENV"
  fi
elif [ "${DRY_RUN:-0}" = 1 ]; then
  run env ONSTART_CMD="$ONSTART_CMD" bash "$OFFER_SEARCH" create "$OFFER_ID"
  NEW_INSTANCE=""
  note "DRY_RUN: no instance created, so nothing is reconciled into $BOX_ENV and no box_ssh is fenced (a real run resolves the new instance's ip/ports here, before it ssh's anywhere)"
else
  # Captured rather than streamed, because the id in this output is the only thing that says which
  # box the rest of the run belongs to. It is echoed below -- success or failure -- so a real run's
  # stdout still carries the create output it always did, and a FAILED create leaves the operator
  # the "Started. {...}" line that may be the only witness of an instance this run will not
  # reconcile. A failed create is therefore never reported as "nothing was created": the provider can accept
  # the offer and fail the call after it, and this script's create is unconditional, so the next
  # thing an operator does on a wrong guess is rent a SECOND box.
  CREATE_FAILED=0
  CREATE_OUT=$(env ONSTART_CMD="$ONSTART_CMD" bash "$OFFER_SEARCH" create "$OFFER_ID") || CREATE_FAILED=1
  [ -z "$CREATE_OUT" ] || printf '%s\n' "$CREATE_OUT"
  if [ "$CREATE_FAILED" = 1 ]; then
    die "offer-search.sh create for offer ${OFFER_ID:-<none>} failed (its own message and output are above) -- DO NOT READ THAT AS NOTHING WAS RENTED: an offer can be accepted before the call fails, and an instance that exists is billed from that moment. Check \`hostctl show instances\` for an instance created from offer ${OFFER_ID:-<none>} BEFORE re-running (this script's create is unconditional, so a re-run rents a SECOND box): if one is there, take its id, resume THAT instance -- never re-run without --instance -- with: $(resume_cmd '<its id>')"
  fi
  # The id is the LAST line of create's stdout that is nothing but digits: offer-search.sh's own
  # contract is that the bare id is its final line, and taking the last all-digits line rather than
  # literally the last line means nothing printed after it can hide it. `|| true` because "no
  # all-digit line at all" is what the case below refuses, not something to abort on under
  # -e/-o pipefail.
  NEW_INSTANCE=$(printf '%s\n' "$CREATE_OUT" | grep -E '^[0-9]+$' | tail -1 || true)
  case "$NEW_INSTANCE" in
    ''|*[!0-9]*)
      die "offer-search.sh create printed no instance id (its last line was '$(printf '%s\n' "$CREATE_OUT" | tail -1)') -- refusing to continue: without the id this run cannot tell which box the next step would ssh to, AND the create may well have rented an instance that is now billed. Run \`hostctl show instances\` before re-running anything (a re-run rents a SECOND box): if an instance created from offer ${OFFER_ID:-<none>} is there, take its id and resume THAT instance with: $(resume_cmd '<its id>')" ;;
  esac
  note "created instance $NEW_INSTANCE -- resolving its ip/ports from the provider API, never from $BOX_ENV (a stale box file here is what pointed the box-g incident's bring-up at a live production box)"
fi
# The run owns an instance from here on (empty under DRY_RUN): open its bring-up record, or note the
# resume in the one an earlier run of this bring-up opened (#2661; the record's section above).
[ -z "$NEW_INSTANCE" ] || open_record_open "$NEW_INSTANCE"

note "== attach ${SSH_KEY_PATH}.pub to the instance this run created"
# Before the reconcile, and well before the ssh wait: the key has to be on the instance by the time the
# first ssh is made, and an operator who stops at a refused reconcile (BOX_WRITE=0) is about to ssh to
# this box by hand. It is a WARNING, never a die -- the box is rented already and every remaining step
# of this run still runs against it, so a die here would send the operator to a re-run that rents a
# SECOND box; the ssh wait re-runs this same attach if the box refuses the key, and diagnoses the
# refusal if it persists. The outcome is recorded in ATTACHED/ATTACH_RC (declared with the run's box
# target above) rather than returned: the run continues either way, and the closing summary reports
# what this attach actually did (#391).
if [ "${DRY_RUN:-0}" != 1 ]; then
  ATTACH_OUT=""
  if ATTACH_OUT=$(attach_ssh_key "$NEW_INSTANCE"); then
    ATTACHED=1; ATTACH_RC=0
    note "attached ${SSH_KEY_PATH}.pub to instance $NEW_INSTANCE (hostctl attach ssh) -- the key every box_ssh in this run authenticates with"
  else
    ATTACH_RC=$?
    note "WARNING: could not attach ${SSH_KEY_PATH}.pub to instance $NEW_INSTANCE (hostctl attach ssh exited $ATTACH_RC): $(printf '%s' "$ATTACH_OUT" | tr '\n' ' ') -- the box may still accept this key if the create's own onstart rendered it (a create from THIS box file does inject it), but if it does not, no ssh in this run can reach the box at all. Attach it by hand (hostctl attach ssh $NEW_INSTANCE \"${SSH_KEY_PATH}.pub\"), then continue from the ssh wait -- do NOT re-run bringup.sh, whose create rents a SECOND box; if this run stops, resume it with: $(resume_cmd)"
  fi
else
  note "DRY_RUN: would attach ${SSH_KEY_PATH}.pub to the instance this run created (hostctl attach ssh <instance-id> \"${SSH_KEY_PATH}.pub\"), before the ssh wait -- the only key a sandboxed ssh can use, and a create/re-create is not guaranteed to have carried it"
fi

note "== reconcile $BOX_ENV with the instance this run created"
if [ "${DRY_RUN:-0}" != 1 ]; then
  # The instance's OWN record, addressed by the id just created -- deliberately not
  # `offer-search.sh show`, which reads $INSTANCE_ID out of the box file and would therefore
  # answer about the previous box, i.e. commit the very stale read this step exists to stop.
  #
  # The wait has two clocks (#1704). A fresh instance spends minutes in actual_status `loading` -- the
  # image pull or build (status_msg "Step 7/8 ...") -- with `ports: null`, and that is the box coming
  # up, not failing: the old flat 300s ceiling died there (the provider 56623298, 2026-10-18, still at Step 7/8)
  # and told the operator to re-run, i.e. to rent a second box. So an instance that is loading (or not
  # yet scheduled: actual_status null/`created`) is waited on up to BRINGUP_INSTANCE_WAIT_S in total,
  # its status_msg's last line noted every BRINGUP_INSTANCE_NOTE_S; one that is past loading, or whose
  # record is not readable, gets BRINGUP_PORTS_GRACE_S of not-loading time to publish ip/ssh/door ports.
  # Every die here names the resume line (die's RESUME hint, and the message itself).
  WAIT_MAX="${BRINGUP_INSTANCE_WAIT_S:-2700}"; PORTS_GRACE="${BRINGUP_PORTS_GRACE_S:-300}"
  POLL_S="${BRINGUP_INSTANCE_POLL_S:-5}"; NOTE_S="${BRINGUP_INSTANCE_NOTE_S:-30}"
  case "$WAIT_MAX$PORTS_GRACE$POLL_S$NOTE_S" in
    ''|*[!0-9]*) die "BRINGUP_INSTANCE_WAIT_S/BRINGUP_PORTS_GRACE_S/BRINGUP_INSTANCE_POLL_S/BRINGUP_INSTANCE_NOTE_S must be whole seconds (got '$WAIT_MAX' '$PORTS_GRACE' '$POLL_S' '$NOTE_S')" ;;
  esac
  RESOLVED=""; I_STATUS="<no record yet>"; I_MSG="-"
  T0=$(date +%s); LAST=$T0; NOTLOADING_T0=$T0
  while :; do
    POLL=$(hostctl show instance "$NEW_INSTANCE" --raw 2>/dev/null \
      | python3 -c '
import json, os, sys

raw = sys.stdin.read()
try:
    d = json.loads(raw, strict=False)
except ValueError:
    sys.exit(0)          # not JSON yet (the instance is still being accepted) -- the caller retries
if isinstance(d, list) and len(d) == 1:
    d = d[0]
if not isinstance(d, dict):
    sys.exit(0)
ip = d.get("public_ipaddr") or ""
ports = d.get("ports") or {}
if not isinstance(ports, dict):
    ports = {}

def host_port(cport):
    for key, val in ports.items():
        k = str(key).split("/")[0]              # "1.2.3.4:22/tcp" -> "1.2.3.4:22"
        if k.rsplit(":", 1)[-1] != cport:
            continue
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list) and val:
            hp = val[0].get("HostPort")
            if hp not in (None, ""):
                return str(hp)
    return ""

ssh_port = host_port(os.environ.get("SSH_CONTAINER_PORT", "22"))
door_port = host_port(os.environ.get("DOOR_PORT", ""))
if ip and ssh_port and door_port:
    print("READY", ip, ssh_port, door_port)
    sys.exit(0)
# Not ready: say whether it is still coming up, and what the provider says it is doing (tab-separated, every
# field non-empty -- the shell splits on tabs, which collapse empty fields).
status = d.get("actual_status")
loading = 1 if status in (None, "", "loading", "created") else 0
lines = [l.strip() for l in str(d.get("status_msg") or "").splitlines() if l.strip()]
msg = " ".join(lines[-1].split())[:240] if lines else "-"
print("WAIT\t%d\t%s\t%s" % (loading, status if status not in (None, "") else "null", msg))
' || true)
    case "$POLL" in
      "READY "*) RESOLVED=${POLL#READY }; break ;;
    esac
    NOW=$(date +%s)
    I_LOADING=0
    if [ -n "$POLL" ]; then
      IFS="$(printf '\t')" read -r _ I_LOADING I_STATUS I_MSG <<< "$POLL" || true
    fi
    if [ "$I_LOADING" = 1 ]; then
      NOTLOADING_T0=$NOW
    elif [ $(( NOW - NOTLOADING_T0 )) -ge "$PORTS_GRACE" ]; then
      die "instance $NEW_INSTANCE is not loading (actual_status=$I_STATUS, status_msg: $I_MSG) and reported no ip/ssh/door port for ${PORTS_GRACE}s ($(( NOW - T0 ))s in all) -- read it with \`hostctl show instance $NEW_INSTANCE --raw\`. If it is still coming up, resume this bring-up with: $(resume_cmd)  (that skips the create and repeats this wait; never a plain re-run, which rents a SECOND box). If it has failed, destroy it yourself (\`hostctl destroy instance $NEW_INSTANCE\`) before renting another"
    fi
    if [ $(( NOW - T0 )) -ge "$WAIT_MAX" ]; then
      die "instance $NEW_INSTANCE reported no ip/ssh/door port within ${WAIT_MAX}s (BRINGUP_INSTANCE_WAIT_S; last actual_status=$I_STATUS, status_msg: $I_MSG) -- read it with \`hostctl show instance $NEW_INSTANCE --raw\`. If it is still loading, resume this bring-up with: $(resume_cmd)  (raise BRINGUP_INSTANCE_WAIT_S for a slow image; never a plain re-run, which rents a SECOND box). If it is stuck, destroy it yourself (\`hostctl destroy instance $NEW_INSTANCE\`) before renting another"
    fi
    if [ $(( NOW - LAST )) -ge "$NOTE_S" ]; then
      note "still waiting for instance $NEW_INSTANCE's ip/ports ($(( NOW - T0 ))s; actual_status=$I_STATUS: $I_MSG)"
      LAST=$NOW
    fi
    sleep "$POLL_S"
  done
  read -r NEW_BOX_IP NEW_SSH_PORT NEW_APP_PORT_EXT <<< "$RESOLVED" || \
    die "could not read the instance's ip/ports out of '$RESOLVED' -- refusing to continue"
  for v in "$NEW_BOX_IP" "$NEW_SSH_PORT" "$NEW_APP_PORT_EXT"; do
    [ -n "$v" ] || die "instance $NEW_INSTANCE resolved an empty ip or port ('$RESOLVED') -- refusing to continue"
  done
  case "$NEW_SSH_PORT$NEW_APP_PORT_EXT" in
    *[!0-9]*) die "instance $NEW_INSTANCE resolved a non-numeric port ('$RESOLVED') -- refusing to continue" ;;
  esac
  note "instance $NEW_INSTANCE: ip $NEW_BOX_IP, ssh on $NEW_SSH_PORT, door port $NEW_APP_PORT_EXT"

  if [ "$BOX_IP" = "$NEW_BOX_IP" ] && [ "$SSH_PORT" = "$NEW_SSH_PORT" ] \
     && [ "$APP_PORT_EXT" = "$NEW_APP_PORT_EXT" ] && [ "${INSTANCE_ID:-}" = "$NEW_INSTANCE" ]; then
    note "$BOX_ENV already describes instance $NEW_INSTANCE ($BOX_IP:$SSH_PORT, door $APP_PORT_EXT) -- nothing to write"
  else
    note "$BOX_ENV describes a DIFFERENT box: BOX_IP=${BOX_IP:-<unset>} SSH_PORT=${SSH_PORT:-<unset>} APP_PORT_EXT=${APP_PORT_EXT:-<unset>} INSTANCE_ID=${INSTANCE_ID:-<unset>}, while instance $NEW_INSTANCE is $NEW_BOX_IP:$NEW_SSH_PORT (door $NEW_APP_PORT_EXT)"
    PASTE="BOX_IP=$NEW_BOX_IP SSH_PORT=$NEW_SSH_PORT APP_PORT_EXT=$NEW_APP_PORT_EXT INSTANCE_ID=$NEW_INSTANCE"
    if [ "${BOX_WRITE:-1}" != 1 ]; then
      die "BOX_WRITE=$BOX_WRITE: refusing to run a step against the box $BOX_ENV describes -- set: $PASTE in $BOX_ENV, then resume with: $(resume_cmd)"
    fi
    # Value-only rewrite, in place, keeping each line's own trailing comment and alignment (a
    # hand-written note on this key survives the value being corrected). Written to a temp file in
    # the same directory and renamed over the original, so an interrupted write cannot leave a
    # half-parsed box file behind.
    python3 - "$BOX_ENV" "BOX_IP=$NEW_BOX_IP" "SSH_PORT=$NEW_SSH_PORT" "APP_PORT_EXT=$NEW_APP_PORT_EXT" "INSTANCE_ID=$NEW_INSTANCE" <<'PY' || \
      die "could not write the new instance's ip/ports into $BOX_ENV -- set $PASTE there by hand, then resume with: $(resume_cmd)"
import os, re, sys, tempfile

path = sys.argv[1]
updates = {}
for kv in sys.argv[2:]:
    key, _, val = kv.partition("=")
    updates[key] = val

with open(path) as f:
    orig = f.read()
lines = orig.split("\n")
if lines and lines[-1] == "":
    lines.pop()

out, seen = [], set()
for line in lines:
    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=", line)
    if not m or m.group(1) not in updates:
        out.append(line)
        continue
    key = m.group(1)
    seen.add(key)
    rest = line[m.end():]
    head = "%s=%s" % (key, updates[key])
    # A `#` only starts a comment at a word boundary -- the same rule bash uses when lib.sh
    # sources this file and boxenv.py uses when it parses it.
    hash_at = rest.find("#")
    if hash_at > 0 and rest[hash_at - 1] in " \t":
        col = hash_at + len(line) - len(rest)          # the original line's own column for it
        out.append(head + " " * max(1, col - len(head)) + line[col:])
    else:
        out.append(head)

# A key the file never had (an older box file with no INSTANCE_ID) is appended, commented so its
# provenance is not mistaken for hand-written configuration.
for key, val in updates.items():
    if key not in seen:
        out.append("%s=%s            # written by bringup.sh: the instance this bring-up created" % (key, val))

fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)) or ".", prefix=".bringup-box-")
with os.fdopen(fd, "w") as f:
    f.write("\n".join(out) + "\n")
os.chmod(tmp, os.stat(path).st_mode & 0o7777)
os.replace(tmp, path)
PY
    # Read the file back and require it to say exactly what the instance says. This is what makes a
    # failed or partial write safe: nothing below ever runs on a box file that does not describe the
    # instance this run created -- which is the whole failure mode, one layer down.
    # shellcheck disable=SC1090
    BOXFILE=$(set -a; . "$BOX_ENV"; set +a; printf '%s %s %s %s' "${BOX_IP:-}" "${SSH_PORT:-}" "${APP_PORT_EXT:-}" "${INSTANCE_ID:-}")
    [ "$BOXFILE" = "$NEW_BOX_IP $NEW_SSH_PORT $NEW_APP_PORT_EXT $NEW_INSTANCE" ] || \
      die "$BOX_ENV still does not describe instance $NEW_INSTANCE after the write (it reads '${BOXFILE}', wanted '$NEW_BOX_IP $NEW_SSH_PORT $NEW_APP_PORT_EXT $NEW_INSTANCE') -- set $PASTE in $BOX_ENV by hand, then resume with: $(resume_cmd)"
    note "$BOX_ENV now describes instance $NEW_INSTANCE (BOX_IP/SSH_PORT/APP_PORT_EXT/INSTANCE_ID written; the previous values were named above)"
  fi
  # Either way the box file now names this run's instance, written by this bring-up or by the earlier
  # run of it a resume continues: the record says so, which is what a box-c re-create's resume reads.
  open_record_add "$NEW_INSTANCE" RECONCILED "$(date -u +%Y-%m-%dT%H:%M:%SZ) $BOX_ENV INSTANCE_ID=$NEW_INSTANCE BOX_IP=$NEW_BOX_IP SSH_PORT=$NEW_SSH_PORT"

  # The box file is right, so make this process agree with it: BOX_IP/SSH_PORT are what every
  # box_ssh below resolves, and the guard checks the values this process actually holds.
  BOX_IP=$NEW_BOX_IP; SSH_PORT=$NEW_SSH_PORT; APP_PORT_EXT=$NEW_APP_PORT_EXT; INSTANCE_ID=$NEW_INSTANCE
  export BOX_IP SSH_PORT APP_PORT_EXT INSTANCE_ID
  # lib.sh derived and EXPORTED DOOR_URL from the box file's APP_PORT_EXT while sourcing it, i.e.
  # before the reconcile above corrected that value -- and a child that resolves its door URL from
  # $DOOR_URL first (boxenv.py's door_url()) would inherit
  # the PREVIOUS box's external port over its own box file. That stale port is what killed the
  # incident's front-door check. A DOOR_URL the caller exported itself is left alone, and said out
  # loud, because it is now this run's own responsibility.
  if [ -z "$DOOR_URL_FROM_ENV" ]; then
    export DOOR_URL="https://${HOSTNAME}:${APP_PORT_EXT}"
  else
    note "DOOR_URL was set in the environment ($DOOR_URL) -- leaving it alone rather than re-deriving it from APP_PORT_EXT=$APP_PORT_EXT"
  fi
else
  note "DRY_RUN: skipping the reconcile (nothing was created; a real run resolves the new instance's ip/ports here and fences every box_ssh to them)"
fi

note "== wait for ssh"
if [ "${DRY_RUN:-0}" != 1 ]; then
  assert_box_target   # loud, before the loop below keeps every box_ssh's stderr off the terminal
  T0=$(date +%s); LAST=$T0
  # A key refusal and a box that has not booted look nothing alike on the wire, and this loop's old
  # `2>/dev/null` is what made the first one read as the second: `Permission denied (publickey)` every
  # 10 s for half an hour, then a die whose whole message was "ssh never came up" (reproduced
  # 2026-10-17 -- the failure the attach step above exists to prevent). stderr now lands in a per-run
  # file, named with lib.sh's own run token (`${BOX_TARGET_RUN_ID:-$$}` -- the shell's own pid as a
  # fallback, so the name stays per-run even if a future lib.sh stops exporting the token) so no other
  # run can read or overwrite it, which is what lets the loop quote the last error and act on the one
  # error that means "the box is up and does not accept this key". The ssh itself still runs in THIS
  # shell, never inside a `$( )`: the box-target fence dies inside box_ssh, and a die swallowed into a
  # command substitution would leave the loop waiting out its ceiling on a box the fence had already
  # refused.
  SSH_WAIT_ERR="${TMPDIR:-/tmp}/bringup-ssh-wait.${BOX_TARGET_RUN_ID:-$$}"
  SSH_REFUSAL_T0=""
  until box_ssh true 2>"$SSH_WAIT_ERR"; do
    NOW=$(date +%s); LAST_ERR=$(tr '\n' ' ' < "$SSH_WAIT_ERR" 2>/dev/null || true)
    case "$LAST_ERR" in
      *"Permission denied"*|*publickey*)
        if [ -z "$SSH_REFUSAL_T0" ]; then
          # The box answered and refused the key, so sshd is up: this is not a box still booting.
          # Re-run the attach once -- a just-created instance can be reachable before the key list this
          # run attached is in place -- and it is the same call the attach step above made.
          SSH_REFUSAL_T0=$NOW
          note "the box answered and refused ${SSH_KEY_PATH} (last error: $LAST_ERR) -- re-running hostctl attach ssh for instance $NEW_INSTANCE"
          if ATTACH_OUT=$(attach_ssh_key "$NEW_INSTANCE"); then
            ATTACHED=1; ATTACH_RC=0
            note "re-attached ${SSH_KEY_PATH}.pub to instance $NEW_INSTANCE (hostctl attach ssh) after the refusal above"
          else
            ATTACH_RC=$?
            note "WARNING: hostctl attach ssh $NEW_INSTANCE failed too (exited $ATTACH_RC): $(printf '%s' "$ATTACH_OUT" | tr '\n' ' ')"
          fi
        elif [ $(( NOW - SSH_REFUSAL_T0 )) -ge 300 ]; then
          die "ssh reached instance $NEW_INSTANCE ($BOX_IP:$SSH_PORT) and it refused ${SSH_KEY_PATH} for 300s -- last error: $LAST_ERR. The box is UP and billed: do NOT re-run bringup.sh (its create rents a SECOND box). Attach the key by hand, then resume with $(resume_cmd) --
  hostctl attach ssh $NEW_INSTANCE \"${SSH_KEY_PATH}.pub\"
  ssh -o BatchMode=yes -p $SSH_PORT -i $SSH_KEY_PATH root@$BOX_IP true
-- and if the attach reports success while the box keeps refusing, then this box file's SSH_KEY is not a key that instance accepts at all: read the keys it carries from \`hostctl show instance $NEW_INSTANCE --raw\` and 07-operations.md's \"Reaching a box from a sandboxed worker\"."
        fi ;;
    esac
    [ $(( NOW - T0 )) -ge 1800 ] && die "ssh never came up after 1800s ($BOX_IP:$SSH_PORT) -- last error: ${LAST_ERR:-<none>}"
    [ $(( NOW - LAST )) -ge 60 ] && { note "still waiting for ssh ($(( NOW - T0 ))s${LAST_ERR:+; last: $LAST_ERR})"; LAST=$NOW; }
    sleep 10
  done
  rm -f "$SSH_WAIT_ERR"
  note "ssh is up"
else
  note "DRY_RUN: skipping the ssh wait"
fi

# A resumed box-c re-create whose engine already answers /health is not resumed (#2661). Everything after
# this point re-runs on the box -- stage.sh, the graft step, restart.sh's fork relaunch (a restart), the
# front door -- and an engine that is up may already be taking box-c's traffic (the door points
# box-c's name at this box, and the router reaches it by that name): a resume would restart production
# outside a router maintenance window. Read once, fail closed: an ssh that cannot run the read (exit 255)
# refuses too. The RESUME hint is suppressed for these dies -- re-running the same resume meets the same
# refusal.
if [ "$RESUME_BOX_C__RECREATE" = 1 ] && [ "${DRY_RUN:-0}" != 1 ]; then
  note "== box-c re-create resume: is an engine already serving on instance $NEW_INSTANCE?"
  # With ENGINES every copy's own port is read too: once the fork checkout's `restart.sh --engine all`
  # has run, the copies serve on their ports (box-c: 8002, 8012) and nothing answers on the legacy one.
  HEALTH_URLS="$ENGINE_URL"
  for HEALTH_ID in $ENGINES_IDS; do HEALTH_URLS="$HEALTH_URLS $(engine_select "$HEALTH_ID" && printf '%s' "$ENGINE_URL")"; done
  HEALTH_READ=""
  for HEALTH_URL in $HEALTH_URLS; do
    HEALTH_RC=0
    HEALTH_CODE=$(box_ssh "curl -s -m 10 -o /dev/null -w '%{http_code}' '$HEALTH_URL/health' || true" 2>/dev/null) || HEALTH_RC=$?
    if [ "$HEALTH_RC" != 0 ]; then
      RESUME_HINTED=1
      die "could not read $HEALTH_URL/health on instance $NEW_INSTANCE (ssh exit $HEALTH_RC) -- refusing to resume a box-c re-create blind: if an engine is up there it may be serving box-c's traffic. Check by hand (box_ssh curl -s -o /dev/null -w '%{http_code}' $HEALTH_URL/health) before anything else"
    fi
    if [ "$HEALTH_CODE" = 200 ] && [ -n "$ENGINES_IDS" ]; then
      RESUME_HINTED=1
      die "instance $NEW_INSTANCE already serves: $HEALTH_URL/health answers 200 -- refusing to resume this box-c re-create: the rest of the run re-stages the box and relaunches every copy (restart.sh --engine all fork), which is a production restart if the door already pointed box-c's name here. Finish the remaining steps by hand with box-c's router rows box-c-a and box-c-b out where they touch the engines (a per-pod \`sbr admin backend maintenance <row> --window 3600s --hold-until-recheck\` on every router pod, lifted with \`backend recheck <row>\`; README.md \"The box-d layout on a re-create\", 07-operations.md \"Restart and cutover recipes\"), then close this bring-up's record: rm '$(open_record "$NEW_INSTANCE")'"
    fi
    if [ "$HEALTH_CODE" = 200 ]; then
      RESUME_HINTED=1
      die "instance $NEW_INSTANCE already serves: $HEALTH_URL/health answers 200 -- refusing to resume this box-c re-create: the rest of the run re-stages the box and relaunches its engine (restart.sh fork), which is a production restart if the door already pointed box-c's name here. Finish the remaining steps by hand, inside a router BACKEND-MAINT-box-c window where they touch the engine (README.md \"The box-d layout on a re-create\", 07-operations.md \"Restart and cutover recipes\"), then close this bring-up's record: rm '$(open_record "$NEW_INSTANCE")'"
    fi
    HEALTH_READ="${HEALTH_READ:+$HEALTH_READ, }${HEALTH_CODE:-<nothing>}"
  done
  if [ -z "$ENGINES_IDS" ]; then
    note "no engine answers /health on instance $NEW_INSTANCE (read ${HEALTH_CODE:-<nothing>}) -- resuming the re-create"
  else
    note "no engine answers /health on instance $NEW_INSTANCE ($HEALTH_URLS read $HEALTH_READ) -- resuming the re-create"
  fi
fi

# The onstart's checkpoint download reads HF_TOKEN from /root/.hf.env on the box (#6450: the token is
# never rendered into the onstart or the provider blob), and nothing else writes that file -- so without
# this step a fresh box downloads anonymously (the repo is ungated: it completes, rate-limited). It
# runs as soon as ssh answers, before stage.sh and the download gate. The onstart may already be in
# its first `hf download` attempt by then (it starts at first boot); each retry re-reads the file, so
# the token applies from the next attempt on. The token goes on STDIN only -- printf is a shell
# builtin, so it is never in an argv, and it is never echoed: the file is written under umask 077 to
# a mktemp sibling and renamed into place, then checked by mode and by sha256 against the same line
# hashed here (the digest is compared, never printed).
note "== /root/.hf.env (HF_TOKEN for the onstart's checkpoint download)"
if [ "${DRY_RUN:-0}" != 1 ]; then
  [ -n "$HF_TOKEN_VALUE" ] || die "no HF_TOKEN to write into /root/.hf.env (preflight should have died on this)"
  printf 'HF_TOKEN=%s\n' "$HF_TOKEN_VALUE" | \
    box_ssh 'umask 077; t=$(mktemp /root/.hf.env.XXXXXX) && cat > "$t" && chmod 600 "$t" && mv -f "$t" /root/.hf.env' \
    || die "could not write /root/.hf.env on $BOX -- the onstart's download runs anonymously until it exists; write it by hand (HF_TOKEN=<value>, mode 0600) or resume with: $(resume_cmd)"
  WANT_SHA=$(printf 'HF_TOKEN=%s\n' "$HF_TOKEN_VALUE" | python3 -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')
  GOT=$(box_ssh 'ls -l /root/.hf.env | cut -c1-10; sha256sum < /root/.hf.env | cut -d" " -f1' || true)
  [ "$GOT" = "-rw-------
$WANT_SHA" ] || die "/root/.hf.env on $BOX did not read back as a mode-0600 copy of $ENV_FILE_PATH's HF_TOKEN line (mode/digest mismatch) -- check it by hand"
  note "/root/.hf.env written (mode 0600, sha256 matches $ENV_FILE_PATH's HF_TOKEN line; the value was piped on stdin, never printed)"
else
  if [ -n "$HF_TOKEN_VALUE" ]; then
    note "DRY_RUN: would pipe HF_TOKEN=<from $ENV_FILE_PATH, not shown> on stdin into $BOX:/root/.hf.env (umask 077, mktemp + mv, mode 0600), then check its mode and sha256"
  else
    note "DRY_RUN: would pipe HF_TOKEN on stdin into $BOX:/root/.hf.env (mode 0600) -- $ENV_FILE_PATH has no HF_TOKEN= line, so a real run dies at preflight before this"
  fi
fi

note "== stage.sh"
# stage.sh's last step delivers a non-stock-pinned Caddy binary onto this
# fresh box (caddy-h2/README.md "A fresh box: where the binary comes from").
run bash "$TOOLS_DIR/stage.sh"

note "== download gate"
# The onstart writes TWO markers into /workspace/download.log, and only their ORDER says which one
# is true of the box's checkpoint. `COSMOS_DOWNLOAD_COMPLETE` is appended at the end of the download
# step, and the `===<box>-DOWNLOAD-INCOMPLETE <date> ...===` line immediately AFTER it, when the
# completeness check that follows the marker fails. A box that gave up therefore carries BOTH lines.
# Grepping for the success marker alone passed such a boot as "download complete" -- bringup then
# ran the graft and the boot gates, and the operator met a later "engine never came up" instead of
# this gate's message -- while grepping for the failure line alone would die forever on a box whose
# FIRST boot failed and whose later boot carried the download through, because the log is
# append-only and grep matches any occurrence. So the gate reads the last of the two lines, never a
# bare match: `tail -1` of the two patterns. Both markers are MATCHED AS SUBSTRINGS of that last
# line, never compared for equality: the hf download's own progress output ends in a carriage
# return, not a newline, so the onstart's `echo` can find itself glued to the tail of a progress
# line (`Downloading shard 47/48: 100%\rCOSMOS_DOWNLOAD_COMPLETE`) -- a substring test reads that
# as the marker it is, an `=` test would wait out the whole 6h ceiling on a completed download.
# A failure line is the onstart's last word on that boot (it exits right after writing one), so a
# failure read dies here at once instead of burning that ceiling.
download_marker() {
  box_ssh "grep -a -e COSMOS_DOWNLOAD_COMPLETE -e DOWNLOAD-INCOMPLETE /workspace/download.log 2>/dev/null | tail -1" || true
}
if [ "${DRY_RUN:-0}" != 1 ]; then
  T0=$(date +%s); LAST=$T0
  while :; do
    MARK=$(download_marker)
    # A success marker is the last line only until the same boot appends the failure line behind
    # it, which is microseconds later but can still be a poll apart, so a success read is confirmed
    # once past that window before it ends the gate: the confirming read sees the failure line and
    # the case below dies on it. (A failure read needs no second look -- it is already the last
    # word.) No marker at all -- no log yet, or a log this boot has not written to yet -- waits,
    # which is what the gate always did while the checkpoint downloads.
    case "$MARK" in
      *COSMOS_DOWNLOAD_COMPLETE*) sleep 3; MARK=$(download_marker) ;;
    esac
    case "$MARK" in
      *DOWNLOAD-INCOMPLETE*)
        die "onstart's LAST download.log marker on $BOX is an incomplete checkpoint ('$MARK') -- the onstart appends COSMOS_DOWNLOAD_COMPLETE BEFORE it checks the checkpoint, so an earlier success marker in that log is NOT success. Check /workspace/download.log and /workspace/progress.log on the box. If the checkpoint was brought to complete by hand since that line was written, the line is stale (the onstart appends COSMOS_DOWNLOAD_COMPLETE again only when its own download step runs): verify the checkpoint on the box, then append a COSMOS_DOWNLOAD_COMPLETE line to /workspace/download.log and resume with: $(resume_cmd) (a plain re-run of bringup.sh rents a SECOND box: without --instance its create is unconditional)." ;;
      *COSMOS_DOWNLOAD_COMPLETE*)
        break ;;
    esac
    NOW=$(date +%s)
    [ $(( NOW - T0 )) -ge 21600 ] && die "checkpoint download not complete after 6h -- check /workspace/download.log and /workspace/progress.log on the box"
    [ $(( NOW - LAST )) -ge 60 ] && { p=$(box_ssh "tail -1 /workspace/progress.log 2>/dev/null" || true); note "still downloading ($(( NOW - T0 ))s): $p"; LAST=$NOW; }
    sleep 20
  done
  note "download complete"
else
  note "DRY_RUN: skipping the download gate"
fi

note "== graft (optional): copy our graft from this Mac and rebuild MODEL_DIR"
# Both ends of the graft's applier are named by the box file, so the graft step's paths are a box
# fact rather than literals here: GRAFT_DIR is where GRAFT_SRC's three files are copied, APPLY_GRAFT_PY
# is the box-side path of the kit's grafting-kit/tools/apply_graft.py this step stages onto the box.
# The fallbacks for both are the generic box layout, never a box's own paths:
# <MODEL_PATH>/graft and <APPLY_GRAFT_PY> -- the kit's tools/
# without the kit/ component, the same shape bake.py's own --graft default,
# <MODEL_PATH>/graft, belongs to. A box file is the only source for the real pair. box-c
# names the L3a graft (GRAFT_SRC=<HOME>/workspace/backups/grafts/campaign-1017/graft-1017-l3a,
# GRAFT_DIR=<GRAFT_DIR>), so this step runs there and rebuilds its repack;
# a box file that names a graft names it one directory per graft (c20's is
# <MODEL_PATH>/graft-1019-c20), so an older graft's files are never mixed into the one
# being applied, and box-c's APPLY_GRAFT_PY is <APPLY_GRAFT_PY>, where the
# 1017, 1019-c20 and 1017-l3a graft runs used it.
# APPLY_GRAFT_PY_SRC is this checkout's end of that copy, overridable for a checkout whose kit is
# elsewhere.
# The pair these two names make -- GRAFT_DIR_REMOTE's graft into MODEL_DIR -- was already checked by
# check_graft_build_pair in the preflight above, before anything was rented: a run that got here has
# either a coherent pair or GRAFT_BUILD_MISMATCH_OK=1 saying in so many words that it means to apply
# GRAFT_DIR's graft into a MODEL_DIR of another name.
#
# Where the graft comes from (#2430): GRAFT_SRC, the graft's backup on THIS Mac, already hashed by
# check_graft_source in preflight (edited_tensors.safetensors == GRAFT_SHA256 == its manifest's own
# graft sha). Our files never come from Hugging Face (our internal artifact policy), so there is
# no hf call and no token in this step at all: the three files go over box_scp, the same ssh transport
# the applier rides. Exactly the three the graft IS -- apply_graft.py's load_graft reads
# edited_tensors.safetensors + applier.json + manifest.json, and graft-apply.sh reads applier.json --
# and nothing else from the backup directory (its README/notes never belong in a box's GRAFT_DIR).
#   * A file whose box copy ALREADY hashes to the wanted sha is not copied again: a resumed run
#     (--instance) after a copy that landed does not move 2 GiB twice.
#   * After the copy every one of the three is RE-HASHED ON THE BOX and compared with the value
#     preflight took on the Mac -- GRAFT_SHA256 for edited_tensors.safetensors, the Mac copy's own sha
#     for applier.json and manifest.json. Any difference (a truncated transfer, a Mac file that changed
#     since preflight, a stale box file the copy did not replace) is refused before graft-apply.sh runs:
#     scp's exit status is not the outcome, the bytes are.
GRAFT_FILES="edited_tensors.safetensors applier.json manifest.json"
GRAFT_DIR_REMOTE="${GRAFT_DIR:-<MODEL_PATH>/graft}"
APPLY_GRAFT_PY_REMOTE="${APPLY_GRAFT_PY:-<APPLY_GRAFT_PY>}"
APPLY_GRAFT_PY_SRC="${APPLY_GRAFT_PY_SRC:-$TOOLS_DIR/../../grafting-kit/tools/apply_graft.py}"
# graft_want <file> -- the sha256 the box's copy of <file> must have: preflight's values, never re-read.
graft_want() {
  case "$1" in
    edited_tensors.safetensors) printf '%s' "$GRAFT_SHA256" ;;
    applier.json)               printf '%s' "$GRAFT_SRC_SHA_APPLIER" ;;
    manifest.json)              printf '%s' "$GRAFT_SRC_SHA_MANIFEST" ;;
  esac
}
# graft_box_shas -- `<file> <sha256>` per graft file in GRAFT_DIR on the box, `<file> -` when absent.
# sha256sum reads stdin, the form the applier read-back above it uses too.
graft_box_shas() {
  box_ssh "cd '$GRAFT_DIR_REMOTE' && for f in $GRAFT_FILES; do if [ -f \"\$f\" ]; then printf '%s %s\n' \"\$f\" \"\$(sha256sum < \"\$f\" | cut -d' ' -f1)\"; else printf '%s -\n' \"\$f\"; fi; done" || true
}
if [ -n "${GRAFT_SRC:-}" ]; then
  # Checked BEFORE any ssh: a missing local applier is a mistake in this checkout, and finding it
  # after a 2 GiB graft copy (or after renting a box) wastes the operator's time. Under DRY_RUN it is
  # a warning, so a dry pass still runs without a full checkout.
  if [ ! -f "$APPLY_GRAFT_PY_SRC" ]; then
    if [ "${DRY_RUN:-0}" = 1 ]; then
      note "WARNING: no apply_graft.py at $APPLY_GRAFT_PY_SRC to stage onto $BOX -- DRY_RUN: continuing anyway (a real run dies here: the box needs it to apply $GRAFT_SRC)"
    else
      die "no apply_graft.py at $APPLY_GRAFT_PY_SRC to stage onto $BOX -- GRAFT_SRC=$GRAFT_SRC cannot be applied without it; point APPLY_GRAFT_PY_SRC at the kit's apply_graft.py, or run graft-apply.sh on the box by hand"
    fi
  fi
  if [ "${DRY_RUN:-0}" != 1 ]; then
    note "GRAFT_SRC=$GRAFT_SRC_PATH -- stage $APPLY_GRAFT_PY_SRC to $BOX:$APPLY_GRAFT_PY_REMOTE (sha256-checked: this is the applier the graft will be built with), then copy $GRAFT_FILES from this Mac into $BOX:$GRAFT_DIR_REMOTE over ssh and re-hash them there (edited_tensors.safetensors must be GRAFT_SHA256=$GRAFT_SHA256)"
    box_ssh "mkdir -p '$GRAFT_DIR_REMOTE' '$(dirname "$APPLY_GRAFT_PY_REMOTE")' '$TOOLS_REMOTE'" || die "could not create $(dirname "$APPLY_GRAFT_PY_REMOTE") / $GRAFT_DIR_REMOTE on $BOX"
    box_scp "$APPLY_GRAFT_PY_SRC" "root@$BOX_IP:$APPLY_GRAFT_PY_REMOTE" \
      || die "could not copy $APPLY_GRAFT_PY_SRC to $BOX:$APPLY_GRAFT_PY_REMOTE -- without it on the box graft-apply.sh cannot apply $GRAFT_SRC (it dies at its own APPLY_GRAFT_PY check); fix the copy or set APPLY_GRAFT_PY_SRC, then resume with: $(resume_cmd)"
    box_ssh "chmod 755 '$APPLY_GRAFT_PY_REMOTE'" || die "could not chmod $APPLY_GRAFT_PY_REMOTE on $BOX"
    # The staged file is what the graft is actually applied with, so it is checked, not assumed: a
    # truncated scp would bake a wrong applier into MODEL_DIR, which is worse than no graft at all.
    WANT_APPLIER=$(python3 -c 'import hashlib, sys; sys.stdout.write(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$APPLY_GRAFT_PY_SRC")
    GOT_APPLIER=$(box_ssh "sha256sum < '$APPLY_GRAFT_PY_REMOTE' | cut -d' ' -f1" || true)
    [ "$GOT_APPLIER" = "$WANT_APPLIER" ] || die "$APPLY_GRAFT_PY_REMOTE on $BOX does not read back as a sha256 copy of $APPLY_GRAFT_PY_SRC (wanted $WANT_APPLIER, read '${GOT_APPLIER:-<nothing>}') -- refusing to run a graft with an applier that is not the one in this checkout; resume with: $(resume_cmd)"
    # The copy: only the files whose box copy is not already the wanted bytes.
    BOX_GRAFT_SHAS=$(graft_box_shas)
    for f in $GRAFT_FILES; do
      want=$(graft_want "$f")
      got=$(printf '%s\n' "$BOX_GRAFT_SHAS" | awk -v f="$f" '$1 == f { print $2 }')
      if [ "$got" = "$want" ]; then
        note "graft: $BOX:$GRAFT_DIR_REMOTE/$f already hashes to $want -- not copied again"
        continue
      fi
      note "graft: copying $GRAFT_SRC_PATH/$f -> $BOX:$GRAFT_DIR_REMOTE/$f"
      box_scp "$GRAFT_SRC_PATH/$f" "root@$BOX_IP:$GRAFT_DIR_REMOTE/$f" \
        || die "could not copy $GRAFT_SRC_PATH/$f to $BOX:$GRAFT_DIR_REMOTE/$f over ssh -- check the transfer, then resume with: $(resume_cmd)"
    done
    # The outcome, re-hashed on the box: every file, whether this run copied it or found it in place.
    BOX_GRAFT_SHAS=$(graft_box_shas)
    GRAFT_BAD=""
    for f in $GRAFT_FILES; do
      want=$(graft_want "$f")
      got=$(printf '%s\n' "$BOX_GRAFT_SHAS" | awk -v f="$f" '$1 == f { print $2 }')
      [ "$got" = "$want" ] || GRAFT_BAD="$GRAFT_BAD
  $f: wanted $want, read ${got:-<nothing>}"
    done
    [ -z "$GRAFT_BAD" ] || die "the graft in $GRAFT_DIR_REMOTE on $BOX does not re-hash to the Mac copy preflight checked ($GRAFT_SRC_PATH, GRAFT_SHA256=$GRAFT_SHA256) -- refusing to apply it:$GRAFT_BAD
a truncated or interrupted copy, or a Mac file that changed since preflight; fix it, then resume with: $(resume_cmd)"
    note "graft copied and re-hashed on $BOX (edited_tensors.safetensors $GRAFT_SHA256, applier.json $GRAFT_SRC_SHA_APPLIER, manifest.json $GRAFT_SRC_SHA_MANIFEST) -- running graft-apply.sh (STOCK_DIR=${STOCK_DIR:-$MODEL_DIR} GRAFT_DIR=$GRAFT_DIR_REMOTE MODEL_DIR=$MODEL_DIR APPLY_GRAFT_PY=$APPLY_GRAFT_PY_REMOTE VERIFY_ALL=${GRAFT_VERIFY_ALL:-0})"
    box_ssh "STOCK_DIR='${STOCK_DIR:-$MODEL_DIR}' GRAFT_DIR='$GRAFT_DIR_REMOTE' MODEL_DIR='$MODEL_DIR' APPLY_GRAFT_PY='$APPLY_GRAFT_PY_REMOTE' VERIFY_ALL='${GRAFT_VERIFY_ALL:-0}' bash '$TOOLS_REMOTE/graft-apply.sh'" \
      || die "graft-apply.sh failed on $BOX -- MODEL_DIR=$MODEL_DIR was not (re)written; check the graft-apply output above, then resume with: $(resume_cmd)"
    note "graft applied -- MODEL_DIR=$MODEL_DIR ready; the fork checkout below relaunches the engine on FORK_TREE serving it, and the engine boot gates after it confirm (no separate restart triggered here)"
  else
    note "DRY_RUN: GRAFT_SRC=$GRAFT_SRC set -- would stage $APPLY_GRAFT_PY_SRC to $BOX:$APPLY_GRAFT_PY_REMOTE (sha256-checked), copy $GRAFT_FILES from ${GRAFT_SRC_PATH:-$GRAFT_SRC} on this Mac into $BOX:$GRAFT_DIR_REMOTE over ssh (box_scp) and re-hash them there (edited_tensors.safetensors == GRAFT_SHA256=${GRAFT_SHA256:-<unset>}), then run graft-apply.sh with STOCK_DIR=${STOCK_DIR:-$MODEL_DIR} GRAFT_DIR=$GRAFT_DIR_REMOTE MODEL_DIR=$MODEL_DIR APPLY_GRAFT_PY=$APPLY_GRAFT_PY_REMOTE VERIFY_ALL=${GRAFT_VERIFY_ALL:-0}"
  fi
else
  note "GRAFT_SRC empty -- serving STOCK_DIR unmodified, no graft step"
fi

# The fork checkout runs BEFORE the engine boot gates (#1588). The onstart boots the IMAGE's own tree
# (/sgl-workspace/sglang at the image's HEAD) with the staged args, and those args carry the fork's
# own flags (--metric-label-denylist, --short-prefill-*, ...): on gpu1018 (2026-10-18) `sglang serve`
# died at once with `error: unrecognized arguments: ...`, and the old order then polled /health for up
# to an hour before it ever reached this checkout. Checking out FORK_TREE first and relaunching on it
# (restart.sh) puts the engine on the tree its args were written for; the gates below then confirm it.
note "== fork checkout"
RESTART="$TOOLS_DIR/restart.sh"
need "$RESTART"
# The tree the box SERVES is the box file's FORK_TREE -- the COMMIT this box serves: a commit sha,
# or a ref that resolves to one, never a TREE object sha (the ^{commit} peel below rejects a tree
# with 'expected commit type, but the object dereferences to tree type'). It is not the branch tip:
# a tip can outrun the pinned image's package pins (box-c: the box-f tip needs sglang-kernel 0.4.7 and
# sgl-deep-gemm >= 0.2.0 the image lacks, and refuses to boot). Unset, the tip is used.
SERVE_TREE="${FORK_TREE:-fork/$FORK_BRANCH}"
if [ "${DRY_RUN:-0}" != 1 ]; then
  box_ssh "git -C '$SGLANG_TREE' rev-parse --verify '$SERVE_TREE^{commit}'" >/dev/null 2>&1 || \
    die "$SERVE_TREE not found on $BOX -- stage.sh's 'fork remote + fetch' step should have fetched it (FORK_TREE must be reachable from fork/$FORK_BRANCH or another fetched branch); re-run stage.sh"
else
  note "DRY_RUN: would verify $SERVE_TREE via: box_ssh git -C '$SGLANG_TREE' rev-parse --verify '$SERVE_TREE^{commit}'"
fi
RESTART_TAG=fork   # restart.sh writes `===== RESTART fork ...` into SERVE_LOG; wait_engine_ready slices from that line
if [ -z "$ENGINES_IDS" ]; then
  run bash "$RESTART" "$RESTART_TAG" - - --tree "$SERVE_TREE"
  note "fork checkout done ($SERVE_TREE) -- the engine was relaunched on it; the boot gates below confirm it serves"
else
  # A box file with ENGINES refuses a bare restart.sh: every copy is launched in one send, each with
  # --engine-args installed as its sglang-args-<id>.txt and the box's env seed (`default`) as its
  # sglang-env-<id>.txt, and each copy's log gets its own `===== RESTART fork engine <id>` marker.
  # restart.sh takes the box off the onstart's one-engine line (its watchdog held, line stopped) and
  # runs its own per-copy warm gate, prewarm and sampling gate; the gates below re-check each copy.
  run bash "$RESTART" --engine all "$RESTART_TAG" "$ENGINE_ARGS" default --tree "$SERVE_TREE"
  note "fork checkout done ($SERVE_TREE) -- every copy ($ENGINES_IDS) was relaunched on it from $ENGINE_ARGS; the boot gates below confirm each serves"
fi

# SERVE_LOG: the engine's log on the box (restart.sh/cutover.sh append to it, each launch after a
# `===== RESTART <tag> ...` marker line).
SERVE_LOG="${SERVE_LOG:-/workspace/sglang-serve.log}"

# wait_engine_ready [ceiling seconds]: poll /health until 200, and fail FAST -- not after the ceiling --
# when the latest launch can never come up: its log section carries argparse's refusal
# (`unrecognized arguments`, `error: argument `). The section starts at the LATER of the last line-start
# `===== RESTART ` marker and the fork checkout's own marker line (the last line holding
# `===== RESTART $RESTART_TAG `, read once, 1 when absent) -- restart.sh's and cutover.sh's DEAD_CHECK
# rule (#1699), so an earlier launch's death does not condemn this one even when the fork marker landed
# mid-line after engine output with no trailing newline (`^` misses it; its own line number does not).
# Poll interval BRINGUP_HEALTH_POLL_S (default 15 s).
wait_engine_ready() {
  local ceiling="${1:-3600}" t0 last now code dead marker
  marker=$(box_ssh "grep -an '===== RESTART ${RESTART_TAG:-fork} ' '$SERVE_LOG' 2>/dev/null | tail -1 | cut -d: -f1" 2>/dev/null || true)
  case "$marker" in ''|*[!0-9]*) marker=1 ;; esac
  t0=$(date +%s); last=$t0
  while :; do
    code=$(box_ssh "curl -s -m 10 -o /dev/null -w '%{http_code}' '$ENGINE_URL/health'" 2>/dev/null || true)
    [ "$code" = 200 ] && return 0
    dead=$(box_ssh "f='$SERVE_LOG'; n=\$(grep -an '^===== RESTART ' \"\$f\" 2>/dev/null | tail -1 | cut -d: -f1); [ \"\${n:-0}\" -gt $marker ] || n=$marker; tail -n +\$n \"\$f\" 2>/dev/null | grep -aE -m1 'unrecognized arguments|error: argument '" 2>/dev/null || true)
    [ -n "$dead" ] && die "the engine cannot start -- $SERVE_LOG on $BOX says: $dead -- the args do not match the tree it is running (git -C $SGLANG_TREE rev-parse HEAD on the box; the staged args need $SERVE_TREE). Not waiting out the ${ceiling}s /health ceiling."
    now=$(date +%s)
    [ $(( now - t0 )) -ge "$ceiling" ] && die "engine /health never went 200 within ${ceiling}s -- check $SERVE_LOG on the box"
    [ $(( now - last )) -ge 60 ] && { note "still waiting on /health ($(( now - t0 ))s, last=$code)"; last=$now; }
    sleep "${BRINGUP_HEALTH_POLL_S:-15}"
  done
}

# server_info_remote: the box half of the /get_server_info gate (#2994). The engine runs with an API
# key, so an unauthed GET answers 401 -- which is all the old bare `curl` here ever got, so both
# backend checks below always WARNed. An authed body is worse to print than to miss: it carries the
# key itself, in server_args (`api_key`, three times on box-c). So the body never leaves the box:
#   * the key is read from the box's own /etc/environment (root-only, stage.sh) into a shell
#     variable and reaches curl as a header on its STDIN (`-H @-`, curl >= 7.55), via `printf`, a
#     shell builtin -- it is in no argv on the box (/proc/<pid>/cmdline) and none on this Mac;
#   * the body goes straight into `grep -o`, which prints only the two fields the gate reads, plus
#     the HTTP code curl's -w appends on its own line.
# No key on the box -> no header (an engine started without one answers 200 anyway; one with one
# answers 401, which the caller names). The body runs as `bash -s -- <url>` from this heredoc;
# tests/stage-keyholders.test.sh extracts it and runs it against a stub engine.
server_info_remote() {
  box_ssh bash -s -- "$ENGINE_URL/get_server_info" <<'SERVERINFO'
k=$(grep -a '^VLLM_API_KEY=' /etc/environment 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"')
{ [ -z "$k" ] || printf 'Authorization: Bearer %s\n' "$k"; } \
  | curl -s -m 10 -H @- -w '\nhttp_code=%{http_code}\n' "$1" \
  | grep -aoE '"(attention_backend|moe_runner_backend)"[[:space:]]*:[[:space:]]*"[^"]*"|^http_code=[0-9]+$'
SERVERINFO
}

# server_info_gate: reads the two fields server_info_remote printed and WARNs (never dies -- smoke.py
# is the hard gate on them) unless the answer was HTTP 200 with attention_backend=dsv4 and
# moe_runner_backend=flashinfer_mxfp4, the values smoke.py asserts. Only those fields and the code
# are printed.
server_info_gate() {
  local out code attn moe
  out=$(server_info_remote 2>/dev/null || true)
  code=$(printf '%s\n' "$out" | sed -n 's/^http_code=\([0-9]*\)$/\1/p' | tail -1)
  attn=$(printf '%s\n' "$out" | sed -n 's/^"attention_backend"[[:space:]]*:[[:space:]]*"\([^"]*\)"$/\1/p' | head -1)
  moe=$(printf '%s\n' "$out" | sed -n 's/^"moe_runner_backend"[[:space:]]*:[[:space:]]*"\([^"]*\)"$/\1/p' | head -1)
  note "/get_server_info: HTTP ${code:-<none>}, attention_backend=${attn:-<missing>}, moe_runner_backend=${moe:-<missing>} (only these fields are read on the box: the body carries the API key in server_args)"
  case "$code" in
    200) ;;
    401|403) note "WARNING: /get_server_info answered HTTP $code -- the key in $BOX's /etc/environment (VLLM_API_KEY) did not authenticate against the engine, so the backend checks below could not read the answer -- check manually" ;;
    *) note "WARNING: /get_server_info answered HTTP ${code:-nothing} -- check manually" ;;
  esac
  [ "$attn" = dsv4 ] || note "WARNING: /get_server_info's attention_backend is ${attn:-missing}, not dsv4 -- check manually"
  [ "$moe" = flashinfer_mxfp4 ] || note "WARNING: /get_server_info's moe_runner_backend is ${moe:-missing}, not flashinfer_mxfp4 -- check manually"
}

note "== engine boot gates"
if [ -z "$ENGINES_IDS" ]; then
  if [ "${DRY_RUN:-0}" != 1 ]; then
    wait_engine_ready 3600
    server_info_gate
    note "engine boot gates OK"
  else
    note "DRY_RUN: skipping the engine boot gates (wait_engine_ready: /health 200, failing fast on 'unrecognized arguments' in $SERVE_LOG; ceiling 3600s; then server_info_gate: an authed /get_server_info whose body never leaves the box, only attention_backend/moe_runner_backend and the HTTP code)"
  fi
else
  # Per copy: engine_select <id> points ENGINE_URL at the copy's port, and SERVE_LOG becomes the copy's
  # own log, whose `===== RESTART fork engine <id>` marker wait_engine_ready slices from. A SERVE_LOG
  # override names the one-engine log and is not read here. Back on the legacy names (ENGINE_* and
  # SERVE_LOG) after.
  GATE_SERVE_LOG=$SERVE_LOG
  for GATE_ID in $ENGINES_IDS; do
    engine_select "$GATE_ID"
    SERVE_LOG=$ENGINE_SERVE_LOG
    if [ "${DRY_RUN:-0}" != 1 ]; then
      note "engine $GATE_ID ($ENGINE_URL, $SERVE_LOG)"
      wait_engine_ready 3600
      server_info_gate
      note "engine $GATE_ID boot gates OK"
    else
      note "DRY_RUN: skipping engine $GATE_ID's boot gates (wait_engine_ready: $ENGINE_URL/health 200, failing fast on 'unrecognized arguments' in $SERVE_LOG; ceiling 3600s; then server_info_gate on $ENGINE_URL: an authed /get_server_info whose body never leaves the box, only attention_backend/moe_runner_backend and the HTTP code)"
    fi
  done
  engine_select
  SERVE_LOG=$GATE_SERVE_LOG
fi

note "== smoke.py (on the box)"
SMOKE="$TOOLS_DIR/smoke.py"
if [ -f "$SMOKE" ] && [ -z "$ENGINES_IDS" ]; then
  run box_ssh "python3 '$TOOLS_REMOTE/smoke.py'"
elif [ -f "$SMOKE" ]; then
  # Per copy: boxenv.py's ENGINE=<id> points smoke.py at the copy's port and its sglang-args-<id>.txt.
  for SMOKE_ID in $ENGINES_IDS; do
    run box_ssh "ENGINE=$SMOKE_ID python3 '$TOOLS_REMOTE/smoke.py'"
  done
else
  note "SKIP smoke.py -- not built yet (Tools P3); run it manually once it lands"
fi

# The one line that says which box this run ended up with and which key was attached to it: everything
# after the create is addressed by $NEW_INSTANCE rather than by the box file, and the ssh key is the
# thing that made the rest of the run possible, so this is the operator's after-the-fact record of both
# (#314). It states the ATTACH's own outcome, never the happy path unconditionally (#391: both attach
# sites are allowed to fail without stopping the run, so a line that always said "attached" was the one
# an operator would quote as proof of a step that never happened). Reaching this line at all means an
# ssh to the box succeeded -- a box no key could reach dies in the ssh wait -- so the not-attached
# wording says that plainly instead of implying the box was never reachable. Skipped under DRY_RUN,
# where no instance exists to have been attached to.
if [ -n "$NEW_INSTANCE" ]; then
  if [ "$ATTACHED" = 1 ]; then
    note "== bring-up summary: instance $NEW_INSTANCE at $BOX_IP:$SSH_PORT, ssh key ${SSH_KEY_PATH}.pub attached to it, MODEL_DIR=$MODEL_DIR"
  else
    note "== bring-up summary: instance $NEW_INSTANCE at $BOX_IP:$SSH_PORT, ssh key ${SSH_KEY_PATH}.pub NOT attached by this run (the last hostctl attach ssh exited $ATTACH_RC -- see the WARNING above); ssh still worked, so the box already accepts this key from somewhere else (this create's own onstart, or an earlier attach) -- attach it yourself with: hostctl attach ssh $NEW_INSTANCE \"${SSH_KEY_PATH}.pub\", MODEL_DIR=$MODEL_DIR"
  fi
  # The bring-up reached its end: close its record, so the box file's id is never again read as an open
  # re-create (the box-c refusal turns it away from here on).
  rm -f "$(open_record "$NEW_INSTANCE")" 2>/dev/null || true
  note "open bring-up record closed: $(open_record "$NEW_INSTANCE")"
fi
note "== bring-up sequence done"
