#!/bin/bash
# dev box: render onstart.sh.tmpl into $OUT_DIR/onstart.rendered.sh (mode 600, gitignored) and
# print, ON STDOUT ONLY, the exact `hostctl create instance --onstart-cmd` fragment. offer-search.sh's
# `create` step invokes this script and captures its stdout verbatim: every diagnostic below
# (round-trip check, bash -n, byte counts) goes to stderr via lib.sh's note()/die(), and this
# script's own only stdout write is the final printf.
#
# The fragment is
#   echo <b64> | base64 -d | ( gzip -dc || python3 -c <decompressor> ) > /root/onstart-<box>.sh && bash /root/onstart-<box>.sh
# where <b64> is the render through `gzip -9n` and then base64. It is compressed because the provider's API
# caps this field at HOST_ONSTART_CMD_MAX bytes and base64 alone costs 4/3 of the render, so no
# render this template can carry would fit -- #643, and the cap gate at the end of this script is
# what keeps the next byte of template growth from failing a rent at the API instead of here. The
# decode arms (`gzip -dc`, the python fallback, and the `&&` before `bash`) are explained at the
# printf itself; the blob is opaque in the create call, so the render's own copy stays on disk
# beside it as onstart.rendered.sh.
#
# usage: build-onstart.sh <args-file> [--out-dir DIR]
#   <args-file>  a tools/args/*.args line (tuning flags only -- see relaunch.sh's header for the
#                composed `sglang serve` command this becomes part of).
#   --out-dir    where to write the rendered files (default: this tools/ directory, beside
#                lib.sh, so a fresh box's first boot and stage.sh agree on what the tool set
#                looked like at create time).
#
# No secret is rendered (#5928/#6450): the onstart's hf download reads HF_TOKEN on the box from
# /root/.hf.env (root, 0600) at run time, so the rendered script and the provider --onstart-cmd blob
# carry no token. HF_TOKEN is not needed here; when the caller has one exported (bringup.sh does,
# so this check runs against the real value), the render is refused if that value appears anywhere in
# it. The token's only use is that download of the PUBLIC upstream checkpoint: our graft never comes
# from Hugging Face (our internal artifact policy).
. "$(dirname "$0")/lib.sh"

# The checkpoint this tool set's onstart downloads, and the revision it is downloaded AT. Not
# box.env values: every box under box-c/ serves this one model, at this one revision. Bump all three
# constants here if that ever changes -- never by adding a per-box override for what is a
# model-identity fact, not a box fact.
#
# HF_REV is that checkpoint's sha of record: the revision the grafting kit's model notes
# name for this repo, and the sha HF's own
# /api/models answered when this pin was set (lastModified 2026-10-17T05:46:55Z -- the repo has had
# one revision, so the pin names what every box has downloaded anyway). It is the onstart's
# `hf download --revision`, because the unpinned form
# takes the repo's DEFAULT-BRANCH TIP: a tip that moves puts different bytes in STOCK_DIR -- whose
# `<repo name>` directory already claims this build -- and nothing in the boot notices, since the
# gate below counts FILES and apply_graft.py verifies a graft against the manifest that ARRIVES WITH
# IT. (apply_graft.py does refuse a stock shard the graft touches whose bytes applier.json's source
# hashes disagree with, so a shifted EDITED shard is caught at graft time; a shifted shard the graft
# never touches is not, and the engine serves it.) The graft is pinned too, but not here and not by
# a revision: bringup.sh copies it over ssh from the Mac (the box file's GRAFT_SRC) and refuses any
# copy whose edited_tensors.safetensors is not the box file's GRAFT_SHA256, on the Mac and again on
# the box. This pin is the stock-side guard; the graft's is a sha256 (ledger #470, #491, #2430).
HF_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
HF_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
SHARD_COUNT=48

# the provider's create API cap on --onstart-cmd, in BYTES of the string it is handed (its own field name
# for that string is `args`). A create one byte over it answers
#   400 error 400/3471: Invalid args: len(image) > 1024, or len(args) > 16384, or len(label) > 256
# and rents NOTHING -- measured on a real offer, 2026-10-17T20:53Z, after every gate in this
# script was green (24373-byte fragment: 18218-byte render, 24292 bytes of base64). The literal is
# copied from that answer, and offer-search.sh's cmd_create carries the same one as the gate in front
# of the CLI call: if the provider ever moves the number, both move -- the comment there names this one.
# WHY THE PAYLOAD IS COMPRESSED. base64 is 4/3 of what it encodes, so the largest render that could
# ever have travelled plain was 12288 bytes, and the template passed that on 2026-10-17 (14546 bytes
# of source then; 18218 rendered). Trimming the template was the other way out and was NOT taken:
# its comments are the onstart's own documentation on the box -- box-c-STOCK-PIN and box-c-HYGIENE are
# read there during a failure -- and the last four ledger items (#491, #500, #509, #510) spent their
# bytes adding exactly that prose. gzip -9n buys the margin back instead: the same render is ~7.3 KB
# compressed, ~9.7 KB of base64, ~10 KB of fragment, leaving ~6 KB under the cap for the template to
# grow into. The cost is honest and bounded: the blob is opaque in the create call, and the box
# needs gzip (or python3) to decode it -- which is why the fragment carries the fallback and the
# `&&` that turns a failed decode into "the onstart never ran" instead of "half of it ran".
HOST_ONSTART_CMD_MAX=16384

# The fallback decompressor's source, spliced into the fragment's `gzip -dc || ...` arm: no quotes
# and nothing for the box's shell to expand, reads the gzip stream on stdin, writes the script to
# stdout -- the same interface as `gzip -dc`, which is why one parenthesized group can carry both.
# It is a fallback, not the primary: `gzip` is what every image this deployment has ever run carries,
# and a stream `gzip` refused (corrupt, truncated) is one python's gzip.decompress refuses too, so
# the arm only ever earns its keep on an image that turned out to have no gzip at all.
GZIP_FALLBACK_PY='import gzip,sys;sys.stdout.buffer.write(gzip.decompress(sys.stdin.buffer.read()))'

[ $# -ge 1 ] || die "usage: build-onstart.sh <args-file> [--out-dir DIR]"
ARGS_FILE=$1; shift
OUT_DIR="$TOOLS_DIR"
if [ "${1:-}" = "--out-dir" ]; then
  # A bare `--out-dir` used to die inside `OUT_DIR=$2` with bash's own "unbound variable" under
  # `set -u` -- a parser message, never this tool's usage line. Refuse it here instead.
  [ $# -ge 2 ] || die "--out-dir needs a directory: usage: build-onstart.sh <args-file> [--out-dir DIR]"
  OUT_DIR=$2; shift 2
fi
# A leftover argument is REFUSED, never ignored: `--out_dir`/`--outdir`/`--out-dir=DIR` typos would
# otherwise render into the default tools/ directory while the operator believed it landed where
# they asked. No caller in this tool set passes a third argument, so this only ever fires on a typo.
# The message names $ARGS_FILE -- the token that was actually consumed as the args file -- so the
# flag-before-positional slip (`build-onstart.sh --out-dir DIR args-file`) reads as what it is
# instead of an unexplained "extra argument(s)" whose first word is the DIRECTORY.
[ $# -eq 0 ] || die "unexpected extra argument(s) after the args file $ARGS_FILE: $*  (usage: build-onstart.sh <args-file> [--out-dir DIR])"
[ -f "$ARGS_FILE" ] || die "no such args file: $ARGS_FILE"
[ -r "$ARGS_FILE" ] || die "args file is not readable: $ARGS_FILE"
# STOCK_DIR is optional in the box file: a box that serves the stock checkpoint directly (box-g)
# needs only MODEL_DIR, which already names the stock path. A box that serves something derived
# from stock (a repack, box-c) sets STOCK_DIR explicitly so the onstart never downloads over it.
STOCK_DIR="${STOCK_DIR:-$MODEL_DIR}"
# Every field below is baked into the render as a LITERAL, so an empty one is substituted AWAY
# silently: the placeholder is gone, the python render's leftover-token assert cannot see a value
# that is not there, and the render still "succeeds" while carrying `--model-path ""` or
# `mkdir -p ... ''` -- a box that only fails at first boot, hours into a download. A box file
# missing the line outright is already fatal (lib.sh's `set -u`); this is the `KEY=` case, which
# `set -u` lets through. STOCK_DIR is checked HERE, after its MODEL_DIR fallback, so one check
# covers both spellings of the same requirement.
for _field in "BOX=$BOX" "MODEL_DIR=$MODEL_DIR" "SERVED_MODEL=$SERVED_MODEL" \
              "STOCK_DIR=$STOCK_DIR" "TOOLS_REMOTE=$TOOLS_REMOTE" "CACHE=$CACHE" \
              "VENV=$VENV" "ENGINE_PORT=$ENGINE_PORT"; do
  [ -n "${_field#*=}" ] || die "empty ${_field%%=*} in $BOX_ENV -- this render bakes it in as a literal, and an empty value would be substituted away silently. Fill it in $BOX_ENV and re-run."
done
# ENGINES (box-c's 2xTP4, lib.sh) is never baked in: the render is create-frozen, so the onstart reads
# the STAGED box file's ENGINES at every boot and starts `relaunch.sh <id>` per engine on the ports
# lib.sh's engine_select derives (onstart.sh.tmpl's box-c-ENGINES block). The one port this render
# bakes, ENGINE_PORT, is the legacy line's: the first-boot inline launcher, which runs before anything
# is staged and knows one engine. An ENGINES this box file carries is still checked here (lib.sh
# refuses a malformed one), so a render never ships for a layout the boot would refuse.
engines_check
[ -z "$ENGINES_IDS" ] || note "ENGINES ($ENGINES_IDS): a staged boot starts relaunch.sh <id> per engine from the box file on the box; ENGINE_PORT=$ENGINE_PORT is baked only into the first-boot inline launcher (the legacy line)"

# Where the served model comes from (ledger #132), all three optional in the box file:
#   ONSTART_MODEL_SOURCE  stock (default): the onstart downloads STOCK_DIR at HF_REV and gates on its
#                         files. copied: the box serves a repack copied onto it whole (box-a), so the
#                         render carries NO download and gates the engine on the served MODEL_DIR's
#                         files plus MODEL_READY_MARKER -- the stock gate would look at a STOCK_DIR the
#                         box never holds and pull the whole checkpoint on every restart.
#   MODEL_READY_MARKER    copied only, and required there: the file the copy's own verification writes
#                         once every shard checked out (a copy pre-sizes its shards, so a file count
#                         alone proves nothing mid-copy). Refused in stock mode, where nothing reads it.
#   ONSTART_GRANT_FILE    a staged copy of a router tunnel's authorized_keys2, restored on every boot
#                         before the door and the model gate (either source).
#   HANGWATCH_CMD         the engine's hang watcher (autopool B1), started on every boot AFTER the
#                         engine launch (relaunch.sh starts the watchdog): `python3 <abs .py> ...`
#                         naming --service, --out <abs dir> and --dump-signal <value>, words of
#                         [A-Za-z0-9._/=-] only. A running copy (same script, same --out) is kept;
#                         empty = none started.
#   FREEZEGC_CMD          the recurring POST /freeze_gc (freeze-gc-timer.sh, #958), started on every boot
#                         after the engine launch: `bash <abs freeze-gc-timer.sh> ...`, words of
#                         [A-Za-z0-9._/=-] only; its --out (default /workspace/freezegc) holds the
#                         timer's flock, so a second start exits at once. Empty = none started.
# The two paths are baked in as literals, so they are shape-checked the way a box file's paths are
# expected to look: absolute, no space, no shell metacharacter.
ONSTART_MODEL_SOURCE="${ONSTART_MODEL_SOURCE:-stock}"
MODEL_READY_MARKER="${MODEL_READY_MARKER:-}"
ONSTART_GRANT_FILE="${ONSTART_GRANT_FILE:-}"
case "$ONSTART_MODEL_SOURCE" in
  stock)  [ -z "$MODEL_READY_MARKER" ] || die "MODEL_READY_MARKER=$MODEL_READY_MARKER in $BOX_ENV but ONSTART_MODEL_SOURCE is stock -- only a copied-model onstart reads it; set ONSTART_MODEL_SOURCE=copied or drop the line" ;;
  copied) [ -n "$MODEL_READY_MARKER" ] || die "ONSTART_MODEL_SOURCE=copied in $BOX_ENV needs MODEL_READY_MARKER=<file the copy's verification writes> -- without it the engine would start on a copy still in flight" ;;
  *)      die "ONSTART_MODEL_SOURCE=$ONSTART_MODEL_SOURCE in $BOX_ENV -- stock or copied only" ;;
esac
for _field in "MODEL_READY_MARKER=$MODEL_READY_MARKER" "ONSTART_GRANT_FILE=$ONSTART_GRANT_FILE"; do
  _v="${_field#*=}"
  [ -z "$_v" ] || [[ "$_v" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "${_field%%=*}=$_v in $BOX_ENV -- an absolute path of [A-Za-z0-9._/-] only (the render bakes it in as a literal)"
done
# HANGWATCH_CMD is baked in as a command line, so it is held to one shape: python3, an absolute .py,
# then plain words; --service (one instance per --out, its own BOOT grace) and --out <abs dir> (the
# running-copy guard keys on the script and that dir). HANGWATCH_PAT is the guard's pgrep pattern,
# the script's last name character bracketed so the pattern never matches a shell that spells it.
# It also needs an explicit --dump-signal: this tree's hangwatch.py defaults to none, but a box-side
# copy older than that (stage.sh never replaces one) defaults to USR2, sent to a rank py-spy cannot
# read whenever its SigCgt shows USR2 caught, and on our engines that catcher is LLVM's (Triton,
# SA_RESETHAND): it resets to SIG_DFL and re-raises, so the rank dies of the dump
# (qa/sglang-takes-20261104/merge-b300/NOTES.md, M1/M1b). `--dump-signal none` is the safe value;
# USR2 only for an engine that registers faulthandler first (that tree's site/). Every value is
# checked against what hangwatch.py's sig_num() accepts on the box (Linux): none, a signal name with
# or without SIG in any case, or a signal number -- never KILL/STOP (9/19), which it refuses -- so a
# typo fails this render, not the watcher at boot. Only names and numbers every Python 3 on Linux
# knows pass (STKFLT/16 is 3.11+); argparse takes the last of several, and each one is checked.
# hw_dump_signal_ok <value>: rc 0 when hangwatch.py would accept it.
hw_dump_signal_ok() {
  local v n
  v=$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')
  [ "$v" = NONE ] && return 0
  if [[ "$v" =~ ^[0-9]+$ ]]; then
    n=$((10#$v))
    { [ "$n" -ge 1 ] && [ "$n" -le 31 ] && [ "$n" -ne 9 ] && [ "$n" -ne 16 ] && [ "$n" -ne 19 ]; } \
      || [ "$n" -eq 34 ] || [ "$n" -eq 64 ]
    return
  fi
  case "${v#SIG}" in
    HUP|INT|QUIT|ILL|TRAP|ABRT|IOT|BUS|FPE|USR1|SEGV|USR2|PIPE|ALRM|TERM|CHLD|CLD|CONT|TSTP|TTIN|TTOU|URG|XCPU|XFSZ|VTALRM|PROF|WINCH|IO|POLL|PWR|SYS|RTMIN|RTMAX) return 0 ;;
  esac
  return 1
}
HANGWATCH_CMD="${HANGWATCH_CMD:-}"
HANGWATCH_SCRIPT="" HANGWATCH_OUT="" HANGWATCH_PAT=""
if [ -n "$HANGWATCH_CMD" ]; then
  [[ "$HANGWATCH_CMD" =~ ^python3\ (/[A-Za-z0-9._/-]+\.py)(\ [A-Za-z0-9._/=-]+)*$ ]] \
    || die "HANGWATCH_CMD=$HANGWATCH_CMD in $BOX_ENV -- 'python3 /abs/path.py' then words of [A-Za-z0-9._/=-] only (the render bakes it in as a command line)"
  HANGWATCH_SCRIPT="${BASH_REMATCH[1]}"
  [[ " $HANGWATCH_CMD " == *" --service "* ]] || die "HANGWATCH_CMD in $BOX_ENV has no --service -- the boot start needs the long-running mode (one instance per --out, its own BOOT grace)"
  [[ " $HANGWATCH_CMD " =~ \ --out\ (/[A-Za-z0-9._/-]+)\  ]] || die "HANGWATCH_CMD in $BOX_ENV has no '--out <absolute dir>' -- the running-copy guard keys on it"
  HANGWATCH_OUT="${BASH_REMATCH[1]%/}"
  _hw_sig_seen=0 _hw_prev=""
  for _w in $HANGWATCH_CMD; do
    _v=""
    if [ "$_hw_prev" = --dump-signal ]; then _v="$_w"
    elif [[ "$_w" == --dump-signal=* ]]; then _v="${_w#--dump-signal=}"
    fi
    if [ "$_hw_prev" = --dump-signal ] || [[ "$_w" == --dump-signal=* ]]; then
      _hw_sig_seen=1
      hw_dump_signal_ok "$_v" || die "HANGWATCH_CMD in $BOX_ENV has --dump-signal '$_v' -- hangwatch.py takes none, a signal name (USR2, SIGUSR2) or a signal number, never KILL/STOP; the watcher would refuse it at boot"
    fi
    _hw_prev="$_w"
  done
  [ "$_hw_prev" != --dump-signal ] || die "HANGWATCH_CMD in $BOX_ENV has no '--dump-signal <value>' (it ends in a bare --dump-signal) -- add --dump-signal none"
  [ "$_hw_sig_seen" = 1 ] || die "HANGWATCH_CMD in $BOX_ENV has no '--dump-signal <value>' -- an older box-side hangwatch.py defaults to USR2, which kills a rank whose USR2 catcher is LLVM's (SA_RESETHAND); add --dump-signal none"
  _hw_base="${HANGWATCH_SCRIPT%.py}"
  HANGWATCH_PAT="${_hw_base%?}[${_hw_base: -1}].py .*--out $HANGWATCH_OUT( |\$)"
fi
# FREEZEGC_CMD is baked in as a command line too: bash, the absolute freeze-gc-timer.sh, then plain words.
# FREEZEGC_OUT is the --out the timer will use (its own default when the command names none): the onstart
# creates it and appends the timer's nohup.log there, and the timer's flock on its timer.lock is the
# running-copy guard, so this render needs no pgrep pattern. Every word after the script is read the way the
# timer's own parser reads it -- --interval-s (an integer >= 30), --port (an integer), --out (an absolute
# dir), each with its value as the next word -- so a typo fails this render, not the timer at boot (it
# exits 2 on any other word, an `--out=DIR` included).
FREEZEGC_CMD="${FREEZEGC_CMD:-}"
FREEZEGC_SCRIPT="" FREEZEGC_OUT=""
if [ -n "$FREEZEGC_CMD" ]; then
  [[ "$FREEZEGC_CMD" =~ ^bash\ (/[A-Za-z0-9._/-]+/freeze-gc-timer\.sh)(\ [A-Za-z0-9._/=-]+)*$ ]] \
    || die "FREEZEGC_CMD=$FREEZEGC_CMD in $BOX_ENV -- 'bash /abs/path/freeze-gc-timer.sh' then words of [A-Za-z0-9._/=-] only (the render bakes it in as a command line)"
  FREEZEGC_SCRIPT="${BASH_REMATCH[1]}"
  [[ " $FREEZEGC_CMD " != *" --once "* ]] || die "FREEZEGC_CMD in $BOX_ENV has --once -- the boot start needs the recurring timer (one POST, then exit, freezes nothing the engine allocates afterwards)"
  FREEZEGC_OUT=/workspace/freezegc
  set -- ${FREEZEGC_CMD#bash $FREEZEGC_SCRIPT}
  while [ $# -gt 0 ]; do
    case "$1" in
      --out|--interval-s|--port) [ $# -ge 2 ] || die "FREEZEGC_CMD in $BOX_ENV ends in a bare $1 -- give it its value" ;;
      *) die "FREEZEGC_CMD in $BOX_ENV has '$1' -- freeze-gc-timer.sh takes --interval-s N, --port P and --out DIR (each value its own word) and exits 2 on anything else" ;;
    esac
    case "$1" in
      --out) [[ "$2" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "FREEZEGC_CMD in $BOX_ENV has --out '$2' -- an absolute dir of [A-Za-z0-9._/-] only"; FREEZEGC_OUT="${2%/}" ;;
      --interval-s) [[ "$2" =~ ^[0-9]+$ ]] && [ "$((10#$2))" -ge 30 ] || die "FREEZEGC_CMD in $BOX_ENV has --interval-s '$2' -- freeze-gc-timer.sh takes an integer >= 30 and refuses anything else at boot" ;;
      --port) [[ "$2" =~ ^[0-9]+$ ]] || die "FREEZEGC_CMD in $BOX_ENV has --port '$2' -- an integer" ;;
    esac
    shift 2
  done
fi

# grep's status is READ, never discarded. Exit 1 (nothing selected) is the empty or comment-only
# args file, and under `set -e` that status used to end the script before the die below could say
# why -- in total silence, rc=1 with no stdout and no message; the emptiness test IS that
# diagnostic, so 1 is tolerated here. Any other status is grep's OWN failure -- the file vanishing
# between the `-f`/`-r` gates above, or an I/O error -- and a `|| true` flattened it into that same
# silence, so the die named the file's contents ("has no flags after stripping comments") when the
# truth was that the file could not be read at all. Both now say what they mean.
grep_rc=0
ARGS=$(grep -v '^#' "$ARGS_FILE") || grep_rc=$?
case "$grep_rc" in
  0) ;;
  1) ;;   # nothing selected: the emptiness die on the next line is its diagnostic, not this one
  *) die "could not read $ARGS_FILE (grep exit $grep_rc) -- it was readable at the gates above, so it vanished or hit an I/O error since; re-run" ;;
esac
[ -n "$ARGS" ] || die "$ARGS_FILE has no flags after stripping comments"

# Two flags are refused in an args file, because on a RE-CREATED box's first boot this file IS the
# serve argv: onstart.sh.tmpl writes __ARGS__ into $TOOLS_REMOTE/sglang-args.txt, the inline launcher
# reads that file into A, and `exec sglang serve "${A[@]}"` puts every word of it on the process
# list. The key guard at the end of this script reads the exec LINE, where these words never appear,
# so it cannot be the gate for them -- these flags arrive through the array, not through that line.
#   --api-key  the leak #2739's judge note on this file names: the engine key on the first-boot argv,
#              where pgrep/ps read it for the whole boot. args/check.sh refuses it in a tools/args
#              file too, but this gate is on the file THIS render was handed (the #500 rule: assert
#              what ships, never what a sister tool would have caught), which may sit outside
#              tools/args and is never run through check.sh by this script.
#   --config   the body passes /etc/sglang-api-key.yaml as --config, and sglang's own merger refuses
#              two of them outright ("Multiple config files specified! Only one allowed."), so a
#              second one here is a first boot that dies on a traceback instead of serving. The flag
#              was harmless in an args file before the key moved into that file; now it is not.
# Matched on $ARGS -- the exact comment-stripped view this render inlines (the `grep -v '^#'` above),
# never the raw file -- so a header comment spelling a flag is not read as one: v4-b200.args names
# `--api-key` in prose at line 19 and renders green. An INDENTED `#` line is a different matter and
# deliberately not exempted: this renderer keeps it (only `^#` is a comment here), so it really does
# travel into the box's sglang-args.txt and onto the exec's argv, and a flag on it is refused as the
# flag it will be there -- args/check.sh's strip is the one that would have called it prose.
for _flag in --api-key --config; do
  if printf '%s\n' "$ARGS" | grep -qE -- "(^|[[:space:]])${_flag}([[:space:]]|=|\$)"; then
    die "$ARGS_FILE carries $_flag -- on a re-created box's first boot that file is the serve argv (the inline launcher reads it into A and execs sglang serve \"\${A[@]}\"), so it cannot hold a flag this render supplies itself: --api-key puts the engine key on the process list, exactly what the key file exists to prevent (#2739), and --config collides with the body's own and sglang refuses two (\"Multiple config files specified! Only one allowed.\"). Drop it from $ARGS_FILE and re-run"
  fi
done

# An empty .pub (or none at all) would render an authorized_keys line that grants nothing, on a box
# nothing can ssh into; `-s` covers both, and the "no public key" wording below is what an operator
# needs to read either way.
PUB_PATH="${SSH_KEY_PATH}.pub"
[ -s "$PUB_PATH" ] || die "no public key at $PUB_PATH"
PUB=$(cat "$PUB_PATH")

# HF_REV is spliced UNQUOTED into the onstart's `hf download --revision <value>` line, so its shape
# is gated here: a commit sha, a tag or a ref is all that is ever wanted. A leading dash would arrive in hf's own argv as another option,
# and a space, quote or semicolon would split or extend the onstart's line on the box -- there is no
# quoting to survive that, because the template line the value lands in is shell text, not an argv.
# It is a constant in THIS file, so the only value that can trip this is a bad hand-edit; the cost of
# missing it is a create-frozen blob whose download dies hours into a boot.
case "$HF_REV" in
  '' | -* | *[!A-Za-z0-9._/-]*)
    die "HF_REV='$HF_REV' is not a plain revision -- the rendered onstart fetches $HF_REPO with it as \`hf download --revision <value>\`, so it may be a commit sha, a tag or a ref (letters, digits, dot, dash, underscore, slash) and nothing else; fix the HF_REV constant in build-onstart.sh" ;;
esac

# gzip is what compresses the payload (HOST_ONSTART_CMD_MAX: base64 alone cannot fit), and what the check
# below refuses is the RENDER, never a box: the gzip it needs is this renderer's own, for the `gzip -9n`
# below, while a served box with none is the fallback's case, not this one's (`gzip -dc || python3 -c
# <decompressor>`, GZIP_FALLBACK_PY). It refuses BEFORE anything is written: `gzip -9n < render > blob` on
# a missing gzip would fail the pipeline with status 127, which under `set -e`+pipefail ends this script
# with no stdout and no message of its own, and with a `|| true` instead it would ship an EMPTY blob -- a
# create that succeeds on a blob that decodes to nothing and, by the fragment's own `&&`, never runs.
command -v gzip >/dev/null 2>&1 || die "no gzip on PATH -- this renderer ships the onstart as base64 of its gzip stream (the provider's --onstart-cmd cap is $HOST_ONSTART_CMD_MAX bytes, and the uncompressed base64 alone is over it), so it cannot render without gzip; install gzip and re-run"

mkdir -p "$OUT_DIR"
umask 077

# The relaunch body inlined into the onstart: a SELF-CONTAINED equivalent of relaunch.sh with
# every box.env value baked in as a literal, EXCEPT MODEL_DIR/SERVED_MODEL -- those two are read
# at runtime from $TOOLS_REMOTE/serve-model.env when stage.sh has written one, falling back to
# this render's own box values only when it has not (see onstart.sh.tmpl's header). Those literals are
# only the fallback for a serve-model.env-less box (a genuinely fresh create, before stage.sh ever ran),
# never authoritative after that: the onstart is create-frozen, so only a re-render changes them, while
# serve-model.env follows a stage.sh re-run -- what stops a container restart from resurrecting whatever
# this onstart was last rendered to serve. (Kept up here, not in the body: the provider's onstart cap.) Everything
# else here cannot source lib.sh the way the staged relaunch.sh does -- lib.sh and boxes/$BOX.env
# are not on the box yet at first boot, stage.sh puts them there afterwards -- so this is
# generated fresh here, not read out of relaunch.sh.
# `read -r -d ''` rather than `$(cat <<EOF ...)`: a `case` statement inside a heredoc nested
# inside a `$(...)` command substitution hits a real bash parser limitation (confirmed: bash -n
# reports a spurious `unexpected token ';;'` even though the heredoc is never executed as code by
# the outer script) -- `read -d ''` takes the same unquoted-heredoc expansion without that nesting.
read -r -d '' RELAUNCH_BODY <<EOF_BODY || true
KEY=\$(grep -a '^VLLM_API_KEY=' /etc/environment | head -1 | cut -d= -f2- | tr -d '"')
[ -n "\$KEY" ] || { echo NO_KEY; exit 3; }
# The key never rides this argv (#2739): it goes into a root-only file, and sglang's own --config
# merges it in as --api-key before argparse runs, so the engine is authenticated exactly as before
# while pgrep/ps (and every process-list record) show only the path. The file is 0600, one line
# \`api-key: "<key>"\`, temp-then-rename: the same path and shape the staged relaunch.sh writes, so
# the first boot and every staged launch after it hold the key in that one file and no argv. A key
# carrying a character outside this set is refused rather than escaped; every key this tool set
# mints (openssl rand -hex 16) is hex.
KEY_FILE=/etc/sglang-api-key.yaml
case "\$KEY" in
  *[!A-Za-z0-9._~+/=-]*) echo KEY_SHAPE; exit 6 ;;
esac
rm -f "\$KEY_FILE.tmp.\$\$"
( umask 077; printf 'api-key: "%s"\n' "\$KEY" > "\$KEY_FILE.tmp.\$\$" ) || { echo NO_KEYFILE; exit 7; }
mv -f "\$KEY_FILE.tmp.\$\$" "\$KEY_FILE" || { echo NO_KEYFILE; exit 7; }
chmod 600 "\$KEY_FILE" || { echo NO_KEYFILE; exit 7; }
A=\$(grep -v '^#' "$TOOLS_REMOTE/sglang-args.txt")
[ -n "\$A" ] || { echo NO_ARGS; exit 4; }
command -v sglang >/dev/null 2>&1 || export PATH="$VENV/bin:\$PATH"
command -v sglang >/dev/null 2>&1 || { echo NO_SGLANG_ON_PATH; exit 5; }
export XDG_CACHE_HOME="$CACHE" HF_HUB_OFFLINE=1 SGLANG_DISABLE_USAGE_STATS=1 DO_NOT_TRACK=1
export PATH="$VENV/bin:\$PATH"
if [ -f "$TOOLS_REMOTE/sglang-env.txt" ]; then
  while IFS= read -r line; do
    case "\$line" in ''|'#'*) ;; *) export "\$line" ;; esac
  done < "$TOOLS_REMOTE/sglang-env.txt"
fi
# The newline before \`)\` is the one \`\$(grep ...)\` stripped: without it a trailing \` # ...\` comment on
# the last kept line comments out the \`)\` too (relaunch.sh's own eval, tests/relaunch-args-comments.test.sh).
eval "A=(\$A"\$'\n'")"
# No key in the environ the watchdog and the engine inherit (#3003, as relaunch.sh): the engine reads
# \$KEY_FILE, the watchdog reads /etc/environment itself. After the sglang-env.txt loop, so a stray
# line there cannot put one back; this copy is also relaunch.sh until stage.sh replaces it, and a
# restart.sh ssh session exports the key through pam_env.
unset VLLM_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN CF_DNS_TOKEN KEY
# No prod traffic at rest on a rented box (#1300): RLIMIT_CORE 0 and relaunch.sh's own deploy gate (its
# rest-gate block, inlined below by this renderer) over the args, the environment and the core limit,
# before the watchdog or the engine starts. A refusal names each flag and exits 8: no engine starts. So
# does a gate that exits non-zero for any reason (#1348, as relaunch.sh): its status is checked.
ulimit -c 0 2>/dev/null || true
__REST_GATE__
RG_RC=0; RG_OUT=\$(RG_E=\$(rg_env_now) && rest_gate "$TOOLS_REMOTE/sglang-args.txt" "\$RG_E" "\${A[@]}" && rg_core) || RG_RC=\$?
[ "\$RG_RC" = 0 ] || RG_OUT="gate exit \$RG_RC: it could not finish, nothing is proved clean \$RG_OUT"
[ -z "\$RG_OUT" ] || { echo "REFUSED (no prod traffic at rest) -- no engine started:"; echo "\$RG_OUT"; exit 8; }
# serve-model.env (stage.sh's, or an operator's) wins over these two render-time literals.
SM_MODEL_DIR="$MODEL_DIR"; SM_SERVED_MODEL="$SERVED_MODEL"
if [ -f "$TOOLS_REMOTE/serve-model.env" ]; then
  while IFS= read -r line; do
    case "\$line" in
      MODEL_DIR=*) SM_MODEL_DIR="\${line#MODEL_DIR=}" ;;
      SERVED_MODEL=*) SM_SERVED_MODEL="\${line#SERVED_MODEL=}" ;;
    esac
  done < "$TOOLS_REMOTE/serve-model.env"
fi
echo "===== sglang serve \$(date -u) box=$BOX (first-boot inline launcher) model=\$SM_MODEL_DIR served=\$SM_SERVED_MODEL ====="
if [ -f "$TOOLS_REMOTE/watchdog.sh" ]; then
  pgrep -f "$TOOLS_REMOTE/watchdo[g].sh" >/dev/null || ( nohup setsid bash "$TOOLS_REMOTE/watchdog.sh" >> "/workspace/$BOX-watchdog.log" 2>&1 < /dev/null & )
fi
exec sglang serve "\${A[@]}" --model-path "\$SM_MODEL_DIR" --served-model-name "\$SM_SERVED_MODEL" --host 127.0.0.1 --port "$ENGINE_PORT" --config "\$KEY_FILE"
EOF_BODY
# relaunch.sh's rest-gate block, verbatim but for its comment lines and indentation, in place of __REST_GATE__ (split
# and joined, never a ${//} substitution: bash 5.2's patsub_replacement would read the block's `&`s).
# Indentation dropped too (the block has no heredoc): the provider's 16384-byte onstart cap is tight.
REST_GATE_BLOCK=$(awk '/^# >>> rest-gate/ { on = 1; next } /^# <<< rest-gate/ { on = 0 } on && !/^[[:space:]]*#/ { sub(/^[[:space:]]+/, ""); print }' "$TOOLS_DIR/relaunch.sh")
grep -q '^rest_gate() {' <<< "$REST_GATE_BLOCK" && grep -q '^rg_core() {' <<< "$REST_GATE_BLOCK" \
  || die "relaunch.sh has no complete '# >>> rest-gate' ... '# <<< rest-gate' block -- the first-boot launcher would start an engine ungated (#1300)"
[[ "$RELAUNCH_BODY" == *__REST_GATE__* ]] || die "the inline launcher body lost its __REST_GATE__ line"
RELAUNCH_BODY="${RELAUNCH_BODY%%__REST_GATE__*}$REST_GATE_BLOCK${RELAUNCH_BODY#*__REST_GATE__}"

python3 - "$TOOLS_DIR/onstart.sh.tmpl" "$OUT_DIR/onstart.rendered.sh" "$PUB" "$ARGS" "$RELAUNCH_BODY" \
          "$BOX" "$STOCK_DIR" "$TOOLS_REMOTE" "$CACHE" "$SHARD_COUNT" "$HF_REPO" "$HF_REV" \
          "$ONSTART_MODEL_SOURCE" "$MODEL_DIR" "$MODEL_READY_MARKER" "$ONSTART_GRANT_FILE" \
          "$HANGWATCH_CMD" "$HANGWATCH_SCRIPT" "$HANGWATCH_OUT" "$HANGWATCH_PAT" \
          "$FREEZEGC_CMD" "$FREEZEGC_SCRIPT" "$FREEZEGC_OUT" <<'PY'
import os, pathlib, re, sys
(tpl_path, out_path, pub, args, relaunch,
 box, stock_dir, tools_remote, cache, shard_count, hf_repo, hf_rev,
 source, model_dir, ready_marker, grant_file,
 hw_cmd, hw_script, hw_out, hw_pat,
 fg_cmd, fg_script, fg_out) = sys.argv[1:24]
# The template's sections first, on the TEMPLATE (so nothing substituted in below -- an args line, the
# launcher body -- can ever read as a section line): a line that is exactly `#@<name>{` opens one,
# exactly `#@}` closes it, and its lines are kept only when <name> is on for this box. Unbalanced or
# unknown sections fail the render.
keep = {'stock': source == 'stock', 'copied': source == 'copied', 'grant': bool(grant_file),
        'hangwatch': bool(hw_cmd), 'freezegc': bool(fg_cmd)}
lines, stack = [], []
for n, line in enumerate(pathlib.Path(tpl_path).read_text().splitlines(keepends=True), 1):
    m = re.fullmatch(r'#@(\w+)\{\n?', line)
    if m:
        assert m.group(1) in keep, f"template line {n}: unknown section {m.group(1)!r}"
        stack.append(keep[m.group(1)])
        continue
    if line.rstrip('\n') == '#@}':
        assert stack, f"template line {n}: #@}} closes no section"
        stack.pop()
        continue
    if all(stack):
        lines.append(line)
assert not stack, "template: a section is never closed"
tpl = ''.join(lines)
# A placeholder only its own section may carry: outside it, an off value would substitute to ''.
for token, on in (('__READY_MARKER__', keep['copied']), ('__GRANT_FILE__', keep['grant']),
                  ('__HANGWATCH_CMD__', keep['hangwatch']), ('__HANGWATCH_SCRIPT__', keep['hangwatch']),
                  ('__HANGWATCH_OUT__', keep['hangwatch']), ('__HANGWATCH_PAT__', keep['hangwatch']),
                  ('__FREEZEGC_CMD__', keep['freezegc']), ('__FREEZEGC_SCRIPT__', keep['freezegc']),
                  ('__FREEZEGC_OUT__', keep['freezegc'])):
    assert on or token not in tpl, f"{token} outside its section in the template"
out = (tpl.replace('__SSH_PUBKEY__', pub)
          .replace('__ARGS__', args)
          .replace('__RELAUNCH__', relaunch)
          .replace('__HF_REPO__', hf_repo)
          .replace('__HF_REV__', hf_rev)
          .replace('__SHARD_COUNT__', shard_count)
          .replace('__BOX__', box)
          .replace('__STOCK_DIR__', stock_dir)
          .replace('__MODEL_DIR__', model_dir)
          .replace('__READY_MARKER__', ready_marker)
          .replace('__GRANT_FILE__', grant_file)
          .replace('__HANGWATCH_CMD__', hw_cmd)
          .replace('__HANGWATCH_SCRIPT__', hw_script)
          .replace('__HANGWATCH_OUT__', hw_out)
          .replace('__HANGWATCH_PAT__', hw_pat)
          .replace('__FREEZEGC_CMD__', fg_cmd)
          .replace('__FREEZEGC_SCRIPT__', fg_script)
          .replace('__FREEZEGC_OUT__', fg_out)
          .replace('__TOOLS_REMOTE__', tools_remote)
          .replace('__CACHE__', cache))
leftover = re.findall(r'__[A-Z_]+__', out)
assert not leftover, f"leftover placeholder tokens: {leftover}"
hf_token = os.environ.get('HF_TOKEN', '')
assert not hf_token or hf_token not in out, "the exported HF_TOKEN value appears in the render -- the onstart reads it from /root/.hf.env on the box, never inline"
pathlib.Path(out_path).write_text(out)
PY

chmod 600 "$OUT_DIR/onstart.rendered.sh"
# The blob is the render through gzip -9n, THEN base64 (HOST_ONSTART_CMD_MAX). -n keeps gzip's
# name/mtime out of the header, so one tree renders to one blob byte for byte -- the same property
# the base64-only blob had, and what makes "the create carries THIS render" checkable at all. The
# artifact keeps its name: it IS a base64 blob, of a gzip stream, and nothing in the tree but this
# script and tests/build-onstart.test.sh reads it (stage.sh excludes it from the rsync).
B64="$OUT_DIR/onstart.rendered.b64"
gzip -9n < "$OUT_DIR/onstart.rendered.sh" | base64 | tr -d '\n' > "$B64"
chmod 600 "$B64"
# Round trip through BOTH stages, in the order the box decodes them. This is the check that catches
# the payload that would otherwise be silently un-bootable: an empty or truncated blob (a gzip that
# was not there, a truncating filesystem) whose only other symptom is a box that rents, bills, and
# runs no onstart at all -- the fragment's `&&` means a failed decode runs NOTHING, so this is the
# last place the failure can be seen for free.
base64 -d < "$B64" | gzip -dc | cmp - "$OUT_DIR/onstart.rendered.sh" || die "base64/gzip round-trip FAILED -- the blob the create would carry does not decode back to this render under the fragment's own recipe (base64 -d | gzip -dc); fix the payload build above before any create uses it (an un-decodable blob is a rent that boots and runs nothing)"
note "round-trip OK (base64 -d | gzip -dc reproduces the render)"
# The blob is the render byte for byte (round-trip above), so checking the render checks the blob.
if [ -n "${HF_TOKEN:-}" ] && grep -qF -- "$HF_TOKEN" "$OUT_DIR/onstart.rendered.sh"; then
  die "the exported HF_TOKEN value appears in the render (and so in the --onstart-cmd blob) -- never inline it"
fi
# HF_TOKEN is fetched on the box from /root/.hf.env (root, 0600) -- #5928's whole point -- so the
# render must READ that file, not merely mention it. onstart.sh.tmpl's box-c-HYGIENE comment (118)
# spells `/root/.hf.env` in prose, so a whole-file grep is satisfied by that comment alone and stays
# satisfied after an edit drops the read from the download line: the same defect as the pin check
# below, in the check beside it (#500's judge note on #491). The pattern is anchored to a COMMAND
# line -- one whose first non-blank character is not `#` -- which is what a read looks like. It does
# NOT require the read to sit on the download line itself: reading the token one line above and
# handing it to the download is the same fact, and this check is about where the token comes from,
# not where it is spelled.
if [ "$ONSTART_MODEL_SOURCE" = stock ]; then
grep -qE '^[[:space:]]*[^[:space:]#].*/root/[.]hf[.]env' "$OUT_DIR/onstart.rendered.sh" || die "rendered onstart does not read /root/.hf.env on any command line -- fix onstart.sh.tmpl (the onstart takes HF_TOKEN from the box's own root-only file at run time; a comment naming it is not a read, and the download would run tokenless)"
note "no inlined HF token OK (hf download reads /root/.hf.env on the box)"
# The pin is asserted on the RENDER, not assumed from the template: an unpinned `hf download` takes
# the repo's default-branch tip, and STOCK_DIR's `<repo name>` directory already claims this build,
# so a template that dropped the flag would be a silent create-frozen drift -- and the blob the provider
# stores is this render byte for byte (round-trip above), so checking the render checks the blob.
#
# ...on the LINE THAT DOWNLOADS, not anywhere in the render (#500's judge note on #491). This check
# used to be `grep -qF -- "--revision $HF_REV"` over the whole file, and onstart.sh.tmpl's own
# retry-loop comment (119-120) spells `--revision __HF_REV__`, which substitutes to the same text:
# an edit that dropped the flag from the download line and left that comment passed the guard and
# shipped a tip-pinned blob -- the drift this check exists to catch, passing on the prose alone. So
# the render's comments are dropped first and every command line that fetches $HF_REPO must name
# `--revision $HF_REV`. Flag ORDER is deliberately not asserted -- hf takes its options in any order
# (that retry comment records why `--revision` is written where it is) -- so a check demanding one
# exact argv spelling would refuse a correctly pinned render the day someone reordered the flags. A
# render that fetches $HF_REPO on no command line at all is refused too: "no unpinned download line"
# must not be true merely because there is no download line, and a template that stopped downloading
# would otherwise fail open in a new way. The `|| true` on each grep below is grep's exit 1 for
# "nothing selected" under `set -e` (the status handling at the top of this file) -- an empty result
# is the case the next line refuses, in its own words.
pin_lines=$(grep -nvE '^[[:space:]]*#' "$OUT_DIR/onstart.rendered.sh" | grep -F "hf download $HF_REPO" || true)
[ -n "$pin_lines" ] || die "rendered onstart runs no \`hf download $HF_REPO\` on a command line -- fix onstart.sh.tmpl (downloading that checkpoint into STOCK_DIR is this onstart's first job, and the blob the provider stores is this render byte for byte, so a render that downloads nothing is not a pinned one)"
pin_unpinned=$(printf '%s\n' "$pin_lines" | grep -vF -- "--revision $HF_REV" || true)
[ -z "$pin_unpinned" ] || die "rendered onstart does not pin the checkpoint download to HF_REV=$HF_REV -- fix onstart.sh.tmpl (every command line that runs \`hf download $HF_REPO\` must name --revision $HF_REV: a comment spelling the flag is not a pin, and unpinned the download takes $HF_REPO's default-branch tip, landing a different revision in STOCK_DIR under this build's name). The command line(s) with no pin: $pin_unpinned"
note "checkpoint download pinned OK (hf download $HF_REPO ... --revision $HF_REV on the download line, the revision $HF_REPO's name claims)"
else
# COPIED: the render must download NOTHING -- no `hf download` and no hub install on any command line
# (comments may still name them) -- and must gate the engine on the marker: a command line that tests
# `-e "<MODEL_READY_MARKER>"` together with the file-count gate.
copied_cmds=$(grep -vE '^[[:space:]]*#' "$OUT_DIR/onstart.rendered.sh" || true)
! printf '%s\n' "$copied_cmds" | grep -qE 'hf download|pip install[^|;&]*huggingface_hub' || die "rendered copied-model onstart still downloads (an \`hf download\` or hub install on a command line) -- fix onstart.sh.tmpl (ONSTART_MODEL_SOURCE=copied keeps the download inside #@stock{ ... #@})"
printf '%s\n' "$copied_cmds" | grep -qF -- "if ! complete || [ ! -e \"$MODEL_READY_MARKER\" ]" || die "rendered copied-model onstart has no gate on MODEL_READY_MARKER=$MODEL_READY_MARKER -- fix onstart.sh.tmpl (the engine must wait for the copy's verification, not just a file count)"
note "copied model OK (no download on any command line; the engine waits for $MODEL_READY_MARKER and the file-count gate)"
fi
# ORDER: the door (Caddy) and, when set, the tunnel grant come up BEFORE the model gate -- the first
# command line that opens `if ! complete` in either source -- so a restart that cannot serve (a download
# that fails, a copy not verified) still has its door and tunnel. Line numbers of command lines only.
gate_ln=$(grep -nvE '^[[:space:]]*#' "$OUT_DIR/onstart.rendered.sh" | grep -m1 -E '^[0-9]+:[[:space:]]*if ! complete' | cut -d: -f1 || true)
[ -n "$gate_ln" ] || die "rendered onstart has no model gate (\`if ! complete\` on a command line) -- fix onstart.sh.tmpl"
door_ln=$(grep -nvE '^[[:space:]]*#' "$OUT_DIR/onstart.rendered.sh" | grep -m1 -F "bash $TOOLS_REMOTE/caddy-relaunch.sh" | cut -d: -f1 || true)
[ -n "$door_ln" ] && [ "$door_ln" -lt "$gate_ln" ] || die "rendered onstart does not start Caddy (bash $TOOLS_REMOTE/caddy-relaunch.sh) before its model gate (line $gate_ln; Caddy: ${door_ln:-none}) -- fix onstart.sh.tmpl"
if [ -n "$ONSTART_GRANT_FILE" ]; then
  grant_ln=$(grep -nvE '^[[:space:]]*#' "$OUT_DIR/onstart.rendered.sh" | grep -m1 -F "cp $ONSTART_GRANT_FILE /root/.ssh/authorized_keys2" | cut -d: -f1 || true)
  [ -n "$grant_ln" ] && [ "$grant_ln" -lt "$gate_ln" ] || die "rendered onstart does not restore authorized_keys2 from ONSTART_GRANT_FILE=$ONSTART_GRANT_FILE before its model gate (line $gate_ln; restore: ${grant_ln:-none}) -- fix onstart.sh.tmpl"
fi
note "boot order OK (${ONSTART_GRANT_FILE:+tunnel grant line $grant_ln, }Caddy line $door_ln, before the model gate at line $gate_ln; source $ONSTART_MODEL_SOURCE)"
# HANGWATCH (autopool B1): when set, the watcher starts on a command line AFTER the last engine launch
# (relaunch.sh, which starts the watchdog) -- engine, watchdog, then the watcher of both.
if [ -n "$HANGWATCH_CMD" ]; then
  launch_ln=$(grep -nvE '^[[:space:]]*#' "$OUT_DIR/onstart.rendered.sh" | grep -F "nohup setsid bash $TOOLS_REMOTE/relaunch.sh" | tail -1 | cut -d: -f1 || true)
  hw_ln=$(grep -nvE '^[[:space:]]*#' "$OUT_DIR/onstart.rendered.sh" | grep -m1 -F "nohup setsid $HANGWATCH_CMD " | cut -d: -f1 || true)
  [ -n "$launch_ln" ] && [ -n "$hw_ln" ] && [ "$hw_ln" -gt "$launch_ln" ] || die "rendered onstart does not start HANGWATCH_CMD after its last engine launch (relaunch.sh line ${launch_ln:-none}; hangwatch: ${hw_ln:-none}) -- fix onstart.sh.tmpl"
  note "hangwatch OK (line $hw_ln, after the last relaunch.sh launch at line $launch_ln; guard: pgrep -f '$HANGWATCH_PAT')"
else
  note "hangwatch: none (no HANGWATCH_CMD in $BOX_ENV)"
fi
# FREEZEGC (#958): when set, the timer starts on a command line AFTER the last engine launch, like the watcher.
if [ -n "$FREEZEGC_CMD" ]; then
  launch_ln=$(grep -nvE '^[[:space:]]*#' "$OUT_DIR/onstart.rendered.sh" | grep -F "nohup setsid bash $TOOLS_REMOTE/relaunch.sh" | tail -1 | cut -d: -f1 || true)
  fg_ln=$(grep -nvE '^[[:space:]]*#' "$OUT_DIR/onstart.rendered.sh" | grep -m1 -F "nohup setsid $FREEZEGC_CMD >> $FREEZEGC_OUT/nohup.log" | cut -d: -f1 || true)
  [ -n "$launch_ln" ] && [ -n "$fg_ln" ] && [ "$fg_ln" -gt "$launch_ln" ] || die "rendered onstart does not start FREEZEGC_CMD after its last engine launch (relaunch.sh line ${launch_ln:-none}; freezegc: ${fg_ln:-none}) -- fix onstart.sh.tmpl"
  note "freezegc OK (line $fg_ln, after the last relaunch.sh launch at line $launch_ln; out $FREEZEGC_OUT, one copy by its timer.lock)"
else
  note "freezegc: none (no FREEZEGC_CMD in $BOX_ENV)"
fi
bash -n "$OUT_DIR/onstart.rendered.sh" || die "rendered onstart has a syntax error -- fix onstart.sh.tmpl"
note "bash -n OK"
# The api key is off the first-boot argv (#2739): the render's exec line must name the key FILE
# (`--config "$KEY_FILE"`) and no `--api-key`, or a re-created box shows the key to `pgrep`/`ps`
# until its first `relaunch.sh`. Asserted on the COMMAND line, not the render as a whole -- the
# body's own comment above spells `--api-key` in prose (what the merger makes of the file), and a
# whole-file needle would be satisfied by that comment alone, the defect #500's note on #491
# records. `--api-key` anywhere on the exec line is refused: the file and the flag commute, but
# the flag is exactly the leak this line exists to remove. The args-file gate at the top of this
# script is the other half of the same assertion: a key spelled in an args file never appears on
# this line (it arrives through "${A[@]}"), so the exec-line needle cannot see it.
served_line=$(grep -vE '^[[:space:]]*#' "$OUT_DIR/onstart.rendered.sh" | grep -F 'exec sglang serve' || true)
printf '%s\n' "$served_line" | grep -qF -- '--config "$KEY_FILE"' || die "rendered onstart's exec line names no --config \"\$KEY_FILE\" -- fix build-onstart.sh's RELAUNCH_BODY (the API key rides a 0600 file read from /etc/environment at boot, never the argv: the first-boot launcher must write that file and hand the engine its path). The exec line: ${served_line:-<none>}"
# The flag alone is not the guarantee: `--config` has to name a file the SAME body writes, or the
# engine starts against a path that does not exist -- a named death at first boot, after the rent is
# paid, for a render every gate above would call green. So the comment-stripped render must carry the
# writer's own redirect (`> "$KEY_FILE.tmp.$$"`): the assignment alone is a variable, and the exec
# alone is a path; the redirect is the fact that the key reaches the file the exec names.
render_cmds=$(grep -vE '^[[:space:]]*#' "$OUT_DIR/onstart.rendered.sh" || true)
printf '%s\n' "$render_cmds" | grep -qF -- '> "$KEY_FILE.tmp.$$"' || die "rendered onstart's exec line passes --config \"\$KEY_FILE\" but no command line writes that file -- fix build-onstart.sh's RELAUNCH_BODY (the engine would start against a path that does not exist: the key has to be written into it, at 0600, before the exec, or a box that rents and downloads serves nothing)"
if printf '%s\n' "$served_line" | grep -qF -- '--api-key'; then
  die "rendered onstart's exec line still carries --api-key -- fix build-onstart.sh's RELAUNCH_BODY (that puts the key on a re-created box's first-boot argv, where pgrep/ps read it; pass --config \"\$KEY_FILE\" instead). The exec line: $served_line"
fi
note "api key off the first-boot argv OK (exec line names --config \"\$KEY_FILE\", no --api-key; the body writes that file before the exec)"
note "rendered: $(wc -c < "$OUT_DIR/onstart.rendered.sh") bytes, gzip+base64: $(wc -c < "$B64") bytes, md5 $(md5 -q "$OUT_DIR/onstart.rendered.sh" 2>/dev/null || md5sum "$OUT_DIR/onstart.rendered.sh" | cut -d' ' -f1)"

# STDOUT ONLY from here: the hostctl --onstart-cmd fragment (offer-search.sh captures this).
#
# Every piece of the shape, and what each one is holding up:
#   echo <b64> | base64 -d          the blob above, back to gzip bytes on the box. `echo` unquoted
#                                   is safe for base64's own alphabet (no glob, no space, no
#                                   backslash), which is why this needs no quoting -- and why any
#                                   other alphabet would.
#   ( gzip -dc || python3 -c ... )  one stdin, two decompressors. `gzip -dc` is the normal arm; the
#                                   python one covers an image without gzip, whose failure mode is
#                                   silent and expensive (a box rented and billed with an empty
#                                   onstart), and python3 is the one interpreter an sglang image is
#                                   certain to have. A corrupt or truncated stream fails BOTH, and
#                                   the group's status is then the fallback's non-zero one -- which
#                                   is what the `&&` below reads. The fallback only ever sees a
#                                   COMPLETE stream: a missing gzip consumes nothing, so the pipe's
#                                   bytes are all still there for python.
#   > /root/onstart-<box>.sh        the path the uncompressed fragment wrote to, unchanged: every
#                                   recipe and habit that reads/cats/diffs the onstart on a box
#                                   keeps working, and reads the script, not a blob.
#   && bash /root/onstart-<box>.sh  RUN ONLY AFTER THE DECODE SUCCEEDED. A `;` here would run
#                                   whatever a failed decode left behind -- half an onstart (sshd
#                                   fixed, checkpoint never downloaded) or an empty file -- and
#                                   nothing in the boot log would say so.
ONSTART_CMD=$(printf 'echo %s | base64 -d | ( gzip -dc || python3 -c "%s" ) > /root/onstart-%s.sh && bash /root/onstart-%s.sh' \
                     "$(cat "$B64")" "$GZIP_FALLBACK_PY" "$BOX" "$BOX")
# THE CAP GATE (#643). the provider answers 400 and rents nothing when the field is longer than
# HOST_ONSTART_CMD_MAX, and that refusal happens at the API, after an offer was picked -- the state
# the 2026-10-17 rent died in, with all four content gates above green (they read the render's
# CONTENT: the token's absence, the revision pin, the syntax; none of them ever looked at length).
# The measurement is of the exact string about to be printed, WITHOUT the trailing newline: both
# callers in the tree capture it with `$( )`, which strips it, and the API counts the string it is
# handed, not this printf. The die is deliberate over a warning: the fragment is already compressed,
# so exceeding the cap means the template itself has to shrink, and a create that got the warning
# and went ahead is a rent that fails with the operator mid-run. The numbers in the message are
# there because the alternative -- the API's own "len(args) > 16384" -- names a byte cap and no
# cause, and because a fragment this size is otherwise invisible (it is 10 KB of base64 on one line).
ONSTART_CMD_BYTES=$(printf '%s' "$ONSTART_CMD" | wc -c | tr -d '[:space:]')
[ "$ONSTART_CMD_BYTES" -le "$HOST_ONSTART_CMD_MAX" ] || die "the --onstart-cmd fragment is $ONSTART_CMD_BYTES bytes, over the provider's $HOST_ONSTART_CMD_MAX-byte cap -- hostctl create instance would answer 400 'len(args) > $HOST_ONSTART_CMD_MAX' and rent nothing (that is the failure this gate exists to be earlier than). The payload is already gzip+base64, so the excess is the template's own size: shrink onstart.sh.tmpl, or raise HOST_ONSTART_CMD_MAX in this script AND its twin in offer-search.sh if the provider has raised the cap. Render $(wc -c < "$OUT_DIR/onstart.rendered.sh") bytes, gzip+base64 $(wc -c < "$B64") bytes, fragment $ONSTART_CMD_BYTES bytes of $HOST_ONSTART_CMD_MAX"
note "onstart-cmd fragment: $ONSTART_CMD_BYTES / $HOST_ONSTART_CMD_MAX bytes ($(( HOST_ONSTART_CMD_MAX - ONSTART_CMD_BYTES )) under the cap; render $(wc -c < "$OUT_DIR/onstart.rendered.sh") bytes, gzip+base64 $(wc -c < "$B64") bytes)"
printf '%s\n' "$ONSTART_CMD"
