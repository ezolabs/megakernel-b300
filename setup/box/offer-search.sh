#!/bin/bash
# offer-search.sh — search the provider for a rentable offer of this box's own GPU shape, create an
# instance from one, and inspect a running instance. Box-parametrized capture of the
# offer-search and create recipe run by hand in
# box-c/archive/box-f.md §5c and box-g.md §5 -- no literal
# instance/offer id and no literal GPU shape anywhere here: the GPU name, the GPU count, the
# disk size, the instance label, the pinned image digest and the door port all come from the
# box file (boxes/$BOX.env, via lib.sh), so a different shape is a different box file and
# nothing else in here changes.
#
# Three subcommands:
#
#   offer-search.sh [search] [--gpu <gpu_name>] [--num-gpus N] [--pick-best]
#       `hostctl search offers` for a verified, rentable offer of this box's GPU shape
#       ($GPU_NAME x $NUM_GPUS, with at least $DISK_GB of free host disk),
#       sorted by $/hr ascending (`-o dph`). --gpu/--num-gpus override the box file's own name
#       and count for this one invocation (probing a different shape); the disk floor is not
#       overridable -- it has to stay the same number create will ask for with --disk.
#       An offer with `bw_nvlink` > 0 is kept; one at 0 is a PCIe-only host with no NVLink mesh
#       between the ranks, which cannot carry the tensor-parallel all-reduce this model runs,
#       however cheap it is -- so it is dropped, and the drop is COUNTED AND NAMED on stderr.
#       The filter runs over the rows the API returned rather than inside the query string, so
#       the rejected ids stay visible instead of vanishing server-side; a row that reports no
#       `bw_nvlink` at all is refused too, never assumed good.
#       When the box file sets `HOST_REQUIRE_STATIC_IP=1`, a SECOND filter runs the same way over
#       the NVLink-surviving rows: an offer whose `static_ip` is not `True` is dropped, counted
#       and named on stderr, never silently kept. This is off by default (no box file is broken
#       by upgrading to this script) -- but every box whose `dns.sh` manages a DNS-only
#       `A $HOSTNAME -> $BOX_IP` record needs it on, because a non-static address silently
#       invalidates that record on any host restart (re-read the instance, then re-point the A
#       record by hand). `--pick-best` picks from the same doubly-filtered rows, so a box that
#       asks for a static IP never gets handed one that isn't.
#       Prints one line per surviving offer: id, dph_total, gpu_frac, duration (days),
#       cuda_max_good, disk space, inet_down, bw_nvlink, static_ip, geolocation -- the fields
#       box-f.md/box-g.md's own preflight reads by hand before every create. --pick-best prints
#       ONLY the cheapest surviving offer's bare id (the first row, since the query above is
#       already sorted by dph ascending) -- nothing else on stdout, the filters' own summary
#       lines still go to stderr -- for a caller (bringup.sh) that wants to script the pick
#       instead of reading the table; exits 2 if no offer matched.
#
#   offer-search.sh create <offer-id> [<args-file>]
#       `hostctl create instance <offer-id> --image $CONTAINER_IMAGE --disk $DISK_GB --ssh
#       --direct --cancel-unavail --label $INSTANCE_LABEL --env "-p $DOOR_PORT:$DOOR_PORT -e
#       VLLM_API_KEY=<secret> -e XDG_CACHE_HOME=$CACHE" --onstart-cmd <rendered blob>`.
#       The onstart-cmd fragment comes from, in order: $ONSTART_CMD if already set (bringup.sh's
#       own mode -- it pre-renders this itself via `build-onstart.sh <args-file>` so the fragment
#       matches the boot's own args file); else `build-onstart.sh <args-file>` run right here when
#       an <args-file> was passed on this command line (standalone use, no bringup.sh in front);
#       else this refuses -- build-onstart.sh REQUIRES an args file as $1, so this never calls it
#       with none.
#       The fragment itself is opaque here on purpose (#643): it is base64 of the onstart's gzip
#       stream, built and explained in build-onstart.sh, and this subcommand neither builds nor
#       decodes it. What it does own is the size gate below -- the last thing between a fragment and
#       a rent. the provider's create API answers 400 and rents NOTHING when this field passes
#       HOST_ONSTART_CMD_MAX, so the gate refuses here first, in this tool's words with the byte
#       count, instead of leaving the operator with the API's own `len(args) > 16384` after they
#       had already picked an offer. build-onstart.sh refuses to even PRINT an over-cap fragment,
#       so this gate only ever fires on a caller-supplied $ONSTART_CMD -- the path where no
#       renderer ran at all (bringup.sh always renders, so in the tree's own flow these two gates
#       are the same check twice, deliberately).
#
#       STDOUT CONTRACT: the CLI's own create output, with the instance's api key withheld, and
#       then the new instance id ALONE on the last line -- the bare-id-last convention bringup.sh's
#       extraction reads. The id is read out of that output tolerantly, because the CLI's shape has
#       moved under this script: `hostctl create instance` prints a "Started. {...}" sentence
#       wrapping a PYTHON REPR dict (single quotes, not JSON -- box-c/archive/box-g.md:459 and the
#       2026-10-17 create this parse was fixed against), while the older CLI printed a JSON object
#       (fleet-history/glm-5.3-flash/box-h.md:2801). So: json.loads (strict=False, the house
#       convention) then ast.literal_eval on the whole output and on its first-`{`..last-`}` slice,
#       then a regex on `new_contract` as a last resort. An output with NO readable id is a FAILURE
#       (die, exit 1), not a silent success -- an unread id is exactly how a create gets orphaned,
#       and the failure message says the instance may already be rented and billed, so nobody
#       re-runs a create that would rent a SECOND box. The instance_api_key the CLI also prints is
#       withheld (it is the instance's own key, never the account key; no caller here reads it, and
#       this stdout is captured into a log by bringup.sh) -- re-read it with `show instance <id>
#       --raw` if it is ever needed.
#
#   offer-search.sh show
#       `hostctl show instance $INSTANCE_ID --raw`, parsed with `json.loads(s,
#       strict=False)` (docker progress control characters in status_msg while an image
#       pulls -- the same house convention as the sibling sniper and poll scripts). Prints ip, ports,
#       status and end_date.
#
# The search query itself keeps `rentable=True verified=True`: `rentable` is the documented
# trigger (an offer that is not rentable is not a candidate at all), and `verified` holds the
# search to hosts the provider has attested. `verified` is deliberately restrictive -- on a thin market
# it can return a single row for a whole shape -- and it is a default of the query in
# cmd_search, not a flag here: widening it is an edit of that query, made knowingly.
#
# `end_date` (printed by `show`) is the rental-window deadline on the instance's own record --
# the offer's own duration promise, and nothing else. It is not a free-since hint and nothing
# here reads it as one.
#
# Box-file variables this script reads (boxes/$BOX.env, via lib.sh):
#   GPU_NAME  the GPU shape to search for, as the provider spells it (`gpu_name=`, e.g. B300, B200).
#   NUM_GPUS  how many of them one offer must bundle (`num_gpus=`) -- 8 for an 8x B200 box.
#   DISK_GB   instance disk in GB, one number for both sides: the free-host-disk floor the
#                  search applies (`disk_space>=`) and create's own `--disk`, so the offer the
#                  search picks can always hold the disk create asks for. --disk is the size
#                  REQUESTED, not the host's capacity, so a tight number is what fills the disk
#                  partway through a long boot: the checkpoint is 510 GB (476 GiB) and the bake
#                  writes a whole second copy of it, plus HF staging and cache, before repacking
#                  -- a box that runs the bake sizes this well above two copies of the model.
#   HOST_REQUIRE_STATIC_IP  optional, default unset (off). Set to `1` when this box's `dns.sh`
#                  manages a DNS-only `A $HOSTNAME -> $BOX_IP` record: `search`/--pick-best then
#                  drop every offer whose `static_ip` is not `True`, counted and named on stderr,
#                  the same way a PCIe-only (no-NVLink) offer is dropped. Left unset, no offer is
#                  excluded on this basis -- a box file that predates this key keeps searching
#                  exactly as it did before.
#   INSTANCE_LABEL     the instance's the provider label (`--label`); default <box>-dsv41-<gpu_name>, e.g.
#                  box-g-dsv41-b300, so a label always follows the shape the box file
#                  names and a B200 box can never be labelled b300 by a literal left in here.
#   CONTAINER_IMAGE     pinned engine image, registry/repo@sha256:digest (box-f.md §1's dev-dsv41
#                  digest as of this writing: see the default below, override in the box
#                  file if a newer image is pinned)
#   DOOR_PORT      the single mapped container port (both host and container side use the
#                  same number, matching box-g/box-f's own --env "-p N:N")
#   CACHE          XDG_CACHE_HOME path inside the container, e.g. /workspace/cache-box-g
# show/create also need:
#   INSTANCE_ID required by `show` (the numeric the provider instance id, set once a box exists)
# ENV_FILE (already required by lib.sh) must carry a VLLM_API_KEY= line for `create`
# (read via lib.sh's `secret` helper -- never echoed).
#
# The four shape keys above are optional each with the default below, which is box-g's own 4x B300
# shape: an older box file that names none of them keeps working, and a box file that names its
# own shape (box-c's 8x B200) wins outright with no edit in here. A fallback that actually kicks in
# is announced on stderr -- a box file that never named its shape is reported as such rather than
# quietly searched as a 4x B300.
#
# DRY_RUN=1: prints the exact `hostctl` command line instead of running it (search still
# runs read-only against the live API even under DRY_RUN, since it changes nothing;
# create's `hostctl create instance` call itself is what DRY_RUN suppresses).
#
# Usage: offer-search.sh [search] [--gpu <gpu_name>] [--num-gpus N] [--pick-best]
#        offer-search.sh create <offer-id> [<args-file>]
#        offer-search.sh show
set -euo pipefail
. "$(dirname "$0")/lib.sh"

usage() {
  echo "usage: $0 [search] [--gpu <gpu_name>] [--num-gpus N] [--pick-best]" >&2
  echo "       $0 create <offer-id> [<args-file>]" >&2
  echo "       $0 show" >&2
}

# A number the provider takes as a number, refused here (naming the box file) rather than halfway
# through a create.
require_uint() {
  case "$2" in
    ''|*[!0-9]*) die "$1 must be a positive integer, got '$2' (set $1= in $BOX_ENV)" ;;
  esac
  [ "$2" -ge 1 ] || die "$1 must be at least 1, got '$2' (set $1= in $BOX_ENV)"
}

# The instance label: $INSTANCE_LABEL when the box file gives one, else the historical
# <box>-dsv41-<gpu_name> shape built from the GPU name the box file itself names.
offer_label() {
  if [ -n "${INSTANCE_LABEL:-}" ]; then
    printf '%s' "$INSTANCE_LABEL"
  else
    printf '%s-dsv41-%s' "$BOX" "$(printf '%s' "$GPU_NAME" | tr '[:upper:]' '[:lower:]')"
  fi
}

# This box's own GPU shape -- the only place a GPU count or a disk size is decided. A box file
# that names its own shape (an 8x B200 box: GPU_NAME=B200 NUM_GPUS=8) wins; the
# fallbacks are box-g's 4x B300 shape, for a box file that predates these keys.
shape_keys_set=1
if [ -z "${GPU_NAME:-}" ] || [ -z "${NUM_GPUS:-}" ] || [ -z "${DISK_GB:-}" ]; then
  shape_keys_set=0
fi
# ... and a fallback that actually gets used is said out loud, on the two subcommands that rent
# something, because a B200 box left on the B300 default would search the wrong book.
warn_shape_fallback() {
  [ "$shape_keys_set" = 1 ] || note "no GPU_NAME/NUM_GPUS/DISK_GB in $BOX_ENV: falling back to the 4x B300 shape (4 x B300, 800 GB) -- set them there to rent any other shape"
}
: "${GPU_NAME:=B300}"
: "${NUM_GPUS:=4}"
: "${DISK_GB:=800}"      # the 4x B300 default: 476 GiB checkpoint + image layers, logs, cache; a box that runs the bake sets it above two copies of the 510 GB model
require_uint NUM_GPUS "$NUM_GPUS"
require_uint DISK_GB "$DISK_GB"

# The dev-dsv41 image pinned in box-f.md §1 / §5c and box-g.md §5c (same digest both arms boot
# from). Override CONTAINER_IMAGE in the box file if a newer build is ever re-pinned.
: "${CONTAINER_IMAGE:=lmsysorg/sglang:dev-dsv41@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860}"

# the provider's create API cap on --onstart-cmd, in BYTES of the string it is handed (`args` in the API's
# own wording). This literal is build-onstart.sh's HOST_ONSTART_CMD_MAX, where the full story lives:
# the 2026-10-17 rent that answered `400 error 400/3471: Invalid args: len(image) > 1024, or
# len(args) > 16384, or len(label) > 256` with a 24373-byte fragment whose uncompressed base64 was
# over the cap, after every one of that script's content gates was green. The two are a matched pair
# and a change to one is a change to both. Deliberately NOT overridable from the box file: a box file
# able to raise it is a box file that can ask the API for a create the API refuses.
HOST_ONSTART_CMD_MAX=16384

cmd_search() {
  warn_shape_fallback
  local gpu="$GPU_NAME"
  local num_gpus="$NUM_GPUS"
  local pick_best=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --gpu) [ $# -ge 2 ] || { usage; exit 2; }; gpu=$2; shift 2 ;;
      --num-gpus) [ $# -ge 2 ] || { usage; exit 2; }; require_uint --num-gpus "$2"; num_gpus=$2; shift 2 ;;
      --pick-best) pick_best=1; shift ;;
      *) usage; exit 2 ;;
    esac
  done
  local require_static_ip="${HOST_REQUIRE_STATIC_IP:-0}"
  [ "$pick_best" = 1 ] || note "searching verified, rentable ${num_gpus}x $gpu offers (>=${DISK_GB} GB free host disk), sorted by dph"
  hostctl search offers \
    "gpu_name=$gpu num_gpus=$num_gpus rentable=True verified=True disk_space>=${DISK_GB}" \
    -o 'dph' --raw \
  | PICK_BEST="$pick_best" REQUIRE_STATIC_IP="$require_static_ip" python3 -c '
import json, os, sys
offers = json.loads(sys.stdin.read(), strict=False)

# bw_nvlink is the offer NVLink bandwidth in Gb/s; 0 is a PCIe-only host, whose ranks have no
# NVLink mesh for the tensor-parallel all-reduce this model runs. The filter runs here, over the
# rows the API returned, not inside the query string -- a rejected offer is counted and named
# below instead of vanishing server-side. A row that reports no bw_nvlink is refused, never
# assumed good.
def nvlink_class(o):
    v = o.get("bw_nvlink")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return "absent"
    return "ok" if v > 0 else "zero"

kept = [o for o in offers if nvlink_class(o) == "ok"]
zero = [o for o in offers if nvlink_class(o) == "zero"]
absent = [o for o in offers if nvlink_class(o) == "absent"]

def ids(rows, cap=12):
    return ",".join(str(o.get("id")) for o in rows[:cap]) + ("..." if len(rows) > cap else "")

line = "[%s] nvlink filter: %d of %d offer(s) keep bw_nvlink>0" % (
    os.environ.get("BOX", "?"), len(kept), len(offers))
if zero:
    line += "; %d excluded: bw_nvlink=0 (PCIe-only, no NVLink -- unusable for the TP all-reduce) ids %s" % (
        len(zero), ids(zero))
if absent:
    line += "; %d excluded: no bw_nvlink reported ids %s" % (len(absent), ids(absent))
print(line, file=sys.stderr)

# static_ip -- a criterion the plan requires whenever the box file asks for it (dns.sh manages a
# DNS-only A record that a non-static address silently invalidates on any host restart). Off by
# default so a box file that predates this key keeps searching exactly as it did before; when on,
# this runs the same way as the NVLink filter above -- over the NVLink survivors, counted and
# named on stderr, never a silent drop.
if os.environ.get("REQUIRE_STATIC_IP") == "1":
    before = kept
    kept = [o for o in before if o.get("static_ip") is True]
    non_static = [o for o in before if o.get("static_ip") is not True]
    sline = "[%s] static-ip filter: %d of %d nvlink-surviving offer(s) keep static_ip=True" % (
        os.environ.get("BOX", "?"), len(kept), len(before))
    if non_static:
        sline += "; %d excluded: static_ip is not True (a DNS-only A record would silently go stale on restart) ids %s" % (
            len(non_static), ids(non_static))
    print(sline, file=sys.stderr)

if os.environ.get("PICK_BEST") == "1":
    # already sorted by dph ascending (the -o dph above) -- kept[0] is the cheapest usable one.
    if not kept:
        sys.exit(2)
    print(kept[0].get("id"))
    sys.exit(0)
if not kept:
    print("(no offers matched)")
    sys.exit(0)
hdr = ("id", "dph_total", "gpu_frac", "duration_d", "cuda_max_good", "disk_space", "inet_down", "bw_nvlink", "static_ip", "geolocation")
print("  ".join(f"{h:>12}" if h != "geolocation" else h for h in hdr))
for o in kept:
    dur_d = (o.get("duration") or 0) / 86400.0
    dph_total = o.get("dph_total")
    row = [
        str(o.get("id")),
        f"{dph_total:.3f}" if dph_total is not None else "-",
        str(o.get("gpu_frac")),
        f"{dur_d:.1f}",
        str(o.get("cuda_max_good")),
        str(o.get("disk_space")),
        str(o.get("inet_down")),
        str(o.get("bw_nvlink")),
        str(o.get("static_ip")),
        str(o.get("geolocation")),
    ]
    print("  ".join(f"{v:>12}" if i < len(hdr) - 1 else v for i, v in enumerate(row)))
'
}

# ------------------------------------------------------------ create's output, before it is printed
# The CLI's create output carries the instance's OWN api key. Nothing in this tool set reads it, and
# create's stdout is captured into a log line by bringup.sh, so the value is masked on every path
# that prints that output -- success, failure, or no-id-readable-at-all. The mask keeps the key's own
# quoting, so the line is still the literal it was (a JSON object or a Python repr) and the id parse
# in cmd_create still sees an intact dict; output whose key it cannot match passes through unchanged
# rather than being lost, which is why this always exits 0 -- a display shim, never a gate.
mask_instance_key() {
  # shellcheck disable=SC2016   # the python source is not shell -- its backticks and %s are its own, not expansions
  python3 -c '
import os, re, sys

raw = sys.stdin.read()
MASK = re.compile(r"""(instance_api_key["\x27]?\s*:\s*["\x27])[^"\x27]*(["\x27])""")
masked = MASK.sub(r"\1<withheld>\2", raw)
if masked != raw:
    print("[%s] create output: instance_api_key withheld (that is the instance key, not the account key; re-read it with `hostctl show instance <id> --raw` if it is ever needed)" % os.environ.get("BOX", "?"), file=sys.stderr)
sys.stdout.write(masked + ("" if masked.endswith("\n") else "\n"))
'
}

cmd_create() {
  warn_shape_fallback
  [ $# -ge 1 ] && [ $# -le 2 ] || { usage; exit 2; }
  local offer_id=$1
  local args_file="${2:-}"
  : "${DOOR_PORT:?set DOOR_PORT=<port> in $BOX_ENV}"
  : "${CACHE:?set CACHE=<container cache path> in $BOX_ENV}"

  local api_key
  api_key=$(secret VLLM_API_KEY)
  [ -n "$api_key" ] || die "no VLLM_API_KEY in $ENV_FILE_PATH"

  local onstart_cmd="${ONSTART_CMD:-}"
  if [ -z "$onstart_cmd" ]; then
    local build_onstart="$TOOLS_DIR/build-onstart.sh"
    # build-onstart.sh REQUIRES an <args-file> as $1 -- never call it with none.
    if [ -n "$args_file" ] && [ -x "$build_onstart" ]; then
      onstart_cmd=$("$build_onstart" "$args_file")
    elif [ "${DRY_RUN:-0}" = 1 ]; then
      # The placeholder keeps build-onstart.sh's shape (#643: base64 of a gzip stream, decoded with
      # a gzip/python fallback and run only if the decode succeeded) with a marker in place of the
      # blob, so a DRY_RUN transcript still shows what a real create would send. The marker is not
      # base64 and nothing runs it -- DRY_RUN only prints it.
      onstart_cmd='echo <no ONSTART_CMD/args-file given -- DRY_RUN placeholder> | base64 -d | ( gzip -dc || python3 -c "import gzip,sys;sys.stdout.buffer.write(gzip.decompress(sys.stdin.buffer.read()))" ) > /root/onstart.sh && bash /root/onstart.sh'
      note "DRY_RUN: no ONSTART_CMD and no <args-file> given (and/or $build_onstart missing); using a placeholder --onstart-cmd"
    else
      die "no --onstart-cmd source: pass ONSTART_CMD=<rendered fragment> (bringup.sh's own mode, via build-onstart.sh <args-file>) or an <args-file> here directly: $0 create $offer_id <args-file>"
    fi
  fi

  # THE CAP GATE (#643), immediately before the one call it guards -- nothing between here and the
  # CLI touches the fragment, so this is the last point at which an oversized one is still free.
  # Measured on the exact string that would be handed to `--onstart-cmd` (bash's ${#var} counts
  # CHARACTERS under a UTF-8 locale, so the byte count is the honest one here, as it is in the API).
  # A refusal is a rent that never happened: the alternative is the API's 400 arriving after the
  # offer was chosen, which is how the 2026-10-17 rent died (400 error 400/3471: Invalid args:
  # len(args) > 16384) with no local gate ever having looked at length.
  local cmd_bytes
  cmd_bytes=$(printf '%s' "$onstart_cmd" | wc -c | tr -d '[:space:]')
  [ "$cmd_bytes" -le "$HOST_ONSTART_CMD_MAX" ] || die "the --onstart-cmd for this create is $cmd_bytes bytes, over the provider's $HOST_ONSTART_CMD_MAX-byte cap: the API would answer 400 'len(args) > $HOST_ONSTART_CMD_MAX' and rent NOTHING. If this came from ONSTART_CMD, re-render it with build-onstart.sh (its fragment is gzip+base64 and fits); if it was built by hand, pass an <args-file> to this subcommand instead so the render's own cap gate runs"
  note "onstart-cmd: $cmd_bytes / $HOST_ONSTART_CMD_MAX bytes"

  local label
  label=$(offer_label)
  local env_str="-p $DOOR_PORT:$DOOR_PORT -e VLLM_API_KEY=$api_key -e XDG_CACHE_HOME=$CACHE"

  note "creating instance from offer $offer_id (label=$label, image=$CONTAINER_IMAGE, disk=${DISK_GB} GB)"
  local out
  # Captured rather than streamed (the id in it is the whole point of this subcommand), and a
  # failure at the CLI is NOT read as "nothing was rented": the provider can accept the offer and fail the
  # call after it, so the captured output is printed before any failure path speaks -- it is the
  # only witness of an instance nobody would otherwise go looking for.
  if ! out=$(run hostctl create instance "$offer_id" \
    --image "$CONTAINER_IMAGE" \
    --disk "$DISK_GB" --ssh --direct --cancel-unavail \
    --label "$label" \
    --env "$env_str" \
    --onstart-cmd "$onstart_cmd"); then
    [ -z "$out" ] || printf '%s\n' "$out" | mask_instance_key
    die "create for offer $offer_id failed at the CLI (its output is above, if it printed any) -- DO NOT READ THAT AS NOTHING WAS RENTED: an offer can be accepted before the call fails, and an instance that exists is billed from that moment. Check \`hostctl show instances\` for an instance created from offer $offer_id (label=$label) BEFORE re-running anything -- re-running a create rents a SECOND box. If one is there, take its id and its ip/ports from \`hostctl show instance <id> --raw\` and fill INSTANCE_ID/BOX_IP/SSH_PORT/APP_PORT_EXT in $BOX_ENV by hand."
  fi
  if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%s\n' "$out"
    return 0
  fi
  # The id comes out of the CLI's output into a variable first, then the output is echoed (key
  # withheld), then the id is printed LAST -- the contract bringup.sh's create step and its reconcile
  # read. The parser reads the RAW output, never the masked echo: it is evidence, and masking it is a
  # display concern. A non-zero status here is either the parser reporting no readable id (its own
  # exit 3) or python itself failing; the die below treats both alike, because either way this run
  # does not know which instance it just rented.
  local cid="" parse_rc=0
  # shellcheck disable=SC2016   # the python source is not shell -- its backticks and %s are its own, not expansions
  cid=$(printf '%s\n' "$out" | python3 -c '
import ast, json, os, re, sys

raw = sys.stdin.read()

# An instance id is a positive whole number. bool is an int subclass in Python, so a True here is
# refused rather than silently read as 1.
def as_id(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v > 0 else None
    if isinstance(v, str) and v.strip().isdigit():
        n = int(v.strip())
        return n if n > 0 else None
    return None

def loads_json(text):
    # The house convention for a CLI payload with docker progress control characters in it.
    return json.loads(text, strict=False)

cid = None
success = None
# The CLI sentence wrapper ("Started. {...}") defeats both parsers, so each candidate text is tried
# whole, then as its first-`{`..last-`}` slice: json.loads (strict=False) first, then
# ast.literal_eval (the evidence shape is a Python repr, which json refuses). A regex over the raw
# text is the last resort, for an output whose framing defeats both -- it is deliberately narrow
# (`new_contract` as a quoted or bare key, then digits) so it cannot pick a number out of prose.
spans = [raw]
open_at, close_at = raw.find("{"), raw.rfind("}")
if open_at != -1 and close_at > open_at:
    spans.append(raw[open_at:close_at + 1])
for text in spans:
    for load in (loads_json, ast.literal_eval):
        try:
            d = load(text)
        except Exception:   # ValueError from json; ValueError/SyntaxError/TypeError from literal_eval
            continue
        if not isinstance(d, dict):
            continue
        got = as_id(d.get("new_contract"))
        if got is not None:
            cid = got
            success = d.get("success")
            break
    if cid is not None:
        break
if cid is None:
    m = re.search(r"""new_contract["\x27]?\s*:\s*["\x27]?(\d+)""", raw)
    if m:
        cid = as_id(m.group(1))
if cid is None:
    sys.exit(3)             # no readable id anywhere -- the caller dies, loudly, below
if success is False:
    print("[%s] create reported success=False and still named instance %d -- check that instance with `hostctl show instance %d --raw` before trusting the id below" % (os.environ.get("BOX", "?"), cid, cid), file=sys.stderr)
print(cid)
') || parse_rc=$?
  # Echoed before either failure or the id: a create whose id could not be read still shows the
  # "Started. {...}" line that names the instance this run will not go on to reconcile.
  printf '%s\n' "$out" | mask_instance_key
  [ "$parse_rc" = 0 ] || die "create for offer $offer_id: no readable instance id in the output above (id parser exit $parse_rc) -- THE INSTANCE MAY ALREADY EXIST AND BE BILLED, and re-running a create rents a SECOND box. Find it before doing anything else: \`hostctl show instances\` (offer $offer_id, label=$label), then take its id and ip/ports from \`hostctl show instance <id> --raw\` and set INSTANCE_ID/BOX_IP/SSH_PORT/APP_PORT_EXT in $BOX_ENV by hand."
  printf '%s\n' "$cid"
}

cmd_show() {
  : "${INSTANCE_ID:?set INSTANCE_ID=<instance id> in $BOX_ENV}"
  hostctl show instance "$INSTANCE_ID" --raw \
  | python3 -c '
import json, sys
d = json.loads(sys.stdin.read(), strict=False)
print("ip:", d.get("public_ipaddr"))
print("ports:", d.get("ports"))
print("status:", d.get("actual_status"), "|", d.get("status_msg"))
# end_date is the deadline of this rental window on the instance record -- when the duration
# the host promised runs out. It is NOT a hint about when the offer was last free (the audit
# measured it as no such signal), so nothing here or above reads it as one.
print("end_date:", d.get("end_date"))
'
}

case "${1:-search}" in
  search) shift || true; cmd_search "$@" ;;
  create) shift; cmd_create "$@" ;;
  show) cmd_show ;;
  *) usage; exit 2 ;;
esac