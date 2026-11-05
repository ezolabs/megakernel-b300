#!/bin/bash
# install-onstart.sh -- autopool B1 (box-c/qa/autopool-20261030/DESIGN.md §3 B1): render this tree's
# onstart for a box and install it IN PLACE as the box's /root/onstart.sh.
# the provider's /.launch runs `bash /etc/rc.local; bash /root/onstart.sh` on EVERY container start (a console
# stop -> start, a host reboot). On an instance whose `onstart` field is empty (box-a, box-b: created
# with the fragment, then cleared -- `hostctl show instance --raw` reads onstart=None) that file on the
# box disk IS the boot hook, so a new render reaches the next start only by being written there.
# Never `hostctl update instance` / `recycle` for this: those re-create the container.
#
# usage:
#   BOX=<box> bash autopool/install-onstart.sh <args-file>            # install
#   DRY_RUN=1 BOX=<box> bash autopool/install-onstart.sh <args-file>  # render + compare only
# <args-file> is build-onstart.sh's (tools/args/*.args): the render's first-boot args, used by the
# onstart only when the box has no sglang-args.txt (a staged box always has one).
# What it does: renders with build-onstart.sh into a scratch dir (every render gate applies); reads
# the box's /root/onstart.sh sha256; equal -> no-op. Otherwise keeps the old file as
# /root/onstart.<UTC stamp>.sh, writes the render beside it, `bash -n`s it there, gives it the old
# file's mode and renames it over /root/onstart.sh, then reads the sha256 back. Refused when
# /root/onstart.sh is a symlink (what it points at is the hook then -- decide by hand). It never runs
# the onstart and never touches an engine, the watchdog or hangwatch: the render takes effect at the
# NEXT container start.
. "$(dirname "$0")/../lib.sh"

ARGS="${1:?usage: BOX=<box> bash $0 <args-file>}"
[ -f "$ARGS" ] || die "no args file $ARGS"
OUT=$(mktemp -d "${TMPDIR:-/tmp}/install-onstart.$BOX.XXXXXX") || die "cannot make a scratch dir"
trap 'rm -rf "$OUT"' EXIT
BOX="$BOX" BOX_ENV="$BOX_ENV" env -u BOX_TARGET_RUN_ID bash "$TOOLS_DIR/build-onstart.sh" "$ARGS" --out-dir "$OUT" > "$OUT/fragment" 2> "$OUT/render.log" \
  || { cat "$OUT/render.log" >&2; die "build-onstart.sh refused the render (above)"; }
R="$OUT/onstart.rendered.sh"
NEW_SHA=$(shasum -a 256 < "$R" | cut -d' ' -f1)
grep -E 'hangwatch|freezegc' "$OUT/render.log" | sed 's/^\[[^]]*\] /render: /' >&2 || true
note "render: $(wc -c < "$R" | tr -d ' ') bytes, sha256 $NEW_SHA"

# shellcheck disable=SC2016  # expands on the box
CUR=$(box_ssh 'if [ -L /root/onstart.sh ]; then echo "LINK $(readlink /root/onstart.sh)"; elif [ -f /root/onstart.sh ]; then echo "FILE $(sha256sum < /root/onstart.sh | cut -d" " -f1) $(stat -c %a /root/onstart.sh)"; else echo NONE; fi' < /dev/null) \
  || die "could not read /root/onstart.sh on the box"
read -r KIND A1 A2 _ <<< "$CUR"
case "$KIND" in
  LINK) die "/root/onstart.sh on the box is a symlink to $A1 -- the hook is that target; not replacing a link here" ;;
  NONE) OLD_SHA="-"; MODE=755 ;;
  FILE) OLD_SHA="$A1"; MODE="$A2" ;;
  *)    die "unexpected answer from the box: $CUR" ;;
esac
note "box: /root/onstart.sh sha256 $OLD_SHA (mode $MODE)"
if [ "$OLD_SHA" = "$NEW_SHA" ]; then
  note "no-op: the box's /root/onstart.sh is already this render"
  echo "ONSTART_BEFORE=$OLD_SHA"; echo "ONSTART_AFTER=$NEW_SHA"
  exit 0
fi
if [ "${DRY_RUN:-0}" = 1 ]; then
  note "DRY: would keep the old file as /root/onstart.<UTC>.sh and install the render (sha256 $NEW_SHA, mode $MODE)"
  echo "ONSTART_BEFORE=$OLD_SHA"; echo "ONSTART_AFTER=$NEW_SHA"
  exit 0
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
box_ssh "set -e; n=/root/.onstart.sh.new.$STAMP
  [ ! -f /root/onstart.sh ] || cp -p /root/onstart.sh /root/onstart.$STAMP.sh
  cat > \$n; bash -n \$n; chmod $MODE \$n; mv -f \$n /root/onstart.sh" < "$R" \
  || die "install failed on the box (the old /root/onstart.sh stays in place unless the final mv ran)"
AFTER=$(box_ssh 'sha256sum < /root/onstart.sh | cut -d" " -f1' < /dev/null) || die "could not read the installed sha256 back"
[ "$AFTER" = "$NEW_SHA" ] || die "the box's /root/onstart.sh reads sha256 $AFTER after the install, not the render's $NEW_SHA"
note "installed: /root/onstart.sh sha256 $OLD_SHA -> $AFTER (the old file kept as /root/onstart.$STAMP.sh); it runs at the next container start"
echo "ONSTART_BEFORE=$OLD_SHA"; echo "ONSTART_AFTER=$AFTER"
