#!/bin/bash
# box-c-RESTART-SAFETY (#6325, 2026-09-29): this template carries the serve-model.env launcher plus #5928's two
# hygiene edits -- the on-box onstart copy box-c was hand-swapped to (sha d39d87db...) is now what a render yields.
# box-c-STAGED-LAUNCHER (#6327, 2026-09-29): when /workspace/tools/relaunch.staged.sh (a copy of stage.sh's
# lib.sh launcher) is present, a restart restores it as relaunch.sh instead of writing the inline copy, and
# exports BOX (pid 1's env has none; lib.sh dies without it). Without the staged copy it writes the inline one.
# box-c-BOX-BOTH-ARMS (#6435, 2026-09-29): BOX is exported ABOVE the launcher choice, so the inline arm's watchdog
# (watchdog.sh sources lib.sh) also comes up. A relaunch.sh that is already the staged lib.sh launcher is kept
# as-is and re-copied to relaunch.staged.sh, so a later stage.sh of a newer launcher never gets rolled back to
# an older backup; the backup is restored only when relaunch.sh is not the staged launcher.
# box-c-HUB-PIN (ledger #66, #76; 2026-10-17): the download gate's hub install is pinned to the range the
# image's own transformers needs, and the hub/transformers probe is a function called from BOTH arms of
# that gate -- a restart whose checkpoint already landed skips the download whole, and that boot path is
# the one the marker exists to instrument. HF_HUB_ENABLE_HF_TRANSFER=1 went with the pin, beside the
# hf_transfer extra it paired with: both are dead on the hub 1.x line. The runtime detail is at the
# install and the probe themselves, inside the gate below.
# box-c-STOCK-PIN (ledger #491, 2026-10-17): the checkpoint download names its revision. Unpinned,
# `hf download deepseek-ai/DeepSeek-V4.1-Flash-1017` takes that repo's DEFAULT-BRANCH TIP, so STOCK_DIR -- a directory whose
# `<repo name>` path already claims this build -- would hold whatever the tip happened to be at the
# boot. 1bae0e1088993b9b82de08f59de5be5a34e42170 is build-onstart.sh's HF_REV, the checkpoint's sha of record
# (the grafting kit's model notes name the same revision), and
# the graft download bringup.sh runs is pinned the same way (#470). Every retry below passes it, so
# a tip that moves mid-boot cannot land a second revision's shards beside the first's. It pins the
# repo; it is not a gate: the file-count gate below counts files, and apply_graft.py's own stock
# check covers only the shards a graft touches.
# Box onstart TEMPLATE (box-parametrized, ported from the earlier per-box onstart scripts) --
# rendered by build-onstart.sh, which substitutes every double-underscored placeholder below
# (never spell one out in a comment here -- the renderer replaces every occurrence textually, and
# a leftover one fails the render) and emits the gzip+base64 blob that `hostctl create instance
# --onstart-cmd` carries. The
# onstart is create-frozen, so everything a first boot needs is here: the sshd key-mode fix, the
# checkpoint download with a file-count gate, the args file, the launcher. The rest of the tool
# set (lib.sh, boxes/, watchdog.sh, cutover.sh, restart.sh, probes) is staged over rsync/scp by
# stage.sh once ssh works -- this script's inlined launcher starts the watchdog only when its
# file already exists, which it never does on a first boot.
#
# STOCK_DIR vs MODEL_DIR: the download above always targets STOCK_DIR (the stock checkpoint
# deepseek-ai/DeepSeek-V4.1-Flash-1017 names, at 1bae0e1088993b9b82de08f59de5be5a34e42170). What sglang serve actually loads is MODEL_DIR, read by the
# inlined launcher below from /workspace/tools/serve-model.env when stage.sh has written one (from the box file's
# current MODEL_DIR/SERVED_MODEL), else it falls back to this render's own baked-in box values.
# The split exists because onstart is create-frozen but a box's served model is not: without it, a
# container restart re-runs this exact script and the inlined launcher would serve whatever
# MODEL_DIR/SERVED_MODEL were baked in at the last `build-onstart.sh` render -- on a box switched
# to a repack after create, that resurrects the stock engine. serve-model.env is the fix that needs
# no re-render: `stage.sh` (or a hand-write on the box) updates it and the very next relaunch --
# inline-launcher or the staged tools/relaunch.sh, both honour it -- picks up the current model.
#
# MODEL SOURCE (ledger #132, 2026-10-26): the box file's ONSTART_MODEL_SOURCE picks one of two arms at
# RENDER time -- build-onstart.sh keeps the lines between a `#@stock{` or `#@copied{` line and its `#@}`
# only for the matching source, and the lines between `#@grant{` and `#@}` only when the box file sets
# ONSTART_GRANT_FILE. `stock` (the default) downloads STOCK_DIR as above. `copied` is a box that serves
# a repack COPIED onto it (box-a: from box-c): its render carries no download at all and gates the engine
# on the served MODEL_DIR's files plus the copy's MODEL_READY_MARKER. In both, sshd, the tunnel grant
# and Caddy come up BEFORE the model gate, so a box that cannot serve still answers ssh, its tunnel and
# its door.
echo "===box-b-START $(date -u) ==="
# No core dumps from anything this boot starts (#1300: no prod traffic at rest on a rented box). The
# box's core_pattern writes to the HOST's disk (/var/lib/hostctl_kaalia/data/core-%e.%p.%h.%t), and a
# crashed engine's or Caddy's core holds requests; RLIMIT_CORE 0 here is inherited by Caddy, the
# watchdog and every engine launcher below (relaunch.sh sets it again and its gate refuses any other).
ulimit -c 0
# box-c-HYGIENE (#5928, 2026-09-29): the container env's VLLM_API_KEY is the create-time key (pre-#5645
# rotation). An existing /etc/environment VLLM_API_KEY line is the live, rotated key and is kept; the
# env copy is appended only when the file has no VLLM_API_KEY line at all (a fresh filesystem).
env | grep -E '^(XDG_CACHE_HOME|SGLANG_)' >> /etc/environment
grep -q '^VLLM_API_KEY=' /etc/environment 2>/dev/null || env | grep '^VLLM_API_KEY=' >> /etc/environment
echo "--- fix: keys + modes (the injector writes authorized_keys as uid 112 mode 644)"
mkdir -p /root/.ssh; touch /root/.ssh/authorized_keys
printf '%s\n' "$SSH_PUBLIC_KEY" "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHsE+F7XdgTwYGcFjpAiE69j/pnCykhhYK/rAEICSFk7 ablit1017" | while IFS= read -r k; do [ -n "$k" ] && ! grep -qxF "$k" /root/.ssh/authorized_keys && printf '%s\n' "$k" >> /root/.ssh/authorized_keys; done
chown -R 0:0 /root/.ssh; chmod 755 /root; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys
mkdir -p /etc/ssh/sshd_config.d; printf 'StrictModes no\n' > /etc/ssh/sshd_config.d/zz-box-b.conf
pkill -x sshd 2>/dev/null; sleep 1
/usr/sbin/sshd 2>&1 && echo "sshd restarted rc=0" || echo "sshd restart FAILED"
stat -c 'after: %U:%G %a %n' /root/.ssh/authorized_keys
echo "===box-b-FIX-DONE==="
# TUNNEL GRANT (ONSTART_GRANT_FILE): a router tunnel's key lives in authorized_keys2, which the provider's key
# injector never writes; the staged copy (stage.sh's rsync never deletes it) puts it back whenever it is
# missing or different. Here, after the sshd fix (which chowns /root/.ssh) and before the door and the
# model gate, so the tunnel survives a restart that cannot serve. A missing copy is said, never fatal.
if [ -s /workspace/tools/authorized_keys2.box-b-kv ]; then
  cmp -s /workspace/tools/authorized_keys2.box-b-kv /root/.ssh/authorized_keys2 || { cp /workspace/tools/authorized_keys2.box-b-kv /root/.ssh/authorized_keys2 && echo "===box-b-GRANT: authorized_keys2 restored from /workspace/tools/authorized_keys2.box-b-kv==="; }
  chown 0:0 /root/.ssh/authorized_keys2; chmod 600 /root/.ssh/authorized_keys2
else
  echo "===box-b-GRANT-MISSING: no /workspace/tools/authorized_keys2.box-b-kv -- authorized_keys2 not restored==="
fi
# DOOR: Caddy (tls.sh's caddy-relaunch.sh, the allowlisted Caddyfile) before the model gate, so the
# door is back on a restart even when the engine cannot start; the watchdog keeps it up once the
# launcher runs. Absent on a first boot (tls.sh has not run yet), so a no-op there. Same process
# match as watchdog.sh's.
if [ -x /workspace/tools/caddy-relaunch.sh ] && ! pgrep -f "/workspace/tools/cadd[y]-relaunch.sh|cadd[y] run" >/dev/null; then
  bash /workspace/tools/caddy-relaunch.sh >> /workspace/box-b-caddy.log 2>&1; echo "===box-b-DOOR: caddy-relaunch.sh rc=$?==="
fi
# IPROUTE2: relaunch.sh's per-engine port check reads `ss -ltn` (it falls back to /proc/net/tcp
# without it), and the image ships no iproute2 -- a recreated container lost the one installed by hand.
# Best-effort and never fatal, and in the BACKGROUND (its own session, like the engines): ss is a
# nicety, never a need, and an apt that cannot reach a mirror would otherwise hold this boot -- the
# model gate and every engine launch behind it -- for up to the two 300 s timeouts. One attempt per
# boot, never retried. Started before the model gate so even a box that cannot serve gets it; the
# start is said at once, in this boot log; the background job writes everything -- apt's own output,
# then the outcome line -- to the box's apt log, like every other background job here, never to this
# boot log.
iproute2_install() {
  if timeout 300 apt-get update -qq && DEBIAN_FRONTEND=noninteractive timeout 300 apt-get install -y -qq iproute2 && command -v ss >/dev/null 2>&1; then
    echo "===box-b-IPROUTE2: installed ($(command -v ss))==="
  else
    echo "===box-b-IPROUTE2-MISSING: apt-get install iproute2 failed (apt's output above) -- relaunch.sh reads /proc/net/tcp instead==="
  fi
}
if ! command -v ss >/dev/null 2>&1; then
  nohup setsid bash -c "$(declare -f iproute2_install); iproute2_install" >> /workspace/box-b-apt.log 2>&1 < /dev/null &
  echo "===box-b-IPROUTE2-INSTALLING: no ss on PATH -- apt-get install iproute2 started in the background (pid $!; output and outcome in /workspace/box-b-apt.log), the engines do not wait for it==="
fi
echo "--- diag"
df -h /dev/shm /workspace 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1
nvidia-smi topo -m 2>&1 | head -12
command -v sglang; command -v nvcc && nvcc --version | tail -1
# COPIED: nothing is downloaded. The gate below counts the directory the launcher will SERVE, which it
# names STOCK in this arm too: serve-model.env's MODEL_DIR read the launcher's way (last line wins), else
# this render's own MODEL_DIR. Never created here -- an absent copy is a not-ready box, not an empty dir.
STOCK="<MODEL_PATH>"
SM=$(sed -n 's/^MODEL_DIR=//p' /workspace/tools/serve-model.env 2>/dev/null | tail -1)
[ -n "$SM" ] && STOCK=$SM
mkdir -p /workspace/tools /workspace/cache-mk29
# The gate counts FILES, never log lines: 48 shards + index + config + tokenizer, 0 .incomplete.
complete() { [ "$(ls -1 $STOCK/model-*.safetensors 2>/dev/null | wc -l)" -ge 48 ] && [ -f "$STOCK/model.safetensors.index.json" ] && [ -f "$STOCK/config.json" ] && [ -f "$STOCK/tokenizer.json" ] && [ -f "$STOCK/tokenizer_config.json" ] && [ "$(find "$STOCK" -name '*.incomplete' 2>/dev/null | wc -l)" -eq 0 ]; }
# Name the hub/transformers pair the download and the serve both need, and let a venv whose two cannot
# be imported together say so HERE, in the boot log, instead of only in the engine's ImportError later.
# A function because BOTH arms of the download gate below call it: the fresh-boot arm right after the
# pinned install, and the restart arm, where the gate is skipped whole. Inline in the gate it reported
# nothing on a container restart of a box whose checkpoint had already landed -- the one boot path this
# marker exists to instrument, a venv that drifted back into the ImportError since the last boot.
# `python` is the same PATH-resolved interpreter as the `pip` install below -- the image puts the venv's
# bin first, which is how the un-pinned install reached the venv's transformers at all. Diagnostic only:
# nothing here aborts the boot, since the six download attempts below may still complete. Same
# `===box-MARKER===` shape as the xgrammar line further down. `hf_xet` is probed SEPARATELY, never
# folded into the pair above: a missing xet backend only drops the pull to plain HTTPS, so it must never
# raise the IMPORT-FAILED marker -- the healthy line reports it as a field instead, and the failing one
# claims no cause at all, just the error line the next `sglang serve` would die on.
hub_probe() {
  if python -c 'import transformers, huggingface_hub' 2>/dev/null; then
    xet=present; python -c 'import hf_xet' 2>/dev/null || xet='MISSING (this pull falls back to plain HTTPS)'
    echo "===box-b-HUB: huggingface_hub $(python -c 'import huggingface_hub; print(huggingface_hub.__version__)') / transformers $(python -c 'import transformers; print(transformers.__version__)') / hf_xet $xet==="
  else
    echo "===box-b-HUB-IMPORT-FAILED: $(python -c 'import transformers, huggingface_hub' 2>&1 | grep -m1 -E '^[A-Za-z_.]*Error' | cut -c1-240 || echo 'no error line -- see /workspace/setup.log')==="
  fi
}
# The copy pre-sizes its shards, so a file count proves nothing mid-copy: the engine also waits for
# MODEL_READY_MARKER, which the copy's own sha256 check writes when every shard verified. Not ready
# exits here, with sshd, the grant and the door already up.
hub_probe
if ! complete || [ ! -e "<MODEL_READY_MARKER>" ]; then
  echo "===box-b-MODEL-NOT-READY $(date -u): MODEL_DIR=$STOCK files $(complete && echo complete || echo INCOMPLETE), <MODEL_READY_MARKER> $([ -e "<MODEL_READY_MARKER>" ] && echo present || echo ABSENT); engine NOT launched==="
  exit 1
fi
echo "===box-b-MODEL-READY: $STOCK (files complete, <MODEL_READY_MARKER> present)==="
# XGRAMMAR (the 0044b roll): the image's editable sglang install sits on xgrammar 0.2.1, and the served fork
# tree (0044 onward, upstream #39026's deepseek_v4_1 structural tag) needs 0.2.8. Bump the venv before the
# first launch so a fresh bring-up matches the live box: the wheel staged in /workspace/wheels when one is
# there, else the package index. --no-deps keeps the image's apache-tvm-ffi 0.1.11 (0.2.8 needs >= 0.1.11).
# A no-op once 0.2.8 is installed; drop this block once the image itself bakes 0.2.8 (ISSUES.md).
XGR_WANT=0.2.8
xgr_have() { /opt/sglang/bin/pip show xgrammar 2>/dev/null | sed -n 's/^Version: //p'; }
if [ "$(xgr_have)" != "$XGR_WANT" ]; then
  if ls /workspace/wheels/xgrammar-$XGR_WANT-*.whl >/dev/null 2>&1; then
    /opt/sglang/bin/pip install --no-deps --no-index /workspace/wheels/xgrammar-$XGR_WANT-*.whl >> /workspace/setup.log 2>&1
  else
    /opt/sglang/bin/pip install --no-deps "xgrammar==$XGR_WANT" >> /workspace/setup.log 2>&1
  fi
fi
echo "===box-b-XGRAMMAR: $(xgr_have) (want $XGR_WANT)==="
mkdir -p /workspace/tools
# The args file wins if it already exists (a tuned line survives a container restart).
[ -s /workspace/tools/sglang-args.txt ] || cat > /workspace/tools/sglang-args.txt <<'EOF_ARGS'
--tp-size 4 --ep-size 1 --mem-fraction-static 0.75 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --cuda-graph-max-bs-decode 64 --max-running-requests 64 --context-length 1048576 --chunked-prefill-size 8192 --reasoning-parser auto --tool-call-parser auto --enable-metrics --trust-remote-code --prefill-decode-interval 4 --enable-cache-report --swa-prefix-tails 2048 --enable-priority-scheduling --disable-priority-preemption --default-priority-value 1 --max-queued-requests 512 --kv-events-config '{"publisher":"zmq","endpoint":"tcp://127.0.0.1:5557","bind":true,"replay_endpoint":"tcp://127.0.0.1:5558"}' --load-publish-endpoint tcp://127.0.0.1:5559 --metric-label-denylist priority model_name --short-prefill-token-threshold 3072 --short-prefill-chunk-size 1024 --short-prefill-max-tokens 4096 --short-prefill-lookahead 8 --short-prefill-swap-limit 16384 --enable-hierarchical-cache --hicache-ratio 2 --hicache-write-policy write_through_selective --hicache-io-backend direct --hicache-mem-layout page_first_direct --hicache-host-memory-mode cache --enable-decoder-swa-bounded-replay --speculative-use-block-verification --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2048 --cuda-graph-prefill-max-context 131072 --flashinfer-allreduce-fusion-backend trtllm --enable-nvidia-megakernel
EOF_ARGS
# box-c-STAGED-LAUNCHER (#6327): prefer the staged lib.sh launcher (its guards: dies without sglang-env.txt,
# `note` logging) over the inline copy below, when it and what it sources are on the box.
# box-c-BOX-BOTH-ARMS (#6435): BOX for whichever launcher runs, and for the watchdog.sh either one starts.
export BOX="${BOX:-box-b}"
LAUNCHER=inline
if [ -r /workspace/tools/lib.sh ] && [ -r /workspace/tools/boxes/box-b.env ] && grep -qF '. "$(dirname "$0")/lib.sh"' /workspace/tools/relaunch.sh 2>/dev/null; then
  LAUNCHER=staged
  cp -p /workspace/tools/relaunch.sh /workspace/tools/relaunch.staged.sh
  echo "===box-b-LAUNCHER: staged (relaunch.sh kept, relaunch.staged.sh refreshed, BOX=$BOX)==="
elif [ -s /workspace/tools/relaunch.staged.sh ] && [ -r /workspace/tools/lib.sh ] && [ -r /workspace/tools/boxes/box-b.env ]; then
  LAUNCHER=staged
  cp -p /workspace/tools/relaunch.staged.sh /workspace/tools/relaunch.sh
  echo "===box-b-LAUNCHER: staged (restored from relaunch.staged.sh, BOX=$BOX)==="
else
echo "===box-b-LAUNCHER: inline (no staged launcher, BOX=$BOX)==="
cat > /workspace/tools/relaunch.sh <<'EOF_RELAUNCH'
KEY=$(grep -a '^VLLM_API_KEY=' /etc/environment | head -1 | cut -d= -f2- | tr -d '"')
[ -n "$KEY" ] || { echo NO_KEY; exit 3; }
# The key never rides this argv (#2739): it goes into a root-only file, and sglang's own --config
# merges it in as --api-key before argparse runs, so the engine is authenticated exactly as before
# while pgrep/ps (and every process-list record) show only the path. The file is 0600, one line
# `api-key: "<key>"`, temp-then-rename: the same path and shape the staged relaunch.sh writes, so
# the first boot and every staged launch after it hold the key in that one file and no argv. A key
# carrying a character outside this set is refused rather than escaped; every key this tool set
# mints (openssl rand -hex 16) is hex.
KEY_FILE=/etc/sglang-api-key.yaml
case "$KEY" in
  *[!A-Za-z0-9._~+/=-]*) echo KEY_SHAPE; exit 6 ;;
esac
rm -f "$KEY_FILE.tmp.$$"
( umask 077; printf 'api-key: "%s"\n' "$KEY" > "$KEY_FILE.tmp.$$" ) || { echo NO_KEYFILE; exit 7; }
mv -f "$KEY_FILE.tmp.$$" "$KEY_FILE" || { echo NO_KEYFILE; exit 7; }
chmod 600 "$KEY_FILE" || { echo NO_KEYFILE; exit 7; }
A=$(grep -v '^#' "/workspace/tools/sglang-args.txt")
[ -n "$A" ] || { echo NO_ARGS; exit 4; }
command -v sglang >/dev/null 2>&1 || export PATH="/opt/sglang-mk29/bin:$PATH"
command -v sglang >/dev/null 2>&1 || { echo NO_SGLANG_ON_PATH; exit 5; }
export XDG_CACHE_HOME="/workspace/cache-mk29" HF_HUB_OFFLINE=1 SGLANG_DISABLE_USAGE_STATS=1 DO_NOT_TRACK=1
export PATH="/opt/sglang-mk29/bin:$PATH"
if [ -f "/workspace/tools/sglang-env.txt" ]; then
  while IFS= read -r line; do
    case "$line" in ''|'#'*) ;; *) export "$line" ;; esac
  done < "/workspace/tools/sglang-env.txt"
fi
# The newline before `)` is the one `$(grep ...)` stripped: without it a trailing ` # ...` comment on
# the last kept line comments out the `)` too (relaunch.sh's own eval, tests/relaunch-args-comments.test.sh).
eval "A=($A"$'\n'")"
# No key in the environ the watchdog and the engine inherit (#3003, as relaunch.sh): the engine reads
# $KEY_FILE, the watchdog reads /etc/environment itself. After the sglang-env.txt loop, so a stray
# line there cannot put one back; this copy is also relaunch.sh until stage.sh replaces it, and a
# restart.sh ssh session exports the key through pam_env.
unset VLLM_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN CF_DNS_TOKEN KEY
# No prod traffic at rest on a rented box (#1300): RLIMIT_CORE 0 and relaunch.sh's own deploy gate (its
# rest-gate block, inlined below by this renderer) over the args, the environment and the core limit,
# before the watchdog or the engine starts. A refusal names each flag and exits 8: no engine starts. So
# does a gate that exits non-zero for any reason (#1348, as relaunch.sh): its status is checked.
ulimit -c 0 2>/dev/null || true
rg_on() { case "$1" in ''|0|[Ff][Aa][Ll][Ss][Ee]|[Nn][Oo]|[Oo][Ff][Ff]|[Nn][Oo][Nn][Ee]|-1) return 1 ;; esac; return 0; }
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
rg_wild() { local h=${1#*://}; h=${h%:*}; case "$h" in '*'|0.0.0.0|::|'[::]') return 0 ;; esac; return 1; }
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
rg_kv_get() { local s=$RG_KVP n=$'\n'; case "$s" in *"$n$1="*) s=${s#*"$n$1="}; printf '%s\n' "${s%%"$n"*}" ;; *) return 1 ;; esac; }
RG_ENDPOINTS=1
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
rg_env_now() { local v; for v in $(compgen -e); do printf '%s=%s\n' "$v" "${!v//$'\n'/ }"; done; }
rg_core() { local c; c=$(ulimit -c); [ "$c" = 0 ] || echo "${1:-launch}: RLIMIT_CORE is $c, not 0 -- a crash would write engine memory to the host's disk (ulimit -c 0)"; }
RG_RC=0; RG_OUT=$(RG_E=$(rg_env_now) && rest_gate "/workspace/tools/sglang-args.txt" "$RG_E" "${A[@]}" && rg_core) || RG_RC=$?
[ "$RG_RC" = 0 ] || RG_OUT="gate exit $RG_RC: it could not finish, nothing is proved clean $RG_OUT"
[ -z "$RG_OUT" ] || { echo "REFUSED (no prod traffic at rest) -- no engine started:"; echo "$RG_OUT"; exit 8; }
# serve-model.env (stage.sh's, or an operator's) wins over these two render-time literals.
SM_MODEL_DIR="<MODEL_PATH>"; SM_SERVED_MODEL="DeepSeek-V4.1-Flash"
if [ -f "/workspace/tools/serve-model.env" ]; then
  while IFS= read -r line; do
    case "$line" in
      MODEL_DIR=*) SM_MODEL_DIR="${line#MODEL_DIR=}" ;;
      SERVED_MODEL=*) SM_SERVED_MODEL="${line#SERVED_MODEL=}" ;;
    esac
  done < "/workspace/tools/serve-model.env"
fi
echo "===== sglang serve $(date -u) box=box-b (first-boot inline launcher) model=$SM_MODEL_DIR served=$SM_SERVED_MODEL ====="
if [ -f "/workspace/tools/watchdog.sh" ]; then
  pgrep -f "/workspace/tools/watchdo[g].sh" >/dev/null || ( nohup setsid bash "/workspace/tools/watchdog.sh" >> "/workspace/box-b-watchdog.log" 2>&1 < /dev/null & )
fi
exec sglang serve "${A[@]}" --model-path "$SM_MODEL_DIR" --served-model-name "$SM_SERVED_MODEL" --host 127.0.0.1 --port "8002" --config "$KEY_FILE"
EOF_RELAUNCH
fi
chmod +x /workspace/tools/relaunch.sh
# box-c-ENGINES (#429/#497): the staged box file's ENGINES (box-c's 2xTP4, lib.sh) boots every engine it
# names, `relaunch.sh <id>` each -- its own GPUs, ports, args/env pair, serve log, warm gate and
# `watchdog.sh <id>` (lib.sh engine_select), so the watchdogs a reboot leaves behind iterate the same
# ids. Read from the box file at BOOT, never baked into this render (it is create-frozen), and only
# under the staged launcher: the inline copy knows one engine. Any doubt boots the legacy line below
# (sglang-args.txt, the TP8 line on ENGINE_PORT): a malformed ENGINES (lib.sh refuses it), or an
# engine whose sglang-args-<id>.txt / sglang-env-<id>.txt is not on the box (restart.sh --engine
# installs them). A rollback that must survive a reboot reverts the box file's ENGINES line and
# re-stages; `restart.sh --single` alone leaves the per-engine pair, and this block, in place.
ENGINE_IDS=""
if [ "$LAUNCHER" = staged ]; then
  if ENGINE_IDS=$(bash -c '. "$1"; engines_list' _ /workspace/tools/lib.sh 2>> /workspace/setup.log); then
    ENGINE_IDS=$(echo $ENGINE_IDS)
  else
    echo "===box-b-ENGINES-REFUSED: lib.sh could not list the box file's ENGINES (see /workspace/setup.log) -- booting the legacy line==="
    ENGINE_IDS=""
  fi
  for id in $ENGINE_IDS; do
    if [ ! -s /workspace/tools/sglang-args-$id.txt ] || [ ! -f /workspace/tools/sglang-env-$id.txt ]; then
      echo "===box-b-ENGINES-INCOMPLETE: ENGINES names '$ENGINE_IDS' but engine $id has no sglang-args-$id.txt/sglang-env-$id.txt -- booting the legacy line==="
      ENGINE_IDS=""
      break
    fi
  done
fi
if [ -n "$ENGINE_IDS" ]; then
  for id in $ENGINE_IDS; do
    echo "===box-b-SERVE-$id: $(grep -v '^#' /workspace/tools/sglang-args-$id.txt)==="
  done
else
echo "===box-b-SERVE: $(grep -v '^#' /workspace/tools/sglang-args.txt)==="
fi
# box-c-KEYHOLDERS (#3003): the serve log carries the key (the engine's server_args= line), so it is
# CREATED 0600 -- in a subshell, so the umask never reaches the launcher or the engine; `>>` keeps the
# mode of a log that already exists. And the container env's VLLM_API_KEY (create-time -e) stops here:
# both launchers read the key from /etc/environment, so the engine's environ never carries it.
unset VLLM_API_KEY
if [ -n "$ENGINE_IDS" ]; then
  for id in $ENGINE_IDS; do
    ( umask 077; : >> /workspace/sglang-serve-$id.log )
    nohup setsid bash /workspace/tools/relaunch.sh "$id" >> /workspace/sglang-serve-$id.log 2>&1 < /dev/null &
    echo "===box-b-ENGINE-$id: relaunch.sh $id started (log /workspace/sglang-serve-$id.log)==="
  done
else
( umask 077; : >> /workspace/sglang-serve.log )
nohup setsid bash /workspace/tools/relaunch.sh >> /workspace/sglang-serve.log 2>&1 < /dev/null &
fi
# box-c-HANGWATCH (autopool B1): the box file's HANGWATCH_CMD, after the launch above (relaunch.sh
# starts the watchdog) -- it holds its own BOOT grace while the engine loads. A running copy (same
# script, same --out: a hand start, a re-run of this file) is kept.
if pgrep -f '/workspace/mk29/harness/hangwatc[h].py .*--out /workspace/mk29/watch/serving( |$)' > /dev/null; then
  echo "===box-b-HANGWATCH: already running (/workspace/mk29/watch/serving) -- kept==="
elif [ ! -f /workspace/mk29/harness/hangwatch.py ]; then
  echo "===box-b-HANGWATCH-MISSING: no /workspace/mk29/harness/hangwatch.py on the box -- not started==="
else
  mkdir -p /workspace/mk29/watch/serving
  nohup setsid python3 /workspace/mk29/harness/hangwatch.py --service --on-hang kill --out /workspace/mk29/watch/serving --dump-signal none >> /workspace/mk29/watch/serving/nohup.log 2>&1 < /dev/null &
  echo "===box-b-HANGWATCH: started (pid $!, out /workspace/mk29/watch/serving)==="
fi
echo "===box-b-END $(date -u) ==="
