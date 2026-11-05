#!/bin/bash
# runs ON the box, staged by stage.sh: this box's direct-TLS front door
# (box-c/05-frontdoor-tls.md). Adapted from a single-engine box's direct-TLS door script in this
# repo's legacy per-box tooling, box-parametrized: the hostname, the container door port, the
# loopback engine port and every other fact come from boxes/$BOX.env via lib.sh -- nothing here
# names a box, an ip or a port literally.
#
# Unlike a six-object Cloudflare-Access shape (Access app, service token, policy, Worker,
# DNS-only origin A + proxied AAAA placeholder + route -- see the legacy frontdoor script for
# another box in this repo's older per-box tooling), this door dials straight to the box:
# $HOSTNAME is itself a DNS-only A record pointed at $BOX_IP, and Caddy here
# terminates TLS on the container's mapped door port ($DOOR_PORT -- the container-internal port
# the provider maps to an external, per-instance port exposed as $APP_PORT_EXT) with a real Let's
# Encrypt certificate issued OFF the box by the operator's own issuer (ACME DNS-01, the
# DNS credential never on the box) and pushed in as files, then
# reverse-proxies to this box's engine on 127.0.0.1:$ENGINE_PORT. No Worker, no Access
# app, no router in front -- the engine's own bearer key is the only gate, narrowed
# by the box file's IP allowlist (TLS_ALLOW; open only on an explicit TLS_ALLOW_ANY=1).
#
# What this script does, idempotently, every time it runs:
#   1. Hold $TOOLS_DIR/caddy to the binary of record, CADDY_EXPECT_SHA256 -- the box file's pin
#      (box-c and box-a carry one), else the stock build-server sha; the patched h2-window build
#      (caddy-h2/) is CADDY_PATCHED_SHA256, pinned per box once that box's binary is
#      swapped. A binary on disk with the pinned sha256 is kept; one with any other sha256 is
#      REFUSED (exit 6, the binary left as it is, nothing downloaded over it) with the two fixes
#      named -- build with caddy-h2/build.sh, or pin CADDY_EXPECT_SHA256 to the running binary's
#      sha. With no binary on disk, the stock build-server download (caddy-dns/cloudflare baked in;
#      the build server has no version-pin query parameter) runs ONLY when CADDY_EXPECT_SHA256 is
#      the stock sha, and the download is installed only if it hashes to it; a patched pin with no
#      binary (a fresh box) is refused, pointing at stage.sh and build.sh -- never an auto-download
#      of stock. stage.sh's last step delivers that binary from the Mac's artifact store
#      (<HOME>/workspace/backups/caddy-h2/) onto a box with none, and bringup.sh stages before either of
#      its tls.sh calls, so on a bring-up this refusal means the stage refused or was skipped.
#      caddy.env's own refusals (step 2's exit 3) are checked before this step touches anything;
#      under DRY_RUN=1 a refusal prints `DRY: would REFUSE` and the preview goes on.
#   2. (Re-)write $TOOLS_DIR/Caddyfile from $TOOLS_DIR/caddy.env (the IP allowlist) -- always
#      rewritten, since caddy.env can change under it, and REFUSED (exit 3, the Caddyfile on disk
#      untouched) when there is no allowlist and no TLS_ALLOW_ANY=1, when caddy.env or the box file
#      still carries a CF_DNS_TOKEN (#1306: the Cloudflare token stays off every rented box -- re-stage,
#      stage.sh writes caddy.env without it), or when `caddy adapt` cannot load the render. Every site
#      serves the box's certificate FILES, `tls $TOOLS_REMOTE/certs/$HOSTNAME.crt <.key>`: one cert whose
#      SANs carry every name the door serves, issued OFF the box by the operator's own issuer and
#      pushed in as files -- Caddy solves no ACME challenge and holds
#      no DNS credential. Before the write, the pair is REFUSED (exit 5, the Caddyfile on disk
#      untouched) when it is missing, expired, its key does not match, or a site's name is not on it
#      -- push one in first. With no ENGINES in the box file there is one
#      engine, on 127.0.0.1:$ENGINE_PORT, behind one site; with ENGINES (box-c's 2xTP4 layout) each
#      engine also gets its own site on its own name, and $HOSTNAME balances across the copies
#      (see TLS_SITES below; TLS_PRESTAGE=1 renders the engine sites ahead of the cutover).
#   3. Write the launcher $TOOLS_DIR/caddy-relaunch.sh (nohup/setsid, XDG_DATA_HOME pinned
#      under /workspace for Caddy's own state), and run it once if Caddy is not already up. If
#      Caddy IS already running, the rewritten Caddyfile is picked up by `caddy run --watch` on its
#      own; the launcher is only needed for a cold start or a forced restart (watchdog.sh calls it
#      directly, not this script). The launcher reads no caddy.env and carries no token: the
#      certificate is a file and the allowlist is baked into the Caddyfile. A set
#      CADDY_H2_RECV_BUFFER_PER_CONNECTION / _PER_STREAM (box file or environment; bytes, digits
#      only, else exit 3 before anything is written) is baked into it as an export -- the patched
#      binary's HTTP/2 receive windows (caddy-h2/README.md), which caddy.env cannot carry because
#      stage.sh rewrites it whole. Then, for every name the Caddyfile serves, it polls up to 60s
#      for the TLS handshake on 127.0.0.1:$DOOR_PORT to serve the certificate file's own sha256
#      fingerprint (with Caddy already up and a name serving another one, a `caddy reload --force`
#      first -- --watch reloads only a changed Caddyfile, never a changed certificate file), prints
#      each issuer/notAfter, and exits 5 on a timeout.
#
# Reads:
#   $TOOLS_DIR/caddy.env  -- TLS_ALLOW (space-separated CIDRs; a CF_DNS_TOKEN line in it is
#     REFUSED, exit 3), written there by stage.sh from the box file's TLS_ALLOW line, which
#     lib.sh also loads -- only those source IPs reach the reverse proxy and everyone else gets a
#     403 at the edge of this box). With no TLS_ALLOW this script refuses to render a Caddyfile
#     unless TLS_ALLOW_ANY=1 (box file or environment) asks for a door that answers the whole
#     internet with the engine's own bearer as the only gate.
#   $TOOLS_REMOTE/certs/$HOSTNAME.crt / .key -- the door's certificate (full chain) and key, 600,
#     written by the operator's certificate install; required by a setup run, not by --render.
#   DOOR_BEARER_FILE      -- (box file or environment; optional) an absolute path on the box to a file
#     holding one bearer key, no trailing newline: every site then answers 401 (WWW-Authenticate:
#     Bearer) ahead of its warm gate and allowlist unless the request carries `Authorization: Bearer
#     <that key>` -- Caddy's own gate, for an engine with no bearer of its own. Read per request
#     ({file.PATH}): a rotated key needs no restart; a file emptied or unreadable after the render
#     fails closed (a bare or blank bearer never passes). A bad path, a missing, empty or
#     whitespace-only file, or one ending in a newline is REFUSED (exit 3). Unset, the render is
#     byte-identical to a render without the key.
#   /etc/environment      -- VLLM_API_KEY, read only by --check below.
#   TLS_PRESTAGE=1        -- (environment) with ENGINES set, render every engine's site routed to
#     the legacy ENGINE_PORT behind the legacy engine.warming, the $HOSTNAME site unchanged: the
#     engine sites render while the one TP8 engine still serves (the certificate already carries
#     every engine name: the certificate covers the ENGINES names whether or not they serve yet).
#
# usage:
#   tls.sh            setup: download + write Caddyfile + write launcher + launch if down
#   tls.sh --render   print the Caddyfile this run would write, and nothing else: no download,
#                     no write, no launch (the same caddy.env refusals, exit 3)
#   tls.sh --binary   step 1 only: check (and on a fresh box with the stock pin, download) the
#                     caddy binary against CADDY_EXPECT_SHA256 -- the swap check, exit 6 on a
#                     refusal; no caddy.env read, no Caddyfile, no launch
#   tls.sh --check    one-shot, for every name the Caddyfile serves ($HOSTNAME, and with ENGINES
#                     each engine's name): curl https://<name>:$DOOR_PORT/v1/models on this box's own
#                     loopback (--resolve, never -k) with the bearer, print the response code
#                     and the cert issuer read straight off the TLS handshake. This is the
#                     loopback-on-the-box shortcut; the same check can be run from off the box,
#                     over the real DNS name and the mapped port.
. "$(dirname "$0")/lib.sh"

CADDY_BIN="$TOOLS_DIR/caddy"
CADDYFILE="$TOOLS_DIR/Caddyfile"
ENVF="$TOOLS_DIR/caddy.env"
# The box path, as the warm-gate roots below: tls.sh runs from $TOOLS_REMOTE on the box.
CERT_DIR="${CERT_DIR:-$TOOLS_REMOTE/certs}"
CERT_CRT="$CERT_DIR/$HOSTNAME.crt"
CERT_KEY="$CERT_DIR/$HOSTNAME.key"
LAUNCHER="$TOOLS_DIR/caddy-relaunch.sh"
CADDY_DATA_DIR=/workspace/caddy-data
CADDY_LOG=/workspace/caddy.log
CADDY_VERSION=v2.11.7      # expected `caddy version` -- printed as a warning (not fatal) on mismatch
CF_MODULE_VERSION=v0.2.4   # caddy-dns/cloudflare, pinned alongside; bump both together
# The binaries of record, by sha256 -- the sha is the pin, `caddy version` only a label.
# Patched: v2.11.7 + caddy-dns/cloudflare v0.2.4 + local patch (h2 receive window 16/8 MiB),
# go1.26.10, xcaddy v0.4.5; recipe tools/caddy-h2/ (build.sh prints this sha; README.md says why).
CADDY_PATCHED_SHA256="${CADDY_PATCHED_SHA256:-b843f7bb8c43bae5b8b1b2631993a26c8dfeda1dabaa865bc7fda4a2e8d4d884}"
# Stock: caddyserver.com's build-server v2.11.7 + caddy-dns/cloudflare v0.2.4, go1.26.4 -- the
# binary the boxes served before the patched one; the rollback pin, and the only pin a fresh box
# may download (DOWNLOAD_URL below; a build-server rebuild hashes differently and is refused --
# taking one on purpose is CADDY_STOCK_SHA256=<its sha> CADDY_EXPECT_SHA256=<its sha>, the sha the
# refusal printed).
CADDY_STOCK_SHA256="${CADDY_STOCK_SHA256:-22e29521abb1ce00c030192f3c6f9126ecd84c431a672872d04095c1b1c15aba}"
# The pin this run holds the binary to: the box file's CADDY_EXPECT_SHA256 line (lib.sh sources it,
# and its value wins over one exported in the calling shell), else the environment's, else the stock
# sha -- so a box that pins nothing keeps today's stock binary. The front-door deploy flips a box's
# line to CADDY_PATCHED_SHA256 only after that box's binary swap (caddy-h2/README.md "Swap it in"):
# the pin follows the binary, so between the `mv` and the box file's re-stage a run here refuses
# (exit 6) the patched binary under the still-stock pin -- the swap runs no tls.sh in that window.
CADDY_EXPECT_SHA256="${CADDY_EXPECT_SHA256:-$CADDY_STOCK_SHA256}"
DOWNLOAD_URL="https://caddyserver.com/api/download?os=linux&arch=amd64&p=github.com%2Fcaddy-dns%2Fcloudflare&idempotent=true"

# caddy.env (TLS_ALLOW) and the refusals that guard the render: exit 3, the Caddyfile on disk
# untouched. The Cloudflare DNS token never sits on a rented box (#1306: the provider's host has root, and the
# token can rewrite <DOMAIN>): a caddy.env or a box file still carrying a CF_DNS_TOKEN line is
# refused, naming the file -- a re-stage writes caddy.env without it. caddy.env itself is optional now
# (it only repeats the box file's TLS_ALLOW), so a box staged before it existed still renders.
load_caddy_env() {
  local f
  for f in "$ENVF" "$BOX_ENV"; do
    if [ -f "$f" ] && grep -Eq '^[[:space:]]*(export[[:space:]]+)?CF_DNS_TOKEN=' "$f"; then
      echo "REFUSED: $f carries a CF_DNS_TOKEN line -- the Cloudflare token never sits on a rented box (#1306): the door serves certificate files the operator pushes. Re-stage (stage.sh writes caddy.env without it) and delete the line; nothing written"
      exit 3
    fi
  done
  if [ -f "$ENVF" ]; then set -a; . "$ENVF"; set +a; fi
  # Never render an open door by omission. TLS_ALLOW reaches here from boxes/$BOX.env (lib.sh) and from
  # caddy.env (stage.sh writes the box file's value there), so a missing one means the box file has no
  # allowlist: refuse and keep the Caddyfile on disk (Caddy's --watch keeps serving it) rather than
  # write one that answers the whole internet. TLS_ALLOW_ANY=1 (box file or environment) is the one
  # explicit way to render a door with no allowlist.
  if [ -z "${TLS_ALLOW:-}" ] && [ "${TLS_ALLOW_ANY:-0}" != 1 ]; then
    echo "no TLS_ALLOW in $BOX_ENV or $ENVF -- refusing to write a Caddyfile open to the whole internet; add TLS_ALLOW=\"<cidr> <cidr> ...\" to the box file and re-stage, or set TLS_ALLOW_ANY=1 to render an open door on purpose"
    exit 3
  fi
  # The optional door bearer gate (DOOR_BEARER_FILE, box file or environment): Caddy itself answers 401
  # unless the request carries `Authorization: Bearer <the file's contents>`, for an engine with no
  # bearer of its own. Refused (exit 3) unless the path is absolute and plain (it is spliced into the
  # Caddyfile), the file exists and is non-empty (an empty one would match `Bearer ` and nothing else,
  # failing closed, but a door nobody can reach is a mistake to name now), and it ends without a
  # newline: {file.PATH} is the file's bytes verbatim, so a trailing \n would never match a header.
  if [ -n "${DOOR_BEARER_FILE:-}" ]; then
    [[ "$DOOR_BEARER_FILE" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "DOOR_BEARER_FILE='$DOOR_BEARER_FILE' -- an absolute path of [A-Za-z0-9._/-] only; refusing to write it into the Caddyfile"; exit 3; }
    [ -s "$DOOR_BEARER_FILE" ] || { echo "DOOR_BEARER_FILE $DOOR_BEARER_FILE is missing or empty -- refusing to render a door gated on no key"; exit 3; }
    grep -q '[^[:space:]]' "$DOOR_BEARER_FILE" || { echo "DOOR_BEARER_FILE $DOOR_BEARER_FILE is missing or empty (whitespace only) -- refusing to render a door gated on no key"; exit 3; }
    [ "$(tail -c 1 "$DOOR_BEARER_FILE" | od -An -tx1 | tr -d ' \n')" != 0a ] || { echo "DOOR_BEARER_FILE $DOOR_BEARER_FILE ends in a newline -- {file.PATH} is read verbatim, so no header would ever match; rewrite it with printf '%s' (no trailing newline)"; exit 3; }
  fi
  for cidr in ${TLS_ALLOW:-}; do
    [[ "$cidr" =~ ^[0-9A-Fa-f:.]+(/[0-9]+)?$ ]] || { echo "TLS_ALLOW holds '$cidr', not an IP or CIDR -- refusing to write it into the Caddyfile"; exit 3; }
  done
}

# The door's sites, one TLS_SITES entry each: "<host> <warming files, comma-separated> <upstream
# ports, comma-separated>", the legacy $HOSTNAME site first. With no ENGINES (box-a, every other
# box, box-c's TP8 rollback) there is ONE site -- https://$HOSTNAME:$DOOR_PORT ->
# 127.0.0.1:$ENGINE_PORT behind engine.warming -- rendered byte for byte as it always was. With
# ENGINES (box-c's 2xTP4 layout, lib.sh) each engine gets its own site, https://$ENGINE_HOST:$DOOR_PORT
# (box-ca.<DOMAIN>, the name the router dials for that copy) -> 127.0.0.1:<its port>
# behind engine-<id>.warming, and the legacy $HOSTNAME site fronts the copies together: lb_policy
# first across every engine port, passive health so a copy that is down is skipped, and a 503 only
# while EVERY copy is warming. TLS_PRESTAGE=1 (the window's prep, before the 2xTP4 cutover)
# renders the engine sites while the TP8 engine still serves: each one routes to the legacy
# ENGINE_PORT behind the legacy engine.warming, and the legacy site stays exactly the one-engine
# site, so the engine names get their certificates ahead of the cutover without moving any live
# traffic.
TLS_SITES=()
tls_build_sites() {
  local id ports="" warms="" engine_sites=()
  TLS_SITES=()
  engines_check
  if [ -z "$ENGINES_IDS" ]; then
    [ "${TLS_PRESTAGE:-0}" != 1 ] || note "TLS_PRESTAGE=1, but $BOX_ENV sets no ENGINES -- nothing to prestage; rendering the one-engine door"
    TLS_SITES=("$HOSTNAME engine.warming $ENGINES_LEGACY_PORT")
    return 0
  fi
  for id in $ENGINES_IDS; do
    engine_select "$id"
    if [ "${TLS_PRESTAGE:-0}" = 1 ]; then
      engine_sites+=("$ENGINE_HOST engine.warming $ENGINES_LEGACY_PORT")
    else
      engine_sites+=("$ENGINE_HOST ${ENGINE_WARMING##*/} $ENGINE_PORT")
      warms="${warms:+$warms,}${ENGINE_WARMING##*/}"
      ports="${ports:+$ports,}$ENGINE_PORT"
    fi
  done
  engine_select   # back to the legacy names: everything below reads $ENGINE_PORT as the legacy one
  # The legacy site first, so a prestage render is the one-engine Caddyfile with sites appended.
  if [ "${TLS_PRESTAGE:-0}" = 1 ]; then
    TLS_SITES=("$HOSTNAME engine.warming $ENGINES_LEGACY_PORT" "${engine_sites[@]}")
  else
    TLS_SITES=("$HOSTNAME $warms $ports" "${engine_sites[@]}")
  fi
  return 0
}

# reverse_proxy to one or more loopback ports, at indent $1. One port is today's block; several
# (the legacy site over ENGINES) add the load-balancing lines.
tls_proxy() {
  local in="$1" ports="$2" p up=""
  for p in ${ports//,/ }; do up="${up:+$up }127.0.0.1:$p"; done
  echo "${in}reverse_proxy $up {"
  case "$ports" in
    *,*)
      echo "${in}	# Several engines: the first one up takes every request. A failed dial or round trip"
      echo "${in}	# marks that copy down for fail_duration, and the request is retried on the next"
      echo "${in}	# for up to lb_try_duration -- one copy restarting never fails the shared name."
      echo "${in}	lb_policy first"
      echo "${in}	lb_try_duration 10s"
      echo "${in}	fail_duration 30s"
      echo "${in}	max_fails 1"
      ;;
  esac
  echo "${in}	flush_interval -1"
  echo "${in}	transport http {"
  echo "${in}		read_timeout 0"
  echo "${in}		write_timeout 0"
  echo "${in}		response_header_timeout 0"
  echo "${in}		keepalive 4s"
  echo "${in}	}"
  echo "${in}}"
}

# One site block: tls_site <host> <warming files, comma-separated> <upstream ports, comma-separated>.
tls_site() {
  local host="$1" warm="$2" ports="$3" w first=1
  echo "https://$host:$DOOR_PORT {"
  echo "	tls $CERT_CRT $CERT_KEY"
  # The warm gate (#379): relaunch.sh writes $TOOLS_REMOTE/engine$SFX.warming at every engine start
  # and warm-short.py removes it after the short in-process warm (or at its timeout). While it
  # exists EVERY path answers 503 -- the router's /v1/loads liveness, /v1/chat/completions probes,
  # /get_server_info and /v1/models alike -- so the router sees no READY backend on a cold
  # process. No flag, no gate: the door is open by default.
  case "$warm" in
    *,*)
      # Several warming files (the legacy site over ENGINES): the matcher holds only while EVERY
      # one exists. The first is a plain file matcher; each further one is not { not { file } },
      # a different matcher type in the same set, so Caddy ANDs it with the first.
      echo "	@warming {"
      for w in ${warm//,/ }; do
        if [ "$first" = 1 ]; then
          echo "		file {"
          echo "			root $TOOLS_REMOTE"
          echo "			try_files /$w"
          echo "		}"
          first=0
        else
          echo "		not {"
          echo "			not {"
          echo "				file {"
          echo "					root $TOOLS_REMOTE"
          echo "					try_files /$w"
          echo "				}"
          echo "			}"
          echo "		}"
        fi
      done
      echo "	}"
      ;;
    *)
      echo "	@warming file {"
      echo "		root $TOOLS_REMOTE"
      echo "		try_files /$warm"
      echo "	}"
      ;;
  esac
  # The door bearer gate (optional; unset, the Caddyfile is byte-identical to before): ahead of the
  # warm gate and the allowlist, a request without `Authorization: Bearer <DOOR_BEARER_FILE's
  # contents>` gets 401. {file.PATH} is read per request, so a rotated key needs no Caddy restart. A
  # request passes only when BOTH hold (`not` over a block negates the AND): the header equals the
  # file's key, AND it carries a non-blank bearer -- so a key file emptied or made unreadable after
  # the render, which turns the first into `Bearer `, can never let a bare or blank bearer through.
  # The path is validated in load_caddy_env.
  if [ -n "${DOOR_BEARER_FILE:-}" ]; then
    echo "	@nokey {"
    echo "		not {"
    echo "			header Authorization \"Bearer {file.$DOOR_BEARER_FILE}\""
    echo "			header_regexp Authorization \"^Bearer [^[:space:]]\""
    echo "		}"
    echo "	}"
    echo "	handle @nokey {"
    echo "		header WWW-Authenticate Bearer"
    echo "		respond 401"
    echo "	}"
  fi
  if [ -n "${TLS_ALLOW:-}" ]; then
    echo "	@allowed remote_ip $TLS_ALLOW"
    echo "	handle @allowed {"
    echo "		handle @warming {"
    echo "			header Retry-After 15"
    echo "			respond \"engine warming\" 503"
    echo "		}"
    echo "		handle {"
    echo "			# flush_interval -1: stream every chunk to the client at once, required for"
    echo "			# SSE (text/event-stream) token-by-token completions -- the default buffers."
    echo "			# No request_body size limit is set: Caddy imposes none by default on a"
    echo "			# reverse-proxied request, deliberate here for long contexts. Generous"
    echo "			# transport timeouts: a 1M-token prefill can take minutes before the first"
    echo "			# byte comes back, and Caddy's reverse_proxy defaults are far shorter."
    tls_proxy "			" "$ports"
    echo "		}"
    echo "	}"
    echo "	handle {"
    echo "		respond 403"
    echo "	}"
  else
    echo "	handle @warming {"
    echo "		header Retry-After 15"
    echo "		respond \"engine warming\" 503"
    echo "	}"
    echo "	handle {"
    echo "		# flush_interval -1: stream every chunk to the client at once, required for SSE"
    echo "		# (text/event-stream) token-by-token completions -- the default buffers."
    echo "		# No request_body size limit is set: Caddy imposes none by default on a"
    echo "		# reverse-proxied request, deliberate here for long contexts. Generous transport"
    echo "		# timeouts: a 1M-token prefill can take minutes before the first byte comes back,"
    echo "		# and Caddy's reverse_proxy defaults are far shorter."
    tls_proxy "		" "$ports"
    echo "	}"
  fi
  echo "}"
}

# The whole Caddyfile: every site, a blank line between two.
tls_render() {
  local s n=0
  for s in "${TLS_SITES[@]}"; do
    [ "$n" = 0 ] || echo
    # shellcheck disable=SC2086
    tls_site $s
    n=$((n + 1))
  done
}

# `caddy adapt` the render before it replaces the Caddyfile on disk: a render Caddy cannot load would
# be refused by --watch (which keeps the old config and logs the error), and the next cold start --
# the watchdog's relaunch, a container restart -- would then fail outright. Exit 3 on a refusal, the
# Caddyfile on disk untouched. Skipped while no binary is on disk yet (a DRY_RUN before the first
# download).
validate_caddyfile() {
  [ -x "$CADDY_BIN" ] || { note "no caddy binary on disk -- the render is not adapted"; return 0; }
  local f out
  f=$(mktemp "${TMPDIR:-/tmp}/tls-caddyfile.XXXXXX")
  printf '%s\n' "$1" > "$f"
  if out=$("$CADDY_BIN" adapt --config "$f" --adapter caddyfile 2>&1 >/dev/null); then
    rm -f "$f"
    note "caddy adapt: the render loads"
    return 0
  fi
  rm -f "$f"
  echo "caddy adapt refused the render -- $CADDYFILE left as it was:"
  printf '%s\n' "$out" | tail -n 5
  exit 3
}

# Step 1: the caddy binary, held to CADDY_EXPECT_SHA256. Before this pin a box with a binary but no
# caddy.sha256 beside it (or with neither) silently re-downloaded the stock build server's binary over
# whatever was there -- on a box serving the patched h2-window build, a quiet revert to stock's 1 MiB
# windows. Now nothing is downloaded over a binary on disk: a mismatch is a refusal (exit 6) naming
# both fixes, and a download happens only on a box with no binary, under the stock pin, and is
# installed only when it hashes to that pin (a build-server rebuild does not, and is refused).
# Under DRY_RUN=1 (which writes and downloads nothing) a refusal is printed as `DRY: would REFUSE` and
# the preview goes on; `--binary` still exits 6 on it.
CADDY_REFUSED=0
caddy_sha() { sha256sum "$1" | awk '{print $1}'; }
caddy_pin_name() {
  case "$1" in
    "$CADDY_PATCHED_SHA256") echo "patched h2-window build, tools/caddy-h2/" ;;
    "$CADDY_STOCK_SHA256") echo "stock build-server binary" ;;
    *) echo "no binary of record" ;;
  esac
}
# The build that makes the patched binary on this box (staged with the tool set); the sha reproduces
# only at BUILD_DIR=/opt/caddy-h2 (caddy-h2/README.md "Build").
CADDY_BUILD_HINT="BUILD_DIR=/opt/caddy-h2 bash $TOOLS_DIR/caddy-h2/build.sh, then mv /opt/caddy-h2/caddy-h2win over $CADDY_BIN, then (if not yet) the box file's pin to CADDY_PATCHED_SHA256, re-staged (the hard cut in caddy-h2/README.md \"Swap it in\"), and re-run tls.sh"
# caddy_refuse <what is wrong> <fix>...: the refusal, the pin and where it came from, one line per fix;
# exit 6 (DRY_RUN=1: printed as a preview, and the caller returns).
caddy_refuse() {
  local fix
  CADDY_REFUSED=1
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY: would REFUSE (exit 6): $1"; else echo "REFUSED: $1"; fi
  shift
  echo "  the pin: CADDY_EXPECT_SHA256=$CADDY_EXPECT_SHA256 ($(caddy_pin_name "$CADDY_EXPECT_SHA256")) -- $(caddy_pin_source)"
  for fix in "$@"; do echo "  fix: $fix"; done
  [ "${DRY_RUN:-0}" = 1 ] || exit 6
}
caddy_pin_source() {
  if grep -q '^CADDY_EXPECT_SHA256=' "$BOX_ENV" 2>/dev/null; then echo "the box file's line ($BOX_ENV)"
  elif [ "$CADDY_EXPECT_SHA256" = "$CADDY_STOCK_SHA256" ]; then echo "the default (the box file pins nothing)"
  else echo "the environment (the box file pins nothing)"; fi
}
caddy_binary() {
  local got
  if [ -e "$CADDY_BIN" ]; then
    got=$(caddy_sha "$CADDY_BIN")
    if [ "$got" != "$CADDY_EXPECT_SHA256" ]; then
      caddy_refuse "$CADDY_BIN hashes to $got ($(caddy_pin_name "$got")), not the pin -- left as it is, nothing downloaded over it" \
        "to serve the pinned build: $( [ "$CADDY_EXPECT_SHA256" = "$CADDY_PATCHED_SHA256" ] && echo "build it on this box with $CADDY_BUILD_HINT" || echo "put the binary that hashes to $CADDY_EXPECT_SHA256 at $CADDY_BIN (the patched one: build it on this box with $CADDY_BUILD_HINT)")" \
        "to keep the binary that is there: set CADDY_EXPECT_SHA256=$got in $BOX_ENV (the box file's line wins over an exported one; re-stage from the dev box)"
      return 0
    fi
    note "caddy binary matches the pin ($(caddy_pin_name "$got")): $got"
    if [ "${DRY_RUN:-0}" != 1 ]; then
      [ -x "$CADDY_BIN" ] || chmod +x "$CADDY_BIN"
      # caddy.sha256 records what is installed: rewritten when a hand-installed binary left it stale.
      [ "$(cat "$CADDY_BIN.sha256" 2>/dev/null)" = "$got  $CADDY_BIN" ] || printf '%s  %s\n' "$got" "$CADDY_BIN" > "$CADDY_BIN.sha256"
    fi
  elif [ "$CADDY_EXPECT_SHA256" = "$CADDY_STOCK_SHA256" ]; then
    note "no caddy binary on disk, stock pin: downloading caddy (expect $CADDY_VERSION) + caddy-dns/cloudflare (expect $CF_MODULE_VERSION)"
    if [ "${DRY_RUN:-0}" = 1 ]; then
      note "DRY: curl -fsSL --connect-timeout 20 --max-time 300 -o $CADDY_BIN.tmp $DOWNLOAD_URL"
    else
      # bounded: an unreachable caddyserver.com held a whole install for 133 s (img3, 2026-11-04)
      curl -fsSL --connect-timeout 20 --max-time 300 -o "$CADDY_BIN.tmp" "$DOWNLOAD_URL"
      got=$(caddy_sha "$CADDY_BIN.tmp")
      if [ "$got" != "$CADDY_STOCK_SHA256" ]; then
        rm -f "$CADDY_BIN.tmp"
        caddy_refuse "the build server served sha256 $got, not the stock binary of record -- not installed (a rebuild there)" \
          "copy the stock binary from a box that has it to $CADDY_BIN" \
          "or build the patched one with $CADDY_BUILD_HINT, its sha pinned in $BOX_ENV" \
          "or take the rebuild on purpose: CADDY_STOCK_SHA256=$got CADDY_EXPECT_SHA256=$got tls.sh (with no CADDY_EXPECT_SHA256 line in the box file)"
      fi
      chmod +x "$CADDY_BIN.tmp"
      mv "$CADDY_BIN.tmp" "$CADDY_BIN"
      printf '%s  %s\n' "$got" "$CADDY_BIN" | tee "$CADDY_BIN.sha256"
    fi
  else
    if [ "$CADDY_EXPECT_SHA256" = "$CADDY_PATCHED_SHA256" ]; then
      caddy_refuse "no caddy binary at $CADDY_BIN, and the pin is the PATCHED h2-window build, which is never downloaded (a fresh box): stock is not fetched in its place" \
        "re-stage from the dev box: stage.sh's last step copies this binary from the Mac's artifact store (<HOME>/workspace/backups/caddy-h2/) onto a box with none, then re-run tls.sh" \
        "build it on this box with $CADDY_BUILD_HINT" \
        "or serve stock on purpose: set CADDY_EXPECT_SHA256=$CADDY_STOCK_SHA256 in $BOX_ENV (re-stage) and re-run tls.sh, which downloads it"
    else
      caddy_refuse "no caddy binary at $CADDY_BIN, and the pin is neither the stock nor the patched build -- nothing to download" \
        "put the binary that hashes to $CADDY_EXPECT_SHA256 at $CADDY_BIN, or build the patched one with $CADDY_BUILD_HINT" \
        "or pin the stock sha in $BOX_ENV: CADDY_EXPECT_SHA256=$CADDY_STOCK_SHA256 (re-stage), which tls.sh downloads"
    fi
    return 0
  fi
  if [ -x "$CADDY_BIN" ]; then
    GOT_VERSION=$("$CADDY_BIN" version 2>/dev/null | awk '{print $1}')
    echo "   $("$CADDY_BIN" version 2>/dev/null)"
    case "$GOT_VERSION" in
      "$CADDY_VERSION"*) : ;;
      *) note "WARNING: caddy reports '$GOT_VERSION', expected '$CADDY_VERSION' -- the sha matches the pin, so the pin and CADDY_VERSION disagree: bump them together" ;;
    esac
  else
    note "no caddy binary on disk yet (DRY_RUN skipped the download)"
  fi
}

check() {
  KEY=$(grep -a '^VLLM_API_KEY=' /etc/environment | head -1 | cut -d= -f2- | tr -d '"')
  [ -n "$KEY" ] || { echo "no VLLM_API_KEY in /etc/environment"; exit 2; }
  tls_build_sites
  local site host rc=0
  # Every site the render holds: $HOSTNAME, and with ENGINES each engine's own name too.
  for site in "${TLS_SITES[@]}"; do
    host="${site%% *}"
    # The bearer reaches curl on its STDIN (`-H @-`, curl >= 7.55) via `printf`, a shell builtin, never
    # as an argument: an argv `-H "Authorization: Bearer $KEY"` sits in /proc/<pid>/cmdline while the
    # call runs (#3003).
    code=$(printf 'Authorization: Bearer %s\n' "$KEY" | curl -s -m 20 -o /dev/null -w '%{http_code}' \
      --resolve "$host:$DOOR_PORT:127.0.0.1" \
      -H @- \
      "https://$host:$DOOR_PORT/v1/models")
    echo "https://$host:$DOOR_PORT/v1/models -> $code"
    issuer=$(echo | openssl s_client -connect "127.0.0.1:$DOOR_PORT" -servername "$host" 2>/dev/null \
      | openssl x509 -noout -issuer 2>/dev/null)
    echo "issuer: ${issuer:-<no certificate read -- caddy down, or no certificate pushed yet>}"
    { [ "$code" = 200 ] && [ -n "$issuer" ]; } || rc=1
  done
  exit "$rc"
}

if [ "${1:-}" = --check ]; then check; fi
if [ "${1:-}" = --render ]; then
  load_caddy_env
  tls_build_sites
  tls_render
  exit 0
fi
if [ "${1:-}" = --binary ]; then
  mkdir -p "$TOOLS_DIR"
  echo "== 1. caddy binary"
  caddy_binary
  [ "$CADDY_REFUSED" = 0 ] || exit 6
  exit 0
fi

mkdir -p "$TOOLS_DIR" "$CADDY_DATA_DIR"

# caddy.env's refusals (exit 3) first: a run that cannot render a door touches no binary either.
load_caddy_env
# The patched binary's HTTP/2 receive-window overrides (caddy-h2/README.md "Overrides"): Caddy reads
# them from its own environment at start, and caddy.env is rewritten whole by every stage.sh, so a
# value set here (the box file, which lib.sh sources, or this run's environment) is baked into the
# launcher as an export; unset, the launcher carries none and the binary's 16/8 MiB defaults hold.
# Checked here, with caddy.env's refusals, before anything is written.
H2_ENV_LINES=""
for h2var in CADDY_H2_RECV_BUFFER_PER_CONNECTION CADDY_H2_RECV_BUFFER_PER_STREAM; do
  h2val="${!h2var:-}"
  [ -n "$h2val" ] || continue
  case "$h2val" in *[!0-9]*) echo "REFUSED: $h2var='$h2val' is not a byte count -- nothing written"; exit 3 ;; esac
  H2_ENV_LINES="${H2_ENV_LINES}export $h2var=$h2val  # from tls.sh's environment / the box file
"
done

echo "== 1. caddy binary"
caddy_binary

echo "== 2. Caddyfile"
tls_build_sites
# The certificate pair every site serves (#1306), checked before the Caddyfile that names it is
# written: present, unexpired, the key matching the certificate, and every site's name on it. A
# refusal is exit 5, the Caddyfile on disk untouched (under DRY_RUN=1 a `DRY: would REFUSE` and the
# preview goes on). The fix is always the same: push a pair from off the box.
cert_refuse() {
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY: would REFUSE (exit 5): $*"; return 0; fi
  echo "REFUSED: $* -- push the door's certificate from off the box (the operator's own issuer); nothing written"
  exit 5
}
cert_check() {
  local site host
  if [ ! -r "$CERT_CRT" ] || [ ! -r "$CERT_KEY" ]; then cert_refuse "no certificate pair at $CERT_CRT / $CERT_KEY"; return 0; fi
  openssl x509 -noout -checkend 0 -in "$CERT_CRT" > /dev/null 2>&1 || { cert_refuse "$CERT_CRT has expired (or does not parse)"; return 0; }
  [ "$(openssl x509 -noout -pubkey -in "$CERT_CRT" 2>/dev/null)" = "$(openssl pkey -pubout -in "$CERT_KEY" 2>/dev/null)" ] \
    || { cert_refuse "$CERT_KEY is not the key of $CERT_CRT"; return 0; }
  for site in "${TLS_SITES[@]}"; do
    host="${site%% *}"
    openssl x509 -noout -checkhost "$host" -in "$CERT_CRT" 2>/dev/null | grep -q 'does match' \
      || cert_refuse "$host is not on $CERT_CRT (its names: $(openssl x509 -noout -ext subjectAltName -in "$CERT_CRT" 2>/dev/null | tail -n +2 | tr -d ' '))"
  done
  note "certificate $CERT_CRT: $(openssl x509 -noout -enddate -in "$CERT_CRT" | tr -d '\n'), covers ${#TLS_SITES[@]} site name(s)"
}
cert_check
if [ -z "$ENGINES_IDS" ]; then
  note "upstream=127.0.0.1:$ENGINE_PORT allow=${TLS_ALLOW:-<none: TLS_ALLOW_ANY=1, open, bearer-only>}"
else
  for site in "${TLS_SITES[@]}"; do
    read -r s_host s_warm s_ports <<< "$site"
    note "site $s_host upstream=127.0.0.1:${s_ports//,/ 127.0.0.1:} gate=${s_warm//,/+}${TLS_PRESTAGE:+ (TLS_PRESTAGE=$TLS_PRESTAGE)}"
  done
  note "allow=${TLS_ALLOW:-<none: TLS_ALLOW_ANY=1, open, bearer-only>}"
fi

CADDY_CONTENT=$(tls_render)
validate_caddyfile "$CADDY_CONTENT"
if [ "${DRY_RUN:-0}" = 1 ]; then
  note "DRY: would write $CADDYFILE:"
  printf '%s\n' "$CADDY_CONTENT"
else
  printf '%s\n' "$CADDY_CONTENT" > "$CADDYFILE"
  note "wrote $CADDYFILE"
  cat "$CADDYFILE"
fi

echo "== 3. launcher"
PKILL_PAT="${CADDY_BIN%caddy}cadd[y]"
[ -z "$H2_ENV_LINES" ] || note "launcher exports the h2 receive-window overrides: $(printf '%s' "$H2_ENV_LINES" | awk '{print $2}' | tr '\n' ' ')"
LAUNCHER_CONTENT=$(cat <<LAUNCHEREOF
#!/bin/bash
# relaunch Caddy for $BOX's direct-TLS front door. Written by tls.sh -- do not hand-edit;
# re-run tls.sh to regenerate (it rewrites this file and the Caddyfile every time). --watch on
# \`caddy run\` means an ordinary Caddyfile edit (e.g. an allowlist change) is picked up live
# without needing this relaunch at all; this script exists for a cold start and for
# watchdog.sh to call when caddy's process is gone outright. XDG_DATA_HOME pins Caddy's own
# state under /workspace. The certificate is a FILE ($CERT_CRT, pushed in from outside
# this kit): Caddy solves no ACME challenge and this launcher loads no caddy.env and no
# token (#1306).
${H2_ENV_LINES}pkill -TERM -f '$PKILL_PAT run' 2>/dev/null
sleep 2
pkill -KILL -f '$PKILL_PAT run' 2>/dev/null
mkdir -p "$CADDY_DATA_DIR"
export XDG_DATA_HOME="$CADDY_DATA_DIR"
echo "===== CADDY (RE)LAUNCH \$(date -u) =====" >> "$CADDY_LOG"
nohup setsid "$CADDY_BIN" run --config "$CADDYFILE" --adapter caddyfile --watch >> "$CADDY_LOG" 2>&1 < /dev/null &
disown
sleep 3
if pgrep -f '$PKILL_PAT run' > /dev/null; then
  echo "caddy running, pid \$(pgrep -f '$PKILL_PAT run' | tr '\n' ' ')"
else
  echo "caddy failed to start -- check $CADDY_LOG"
  exit 1
fi
LAUNCHEREOF
)
if [ "${DRY_RUN:-0}" = 1 ]; then
  note "DRY: would write $LAUNCHER (chmod +x):"
  printf '%s\n' "$LAUNCHER_CONTENT"
else
  printf '%s\n' "$LAUNCHER_CONTENT" > "$LAUNCHER"
  chmod +x "$LAUNCHER"
  note "wrote $LAUNCHER"
fi

# The served certificate, checked against the file: for every name the Caddyfile serves, the TLS
# handshake on 127.0.0.1:$DOOR_PORT must present $CERT_CRT's own sha256 fingerprint within 60s. With
# Caddy already up and a name still serving another certificate (a pair cert-install.sh replaced
# while Caddy was down, or a render that changed nothing --watch would reload), one
# `caddy reload --force` first: Caddy re-reads a manually loaded certificate file only on a reload,
# and --watch reloads only a CHANGED Caddyfile. A timeout prints the names still wrong and the tail
# of the log and exits 5 -- never a caller left believing setup succeeded.
fp_file() { openssl x509 -noout -fingerprint -sha256 -in "$1" 2>/dev/null | cut -d= -f2; }
fp_served() {
  echo | openssl s_client -connect "127.0.0.1:$DOOR_PORT" -servername "$1" 2>/dev/null \
    | openssl x509 -noout -fingerprint -sha256 2>/dev/null | cut -d= -f2
}
served_wrong() {   # prints the names not serving $CERT_CRT
  local want host
  want=$(fp_file "$CERT_CRT")
  for host in "$@"; do [ -n "$want" ] && [ "$(fp_served "$host")" = "$want" ] || echo "$host"; done
}
verify_served() {
  local pending waited=0 host
  note "waiting up to 60s for every name to serve $CERT_CRT ($(fp_file "$CERT_CRT")): $*"
  while :; do
    pending=$(served_wrong "$@")
    [ -n "$pending" ] || break
    if [ "$waited" -ge 60 ]; then
      note "FATAL: after 60s these names do not serve $CERT_CRT: $(echo $pending) -- last 5 lines of $CADDY_LOG:"
      tail -n 5 "$CADDY_LOG" >&2
      exit 5
    fi
    sleep 5; waited=$((waited + 5))
  done
  for host in "$@"; do
    note "cert for $host served: $(echo | openssl s_client -connect "127.0.0.1:$DOOR_PORT" -servername "$host" 2>/dev/null | openssl x509 -noout -issuer -enddate 2>/dev/null | tr '\n' ' ')"
  done
}

CERT_NAMES=()
for site in "${TLS_SITES[@]}"; do CERT_NAMES+=("${site%% *}"); done

if ! pgrep -f "$PKILL_PAT run" > /dev/null; then
  if [ "${DRY_RUN:-0}" = 1 ]; then
    note "DRY: caddy not running -- would run $LAUNCHER, then check every name serves $CERT_CRT"
  else
    note "caddy not running, launching"
    bash "$LAUNCHER"
    verify_served "${CERT_NAMES[@]}"
  fi
else
  note "caddy already running (pid $(pgrep -f "$PKILL_PAT run" | tr '\n' ' ')); Caddyfile rewritten in place and picked up by --watch. Run $LAUNCHER by hand to force a restart."
  if [ "${DRY_RUN:-0}" = 1 ]; then
    note "DRY: would check every name serves $CERT_CRT, with a \`caddy reload --force\` first if one does not"
  else
    sleep 3   # --watch's own reload of a changed Caddyfile first
    if [ -n "$(served_wrong "${CERT_NAMES[@]}")" ]; then
      note "a name does not serve $CERT_CRT yet -- caddy reload --force (re-reads the certificate files)"
      XDG_DATA_HOME="$CADDY_DATA_DIR" "$CADDY_BIN" reload --force --config "$CADDYFILE" --adapter caddyfile \
        || note "caddy reload --force failed -- the check below says whether the door serves the file anyway"
    fi
    verify_served "${CERT_NAMES[@]}"
  fi
fi
