#!/usr/bin/env bash
# sanitize_scan.sh: scan every file under record/ (docs, CSVs, raw copies; .gz read through zcat)
# for identifiers that must not leave the repo. Prints each pattern's hit count and the hits; exit 0 = all clean.
# Usage (from the repo root): bash record/data/scripts/sanitize_scan.sh
set -u
ROOT="${1:-record}"
fail=0
scan() { # name, extended regex, optional exclude regex applied to the matched text
  local name="$1" re="$2" ex="${3:-^$}" hits
  hits=$( { grep -rnoE --exclude=sanitize_scan.sh "$re" "$ROOT"; \
            find "$ROOT" -name '*.gz' | while IFS= read -r gz; do gzip -dc "$gz" | grep -noE "$re" | sed "s#^#$gz:#"; done; } 2>/dev/null | grep -vE ":($ex)\$" )
  local n; n=$(printf '%s' "$hits" | grep -c . || true)
  printf '%-34s %s\n' "$name" "$n"
  if [ "$n" -gt 0 ]; then printf '%s\n' "$hits" | head -20; fail=1; fi
}
echo "files scanned: $(find "$ROOT" -type f | wc -l | tr -d ' ') (gz: $(find "$ROOT" -name '*.gz' | wc -l | tr -d ' '))"
# Box identifiers come from the box files at run time, so this script never carries them itself.
ids=$(for b in box-a box-b; do
        f="setup/box/boxes/$b.env"; [ -f "$f" ] || continue
        grep -E '^(INSTANCE_ID|BOX_IP|SSH_PORT|APP_PORT_EXT|DOOR_PORT|HOSTNAME)=' "$f" | cut -d= -f2- | tr -d '"' | sed 's/[[:space:]]*#.*//'
      done | grep -vE '^$|^(127\.0\.0\.1|localhost|[0-9]{1,4})$|^<[^>]*>$' | sed 's/[.]/\\./g' | sort -u | paste -sd'|' -)
echo "box identifiers read from setup/box/boxes/*.env: $(printf '%s' "$ids" | tr '|' '\n' | grep -c .)"
[ -n "$ids" ] && scan "box ids (instance, IP, ports, host)" "(^|[^0-9A-Za-z])($ids)([^0-9A-Za-z]|\$)"
scan "the provider ids written as 'instance N'"  '(instance|offer|machine|host|contract)[ _=:#]+[0-9]{6,8}'
scan "IPv4 other than loopback/any"     '\b(25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.[0-9]{1,3}\.[0-9]{1,3}\b' '127\.0\.0\.1|0\.0\.0\.0'
scan "door hostnames"                   '[A-Za-z0-9.-]+\.(cinema|ezo)\.dev(:[0-9]+)?'
scan "router keys / tokens"             'sbr_(live|adm|test)_[A-Za-z0-9]+|Bearer [A-Za-z0-9._-]{6,}|hf_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9_-]{10,}'
scan "engine API-key var / kube config" 'VLLM_API_KEY|kubeconfig'
scan "router key / account ids"         'api_key_id|(^|[^0-9.])[0-9]{18,19}([^0-9]|$)'
scan "router pod names"                 'router-s(8|13)-[a-z0-9]{4,}'
scan "ssh logins"                       'root@[A-Za-z0-9.-]+|ssh -p'
exit $fail
