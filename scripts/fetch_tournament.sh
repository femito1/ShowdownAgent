#!/usr/bin/env bash
# Retry starting a stopped pod (its GPU may be in use), copy tournament results, stop it again.
# Usage: setsid nohup scripts/fetch_tournament.sh POD_ID [HOURS=8] &      Log: logs/fetch.log
set -u
POD=$1; HOURS=${2:-8}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PATH=$HOME/.local/bin:$PATH
log() { echo "$(date -u +%T) $*" >> "$ROOT/logs/fetch.log"; }
end=$(( $(date +%s) + HOURS * 3600 ))
until runpodctl pod start "$POD" 2>&1 | grep -qv '"error"'; do
  [ "$(date +%s)" -ge "$end" ] && { log "gave up: GPU never freed"; exit 1; }
  log "host GPU busy, retrying in 10 min"; sleep 600
done
log "pod started"
for i in $(seq 60); do
  I=$(runpodctl ssh info "$POD" 2>/dev/null); echo "$I" | grep -q '"ip"' && break; sleep 10
done
read -r IP PORT < <(echo "$I" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['ip'],d['port'])" 2>/dev/null)
mkdir -p "$ROOT/runs/phase0"
for f in tournament.txt autostop.log; do
  rsync -az -e "ssh -i $HOME/.runpod/ssh/runpodctl-ssh-key -p $PORT -o ConnectTimeout=20 -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
    "root@$IP:/workspace/runs/$f" "$ROOT/runs/phase0/" && log "copied $f"
done
runpodctl pod stop "$POD" >> "$ROOT/logs/fetch.log" 2>&1 && log "pod stopped"
