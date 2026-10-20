#!/usr/bin/env bash
# Wait until the laptop battery reaches THRESHOLD%, then: interrupt training (train.py saves
# final_model.zip on exit), copy runs/ down, delete the pod, remove the cron pull.
# Usage: setsid nohup scripts/save_and_delete_pod.sh IP PORT POD_ID [THRESHOLD=6] &
set -u
IP=$1; PORT=$2; POD=$3; THRESHOLD=${4:-6}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=runs/20261019-140413
SSH=(ssh -i "$HOME/.runpod/ssh/runpodctl-ssh-key" -p "$PORT" -o ConnectTimeout=20 -o BatchMode=yes "root@$IP")
log() { echo "$(date -u +%T) $*" >> "$ROOT/logs/save_and_delete.log"; }

log "waiting for battery <= $THRESHOLD%"
until [ "$(cat /sys/class/power_supply/BAT1/capacity)" -le "$THRESHOLD" ]; do sleep 20; done
log "battery $(cat /sys/class/power_supply/BAT1/capacity)%: stopping training"

"${SSH[@]}" "tmux send-keys -t train C-c; for i in \$(seq 90); do
  [ /workspace/ShowdownAgent/$RUN/final_model.zip -nt /workspace/ShowdownAgent/$RUN/train.log ] && exit 0
  tmux has-session -t train 2>/dev/null || exit 0; sleep 2; done" && log "training stopped"
"${SSH[@]}" "ls -la /workspace/ShowdownAgent/$RUN/final_model.zip" >> "$ROOT/logs/save_and_delete.log" 2>&1

for attempt in 1 2 3; do
  "$ROOT/scripts/pull_once.sh" "$IP" "$PORT" && [ -f "$ROOT/$RUN/final_model.zip" ] && { log "copied (attempt $attempt)"; break; }
  log "copy attempt $attempt failed"; sleep 5
done

if [ -f "$ROOT/$RUN/final_model.zip" ]; then
  PATH=$HOME/.local/bin:$PATH runpodctl pod delete "$POD" >> "$ROOT/logs/save_and_delete.log" 2>&1 && log "pod $POD deleted"
  crontab -l | grep -v gloria-pull | crontab - && log "cron pull removed"
else
  log "final_model.zip not copied: pod left running so nothing is lost"
fi
