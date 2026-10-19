#!/usr/bin/env bash
# Copy runs/ (checkpoints, tensorboard, logs) from the pod every 15 min.
# Usage: scripts/pull_runs.sh IP PORT
set -u
IP=$1; PORT=$2
while true; do
  rsync -az -e "ssh -i $HOME/.runpod/ssh/runpodctl-ssh-key -p $PORT -o ConnectTimeout=20" \
    "root@$IP:/workspace/ShowdownAgent/runs/" "$(dirname "$0")/../runs/" || echo "pull failed $(date)"
  sleep 900
done
