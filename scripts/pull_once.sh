#!/usr/bin/env bash
# One rsync of the pod's runs/ into local runs/ (run from cron while a pod is training).
# Usage: scripts/pull_once.sh IP PORT
ROOT=$(cd "$(dirname "$0")/.." && pwd)
rsync -az -e "ssh -i $HOME/.runpod/ssh/runpodctl-ssh-key -p $2 -o ConnectTimeout=20 -o BatchMode=yes" \
  "root@$1:/workspace/ShowdownAgent/runs/" "$ROOT/runs/" && date -u +"%F %T pulled" >> "$ROOT/logs/pull.log"
