#!/usr/bin/env bash
# Download the latest results pushed by a pod (scripts/sync_results.sh) into runs/remote/.
# Usage: scripts/fetch_results.sh [REMOTE=git@github.com:femito1/gloria-runs.git]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd); DEST=$ROOT/runs/remote
REMOTE=${1:-git@github.com:femito1/gloria-runs.git}
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/gloria_runs_deploy -o IdentitiesOnly=yes"
if [ -d "$DEST/.git" ]; then git -C "$DEST" fetch -q origin && git -C "$DEST" reset -q --hard origin/main
else git clone -q "$REMOTE" "$DEST"; fi
git -C "$DEST" log -1 --format="%s"; tail -3 "$DEST/evals.txt" 2>/dev/null
