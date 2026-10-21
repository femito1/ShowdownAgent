#!/usr/bin/env bash
# Download the latest results pushed by a pod or Slurm job (scripts/sync_results.sh) into runs/remote/.
# The remote only holds the newest checkpoint and pool snapshots, so each fetch also copies them
# into runs/archive/ (never overwritten) to keep a history for tournaments.
# Usage: scripts/fetch_results.sh [REMOTE=git@github.com:femito1/porygon-runs.git]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd); DEST=$ROOT/runs/remote
REMOTE=${1:-git@github.com:femito1/porygon-runs.git}
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/porygon_runs_deploy -o IdentitiesOnly=yes"
if [ -d "$DEST/.git" ]; then git -C "$DEST" fetch -q origin && git -C "$DEST" reset -q --hard origin/main
else git clone -q "$REMOTE" "$DEST"; fi
mkdir -p "$ROOT/runs/archive"
cp -n "$DEST"/ppo_*_steps.zip "$DEST"/pool/*.pt "$ROOT/runs/archive/" 2>/dev/null || true
git -C "$DEST" log -1 --format="%s"; tail -3 "$DEST/evals.txt" 2>/dev/null
