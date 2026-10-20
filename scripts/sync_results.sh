#!/usr/bin/env bash
# Push a run's results (log, tensorboard, best/final model, newest checkpoint, newest pool snapshots)
# to a git remote, so they survive the pod being stopped, deleted, or stuck on a busy host.
# Each push is a single fresh commit (force-pushed), so the remote stays one run-sized snapshot.
# Usage: SYNC_KEY=~/.ssh/key scripts/sync_results.sh RUN_DIR REMOTE [once]     (loop: every 20 min)
# Fetch on any machine: git clone REMOTE   (or: git -C clone fetch && git -C clone reset --hard origin/main)
set -u
RUN=$(cd "$1" && pwd); REMOTE=$2; MODE=${3:-loop}
export GIT_SSH_COMMAND="ssh -i ${SYNC_KEY:?deploy key} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

sync_once() {
  local tmp; tmp=$(mktemp -d)
  cp "$RUN"/train.log "$RUN"/*.zip "$tmp"/ 2>/dev/null
  [ -d "$RUN/tb" ] && cp -r "$RUN/tb" "$tmp/tb"
  ckpt=$(ls -t "$RUN"/checkpoints/*.zip 2>/dev/null | head -1)
  [ -n "$ckpt" ] && cp "$ckpt" "$tmp/"
  mkdir -p "$tmp/pool"
  ls -t "$POOL_DIR"/*.pt 2>/dev/null | head -3 | xargs -r cp -t "$tmp/pool/"
  grep "eval @" "$RUN/train.log" > "$tmp/evals.txt" 2>/dev/null
  (cd "$tmp" && git init -q -b main && git add -A \
     && git -c user.name=gloria-pod -c user.email=pod@localhost commit -q -m "$(basename "$RUN") @ $(date -u +%FT%TZ)" \
     && git push -q -f "$REMOTE" main) && echo "$(date -u +%T) synced $(du -sh "$tmp" | cut -f1)" || echo "$(date -u +%T) sync FAILED"
  rm -rf "$tmp"
}

POOL_DIR=${POOL_DIR:-$RUN/pool}
if [ "$MODE" = once ]; then sync_once; exit; fi
while true; do sleep 1200; sync_once; done
