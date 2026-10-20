#!/usr/bin/env bash
# Start Showdown servers and a detached training run in tmux session "train".
remove] [SYNC_REMOTE=.. SYNC_KEY=..] scripts/pod_train.sh|remove] [SYNC_REMOTE=.. SYNC_KEY=..] scripts/pod_train.sh N_SERVERS N_BATTLES [extra gloria.train args...]
# Watch: tmux attach -t train   |   tensorboard --logdir runs --bind_all
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
N_SERVERS=${1:?n servers}; N_BATTLES=${2:?n battles}; shift 2
BASE=8100  # RunPod images run nginx on 8001/8081/9091
"$ROOT/scripts/start_servers.sh" "$N_SERVERS" $BASE
RUN="runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ROOT/$RUN"
CMD="${PYTHON:-python} -m gloria.train --ports $BASE-$((BASE + N_SERVERS - 1)) --n-battles $N_BATTLES --run-dir $RUN $* 2>&1 | tee $RUN/train.log"
# SYNC_REMOTE=git@github.com:USER/REPO.git SYNC_KEY=deploy_key: push results there every 20 min and
# once more when training ends (scripts/sync_results.sh), so they never depend on this pod.
if [ -n "${SYNC_REMOTE:-}" ]; then
  SYNC="SYNC_KEY=$SYNC_KEY POOL_DIR=${POOL_DIR:-$RUN/pool} scripts/sync_results.sh $RUN $SYNC_REMOTE"
  tmux new-session -d -s sync -c "$ROOT" "$SYNC 2>&1 | tee -a $RUN/sync.log"
  CMD="$CMD; $SYNC once 2>&1 | tee -a $RUN/sync.log"
fi
# POD_EXIT=stop: stop this pod once training exits (its /workspace volume, and runs/ if it lives
# there, is kept). POD_EXIT=remove: delete it (only with runs/ on a network volume).
# Needs RUNPOD_API_KEY in the pod's environment (pass it at deploy time). ssh sessions don't
# inherit the container env, so read it (and RUNPOD_POD_ID) from PID 1. `<verb> pod ID` works
# with both the old runpodctl in RunPod images and the new one.
if [ -n "${POD_EXIT:-}" ]; then
  eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
  [ -n "${RUNPOD_API_KEY:-}" ] || { echo "POD_EXIT set but RUNPOD_API_KEY missing from pod env" >&2; exit 1; }
  CMD="$CMD; sync; sleep 30; RUNPOD_API_KEY=$RUNPOD_API_KEY runpodctl $POD_EXIT pod $RUNPOD_POD_ID"
fi
tmux new-session -d -s train -c "$ROOT" "$CMD"
echo "training in tmux session 'train', logging to $RUN/train.log"
