#!/usr/bin/env bash
# Start Showdown servers and a detached training run in tmux session "train".
# Usage: [POD_EXIT=stop|remove] scripts/pod_train.sh N_SERVERS N_BATTLES [extra gloria.train args...]
# Watch: tmux attach -t train   |   tensorboard --logdir runs --bind_all
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
N_SERVERS=${1:?n servers}; N_BATTLES=${2:?n battles}; shift 2
BASE=8100  # RunPod images run nginx on 8001/8081/9091
"$ROOT/scripts/start_servers.sh" "$N_SERVERS" $BASE
RUN="runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ROOT/$RUN"
CMD="python -m gloria.train --ports $BASE-$((BASE + N_SERVERS - 1)) --n-battles $N_BATTLES --run-dir $RUN $* 2>&1 | tee $RUN/train.log"
# POD_EXIT=stop: stop this pod once training exits (its /workspace volume, and runs/ if it lives
# there, is kept). POD_EXIT=remove: delete it (only with runs/ on a network volume).
# Both need RUNPOD_API_KEY in the pod's environment.
[ -n "${POD_EXIT:-}" ] && CMD="$CMD; sync; sleep 30; runpodctl $POD_EXIT pod \$RUNPOD_POD_ID"
tmux new-session -d -s train -c "$ROOT" "$CMD"
echo "training in tmux session 'train', logging to $RUN/train.log"
