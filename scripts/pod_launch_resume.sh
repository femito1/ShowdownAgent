#!/usr/bin/env bash
# Set up a fresh GPU pod and resume training on it, sized to the vCPU count and the RunPod balance.
# runs/ lives on the pod's /workspace volume and the pod stops itself when training ends, so
# nothing depends on this machine after launch.
# Usage: RESUME=model.zip POOL_SEED=dir_of_pt [TOTAL=150000000] scripts/pod_launch_resume.sh IP PORT COST_PER_HR VCPUS
set -eu
IP=$1; PORT=$2; COST=$3; VCPU=$4
: "${RESUME:?path to checkpoint .zip}" "${POOL_SEED:?dir of pool .pt snapshots}"
TOTAL=${TOTAL:-150000000}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
SSH="ssh -i $HOME/.runpod/ssh/runpodctl-ssh-key -p $PORT -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new"
BAL=$(PATH=$HOME/.local/bin:$PATH runpodctl user | python3 -c "import json,sys;print(json.load(sys.stdin)['clientBalance'])")
HOURS=$(python3 -c "print(round(min(20, max(1, ($BAL - 0.4) / $COST - 0.5)), 1))")
SERVERS=$(( VCPU / 3 )); SELFPLAY=$(( VCPU * 2 )); POOL=$VCPU
echo "balance=$BAL cost=$COST -> max-hours=$HOURS, $SERVERS servers, $SELFPLAY self-play + $POOL pool battles"
SHA=$(git -C pokemon-showdown rev-parse HEAD)  # same Showdown as the vocab in src/gloria/data
$SSH root@$IP "which rsync tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync tmux >/dev/null)
  mkdir -p /root/ShowdownAgent/pokemon-showdown /workspace/runs/resume/pool && ln -sfn /workspace/runs /root/ShowdownAgent/runs
  cd /root/ShowdownAgent/pokemon-showdown && git init -q && git fetch -q --depth 1 https://github.com/smogon/pokemon-showdown $SHA && git checkout -q FETCH_HEAD"
rsync -az -e "$SSH" --exclude __pycache__ src scripts tests pyproject.toml README.md root@$IP:/root/ShowdownAgent/
rsync -az -e "$SSH" "$RESUME" root@$IP:/workspace/runs/resume/model.zip
rsync -az -e "$SSH" "$POOL_SEED"/ root@$IP:/workspace/runs/resume/pool/
SYNC_REMOTE=${SYNC_REMOTE:-git@github.com:femito1/gloria-runs.git}
rsync -az -e "$SSH" "${SYNC_KEY:-$HOME/.ssh/gloria_runs_deploy}" root@$IP:/root/.ssh/gloria_runs_deploy
$SSH root@$IP 'cd /root/ShowdownAgent && bash scripts/pod_setup.sh 2>&1 | tail -1'
$SSH root@$IP "cd /root/ShowdownAgent && POD_EXIT=stop SYNC_REMOTE=$SYNC_REMOTE SYNC_KEY=/root/.ssh/gloria_runs_deploy POOL_DIR=runs/resume/pool scripts/pod_train.sh $SERVERS $SELFPLAY --pool-battles $POOL \
  --pool-dir runs/resume/pool --resume runs/resume/model.zip --total-steps $TOTAL --max-hours $HOURS \
  --checkpoint-every 5000000 > /tmp/launch.log 2>&1; cat /tmp/launch.log; sleep 120
  tmux has-session -t train && grep -q sub-envs runs/2*/train.log && touch /workspace/launched && echo LAUNCHED"
