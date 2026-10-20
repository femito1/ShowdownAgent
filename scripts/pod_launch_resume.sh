#!/usr/bin/env bash
# Usage: /tmp/launch_pod.sh IP PORT COST_PER_HR VCPUS
set -eu
IP=$1; PORT=$2; COST=$3; VCPU=$4
cd /home/nand/Documents/personal-projects/ShowdownAgent
SSH="ssh -i $HOME/.runpod/ssh/runpodctl-ssh-key -p $PORT -o ConnectTimeout=15"
BAL=$(PATH=$HOME/.local/bin:$PATH runpodctl user | python3 -c "import json,sys;print(json.load(sys.stdin)['clientBalance'])")
HOURS=$(python3 -c "print(round(min(20, max(1, ($BAL - 1.0) / $COST - 0.5)), 1))")
SERVERS=$(( VCPU / 3 )); BATTLES=$(( VCPU * 3 ))
echo "balance=$BAL cost=$COST -> max-hours=$HOURS, $SERVERS servers, $BATTLES battles"
$SSH root@$IP 'which rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null)'
rsync -az -e "$SSH" --exclude .venv --exclude .git --exclude runs --exclude logs --exclude 'pokemon-showdown/node_modules' \
  --exclude 'pokemon-showdown/dist' --exclude 'pokemon-showdown/logs' --exclude 'pokemon-showdown/config/config.js' --exclude __pycache__ ./ root@$IP:/root/ShowdownAgent/
$SSH root@$IP 'mkdir -p /workspace/runs/resume && ln -sfn /workspace/runs /root/ShowdownAgent/runs'
rsync -az -e "$SSH" runs/20261019-140413/final_model.zip root@$IP:/workspace/runs/resume/final_model.zip
$SSH root@$IP 'cd /root/ShowdownAgent && bash scripts/pod_setup.sh 2>&1 | tail -1'
$SSH root@$IP "cd /root/ShowdownAgent && POD_EXIT=stop scripts/pod_train.sh $SERVERS $BATTLES --resume runs/resume/final_model.zip --total-steps 60000000 --max-hours $HOURS --checkpoint-every 2500000 > /tmp/launch.log 2>&1; cat /tmp/launch.log; sleep 90; tmux has-session -t train && touch /workspace/launched && echo LAUNCHED"
