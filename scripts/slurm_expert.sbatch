#!/bin/bash
# One expert-iteration round on a Slurm GPU node (porygon.expert): search-vs-search games ->
# shards in RUN/data, then distil them into RUN/distilled.zip and push it (scripts/sync_results.sh).
# Same setup as slurm_train.sbatch, plus a one-time gen4 poke-engine build (Rust in .cargo/.rustup).
#
# Settings in slurm.env (never sbatch --export=..., it gets jobs held on this cluster):
#   EXPERT_MODEL=<zip> EXPERT_GAMES=<battles, default 9000 ~ 1M records> EXPERT_MS=<ms, default 300>
#   sbatch scripts/slurm_expert.sbatch [extra `porygon.expert distill` args]
#SBATCH --job-name=porygon-expert
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
[ -f slurm.env ] && source slurm.env
NODE=v22.22.0
export PATH=$ROOT/.node/bin:$ROOT/.cargo/bin:$PATH CARGO_HOME=$ROOT/.cargo RUSTUP_HOME=$ROOT/.rustup

# --- one-time setup (idempotent) ---
if ! [ -x .node/bin/node ]; then
  mkdir -p .node && curl -fsSL "https://nodejs.org/dist/$NODE/node-$NODE-linux-x64.tar.xz" | tar -xJ --strip-components=1 -C .node
fi
if ! [ -f pokemon-showdown/package.json ]; then
  mkdir -p pokemon-showdown && (cd pokemon-showdown && git init -q && \
    git fetch -q --depth 1 https://github.com/smogon/pokemon-showdown "${SHA:?Showdown commit}" && git checkout -q FETCH_HEAD)
fi
[ -d pokemon-showdown/node_modules ] || (cd pokemon-showdown && npm install --no-audit --no-fund --silent)
if ! [ -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu126
  .venv/bin/pip install -q -e .
fi
if ! .venv/bin/python -c "import poke_engine" 2>/dev/null; then
  [ -x .cargo/bin/cargo ] || curl -fsSL https://sh.rustup.rs | sh -s -- -y -q --no-modify-path --profile minimal
  .venv/bin/pip install -q poke-engine --no-cache-dir \
    --config-settings="build-args=--features poke-engine/gen4 --no-default-features"
fi
.venv/bin/python -c "import torch, poke_engine; assert torch.cuda.is_available(); print('cuda + poke-engine ok')"

# --- servers: Showdown is light here, the engine workers get the cores ---
CORES=$(nproc); N_SERVERS=4; WORKERS=$(( CORES - N_SERVERS - 1 ))
BASE=$(( 20000 + (SLURM_JOB_ID % 2000) * 16 ))
rm -f pokemon-showdown/config/config.js
scripts/start_servers.sh "$N_SERVERS" "$BASE"
trap 'pkill -u "$USER" -f "pokemon-showdown start --skip-build --no-security" || true' EXIT

RUN=runs/$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID-expert; mkdir -p "$RUN"
SYNC_REMOTE=${SYNC_REMOTE:-git@github.com:femito1/porygon-runs.git}
export SYNC_KEY=${SYNC_KEY:-$HOME/.ssh/porygon_runs_deploy}
scripts/sync_results.sh "$RUN" "$SYNC_REMOTE" >> "$RUN/sync.log" 2>&1 &

{
  .venv/bin/python -m porygon.expert generate "${EXPERT_MODEL:?checkpoint to start from}" --out "$RUN/data" \
    --games "${EXPERT_GAMES:-9000}" --ms "${EXPERT_MS:-300}" --ports "$BASE-$((BASE + N_SERVERS - 1))" \
    --workers "$WORKERS"
  .venv/bin/python -m porygon.expert distill "$EXPERT_MODEL" "$RUN/data" --out "$RUN/distilled.zip" "$@"
} 2>&1 | tee "$RUN/train.log" || true
scripts/sync_results.sh "$RUN" "$SYNC_REMOTE" once >> "$RUN/sync.log" 2>&1
