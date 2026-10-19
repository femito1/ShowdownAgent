#!/usr/bin/env bash
# One-time setup on a fresh GPU pod (runpod/pytorch image: CUDA torch + Python preinstalled).
# Usage: bash scripts/pod_setup.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NODE=v22.22.0
if ! command -v node >/dev/null || [[ $(node --version) != v22* ]]; then
  curl -fsSL "https://nodejs.org/dist/$NODE/node-$NODE-linux-x64.tar.xz" | tar -xJ -C /opt
  ln -sf "/opt/node-$NODE-linux-x64/bin/"{node,npm,npx} /usr/local/bin/
fi
apt-get update -qq && apt-get install -y -qq tmux rsync >/dev/null
cd "$ROOT/pokemon-showdown"
[ -f config/config.js ] || cp config/config-example.js config/config.js
npm install --no-audit --no-fund --silent && node build
cd "$ROOT"
PIP_BREAK_SYSTEM_PACKAGES=1 pip install -q -e .   # container-only; keeps the image's CUDA torch (pyproject only needs torch>=2.4)
python -c "import torch, gloria.encoder; print('cuda:', torch.cuda.is_available(), torch.__version__)"
