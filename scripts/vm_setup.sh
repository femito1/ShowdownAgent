#!/usr/bin/env bash
# One-time setup on a plain Ubuntu 24.04 GPU VM with sudo (e.g. Imperial GPUDojo / Azure).
# Code, venv and runs/ live on the OS disk (survives VM stop/start; /mnt is wiped on deallocate).
# Usage (on the VM, from the repo root): bash scripts/vm_setup.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NODE=v22.22.0
if ! command -v node >/dev/null || [[ $(node --version) != v22* ]]; then
  curl -fsSL "https://nodejs.org/dist/$NODE/node-$NODE-linux-x64.tar.xz" | sudo tar -xJ -C /opt
  sudo ln -sf "/opt/node-$NODE-linux-x64/bin/"{node,npm,npx} /usr/local/bin/
fi
sudo apt-get update -qq && sudo apt-get install -y -qq tmux rsync git python3-venv >/dev/null
cd "$ROOT/pokemon-showdown"
[ -f config/config.js ] || cp config/config-example.js config/config.js
npm install --no-audit --no-fund --silent && node build
cd "$ROOT"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -q -e .
.venv/bin/python -c "import torch, gloria.encoder; print('cuda:', torch.cuda.is_available(), torch.__version__)"
