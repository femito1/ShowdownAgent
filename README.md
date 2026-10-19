# GlorIA
A Pokémon Showdown agent trained with self-play reinforcement learning (gen4 random battles).

The training recipe follows Jett Wang, *Winning at Pokémon Random Battles Using Reinforcement
Learning* (MIT MEng thesis, 2024 — [dspace](https://hdl.handle.net/1721.1/153888)), whose PPO
self-play agent reached 1693 Elo (rank 8) on the gen4randombattle ladder:

- **Self-play PPO** (stable-baselines3) where both players of every battle are the learner and
  both trajectories are training data; sparse ±1 win reward; γ = 0.9999, λ = 0.754, clip 0.0829,
  7 epochs, ~40k transitions per update, and the annealed learning rate
  `10^-4.23 / (8x + 1)^1.5` that the thesis found to be the single biggest factor.
- **Markov state encoding**: every multi-turn condition (weather, screens, hazards, boosts,
  status counters, volatile effects) is encoded with its elapsed turns.
- **Action masking**: illegal actions get probability 0 in rollouts and in the PPO update.

Differences from the thesis: a transformer over field + 12 Pokémon tokens instead of an MLP,
learned embeddings for species/moves/items/abilities, a movepool prior for unrevealed opponent
sets (from Showdown's random-battle data), and poke-env's slot-based 10-action space.

## Layout
| | |
|---|---|
| `src/gloria/encoder.py` | battle → flat observation (field token + 12 mon tokens) |
| `src/gloria/model.py` | `BattleTransformer` features extractor, `MaskedPolicy` |
| `src/gloria/env.py` | `SelfPlayEnv` (poke-env PettingZoo env), vectorised self-play, `PolicyPlayer` |
| `src/gloria/train.py` | PPO training with eval-vs-baselines and checkpoints |
| `src/gloria/evaluate.py` | win rates vs Random / MaxBasePower / SimpleHeuristics |
| `src/gloria/play.py` | play on the official server (ladder / challenges) |
| `scripts/` | Showdown server launchers, data builder, RunPod setup |

## Setup
```bash
git submodule update --init            # official smogon/pokemon-showdown
cd pokemon-showdown && npm install && cd ..
python -m venv .venv && .venv/bin/pip install -e .
node scripts/build_data.js             # only if the Showdown checkout changes
```

## Train
```bash
scripts/start_servers.sh 12 8100        # 12 local Showdown servers (stop: scripts/stop_servers.sh)
python -m gloria.train --n-battles 96 --ports 8100-8111 --total-steps 50_000_000
tensorboard --logdir runs
```
Throughput is bound by battle latency in the lockstep vector env, not CPU or GPU: on a 31-vCPU
RTX 3090 pod, 96 concurrent battles give ~1.4k steps/s (~5M steps/hour).

On a fresh RunPod GPU pod: `bash scripts/pod_setup.sh && scripts/pod_train.sh 12 96 --total-steps 50000000`.

## Evaluate / play
```bash
python -m gloria.evaluate runs/<run>/best_model.zip --port 8100 --n-battles 500
python -m gloria.play runs/<run>/best_model.zip --username NAME --password PASS --ladder 50
```
