# Porygon
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

## Results (gen4randombattle, 60M self-play steps, 2026-10-20)
- **vs baseline bots** (training evals, 200 battles): 83.5% vs `SimpleHeuristicsPlayer`, 95% vs
  `MaxBasePowerPlayer` — on par with the thesis (~80% at 40M, ~85% at 150M steps).
- **Checkpoint tournament** (`porygon.tournament`, 100 battles per pairing, Bradley-Terry Elo):
  60M 1711 > 46M 1663 > 31M 1635 > 10M 1493 > heuristic 1457 > 2.5M 1346 > max_power 1194.
  Every checkpoint beats every earlier one, so self-play improves rather than cycles; gains
  slow after ~30M steps.
- **Search** (`porygon.search`, network-guided poke-engine MCTS over 4 sampled opponent sets):
  at 300 ms/move it beats the raw 60M network 75% (200 battles); the tuned defaults (1000 ms,
  prior weight 0.25, battle-history tracking) beat that config 59% head-to-head (150) and the
  heuristic bot 97% (100).
- **vs Foul Play** (pmariglia's open-source search bot, 1000 ms/move, `scripts/vs_foulplay.py`):
  the search player scores 49% (204 battles; 48% when Foul Play searches on 4 cores), the raw
  network 16.5% (200).
- **Ladder**: 13-7 in the first 20 real games (Showdown also counts 5 forfeits from dropped
  connections). Showdown locks datacenter IPs, so ladder games must run from a home connection.

## Layout
| | |
|---|---|
| `src/porygon/encoder.py` | battle → flat observation (field token + 12 mon tokens) |
| `src/porygon/model.py` | `BattleTransformer` features extractor, `MaskedPolicy` |
| `src/porygon/env.py` | `SelfPlayEnv` (poke-env PettingZoo env), vectorised self-play, `PolicyPlayer` |
| `src/porygon/train.py` | PPO training with eval-vs-baselines and checkpoints |
| `src/porygon/evaluate.py` | win rates vs Random / MaxBasePower / SimpleHeuristics |
| `src/porygon/play.py` | play on the official server (ladder / challenges) |
| `src/porygon/pool.py` | opponent pool: past snapshots + baseline bots |
| `src/porygon/tournament.py` | round-robin between checkpoints, Elo |
| `src/porygon/search.py` | `SearchPlayer`: network-guided poke-engine MCTS over sampled determinizations (gen4) |
| `src/porygon/engine_worker.py` | poke-engine search subprocesses used by `search.py` |
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
python -m porygon.train --n-battles 96 --ports 8100-8111 --total-steps 50_000_000
tensorboard --logdir runs
```
Throughput is bound by battle latency in the lockstep vector env, not CPU or GPU: on a 31-vCPU
RTX 3090 pod, 96 concurrent battles give ~1.4k steps/s (~5M steps/hour).

Add `--pool-battles N` to also train against an opponent pool (past snapshots, added every
`--snapshot-every` steps, plus baseline bots); seed it with `python -m porygon.pool CKPTS... --out DIR`.

On RunPod, `scripts/pod_retry_deploy.py` finds a working pod and runs `scripts/pod_launch_resume.sh`;
the pod pushes results to a private git repo every 20 min (`scripts/sync_results.sh`), fetch them
with `scripts/fetch_results.sh`, and stops itself when training ends.

## Evaluate / play
```bash
python -m porygon.evaluate runs/<run>/best_model.zip --port 8100 --n-battles 500
python -m porygon.play runs/<run>/best_model.zip --username NAME --password PASS --ladder 50
```

## Credits
- Training recipe: Jett Wang, *Winning at Pokémon Random Battles Using Reinforcement Learning*
  (MIT MEng thesis, 2024).
- [poke-engine](https://github.com/pmariglia/poke-engine) (MIT) by pmariglia is the battle
  simulator behind `porygon.search`.
- [Foul Play](https://github.com/pmariglia/foul-play) (GPL-3.0) by pmariglia was a reference
  for how a Showdown battle maps to a poke-engine state and is our strongest benchmark; no
  Foul Play code is included in this repository.
- [poke-env](https://github.com/hsahovic/poke-env) and
  [Pokémon Showdown](https://github.com/smogon/pokemon-showdown).
