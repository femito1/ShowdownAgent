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

## Results (gen4randombattle)
- **Raw network** (PPO, 60M steps): 83.5% vs `SimpleHeuristicsPlayer`, 95% vs `MaxBasePowerPlayer`
  (200 battles each) — on par with the thesis (~80% at 40M, ~85% at 150M steps). Continued
  training with the opponent pool holds ~84–86% vs the heuristic bot past 90M steps, but still
  improves head-to-head: 121M beats 101M 56.5% and 60M 60% (200 battles each).
- **Checkpoint tournament** (`porygon.tournament`, 100 battles per pairing, Bradley-Terry Elo):
  60M 1711 > 46M 1663 > 31M 1635 > 10M 1493 > heuristic 1457 > 2.5M 1346 > max_power 1194.
  Every checkpoint beats every earlier one; gains slow after ~30M steps.
- **Search** (`porygon.search`, defaults 1000 ms/move, 4 sampled opponent sets, prior weight
  0.25): beats the raw network ~75% head-to-head and the heuristic bot 97%.
- **vs Foul Play** (pmariglia's search bot, 1000 ms/move): search player 49% (204 battles), raw
  network 16.5% (200).
- **Ladder**: 13-7 with the raw 60M network. Showdown locks datacenter IPs, so ladder games
  must run from a home connection.

## Layout
| | |
|---|---|
| `src/porygon/encoder.py` | battle → flat observation (field token + 12 mon tokens) |
| `src/porygon/model.py` | `BattleTransformer` features extractor, `MaskedPolicy` (per-action pointer head) |
| `src/porygon/env.py` | `SelfPlayEnv` (poke-env PettingZoo env), vectorised self-play, `PolicyPlayer` |
| `src/porygon/pool.py` | opponent pool: past policy snapshots + baseline bots |
| `src/porygon/train.py` | PPO training with eval-vs-baselines, checkpoints, pool snapshots |
| `src/porygon/search.py` | `SearchPlayer`: network-guided poke-engine MCTS over sampled opponent sets |
| `src/porygon/engine_worker.py` | poke-engine search subprocesses used by `search.py` |
| `src/porygon/evaluate.py`, `tournament.py` | win rates vs baseline bots; round-robin Elo |
| `src/porygon/play.py` | play on the official server (ladder / challenges) |
| `src/gloria/` | alias of `porygon` (the old name) so checkpoints saved before the rename load |
| `scripts/slurm_train.sbatch` | training job for the Imperial DoC Slurm cluster |
| `scripts/sync_results.sh`, `fetch_results.sh` | push a run's results to a git repo / download and archive them |
| `scripts/pod_*.sh`, `pod_retry_deploy.py` | the same on RunPod |
| `scripts/vs_foulplay.py`, `ladder_watchdog.sh` | benchmark vs Foul Play; long ladder sessions |

## Setup
```bash
git submodule update --init            # official smogon/pokemon-showdown
cd pokemon-showdown && npm install && cd ..
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/pip install poke-engine --no-cache-dir \
  --config-settings="build-args=--features poke-engine/gen4 --no-default-features"   # for search; needs Rust
node scripts/build_data.js             # only if the Showdown checkout changes
```

## Train
Locally (throughput is bound by battle latency, ~3 cores per Showdown server):
```bash
scripts/start_servers.sh 10 8100        # stop: scripts/stop_servers.sh
python -m porygon.train --n-battles 60 --pool-battles 30 --ports 8100-8109 \
  --resume runs/<run>/final_model.zip --total-steps 400_000_000
```
`--pool-battles` adds battles against frozen snapshots (one added every `--snapshot-every` steps)
and baseline bots; seed a pool with `python -m porygon.pool CKPTS... --out DIR`. `--anchor CKPT`
also evaluates against a fixed earlier checkpoint and picks `best_model` by that score: the
baseline bots saturate around 85%, while head-to-head win rates keep showing progress.

On the Imperial DoC cluster, from the repo copy on `/vol/bitbucket`: put `SHA=<showdown commit>`,
`RESUME=<zip>`, `POOL=<pool dir>`, `TOTAL=<steps>`, `CHAIN=<follow-up jobs>`, optionally
`ANCHOR=<zip or .pt>` (default `RESUME`) in `slurm.env`
(never `sbatch --export`, which gets jobs held), then `sbatch scripts/slurm_train.sbatch`. The job
sets itself up, resumes from the newest checkpoint, uses all cores it gets (30 on a 32-core
request), stops after 70.5 h, and pushes results every 20 min. Fetch them anywhere with
`scripts/fetch_results.sh` (into `runs/remote/`). The push holds only the newest checkpoint and
pool snapshots, so the fetch also copies them into `runs/archive/` (never overwritten); the full
history stays on the cluster (`runs/<run>/checkpoints/` every 5M steps, `$POOL/*.pt` every 2.5M)
for `rsync`.

## Evaluate / play
```bash
python -m porygon.evaluate MODEL --port 8000 --n-battles 500 [--anchor OLD_MODEL]
python -m porygon.tournament MODEL_A MODEL_B ... --n-battles 100   # .zip checkpoints or pool .pt
python -m porygon.search MODEL --opponent {policy,heuristic,max_power,search} --n-battles 100
python scripts/vs_foulplay.py MODEL --player search --opponents fpa0 fpa1 ...   # setup in its docstring
PS_PASSWORD=... scripts/ladder_watchdog.sh USER MODEL TOTAL_GAMES 2              # from a home connection
```

## Roadmap
1. **Finish the current long run** (opponent pool, resumed from 60M; stops at 300M steps, ~2026-10-23):
   fetch it, run a tournament against the 60M/91M/121M models, and re-run the search player vs
   Foul Play with the newest network.
2. **Throughput decay**: steps/s falls steadily within a training job (1.4k → ~770 over 20 h, with
   most cores idle and eval time flat) and recovers on restart, so something builds up inside the
   training processes. Find it locally (py-spy works there), or restart jobs every ~12 h.
3. **Ladder the search player** from a home connection for a real rating (the thesis bot: 1693
   Elo / 79.5% GXE over 200 games). `play.py` still uses `PolicyPlayer`; switch it to
   `SearchPlayer`.
4. **Search fidelity**: poke-engine's gen4 build doesn't model taunt/encore/confusion durations;
   Substitute HP is assumed ¼; occasional `NonFinite` engine panics (sample skipped). Tune time
   per move further (1 s beat 0.3 s; more may help).
5. **Search in training** (expert iteration): use search results as policy targets — the main
   lever left now that gains from raw self-play are slowing.
6. **Throughput**: the SB3 vector env is lockstep and latency-bound; an asynchronous actor setup
   would multiply steps per hour.
7. **gen9randombattle**: encoder, action space (tera) and vocab already support it; start from the
   gen4 model.

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
