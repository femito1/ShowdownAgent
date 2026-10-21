"""Opponent pool: frozen past policies and baseline bots, used alongside pure self-play so the
learner keeps beating old strategies instead of only chasing its latest self (fictitious play).

Seed a pool from existing checkpoints:
    python -m porygon.pool runs/<run>/checkpoints/*.zip --out runs/pool
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder, MaxBasePowerPlayer, Player, SimpleHeuristicsPlayer
from poke_env.environment import SingleAgentWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from porygon.env import PolicyPlayer, SelfPlayEnv, local_server
from porygon.model import MaskedPolicy


class PoolOpponent(Player):
    """Opponent for poke-env's SingleAgentWrapper. Each new battle (reset_battles) picks either a
    baseline bot (p_baseline) or a policy snapshot from pool_dir: half the time one of the
    `recent` newest, otherwise uniformly over all of them."""

    def __init__(self, pool_dir: Path, *, p_baseline: float = 0.2, recent: int = 4,
                 max_cached: int = 6, **kwargs: Any):
        super().__init__(start_listening=False, **kwargs)
        torch.set_num_threads(1)  # one env per worker process: don't oversubscribe CPUs
        self.pool_dir, self.p_baseline, self.recent, self.max_cached = Path(pool_dir), p_baseline, recent, max_cached
        common = dict(battle_format=self.format, start_listening=False)
        self.baselines = [SimpleHeuristicsPlayer(**common), MaxBasePowerPlayer(**common)]
        self.cache: dict[Path, PolicyPlayer] = {}
        self.rng = np.random.default_rng()
        self.current: Player = self.baselines[0]

    def sample(self) -> Player:
        snaps = sorted(self.pool_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if not snaps or self.rng.random() < self.p_baseline:
            return self.baselines[self.rng.integers(len(self.baselines))]
        pick = snaps[-self.recent:] if self.rng.random() < 0.5 else snaps
        path = pick[self.rng.integers(len(pick))]
        if path not in self.cache:
            if len(self.cache) >= self.max_cached:
                self.cache.pop(next(iter(self.cache)))
            policy = MaskedPolicy.load(str(path), device="cpu").eval()
            self.cache[path] = PolicyPlayer(policy, battle_format=self.format, start_listening=False)
        return self.cache[path]

    def reset_battles(self) -> None:
        super().reset_battles()
        self.current = self.sample()

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return self.current.choose_move(battle)


def make_pool_vec_env(n_battles: int, ports: list[int], battle_format: str, pool_dir: Path,
                      shaping: dict[str, float] | None = None, p_baseline: float = 0.2) -> SubprocVecEnv:
    """n_battles concurrent learner-vs-pool battles -> n_battles SB3 sub-envs (one learner side each)."""

    def factory(i: int):
        def make():
            env = SelfPlayEnv(battle_format=battle_format, server_configuration=local_server(ports[i % len(ports)]),
                              shaping=shaping, log_level=40, open_timeout=None, challenge_timeout=None, strict=False)
            return SingleAgentWrapper(env, PoolOpponent(pool_dir, p_baseline=p_baseline, battle_format=battle_format))
        return make

    return SubprocVecEnv([factory(i) for i in range(n_battles)], start_method="spawn")


def save_snapshot(policy: MaskedPolicy, pool_dir: Path, name: str) -> Path:
    pool_dir.mkdir(parents=True, exist_ok=True)
    path = pool_dir / f"{name}.pt"
    policy.save(str(path))  # policy weights + constructor args only (~14 MB vs 46 MB full checkpoint)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    for ckpt in args.checkpoints:
        print(save_snapshot(PPO.load(ckpt, device="cpu").policy, args.out, ckpt.stem))


if __name__ == "__main__":
    main()
