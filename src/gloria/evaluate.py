"""Win rates of a trained policy against poke-env's baseline bots.

    python -m gloria.evaluate runs/<run>/best_model.zip --n-battles 500
"""

import argparse
import asyncio

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from gloria.env import PolicyPlayer, local_server

BASELINES: dict[str, type[Player]] = {
    "random": RandomPlayer,
    "max_power": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


class Evaluator:
    """Long-lived players (one websocket each) reused across evaluations during training."""

    def __init__(self, policy: ActorCriticPolicy, battle_format: str, port: int = 8000,
                 opponents: tuple[str, ...] = ("heuristic",), concurrency: int = 16):
        common = dict(battle_format=battle_format, server_configuration=local_server(port),
                      max_concurrent_battles=concurrency, log_level=40)
        self.player = PolicyPlayer(policy, **common)
        self.opponents = {name: BASELINES[name](**common) for name in opponents}

    def run(self, n_battles: int) -> dict[str, float]:
        results = {}
        for name, opp in self.opponents.items():
            won, done = self.player.n_won_battles, self.player.n_finished_battles
            asyncio.run(self.player.battle_against(opp, n_battles=n_battles))
            results[name] = (self.player.n_won_battles - won) / (self.player.n_finished_battles - done)
            self.player.reset_battles()
            opp.reset_battles()
        return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--format", default="gen4randombattle")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--n-battles", type=int, default=200)
    p.add_argument("--opponents", nargs="+", default=list(BASELINES), choices=list(BASELINES))
    args = p.parse_args()

    policy = PPO.load(args.model, device="cpu").policy.eval()
    results = Evaluator(policy, args.format, args.port, tuple(args.opponents)).run(args.n_battles)
    for name, rate in results.items():
        print(f"vs {name:<10} {rate:6.1%}  ({args.n_battles} battles)")


if __name__ == "__main__":
    main()
