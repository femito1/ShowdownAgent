"""Win rates of a trained policy against poke-env's baseline bots (and optionally a fixed
earlier checkpoint, which keeps measuring progress once the bots are beaten ~85%+).

    python -m porygon.evaluate runs/<run>/best_model.zip --n-battles 500 --anchor runs/<old>/final_model.zip
"""

import argparse
import asyncio

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.policies import ActorCriticPolicy

from porygon.env import PolicyPlayer, local_server
from porygon.model import load_policy

BASELINES: dict[str, type[Player]] = {
    "random": RandomPlayer,
    "max_power": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


class Evaluator:
    """Long-lived players (one websocket each) reused across evaluations during training."""

    def __init__(self, policy: ActorCriticPolicy, battle_format: str, port: int = 8000,
                 opponents: tuple[str, ...] = ("heuristic",), concurrency: int = 16,
                 anchor: ActorCriticPolicy | None = None):
        common = dict(battle_format=battle_format, server_configuration=local_server(port),
                      max_concurrent_battles=concurrency, log_level=40)
        self.player = PolicyPlayer(policy, **common)
        self.opponents: dict[str, Player] = {name: BASELINES[name](**common) for name in opponents}
        if anchor is not None:
            self.opponents = {"anchor": PolicyPlayer(anchor, **common)} | self.opponents

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
    p.add_argument("--anchor", help="also play this checkpoint (.zip or pool .pt)")
    args = p.parse_args()

    anchor = load_policy(args.anchor) if args.anchor else None
    results = Evaluator(load_policy(args.model), args.format, args.port, tuple(args.opponents),
                        anchor=anchor).run(args.n_battles)
    for name, rate in results.items():
        print(f"vs {name:<10} {rate:6.1%}  ({args.n_battles} battles)")


if __name__ == "__main__":
    main()
