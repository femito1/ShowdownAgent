"""Play a trained model on the official Pokémon Showdown server (needs a registered account).

    python -m gloria.play runs/<run>/best_model.zip --username NAME --password PASS --ladder 50
    python -m gloria.play runs/<run>/best_model.zip --username NAME --password PASS --accept 5
    python -m gloria.play runs/<run>/best_model.zip --local --challenge YourName   # local server
"""

import argparse
import asyncio
import os

from poke_env import AccountConfiguration, LocalhostServerConfiguration, ShowdownServerConfiguration
from stable_baselines3 import PPO

from gloria.env import PolicyPlayer


class LoggingPlayer(PolicyPlayer):
    """Prints every finished battle with the running record and ratings."""

    def choose_move(self, battle):
        if battle.turn <= 1 and not getattr(battle, "_gloria_announced", False):
            battle._gloria_announced = True
            print(f"started vs {battle.opponent_username}  {battle.battle_tag}", flush=True)
        return super().choose_move(battle)

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        result = "won " if battle.won else "lost" if battle.lost else "tie "
        print(f"{result} vs {battle.opponent_username:<18} rating {battle.rating} (opp {battle.opponent_rating})"
              f"  record {self.n_won_battles}-{self.n_lost_battles}  {battle.battle_tag}", flush=True)


async def run(args: argparse.Namespace) -> None:
    policy = PPO.load(args.model, device="cpu").policy.eval()
    player = LoggingPlayer(
        policy,
        account_configuration=AccountConfiguration(args.username, args.password or os.environ.get("PS_PASSWORD")),
        battle_format=args.format,
        server_configuration=LocalhostServerConfiguration if args.local else ShowdownServerConfiguration,
        start_timer_on_battle_start=not args.local,
        max_concurrent_battles=args.concurrency,
    )
    if args.ladder:
        await player.ladder(args.ladder)
    elif args.challenge:
        await player.send_challenges(args.challenge, n_challenges=args.n)
    else:
        await player.accept_challenges(None, args.accept)
    print(f"final record: {player.n_won_battles}-{player.n_lost_battles}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--username", required=True)
    p.add_argument("--password", help="or set PS_PASSWORD")
    p.add_argument("--format", default="gen4randombattle")
    p.add_argument("--local", action="store_true", help="use localhost:8000 instead of the official server")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ladder", type=int, metavar="N", help="play N ladder games")
    mode.add_argument("--accept", type=int, metavar="N", help="accept N challenges from anyone")
    mode.add_argument("--challenge", metavar="USER", help="challenge USER --n times")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1, help="simultaneous battles")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
