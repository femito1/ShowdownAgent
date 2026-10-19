"""Play a trained model on the official Pokémon Showdown server (needs a registered account).

    python -m gloria.play runs/<run>/best_model.zip --username NAME --password PASS --ladder 50
    python -m gloria.play runs/<run>/best_model.zip --username NAME --password PASS --accept 5
    python -m gloria.play runs/<run>/best_model.zip --local --challenge YourName   # local server
"""

import argparse
import asyncio

from poke_env import AccountConfiguration, LocalhostServerConfiguration, ShowdownServerConfiguration
from stable_baselines3 import PPO

from gloria.env import PolicyPlayer


async def run(args: argparse.Namespace) -> None:
    policy = PPO.load(args.model, device="cpu").policy.eval()
    player = PolicyPlayer(
        policy,
        account_configuration=AccountConfiguration(args.username, args.password),
        battle_format=args.format,
        server_configuration=LocalhostServerConfiguration if args.local else ShowdownServerConfiguration,
        start_timer_on_battle_start=not args.local,
    )
    if args.ladder:
        await player.ladder(args.ladder)
    elif args.challenge:
        await player.send_challenges(args.challenge, n_challenges=args.n)
    else:
        await player.accept_challenges(None, args.accept)
    for tag, battle in player.battles.items():
        print(f"{tag}: {'won' if battle.won else 'lost'} vs {battle.opponent_username} (rating {battle.rating})")
    print(f"record: {player.n_won_battles}-{player.n_lost_battles}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--username", required=True)
    p.add_argument("--password")
    p.add_argument("--format", default="gen4randombattle")
    p.add_argument("--local", action="store_true", help="use localhost:8000 instead of the official server")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ladder", type=int, metavar="N", help="play N ladder games")
    mode.add_argument("--accept", type=int, metavar="N", help="accept N challenges from anyone")
    mode.add_argument("--challenge", metavar="USER", help="challenge USER --n times")
    p.add_argument("--n", type=int, default=1)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
