"""Our model vs Foul Play (pmariglia's search bot) on a local server.

Start Foul Play instances first, each accepting N challenges:
    python run.py --websocket-uri ws://localhost:8100/showdown/websocket --ps-username foulplay0 \
        --bot-mode accept_challenge --pokemon-format gen4randombattle --run-count N
Then: python scripts/vs_foulplay.py MODEL --port 8100 --opponents foulplay0 foulplay1 --n N
"""

import argparse
import asyncio

from stable_baselines3 import PPO

from gloria.env import PolicyPlayer, local_server


async def play(model: str, port: int, opponent: str, n: int, device: str) -> tuple[int, int]:
    policy = PPO.load(model, device=device).policy.eval()
    player = PolicyPlayer(policy, battle_format="gen4randombattle", server_configuration=local_server(port),
                          max_concurrent_battles=1, log_level=40)
    await player.send_challenges(opponent, n_challenges=n)
    return player.n_won_battles, player.n_finished_battles


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--opponents", nargs="+", required=True)
    p.add_argument("--n", type=int, default=1, help="battles per opponent")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    results = await asyncio.gather(*(play(a.model, a.port, o, a.n, a.device) for o in a.opponents))
    won, done = map(sum, zip(*results))
    print(f"vs Foul Play: won {won}/{done} = {won / max(done, 1):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
