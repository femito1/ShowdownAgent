"""Round-robin between checkpoints (and baseline bots): win-rate matrix + Elo ratings.

Tells whether self-play keeps improving or just cycles (newer checkpoints should beat older ones).

    python -m porygon.tournament runs/a/ppo_10M.zip runs/b/best_model.zip --baselines heuristic max_power
    python -m porygon.tournament runs/archive/*.pt --baselines          # pool snapshots work too
"""

import argparse
import asyncio
import itertools
from pathlib import Path

import numpy as np
from poke_env import AccountConfiguration

from porygon.env import PolicyPlayer, local_server
from porygon.evaluate import BASELINES
from porygon.model import load_policy


def fit_elo(names: list[str], wins: np.ndarray, games: np.ndarray, iters: int = 2000) -> np.ndarray:
    """Bradley-Terry ratings on the Elo scale (mean 1500) by gradient ascent on the log-likelihood."""
    r = np.zeros(len(names))
    for _ in range(iters):
        p = 1 / (1 + 10 ** ((r[None, :] - r[:, None]) / 400))  # P(i beats j)
        r += 20 * ((wins - games * p).sum(1) / np.maximum(games.sum(1), 1))
        r -= r.mean()
    return r + 1500


def label(path: Path) -> str:
    stem = path.stem.replace("ppo_", "").replace("_steps", "")
    return f"{int(stem) / 1e6:.1f}M" if stem.isdigit() else f"{path.parent.name[-6:]}/{stem}"


async def run(args: argparse.Namespace) -> None:
    common = dict(battle_format=args.format, server_configuration=local_server(args.port),
                  max_concurrent_battles=args.concurrency, log_level=40)
    players = {}
    for i, path in enumerate(map(Path, args.models)):
        policy = load_policy(path, args.device)
        players[label(path)] = PolicyPlayer(
            policy, account_configuration=AccountConfiguration(f"porygon{i}x{np.random.randint(1e6)}", None), **common)
    for name in args.baselines:
        players[name] = BASELINES[name](**common)

    names = list(players)
    wins, games = np.zeros((len(names),) * 2), np.zeros((len(names),) * 2)
    for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
        pa, pb = players[a], players[b]
        before = pa.n_won_battles, pa.n_finished_battles
        await pa.battle_against(pb, n_battles=args.n_battles)
        w, n = pa.n_won_battles - before[0], pa.n_finished_battles - before[1]
        wins[i, j], wins[j, i] = w, n - w
        games[i, j] = games[j, i] = n
        print(f"{a:>14} vs {b:<14} {w / n:6.1%}  ({n} battles)", flush=True)

    elo = fit_elo(names, wins, games)
    rate = np.divide(wins, games, out=np.full_like(wins, np.nan), where=games > 0)
    print("\nwin rate of row vs column")
    print(" " * 14 + "".join(f"{n[:9]:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>14}" + "".join("        --" if i == j else f"{rate[i, j]:10.1%}" for j in range(len(names))))
    print("\nElo (Bradley-Terry, mean 1500)")
    for i in np.argsort(-elo):
        print(f"{names[i]:>14} {elo[i]:7.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--baselines", nargs="*", default=["heuristic", "max_power"], choices=list(BASELINES))
    p.add_argument("--format", default="gen4randombattle")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--n-battles", type=int, default=200, help="battles per pairing")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--device", default="cpu")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
