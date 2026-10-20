"""SearchPlayer plays a few battles on a local server (port 8000) without ever falling back."""

import asyncio
import random
from pathlib import Path

import pytest

pe = pytest.importorskip("poke_engine")
from poke_env import AccountConfiguration  # noqa: E402
from poke_env.player import SimpleHeuristicsPlayer  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from gloria.env import local_server  # noqa: E402
from gloria.search import SearchPlayer  # noqa: E402

MODEL = Path(__file__).resolve().parents[1] / "runs/20261019-185359/best_model.zip"


@pytest.mark.skipif(not MODEL.exists(), reason="needs a trained model")
def test_search_battles(n_battles=3):
    policy = PPO.load(MODEL, device="cpu").policy.eval()
    tag = random.randint(0, 10**6)
    common = dict(battle_format="gen4randombattle", server_configuration=local_server(8000),
                  max_concurrent_battles=n_battles, log_level=40)
    player = SearchPlayer(policy, ms=50, samples=2, workers=4,
                          account_configuration=AccountConfiguration(f"stest{tag}", None), **common)
    opponent = SimpleHeuristicsPlayer(account_configuration=AccountConfiguration(f"htest{tag}", None), **common)
    asyncio.run(player.battle_against(opponent, n_battles=n_battles))
    assert player.n_finished_battles == n_battles
    assert player.stats["searched"] > 10 * n_battles
    assert player.stats["fallbacks"] == 0
