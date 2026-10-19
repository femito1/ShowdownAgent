"""Plays random self-play battles on a local server (port 8000) and checks every observation."""

import time

import numpy as np
import pytest
import torch
from gymnasium import spaces

from gloria.encoder import OBS_DIM
from gloria.env import SelfPlayEnv, local_server
from gloria.model import BattleTransformer


@pytest.mark.parametrize("battle_format", ["gen4randombattle", "gen9randombattle"])
def test_random_battles(battle_format, n_battles=5):
    env = SelfPlayEnv(battle_format=battle_format, server_configuration=local_server(8000),
                      log_level=40, strict=False)
    rng = np.random.default_rng(0)
    observations, rewards, start = [], [], time.time()
    for _ in range(n_battles):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {a: np.int64(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
            for a in env.agents:
                assert obs[a]["observation"].shape == (OBS_DIM,)
                assert np.isfinite(obs[a]["observation"]).all()
                observations.append(obs[a]["observation"])
            obs, rew, term, trunc, _ = env.step(actions)
            done = any(term.values()) or any(trunc.values())
        rewards.append(sum(rew.values()))
    env.close()
    print(f"{battle_format}: {len(observations) / (time.time() - start):.0f} obs/s")
    assert all(abs(r) < 1e-6 for r in rewards), "zero-sum terminal rewards"

    space = spaces.Dict({"observation": spaces.Box(-np.inf, np.inf, (OBS_DIM,)),
                         "action_mask": spaces.Box(0, 1, (10,))})
    out = BattleTransformer(space)({"observation": torch.as_tensor(np.stack(observations[:64]))})
    assert torch.isfinite(out).all()
