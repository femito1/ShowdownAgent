"""Async rollouts: concatenated segments must give the same GAE as each trajectory on its own."""

import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer

from porygon.async_env import fill_buffer

OBS_SPACE = spaces.Dict({"observation": spaces.Box(-1, 1, (3,)), "action_mask": spaces.Box(0, 1, (10,))})
GAMMA, LAM = 0.9, 0.8


def transition(v, r, done, a=1):
    return ({"observation": np.zeros(3, np.float32), "action_mask": np.ones(10, np.float32)}, a, v, -0.5, r, done)


def reference_gae(seg, boot):
    adv, gae = [], 0.0
    for i in reversed(range(len(seg))):
        _, _, v, _, r, done = seg[i]
        next_v = boot if i == len(seg) - 1 else seg[i + 1][2]
        nonterminal = 0.0 if done else 1.0
        delta = r + GAMMA * next_v * nonterminal - v
        gae = delta + GAMMA * LAM * nonterminal * gae
        adv.append(gae)
    return adv[::-1]


def make_buffer(size):
    return DictRolloutBuffer(size, OBS_SPACE, spaces.Discrete(10), device="cpu", gae_lambda=LAM, gamma=GAMMA)


def test_segments_match_per_trajectory_gae():
    segments = {
        (0, 0): [transition(0.1, 0.0, False), transition(0.2, 0.05, False), transition(0.3, 0.0, False)],
        (0, 1): [transition(-0.1, 0.0, False), transition(-0.2, -1.0, True), transition(0.0, 0.05, False)],
        (1, 0): [transition(0.4, 1.0, True)],
    }
    boot = {(0, 0): 0.5, (0, 1): 1.0, (1, 0): 0.7}
    expected = [x for k in segments for x in reference_gae(segments[k], boot[k])]
    buf = make_buffer(7)
    fill_buffer(buf, {k: list(v) for k, v in segments.items()}, dict(boot), GAMMA)
    assert buf.advantages.flatten() == pytest.approx(expected, abs=1e-6)
    values = np.array([t[2] for k in segments for t in segments[k]])
    assert buf.returns.flatten() == pytest.approx(np.array(expected) + values, abs=1e-6)
    assert buf.episode_starts.flatten().tolist() == [1, 0, 0, 1, 0, 1, 1]


def test_excess_transitions_become_bootstrap():
    seg = [transition(0.1, 0.0, False), transition(0.2, 0.0, False), transition(0.9, 0.0, False)]
    buf = make_buffer(2)
    fill_buffer(buf, {(0, 0): list(seg)}, {(0, 0): 0.0}, GAMMA)  # 3rd transition dropped: its value 0.9 bootstraps
    assert buf.advantages.flatten() == pytest.approx(reference_gae(seg[:2], 0.9), abs=1e-6)
