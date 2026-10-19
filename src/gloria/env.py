from typing import Any

import numpy as np
import supersuit as ss
import torch
from gymnasium.spaces import Box
from poke_env import ServerConfiguration
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, Player
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecMonitor
from supersuit.vector import MakeCPUAsyncConstructor
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from gloria.encoder import OBS_DIM, embed_battle


def local_server(port: int) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


class SelfPlayEnv(SinglesEnv):
    """Both sides are driven by the learner; every decision of both players is a sample.

    Reward is the sparse +-1 win signal used by Wang (2024) unless shaping weights are given.
    """

    def __init__(self, *, shaping: dict[str, float] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.shaping = shaping or {}
        self.observation_spaces = {
            agent: Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(battle, victory_value=1.0, **self.shaping)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return embed_battle(battle)


def make_selfplay_vec_env(n_battles: int, ports: list[int], battle_format: str,
                          shaping: dict[str, float] | None = None, n_cpus: int | None = None):
    """n_battles concurrent battles -> 2 * n_battles SB3 sub-envs, spread over Showdown servers."""

    def factory(i: int):
        def make():
            env = SelfPlayEnv(
                battle_format=battle_format,
                server_configuration=local_server(ports[i % len(ports)]),
                shaping=shaping,
                log_level=40,
                open_timeout=None,
                challenge_timeout=None,
                strict=False,  # fall back to a random legal move on rare poke-env/server desyncs
            )
            return ss.pettingzoo_env_to_vec_env_v1(env)
        return make

    probe = SelfPlayEnv(battle_format=battle_format, start_listening=False)
    agent = probe.possible_agents[0]
    obs_space, act_space = probe.observation_space(agent), probe.action_space(agent)
    vec = MakeCPUAsyncConstructor(n_cpus or n_battles)(
        [factory(i) for i in range(n_battles)], obs_space, act_space
    )
    return VecMonitor(SB3VecEnvWrapper(vec))


class PolicyPlayer(Player):
    """Plays any SB3 policy trained on SelfPlayEnv observations (eval, ladder, challenges)."""

    def __init__(self, policy: ActorCriticPolicy, *args: Any, deterministic: bool = False,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.deterministic = deterministic

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        obs = {
            "observation": torch.as_tensor(embed_battle(battle)).unsqueeze(0),
            "action_mask": torch.as_tensor(np.array(SinglesEnv.get_action_mask(battle))).unsqueeze(0),
        }
        obs = {k: v.to(self.policy.device) for k, v in obs.items()}
        with torch.no_grad():
            action = self.policy._predict(obs, deterministic=self.deterministic)
        return SinglesEnv.action_to_order(np.int64(action.item()), battle, strict=False)
