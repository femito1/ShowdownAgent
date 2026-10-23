"""Asynchronous rollout collection: battles never wait for each other.

The lockstep SB3 vector env makes every step wait for the slowest of ~150 battles. One Showdown
turn usually takes 20-30 ms, but the slowest of 150 takes ~100 ms, so most cores sit idle. Here
every battle runs in its own worker process and asks for an action as soon as it needs one; the
trainer batches whatever requests are waiting into one network forward pass and answers them.

Each battle side's transitions form a separate trajectory segment. A rollout concatenates the
segments in SB3's rollout buffer (n_envs = 1) with an episode start at every segment boundary and,
where a segment ends mid-episode, gamma * V(next observation) added to its last reward: the
same bootstrapping SB3 uses for time limits, so PPO's GAE and update run unchanged. Battles
pause while PPO trains (on-policy, as in lockstep).
"""

import multiprocessing as mp
from collections import defaultdict
from multiprocessing.connection import Connection, wait
from typing import Any, Callable

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

from porygon.env import SelfPlayEnv, local_server
from porygon.pool import PoolOpponent


def selfplay_battle(port: int, battle_format: str, shaping: dict[str, float] | None) -> SelfPlayEnv:
    return SelfPlayEnv(battle_format=battle_format, server_configuration=local_server(port), shaping=shaping,
                       log_level=40, open_timeout=None, challenge_timeout=None, strict=False)


def pool_battle(port: int, battle_format: str, shaping: dict[str, float] | None, pool_dir, p_baseline: float):
    return SingleAgentWrapper(selfplay_battle(port, battle_format, shaping),
                              PoolOpponent(pool_dir, p_baseline=p_baseline, battle_format=battle_format))


def _worker(conn: Connection, make: CloudpickleWrapper) -> None:
    """One battle. Sends ("obs", [obs per side]), then for every list of actions received, the
    next observations (a new battle's after the end), the rewards and whether the battle ended."""
    env = make.var()
    if isinstance(env, SingleAgentWrapper):
        def reset():
            return [env.reset()[0]]

        def step(actions):
            obs, reward, term, trunc, _ = env.step(np.int64(actions[0]))
            return [obs], [float(reward)], bool(term or trunc)
    else:  # PettingZoo self-play: both sides are the learner and finish together
        agents = env.possible_agents

        def reset():
            obs, _ = env.reset()
            return [obs[a] for a in agents]

        def step(actions):
            obs, rewards, terms, truncs, _ = env.step({a: np.int64(x) for a, x in zip(agents, actions)})
            return [obs[a] for a in agents], [float(rewards[a]) for a in agents], \
                any(terms.values()) or any(truncs.values())
    conn.send(reset())
    try:
        while (actions := conn.recv()) is not None:
            obs, rewards, done = step(actions)
            conn.send((reset() if done else obs, rewards, done))
    finally:
        env.close()


class AsyncBattles(VecEnv):
    """Battle worker processes behind a VecEnv facade (num_envs = 1) for SB3's setup code;
    `AsyncPPO.collect_rollouts` drives them through `collect`."""

    def __init__(self, makers: list[tuple[Callable[[], Any], str]]):
        probe = SelfPlayEnv(battle_format="gen4randombattle", start_listening=False)
        agent = probe.possible_agents[0]
        super().__init__(1, probe.observation_space(agent), probe.action_space(agent))
        self.conns: list[Connection] = []
        self.procs = []
        for make, start_method in makers:
            ctx = mp.get_context(start_method)
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, CloudpickleWrapper(make)), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        self.ready = {c: c.recv() for c in self.conns}  # worker -> observations awaiting actions
        self.index = {c: i for i, c in enumerate(self.conns)}
        self.ep_stats: dict[tuple[int, int], list] = defaultdict(lambda: [0.0, 0])  # running return, length

    def collect(self, policy, n: int, on_transitions: Callable[[int, list[dict]], bool]):
        """Run battles until n transitions are taken. Returns the segments {(worker, side): [...]}
        in collection order, a bootstrap value per segment, and False if a callback stopped it."""
        segments: dict[tuple[int, int], list[tuple]] = defaultdict(list)
        in_flight: dict[Connection, tuple] = {}
        taken, keep_going = 0, True
        while (taken < n and keep_going) or in_flight:
            if self.ready and taken < n and keep_going:
                conns = list(self.ready)
                obs = [o for c in conns for o in self.ready[c]]
                actions, values, log_probs = self._forward(policy, obs)
                k = 0
                for c in conns:
                    m = len(self.ready[c])
                    c.send(actions[k:k + m].tolist())
                    in_flight[c] = (self.ready.pop(c), actions[k:k + m], values[k:k + m], log_probs[k:k + m])
                    k += m
                taken += len(obs)
            for c in wait(list(in_flight)):
                next_obs, rewards, done = c.recv()
                obs, actions, values, log_probs = in_flight.pop(c)
                finished = []
                for side, (o, a, v, lp, r) in enumerate(zip(obs, actions, values, log_probs, rewards)):
                    key = (self.index[c], side)
                    segments[key].append((o, a, v, lp, r, done))
                    stats = self.ep_stats[key]
                    stats[0] += r
                    stats[1] += 1
                    if done:
                        finished.append({"r": stats[0], "l": stats[1]})
                        del self.ep_stats[key]
                self.ready[c] = next_obs
                keep_going = on_transitions(len(obs), finished) and keep_going
        # every worker is waiting again: bootstrap each segment from its side's pending observation
        keys = list(segments)
        pending = [self.ready[self.conns[w]][side] for w, side in keys]
        boot = dict(zip(keys, self._forward(policy, pending)[1])) if pending else {}
        return segments, boot, keep_going

    @staticmethod
    def _forward(policy, obs: list[dict]):
        batch = {k: torch.as_tensor(np.stack([o[k] for o in obs]), device=policy.device) for k in obs[0]}
        with torch.no_grad():
            actions, values, log_probs = policy(batch)
        return actions.cpu().numpy(), values.flatten().cpu().numpy(), log_probs.cpu().numpy()

    # --- VecEnv facade: SB3 only resets it once during setup ---
    def reset(self):
        return {k: np.zeros((1, *s.shape), s.dtype) for k, s in self.observation_space.spaces.items()}

    def step_async(self, actions):
        raise NotImplementedError("AsyncBattles is driven by AsyncPPO.collect_rollouts")

    def step_wait(self):
        raise NotImplementedError("AsyncBattles is driven by AsyncPPO.collect_rollouts")

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    def get_attr(self, attr_name, indices=None):
        return [None]

    def set_attr(self, attr_name, value, indices=None) -> None:
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]


def fill_buffer(buffer: DictRolloutBuffer, segments: dict, boot: dict, gamma: float) -> None:
    """Concatenate segments into a (buffer_size, 1) rollout buffer and compute GAE. Extra
    transitions beyond buffer_size are dropped from the ends of the longest segments (their
    value becomes that segment's bootstrap)."""
    excess = sum(map(len, segments.values())) - buffer.buffer_size
    while excess > 0:
        key = max(segments, key=lambda k: len(segments[k]))
        boot[key] = segments[key].pop()[2]
        if not segments[key]:
            del segments[key]
        excess -= 1
    buffer.reset()
    for key, seg in segments.items():
        for i, (o, a, v, lp, r, done) in enumerate(seg):
            start = i == 0 or seg[i - 1][5]
            if i == len(seg) - 1 and not done:
                r += gamma * float(boot[key])  # segment cut mid-episode
            buffer.add({k: x[None] for k, x in o.items()}, np.array([[a]]), np.array([r], dtype=np.float32),
                       np.array([start], dtype=np.float32), torch.tensor([[v]]), torch.tensor([lp]))
    # every segment's tail is either terminal or already bootstrapped
    buffer.compute_returns_and_advantage(last_values=torch.zeros(1, 1), dones=np.array([True]))


class AsyncPPO(PPO):
    """PPO whose rollouts come from `AsyncBattles` instead of a lockstep VecEnv."""

    def collect_rollouts(self, env: AsyncBattles, callback: BaseCallback, rollout_buffer, n_rollout_steps: int) -> bool:
        self.policy.set_training_mode(False)
        callback.on_rollout_start()

        def on_transitions(n: int, finished: list[dict]) -> bool:
            if finished:
                self.ep_info_buffer.extend(finished)
            ok = True
            for _ in range(n):
                self.num_timesteps += 1
                ok = callback.on_step() and ok
            return ok

        threads = torch.get_num_threads()
        torch.set_num_threads(1)  # many small forwards: extra threads only fight the battle workers for CPU
        try:
            segments, boot, ok = env.collect(self.policy, n_rollout_steps, on_transitions)
        finally:
            torch.set_num_threads(threads)
        if not ok:
            return False
        fill_buffer(rollout_buffer, segments, boot, self.gamma)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True
