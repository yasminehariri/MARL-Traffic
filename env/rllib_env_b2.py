import os, sys
sys.path.insert(0, '/content/MARL-Traffic')

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from env.traci_env import TraCIEnv, TL_IDS, OBS_DIM, PHASES

# 2x2 ring topology neighbours — confirmed from network edge structure:
# A0 -- A1
# |      |
# B0 -- B1
NEIGHBORS = {
    "A0": ["A1", "B0"],
    "A1": ["A0", "B1"],
    "B0": ["A0", "B1"],
    "B1": ["A1", "B0"],
}

MSG_DIM = 4          # [queue_N, queue_S, queue_E, queue_W] — raw, no learning
B2_OBS_DIM = OBS_DIM + MSG_DIM  # 9 + 4 = 13


class MARLTrafficEnvB2(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.route_file = config.get("route_file", None)
        self._env = TraCIEnv()
        self._agents = TL_IDS
        self._agent_ids = set(self._agents)

        single_obs = spaces.Box(low=0.0, high=1.0, shape=(B2_OBS_DIM,), dtype=np.float32)
        single_act = spaces.Discrete(PHASES)

        self.observation_spaces = {tl: single_obs for tl in self._agents}
        self.action_spaces = {tl: single_act for tl in self._agents}
        self.observation_space = single_obs
        self.action_space = single_act

    def _build_augmented_obs(self, obs, messages):
        augmented = {}
        for tl_id in self._agents:
            neighbour_msgs = [messages[n] for n in NEIGHBORS[tl_id]]
            mean_msg = np.mean(neighbour_msgs, axis=0)
            augmented[tl_id] = np.concatenate([obs[tl_id], mean_msg]).astype(np.float32)
        return augmented

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset(route_file=self.route_file)
        messages = self._env.get_messages()
        augmented_obs = self._build_augmented_obs(obs, messages)
        return augmented_obs, {}

    def step(self, action_dict):
        actions = [action_dict[tl] for tl in self._agents]
        obs, reward, done, info = self._env.step(actions)
        messages = self._env.get_messages()
        augmented_obs = self._build_augmented_obs(obs, messages)

        rewards = {tl: reward / len(self._agents) for tl in self._agents}
        dones = {tl: done for tl in self._agents}
        dones["__all__"] = done
        truncated = {tl: False for tl in self._agents}
        truncated["__all__"] = False

        return augmented_obs, rewards, dones, truncated, {}

    def close(self):
        try:
            import traci
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass