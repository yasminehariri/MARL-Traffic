import os, sys
sys.path.insert(0, '/content/MARL-Traffic')

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from env.traci_env import TraCIEnv, TL_IDS, OBS_DIM, PHASES

# One-hot agent ID mapping -- fixed order, must match TL_IDS
AGENT_ID_ONEHOT = {
    "A0": [1.0, 0.0, 0.0, 0.0],
    "A1": [0.0, 1.0, 0.0, 0.0],
    "B0": [0.0, 0.0, 1.0, 0.0],
    "B1": [0.0, 0.0, 0.0, 1.0],
}
AGENT_ID_DIM = 4
E1_OBS_DIM = OBS_DIM + AGENT_ID_DIM  # 9 + 4 = 13


class MARLTrafficEnvE1(MultiAgentEnv):
    """
    Environment wrapper for E1 (learned communication).
    Appends a 4-dim one-hot agent-ID tag to each observation, so the custom
    model (E1CommModel) can reliably identify which agent it's processing
    without depending on RLlib's internal agent-tracking (which is known to
    be unreliable on the first timestep after a reset).
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.route_file = config.get("route_file", None)
        self._env = TraCIEnv()
        self._agents = TL_IDS
        self._agent_ids = set(self._agents)

        single_obs = spaces.Box(low=0.0, high=1.0, shape=(E1_OBS_DIM,), dtype=np.float32)
        single_act = spaces.Discrete(PHASES)

        self.observation_spaces = {tl: single_obs for tl in self._agents}
        self.action_spaces = {tl: single_act for tl in self._agents}
        self.observation_space = single_obs
        self.action_space = single_act

    def _tag_with_agent_id(self, obs):
        tagged = {}
        for tl_id, vec in obs.items():
            id_tag = AGENT_ID_ONEHOT[tl_id]
            tagged[tl_id] = np.array(list(vec) + id_tag, dtype=np.float32)
        return tagged

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset(route_file=self.route_file)
        return self._tag_with_agent_id(obs), {}

    def step(self, action_dict):
        actions = [action_dict[tl] for tl in self._agents]
        obs, reward, done, info = self._env.step(actions)
        tagged_obs = self._tag_with_agent_id(obs)

        rewards = {tl: reward / len(self._agents) for tl in self._agents}
        dones = {tl: done for tl in self._agents}
        dones["__all__"] = done
        truncated = {tl: False for tl in self._agents}
        truncated["__all__"] = False

        return tagged_obs, rewards, dones, truncated, {}

    def close(self):
        try:
            import traci
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass