import os, sys
sys.path.insert(0, '/content/MARL-Traffic')

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from env.traci_env import TraCIEnv, TL_IDS, OBS_DIM, PHASES

NEIGHBORS = {
    "A0": ["A1", "B0"],
    "A1": ["A0", "B1"],
    "B0": ["A0", "B1"],
    "B1": ["A1", "B0"],
}

NUM_NEIGHBORS = 2  # fixed for this 2x2 ring topology
E1_OBS_DIM = OBS_DIM * (1 + NUM_NEIGHBORS)  # 9 own + 9 + 9 neighbours = 27


class MARLTrafficEnvE1(MultiAgentEnv):
    """
    Stateless E1 environment wrapper. Each agent's observation bundles its own
    9-dim obs plus its neighbours' raw 9-dim obs (in a FIXED order, per NEIGHBORS).
    The model recomputes neighbours' messages fresh every forward call --
    no cross-call cache, no dependency on call order, safe under RLlib's
    shuffle-and-multi-epoch training.
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

    def _bundle_with_neighbors(self, obs):
        bundled = {}
        for tl_id, own_vec in obs.items():
            neighbor_vecs = [obs[n] for n in NEIGHBORS[tl_id]]  # fixed order, always 2
            full = list(own_vec)
            for nv in neighbor_vecs:
                full.extend(nv)
            bundled[tl_id] = np.array(full, dtype=np.float32)
        return bundled

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset(route_file=self.route_file)
        return self._bundle_with_neighbors(obs), {}

    def step(self, action_dict):
        actions = [action_dict[tl] for tl in self._agents]
        obs, reward, done, info = self._env.step(actions)
        bundled_obs = self._bundle_with_neighbors(obs)

        rewards = {tl: reward / len(self._agents) for tl in self._agents}
        dones = {tl: done for tl in self._agents}
        dones["__all__"] = done
        truncated = {tl: False for tl in self._agents}
        truncated["__all__"] = False

        return bundled_obs, rewards, dones, truncated, {}

    def close(self):
        try:
            import traci
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass