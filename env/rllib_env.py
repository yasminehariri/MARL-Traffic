import os, sys
sys.path.insert(0, '/content/MARL-Traffic')
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from env.traci_env import TraCIEnv, TL_IDS, OBS_DIM, PHASES

class MARLTrafficEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.route_file = config.get("route_file", None)
        self._env = TraCIEnv()
        self._agents = TL_IDS
        self._agent_ids = set(self._agents)
        single_obs = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        single_act = spaces.Discrete(PHASES)
        self.observation_spaces = {tl: single_obs for tl in self._agents}
        self.action_spaces = {tl: single_act for tl in self._agents}
        self.observation_space = single_obs
        self.action_space = single_act

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset(route_file=self.route_file)
        return {k: np.array(v, dtype=np.float32) for k, v in obs.items()}, {}

    def step(self, action_dict):
        actions = [action_dict[tl] for tl in self._agents]
        obs, reward, done, info = self._env.step(actions)
        obs = {k: np.array(v, dtype=np.float32) for k, v in obs.items()}
        rewards = self._env.agent_rewards(reward)
        dones = {tl: done for tl in self._agents}
        dones["__all__"] = done
        truncated = {tl: False for tl in self._agents}
        truncated["__all__"] = False
        return obs, rewards, dones, truncated, {}

    def close(self):
        try:
            import traci
            if traci.isLoaded():
                traci.close()
        except:
            pass