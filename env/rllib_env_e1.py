import os, sys
sys.path.insert(0, '/content/MARL-Traffic')

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from env.traci_env import TraCIEnv, TL_IDS, OBS_DIM, PHASES

MAX_NEIGHBORS = 4  # true max degree on a 4x4 grid (interior intersections)
E1_OBS_DIM = OBS_DIM + (MAX_NEIGHBORS * OBS_DIM) + MAX_NEIGHBORS  # 9 + 36 + 4 = 49


def build_neighbors_from_net(net_path, tl_ids):
    """
    Auto-derives the NEIGHBORS mapping from the actual network file, rather
    than hand-typing it -- avoids drift/error at 16+ nodes. A traffic light
    B is a 'neighbor' of A if an edge directly connects their junctions.
    """
    import sumolib
    net = sumolib.net.readNet(net_path)
    neighbors = {tl: [] for tl in tl_ids}
    for edge in net.getEdges():
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        if from_node in neighbors and to_node in neighbors and from_node != to_node:
            if to_node not in neighbors[from_node]:
                neighbors[from_node].append(to_node)
    return neighbors


class MARLTrafficEnvE1(MultiAgentEnv):
    """
    Stateless E1 environment wrapper, generalized for variable neighbor
    counts (2x2 through 4x4+). Each agent's observation bundles its own
    9-dim obs, its neighbours' raw 9-dim obs padded to MAX_NEIGHBORS slots,
    and a mask indicating which slots are real neighbours vs. padding.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.route_file = config.get("route_file", None)
        self._env = TraCIEnv()
        self._agents = TL_IDS
        self._agent_ids = set(self._agents)

        net_path = config.get("net_path", os.environ.get("MARL_NET_FILE", "/content/MARL-Traffic/configs/mygrid2x2.net.xml"))
        self.neighbors = config.get("neighbors_override") or build_neighbors_from_net(net_path, self._agents)

        single_obs = spaces.Box(low=0.0, high=1.0, shape=(E1_OBS_DIM,), dtype=np.float32)
        single_act = spaces.Discrete(PHASES)

        self.observation_spaces = {tl: single_obs for tl in self._agents}
        self.action_spaces = {tl: single_act for tl in self._agents}
        self.observation_space = single_obs
        self.action_space = single_act

    def _bundle_with_neighbors(self, obs):
        bundled = {}
        for tl_id, own_vec in obs.items():
            real_neighbors = self.neighbors.get(tl_id, [])
            num_real = len(real_neighbors)

            neighbor_block = np.zeros(MAX_NEIGHBORS * OBS_DIM, dtype=np.float32)
            mask = np.zeros(MAX_NEIGHBORS, dtype=np.float32)

            for i, n_id in enumerate(real_neighbors[:MAX_NEIGHBORS]):
                neighbor_block[i * OBS_DIM:(i + 1) * OBS_DIM] = obs[n_id]
                mask[i] = 1.0

            full = np.concatenate([own_vec, neighbor_block, mask]).astype(np.float32)
            bundled[tl_id] = full
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