import os, sys
sys.path.insert(0, '/content/MARL-Traffic')

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from env.traci_env import TraCIEnv, TL_IDS, OBS_DIM, PHASES, LANE_IDS

MAX_NEIGHBORS = 4  # one slot per approach direction (N, E, S, W)
E1_OBS_DIM = OBS_DIM + (MAX_NEIGHBORS * OBS_DIM) + MAX_NEIGHBORS  # 9 + 36 + 4 = 49


def build_neighbor_slots(net_path, tl_ids):
    """
    Direction-consistent neighbour slots, derived from the network file.

    slots[tl][k] is the traffic light whose outgoing edge feeds the lane that
    action k turns green (LANE_IDS[tl][k]), or None if that lane comes from
    outside the grid. Because LANE_IDS is in action order (N, E, S, W for every
    light), slot k means the same direction for every agent, which a shared
    policy needs. Missing directions are padded and masked.
    """
    import sumolib
    net = sumolib.net.readNet(net_path)
    slots = {}
    for tl in tl_ids:
        row = []
        for lane_id in LANE_IDS[tl]:
            upstream = net.getLane(lane_id).getEdge().getFromNode().getID()
            row.append(upstream if (upstream in tl_ids and upstream != tl) else None)
        slots[tl] = row
    return slots


def build_neighbors_from_net(net_path, tl_ids):
    """
    Real neighbours of each light (no padding), in the same direction order as
    build_neighbor_slots. A traffic light B is a neighbour of A if an edge
    directly connects their junctions. Used by B2 and by checks.
    """
    slots = build_neighbor_slots(net_path, tl_ids)
    return {tl: [n for n in slots[tl] if n is not None] for tl in tl_ids}


class MARLTrafficEnvE1(MultiAgentEnv):
    """
    Stateless E1 environment wrapper, generalized for variable neighbor
    counts (2x2 through 4x4+). Each agent's observation bundles its own
    9-dim obs, its neighbours' raw 9-dim obs in fixed direction slots
    (N, E, S, W), and a mask indicating which slots are real neighbours.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.route_file = config.get("route_file", None)
        self._env = TraCIEnv()
        self._agents = TL_IDS
        self._agent_ids = set(self._agents)

        net_path = config.get("net_path", os.environ.get("MARL_NET_FILE", "/content/MARL-Traffic/configs/mygrid2x2.net.xml"))
        self.neighbor_slots = build_neighbor_slots(net_path, self._agents)
        self.neighbors = {tl: [n for n in self.neighbor_slots[tl] if n is not None] for tl in self._agents}

        single_obs = spaces.Box(low=0.0, high=1.0, shape=(E1_OBS_DIM,), dtype=np.float32)
        single_act = spaces.Discrete(PHASES)

        self.observation_spaces = {tl: single_obs for tl in self._agents}
        self.action_spaces = {tl: single_act for tl in self._agents}
        self.observation_space = single_obs
        self.action_space = single_act

    def _bundle_with_neighbors(self, obs):
        bundled = {}
        for tl_id, own_vec in obs.items():
            neighbor_block = np.zeros(MAX_NEIGHBORS * OBS_DIM, dtype=np.float32)
            mask = np.zeros(MAX_NEIGHBORS, dtype=np.float32)

            for k, n_id in enumerate(self.neighbor_slots[tl_id]):
                if n_id is not None:
                    neighbor_block[k * OBS_DIM:(k + 1) * OBS_DIM] = obs[n_id]
                    mask[k] = 1.0

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

        rewards = self._env.agent_rewards(reward)
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