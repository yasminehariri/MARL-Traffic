import traci

# Network constants — confirmed Day 4
TL_IDS = ["A0", "A1", "B0", "B1"]
LANE_IDS = {
    "A0": ["A1A0_0", "B0A0_0"],
    "A1": ["B1A1_0", "A0A1_0"],
    "B0": ["B1B0_0", "A0B0_0"],
    "B1": ["B0B1_0", "A1B1_0"],
}
NUM_AGENTS = 4
OBS_DIM = 9
PHASES = 4
SUMO_CFG = "simulation.sumocfg"


class TraCIEnv:
    def __init__(self):
        self.agents = TL_IDS
        self.num_agents = NUM_AGENTS

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start(["sumo", "-c", SUMO_CFG])
        return self.get_obs()

    def step(self, actions):
        for i, tl_id in enumerate(self.agents):
            traci.trafficlight.setPhase(tl_id, actions[i])
        traci.simulationStep()
        obs = self.get_obs()
        reward = self.get_reward()
        done = traci.simulation.getTime() >= 3600
        info = {}
        return obs, reward, done, info

    def get_obs(self):
        obs = {}
        for tl_id in self.agents:
            obs[tl_id] = [0.0] * OBS_DIM
        return obs

    def get_reward(self):
        return 0.0

