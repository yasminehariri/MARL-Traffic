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
SUMO_CFG = "configs/simulation.sumocfg"
MAX_PHASE_DURATION = 90.0
MAX_QUEUE = 20.0

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
            lanes = LANE_IDS[tl_id]
            q0 = min(traci.lane.getLastStepHaltingNumber(lanes[0]), MAX_QUEUE) / MAX_QUEUE
            q1 = min(traci.lane.getLastStepHaltingNumber(lanes[1]), MAX_QUEUE) / MAX_QUEUE
            q2 = 0.0  # placeholder — needs cross-directional routes
            q3 = 0.0  # placeholder — needs cross-directional routes
            phase = traci.trafficlight.getPhase(tl_id)
            phase_onehot = [1.0 if phase == i else 0.0 for i in range(PHASES)]
            elapsed = traci.trafficlight.getSpentDuration(tl_id)
            elapsed_ratio = min(elapsed / MAX_PHASE_DURATION, 1.0)
            obs[tl_id] = [q0, q1, q2, q3] + phase_onehot + [elapsed_ratio]
        return obs

    def get_reward(self):
        total_wait = 0.0
        for tl_id in self.agents:
            for lane in LANE_IDS[tl_id]:
                total_wait += traci.lane.getWaitingTime(lane)
        return -total_wait

