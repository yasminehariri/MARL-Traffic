import os

# Point SUMO_HOME at the installed sumo package so SUMO finds its XSD schemas
# and can validate the network/route XML (portable across machines/venvs).
import os
import sumolib
os.environ["SUMO_HOME"] = os.path.dirname(sumolib.__file__)
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

# The net defines 8 phases per light: green phases at even indices (the agent's
# 4 actions) each followed by a 3s yellow phase at the next odd index.
GREEN_PHASES = [0, 2, 4, 6]  # GG, Gr, rG, rG2 — four agent green choices
PHASES = 4
YELLOW_TIME = 3  # simulation steps of yellow inserted on every phase change

class TraCIEnv:
    def __init__(self):
        self.agents = TL_IDS
        self.num_agents = NUM_AGENTS
        self.green = {tl: 0 for tl in TL_IDS}
        self.pending = {tl: 0 for tl in TL_IDS}
        self.yellow_left = {tl: 0 for tl in TL_IDS}
        self.elapsed = {tl: 0 for tl in TL_IDS}

    def reset(self, route_file=None):
        if traci.isLoaded():
            traci.close()
        cmd = ["sumo", "-c", SUMO_CFG, "--tripinfo-output.write-unfinished"]
        if route_file:
            cmd += ["--route-files", route_file]
        traci.start(cmd)
        self.green = {tl: 0 for tl in TL_IDS}
        self.pending = {tl: 0 for tl in TL_IDS}
        self.yellow_left = {tl: 0 for tl in TL_IDS}
        self.elapsed = {tl: 0 for tl in TL_IDS}
        for tl_id in self.agents:
            traci.trafficlight.setPhase(tl_id, 0)
        return self.get_obs()

    def step(self, actions):
        for i, tl_id in enumerate(self.agents):
            target = GREEN_PHASES[max(0, min(actions[i], len(GREEN_PHASES) - 1))]
            if self.yellow_left[tl_id] > 0:
                # Mid-transition: hold yellow, ignore the new action until done.
                self.yellow_left[tl_id] -= 1
                if self.yellow_left[tl_id] == 0:
                    self.green[tl_id] = self.pending[tl_id]
                    traci.trafficlight.setPhase(tl_id, self.green[tl_id])
                else:
                    traci.trafficlight.setPhase(tl_id, self.green[tl_id] + 1)
                self.elapsed[tl_id] = 0
            elif target != self.green[tl_id]:
                # Phase change requested: run the yellow that follows the current green.
                self.pending[tl_id] = target
                self.yellow_left[tl_id] = YELLOW_TIME
                traci.trafficlight.setPhase(tl_id, self.green[tl_id] + 1)
                self.elapsed[tl_id] = 0
            else:
                # Hold the current green.
                traci.trafficlight.setPhase(tl_id, self.green[tl_id])
                self.elapsed[tl_id] += 1
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
            phase_idx = self.green[tl_id] // 2
            phase_onehot = [1.0 if phase_idx == i else 0.0 for i in range(PHASES)]
            elapsed_ratio = min(self.elapsed[tl_id] / MAX_PHASE_DURATION, 1.0)
            obs[tl_id] = [q0, q1, q2, q3] + phase_onehot + [elapsed_ratio]
        return obs

    def get_reward(self):
        total_wait = 0.0
        for tl_id in self.agents:
            for lane in LANE_IDS[tl_id]:
                total_wait += traci.lane.getWaitingTime(lane)
        return -total_wait

