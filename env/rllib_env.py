import os
import random

# Point SUMO_HOME at the installed sumo package so SUMO finds its XSD schemas
# and can validate the network/route XML (portable across machines/venvs).
import sumolib
os.environ["SUMO_HOME"] = os.path.dirname(sumolib.__file__)
import traci

# Network constants

NET_FILE = os.environ.get("MARL_NET_FILE", "/content/MARL-Traffic/configs/mygrid2x2.net.xml")
SUMO_CFG = os.environ.get("MARL_SUMO_CFG", "/content/MARL-Traffic/configs/simulation.sumocfg")

# The net defines 8 phases per light: green phases at even indices (the agent's
# 4 actions) each followed by a 3s yellow phase at the next odd index.
GREEN_PHASES = [0, 2, 4, 6]  # action k -> green phase GREEN_PHASES[k]
PHASES = 4
YELLOW_TIME = 3  # simulation steps of yellow inserted on every phase change

OBS_DIM = 9      # 4 queue values + 4 phase indicators + 1 timer = 9
MAX_PHASE_DURATION = 90.0
MAX_QUEUE = 20.0


def _derive_network_constants(net_path):
    """Read traffic-light IDs and their incoming lanes straight from the network.

    LANE_IDS[tl][k] is the incoming lane that receives green under action k
    (green phase GREEN_PHASES[k]). This keeps queue slot k aligned with action k
    for EVERY agent, which a shared policy needs. (Earlier versions sorted lanes
    alphabetically, which gave each light a different slot-to-action mapping.)
    """
    net = sumolib.net.readNet(net_path, withPrograms=True)
    tl_ids, lane_ids = [], {}
    for tls in net.getTrafficLights():
        tl_id = tls.getID()
        links = sorted(tls.getConnections(), key=lambda c: c[2])  # by link index
        if not links:
            continue
        program = list(tls.getPrograms().values())[0]
        phases = program.getPhases()
        ordered = []
        for g in GREEN_PHASES:
            state = phases[g].state
            served = sorted({links[i][0].getID() for i, s in enumerate(state) if s in "Gg"})
            if len(served) != 1:
                raise ValueError(f"{tl_id}: green phase {g} serves {served}; expected exactly one "
                                 f"incoming lane per green phase")
            ordered.append(served[0])
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"{tl_id}: two actions serve the same lane: {ordered}")
        tl_ids.append(tl_id)
        lane_ids[tl_id] = ordered
    tl_ids.sort()
    return tl_ids, lane_ids


TL_IDS, LANE_IDS = _derive_network_constants(NET_FILE)

NUM_AGENTS = len(TL_IDS)


class TraCIEnv:
    def __init__(self):
        self.agents = TL_IDS
        self.num_agents = NUM_AGENTS
        self.green = {tl: 0 for tl in TL_IDS}
        self.pending = {tl: 0 for tl in TL_IDS}
        self.yellow_left = {tl: 0 for tl in TL_IDS}
        self.elapsed = {tl: 0 for tl in TL_IDS}
        self._rng = random.Random(int(os.environ.get("MARL_ROUTE_SEED", "0")))

    def reset(self, route_file=None):
        # Allow a list of route files: pick one per episode (training seeds)
        if isinstance(route_file, (list, tuple)):
            route_file = self._rng.choice(route_file)
        self.current_route_file = route_file
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
            q = [min(traci.lane.getLastStepHaltingNumber(l), MAX_QUEUE) / MAX_QUEUE for l in lanes]
            phase_idx = self.green[tl_id] // 2
            phase_onehot = [1.0 if phase_idx == i else 0.0 for i in range(PHASES)]
            elapsed_ratio = min(self.elapsed[tl_id] / MAX_PHASE_DURATION, 1.0)
            obs[tl_id] = q + phase_onehot + [elapsed_ratio]
        return obs

    def get_messages(self):
        """
        Hand-crafted communication (B2): each agent's raw queue readings,
        to be broadcast to its neighbours as-is (no learning, no encoding).
        Returns {tl_id: [q0, q1, q2, q3]} — same 4 real queue values used in get_obs(),
        in action order (q_k = queue on the lane that action k turns green).
        """
        messages = {}
        for tl_id in self.agents:
            lanes = LANE_IDS[tl_id]
            q = [min(traci.lane.getLastStepHaltingNumber(l), MAX_QUEUE) / MAX_QUEUE for l in lanes]
            messages[tl_id] = q
        return messages

    def get_reward(self):
        total_wait = 0.0
        for tl_id in self.agents:
            for lane in LANE_IDS[tl_id]:
                total_wait += traci.lane.getWaitingTime(lane)
        return -total_wait