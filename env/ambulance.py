"""
Ambulances and signal preemption (evaluation only; nothing here is used in training).

Ambulances
    One ambulance every INTERVAL seconds (300, 600, ..., 3300 s: 11 per episode). Each one drives from a
    random boundary entry to a random boundary exit on another side of the grid, along the shortest route,
    and must cross at least MIN_JUNCTIONS signalised junctions. The schedule depends only on (grid,
    scenario, seed), so every controller, with and without preemption, gets exactly the same ambulances.
    They are added through TraCI after SUMO starts; the route files are not changed.
    Vehicle: SUMO class "emergency", 6.5 m, drives at the speed limit with no random dawdling, and obeys
    the lights like any vehicle (no SUMO "bluelight" device), so any green it gets is given by the signals.

Preemption (when switched on)
    Detection: a light detects an ambulance as soon as it is on one of the light's approach lanes (one
    block, 186 to 193 m upstream of the stop line), as an optical or radio emitter would. The light then
    serves that approach: its agent's action is replaced by the action that turns that lane green. The
    environment still applies its normal safety rules: a green change always goes through the 3 s
    yellow (never shortened, FHWA Signal Timing Manual 9.1.2) and the 5 s minimum green. If the light is
    in a yellow when the ambulance is detected, the yellow finishes and then goes to the ambulance's
    approach (a return to the previous green after the yellow is allowed, FHWA 9.1.2). When the ambulance
    has crossed the stop line, the light goes straight back to its agent. If several ambulances approach
    one light, the one closest to the stop line is served first.
"""
import random

import sumolib

INTERVAL = 300          # seconds between ambulances
FIRST = 300             # first departure
END = 3600              # episode length
MIN_JUNCTIONS = 3       # signalised junctions an ambulance route must cross
PREFIX = "amb_"         # vehicle id prefix (metrics use it to separate ambulances from other traffic)
VTYPE = "ambulance"
SCENARIO_CODE = {"uniform": 1, "peak": 2, "asym": 3, "surge": 4}


def is_ambulance(vid):
    return vid.startswith(PREFIX)


def candidate_routes(net_file):
    """All boundary-to-boundary shortest routes that cross >= MIN_JUNCTIONS signalised junctions,
    from an entry on one side to an exit on a different side. Sorted, so the list never depends on
    dictionary or file order."""
    net = sumolib.net.readNet(net_file)
    tls = {t.getID() for t in net.getTrafficLights()}
    entries = sorted(e.getID() for e in net.getEdges() if e.getFromNode().getID() not in tls)
    exits = sorted(e.getID() for e in net.getEdges() if e.getToNode().getID() not in tls)
    side = lambda node_id: "".join(c for c in node_id if c.isalpha())   # "left2" -> "left"
    routes = []
    for a in entries:
        for b in exits:
            if side(net.getEdge(a).getFromNode().getID()) == side(net.getEdge(b).getToNode().getID()):
                continue
            path, _ = net.getShortestPath(net.getEdge(a), net.getEdge(b))
            if path is None:
                continue
            crossed = [e.getToNode().getID() for e in path if e.getToNode().getID() in tls]
            if len(crossed) >= MIN_JUNCTIONS:
                routes.append(tuple(e.getID() for e in path))
    if not routes:
        raise ValueError(f"no ambulance route crosses {MIN_JUNCTIONS} junctions in {net_file}")
    return routes


def schedule(net_file, grid, scenario, seed):
    """[(vehicle id, departure time, route edges)] for one episode. Deterministic in (grid, scenario, seed)."""
    routes = candidate_routes(net_file)
    rng = random.Random(f"ambulances|{grid}|{scenario}|{seed}")   # string seeds are hashed deterministically
    out = []
    for i, t in enumerate(range(FIRST, END, INTERVAL)):
        out.append((f"{PREFIX}{i:02d}", float(t), rng.choice(routes)))
    return out


def add_to_simulation(plan):
    """Define the ambulance vehicle type and add the planned ambulances (call right after SUMO starts)."""
    import traci
    traci.vehicletype.copy("DEFAULT_VEHTYPE", VTYPE)
    traci.vehicletype.setVehicleClass(VTYPE, "emergency")
    traci.vehicletype.setShapeClass(VTYPE, "emergency")
    traci.vehicletype.setLength(VTYPE, 6.5)
    traci.vehicletype.setWidth(VTYPE, 2.2)
    traci.vehicletype.setMinGap(VTYPE, 2.5)
    traci.vehicletype.setAccel(VTYPE, 2.6)
    traci.vehicletype.setDecel(VTYPE, 4.5)
    traci.vehicletype.setMaxSpeed(VTYPE, 13.89)
    traci.vehicletype.setSpeedFactor(VTYPE, 1.0)
    traci.vehicletype.setSpeedDeviation(VTYPE, 0.0)
    traci.vehicletype.setImperfection(VTYPE, 0.0)
    traci.vehicletype.setColor(VTYPE, (230, 30, 30, 255))
    for vid, depart, edges in plan:
        traci.route.add(f"route_{vid}", list(edges))
        traci.vehicle.add(vid, f"route_{vid}", typeID=VTYPE, depart=f"{depart:.0f}",
                          departLane="best", departSpeed="max")


class Preemption:
    """Decides, every second, which lights must serve an ambulance, and logs what it did.

    lane_ids: {light: [approach lane for action 0, 1, 2, 3]} (env.traci_env.LANE_IDS)
    """

    def __init__(self, lane_ids):
        self.lane_to = {lane: (tl, k) for tl, lanes in lane_ids.items() for k, lane in enumerate(lanes)}
        self.lane_len = {}
        self.reset()

    def reset(self):
        self.active = set()          # ambulances currently in the network
        self.seconds = 0             # light-seconds under preemption this episode
        self.events = 0              # number of times a light started serving an ambulance
        self.serving = {}            # light -> ambulance it is serving this second (used by the demo recorder)

    def targets(self):
        """{light: action} for every light that has an ambulance on one of its approaches now."""
        import traci
        for vid in traci.simulation.getDepartedIDList():
            if is_ambulance(vid):
                self.active.add(vid)
        for vid in traci.simulation.getArrivedIDList():
            self.active.discard(vid)
        best = {}                    # light -> (distance to stop line, ambulance, action)
        for vid in sorted(self.active):
            lane = traci.vehicle.getLaneID(vid)
            hit = self.lane_to.get(lane)
            if hit is None:          # inside a junction or on an exit edge: no light to preempt
                continue
            tl, k = hit
            if lane not in self.lane_len:
                self.lane_len[lane] = traci.lane.getLength(lane)
            dist = self.lane_len[lane] - traci.vehicle.getLanePosition(vid)
            if tl not in best or dist < best[tl][0]:
                best[tl] = (dist, vid, k)
        serving = {tl: vid for tl, (_, vid, _) in best.items()}
        self.events += sum(1 for tl, vid in serving.items() if self.serving.get(tl) != vid)
        self.seconds += len(serving)
        self.serving = serving
        return {tl: k for tl, (_, _, k) in best.items()}


def wrap_choose(choose, preemption, traci_env):
    """Agent actions, with every light that has an ambulance on an approach serving that approach.

    The agents still choose for every light first, so sampled actions of the other lights are drawn
    exactly as without preemption. traci_env is the TraCIEnv: if a preempted light is in its yellow,
    the green that follows the yellow is set to the ambulance's approach.
    """
    from env.traci_env import GREEN_PHASES

    def chosen(obs, t):
        actions = dict(choose(obs, t))
        for tl, k in preemption.targets().items():
            actions[tl] = k
            if traci_env.yellow_left[tl] > 0:
                traci_env.pending[tl] = GREEN_PHASES[k]
        return actions
    return chosen


def trip_metrics(tripinfo_file, plan):
    """Ambulance results from SUMO's tripinfo file (written with --tripinfo-output.write-unfinished).

    travel time = arrival - planned departure (includes waiting to enter the network); an ambulance
    still on the road at the end counts END - planned departure, so a blocked ambulance is never
    dropped from the average.
    """
    import xml.etree.ElementTree as ET
    info = {t.get("id"): t for t in ET.parse(tripinfo_file).getroot().findall("tripinfo")
            if is_ambulance(t.get("id"))}
    times, stops, waits, arrived = [], [], [], 0
    for vid, depart, _ in plan:
        t = info.get(vid)
        arr = float(t.get("arrival", -1)) if t is not None else -1.0
        if arr >= 0:
            arrived += 1
            times.append(arr - depart)
            stops.append(int(t.get("waitingCount", 0)))
            waits.append(float(t.get("waitingTime", 0)) + float(t.get("departDelay", 0)))
        else:
            times.append(END - depart)
    n = len(plan)
    return dict(amb_n=n, amb_arrived=arrived,
                amb_travel_time=round(sum(times) / n, 3),
                amb_stops=round(sum(stops) / max(1, len(stops)), 3),
                amb_wait=round(sum(waits) / max(1, len(waits)), 3))
