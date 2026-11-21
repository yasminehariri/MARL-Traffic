"""Checks the ambulance and preemption code (env/ambulance.py) on the real 4x4 network.

1. Schedule: deterministic, 11 ambulances, valid connected routes crossing at least 3 junctions.
2. One full peak episode (seed 0) driven by the tuned fixed-time cycle, run twice with the same
   ambulances: preemption OFF and ON. Checked every second:
   OFF  every action is exactly the controller's action.
   ON   a light with an ambulance on one of its approaches gets the action of that approach
        (the closest ambulance if several); every other light gets exactly the controller's action;
        every green change goes through the full 3 s yellow; the ambulance's green comes within
        5 s (minimum green) + 3 s (yellow) + 3 s (a yellow already running) = 11 s; after the
        ambulance has crossed, the light is no longer overridden.
        Once an ambulance's approach is green, it stays green until the ambulance has crossed.
   Both: ambulances are left out of the traffic metrics; preemption gets at least as many ambulances
        through, with fewer stops and a shorter travel time.

Run from the repo root:  python tests/test_ambulance.py
It uses its own SUMO config and output files in a temporary folder, so it never touches a running
evaluation.
"""
import os, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="amb_test_")
NET = os.path.join(REPO, "configs", "mygrid4x4.net.xml")
ROUTE = os.path.join(REPO, "configs", "routes_4x4_peak_seed0.rou.xml")
CFG = os.path.join(TMP, "sim.sumocfg")
TRIP = os.path.join(TMP, "tripinfo.xml")
open(CFG, "w").write(f"""<configuration><input><net-file value="{NET}"/></input>
<time><begin value="0"/><end value="3600"/><step-length value="1"/></time>
<output><tripinfo-output value="{TRIP}"/></output></configuration>""")
os.environ.update(MARL_NET_FILE=NET, MARL_SUMO_CFG=CFG, MARL_MIN_GREEN="5", MARL_REWARD="local")
os.environ.pop("MARL_BLIND", None)
sys.path.insert(0, REPO)

import sumolib  # noqa: E402
import traci  # noqa: E402
from env import ambulance  # noqa: E402
from env.traci_env import TraCIEnv, TL_IDS, LANE_IDS, GREEN_PHASES, YELLOW_TIME, MIN_GREEN  # noqa: E402

PERIOD = 13   # tuned controller: 10 s greens + 3 s yellow
MAX_WAIT = MIN_GREEN + YELLOW_TIME + YELLOW_TIME


def check_schedule():
    net = sumolib.net.readNet(NET)
    tls = set(TL_IDS)
    a = ambulance.schedule(NET, "4x4", "peak", 0)
    assert a == ambulance.schedule(NET, "4x4", "peak", 0), "schedule not deterministic"
    assert a != ambulance.schedule(NET, "4x4", "peak", 1), "different seeds must give different ambulances"
    assert len(a) == 11 and [t for _, t, _ in a] == [float(t) for t in range(300, 3600, 300)]
    for vid, _, edges in a:
        es = [net.getEdge(e) for e in edges]
        for e1, e2 in zip(es, es[1:]):
            assert e2 in e1.getOutgoing(), f"{vid}: {e1.getID()} -> {e2.getID()} not connected"
        assert es[0].getFromNode().getID() not in tls and es[-1].getToNode().getID() not in tls
        assert sum(e.getToNode().getID() in tls for e in es) >= 3
    print("1 schedule: OK (deterministic, 11 ambulances every 300 s, connected boundary-to-boundary routes, >= 3 junctions)")
    return a


def episode(plan, preempt):
    env = TraCIEnv()
    env.reset(route_file=ROUTE)
    ambulance.add_to_simulation(plan)
    pre = ambulance.Preemption(LANE_IDS)
    controller = lambda obs, t: {tl: (t // PERIOD) % 4 for tl in TL_IDS}
    choose = ambulance.wrap_choose(controller, pre, env) if preempt else controller
    lane_to = {lane: (tl, k) for tl, lanes in LANE_IDS.items() for k, lane in enumerate(lanes)}
    waiting = {}                 # light -> (ambulance, first second it was overridden)
    yellow_run = {tl: 0 for tl in TL_IDS}
    last_phase = {tl: 0 for tl in TL_IDS}
    n_override = waits = 0
    max_wait = 0
    obs = None
    for t in range(3600):
        expect = controller(obs, t)
        acts = choose(obs, t)
        # what the rule must do, computed independently from TraCI
        on_approach = {}
        for vid in traci.vehicle.getIDList():
            if not ambulance.is_ambulance(vid):
                continue
            hit = lane_to.get(traci.vehicle.getLaneID(vid))
            if hit:
                tl, k = hit
                d = traci.lane.getLength(traci.vehicle.getLaneID(vid)) - traci.vehicle.getLanePosition(vid)
                if tl not in on_approach or d < on_approach[tl][0]:
                    on_approach[tl] = (d, vid, k)
        for tl in TL_IDS:
            if preempt and tl in on_approach:
                assert acts[tl] == on_approach[tl][2], f"t={t} {tl}: action {acts[tl]}, ambulance needs {on_approach[tl][2]}"
                n_override += 1
                vid, k = on_approach[tl][1], on_approach[tl][2]
                if waiting.get(tl, (None,))[0] != vid:
                    waiting[tl] = (vid, t, False)           # ambulance, first second served, green reached yet
                _, t0, reached = waiting[tl]
                if env.green[tl] == GREEN_PHASES[k] and env.yellow_left[tl] == 0:
                    if not reached:
                        max_wait = max(max_wait, t - t0)
                        waiting[tl] = (vid, t0, True)
                else:
                    assert not reached, f"t={t} {tl}: green taken away from ambulance {vid}"
                    assert t - t0 <= MAX_WAIT, f"t={t} {tl}: ambulance {vid} waited over {MAX_WAIT} s for green"
            else:
                assert acts[tl] == expect[tl], f"t={t} {tl}: action changed without an ambulance"
                waiting.pop(tl, None)
        obs, *_ = env.step([acts[tl] for tl in TL_IDS])
        for tl in TL_IDS:                         # yellow is never shortened
            ph = traci.trafficlight.getPhase(tl)
            if ph % 2 == 1:
                yellow_run[tl] += 1
            else:
                if ph != last_phase[tl]:
                    assert yellow_run[tl] == YELLOW_TIME, f"t={t} {tl}: green changed after {yellow_run[tl]} s of yellow"
                yellow_run[tl] = 0
                last_phase[tl] = ph
    traci.close()
    import xml.etree.ElementTree as ET
    import time
    time.sleep(1)
    ids = [x.get("id") for x in ET.parse(TRIP).getroot().findall("tripinfo")]
    planned = {v.get("id") for v in ET.parse(ROUTE).getroot().iter("vehicle")}
    amb_ids = [i for i in ids if ambulance.is_ambulance(i)]
    others = [i for i in ids if not ambulance.is_ambulance(i)]
    assert set(amb_ids) <= {v for v, _, _ in plan} and set(others) <= planned, "ambulances must be separable from the traffic"
    m = ambulance.trip_metrics(TRIP, plan)
    return m, pre, n_override, max_wait


if __name__ == "__main__":
    plan = check_schedule()
    off, _, n_off, _ = episode(plan, preempt=False)
    print(f"2 preemption OFF: OK (every action is the controller's own) | ambulances arrived {off['amb_arrived']}/11, "
          f"travel time {off['amb_travel_time']:.1f} s, stops {off['amb_stops']:.2f}")
    on, pre, n_on, max_wait = episode(plan, preempt=True)
    print(f"3 preemption ON: OK ({n_on} light-seconds overridden, each exactly as the rule says; every other "
          f"action untouched; yellow always 3 s; ambulance green within {max_wait} s, limit {MAX_WAIT}) | "
          f"arrived {on['amb_arrived']}/11, travel time {on['amb_travel_time']:.1f} s, stops {on['amb_stops']:.2f}")
    assert pre.seconds == n_on, (pre.seconds, n_on)
    assert on["amb_arrived"] >= off["amb_arrived"]
    assert on["amb_stops"] < off["amb_stops"] and on["amb_travel_time"] < off["amb_travel_time"], (on, off)
    print(f"4 effect: OK (with preemption: {on['amb_arrived']} vs {off['amb_arrived']} of 11 ambulances arrived, fewer stops, shorter travel time)")
    print("ALL AMBULANCE CHECKS PASSED")
