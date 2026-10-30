"""
Regression tests that fail loudly if a fixed bug ever comes back.
Run from the repo root:  python tests/test_fixes.py
"""
import os, re, sys, csv, subprocess, tempfile, traceback
import xml.etree.ElementTree as ET

REPO = "/content/MARL-Traffic"
sys.path.insert(0, REPO)

# 4x4 grid, 5 s minimum green — set BEFORE importing env (read at import time)
os.environ["MARL_NET_FILE"] = f"{REPO}/configs/mygrid4x4.net.xml"
os.environ["MARL_SUMO_CFG"] = f"{REPO}/configs/simulation_4x4.sumocfg"
os.environ["MARL_MIN_GREEN"] = "5"

import numpy as np
import sumolib
from env.traci_env import (TraCIEnv, TL_IDS, LANE_IDS, OBS_DIM, GREEN_PHASES,
                           MIN_GREEN, _derive_network_constants)
from env.rllib_env_e1 import MARLTrafficEnvE1, build_neighbors_from_net, MAX_NEIGHBORS

NET_2X2 = f"{REPO}/configs/mygrid2x2.net.xml"
NET_4X4 = f"{REPO}/configs/mygrid4x4.net.xml"
END = 3600
RANDOM_UNIFORM_SEED0_DELAY = 91.225   # 4x4 random, uniform seed0, min green 0 (deterministic)


def _lanes_green_in_phase(tls, phase_idx):
    links = sorted(tls.getConnections(), key=lambda c: c[2])
    state = list(tls.getPrograms().values())[0].getPhases()[phase_idx].state
    return sorted({links[i][0].getID() for i, s in enumerate(state) if s in "Gg"})


def test_lane_order():
    for grid, net_path in [("2x2", NET_2X2), ("4x4", NET_4X4)]:
        net = sumolib.net.readNet(net_path, withPrograms=True)
        _, lane_ids = _derive_network_constants(net_path)
        for tls in net.getTrafficLights():
            tl = tls.getID()
            if tl not in lane_ids:
                continue
            for k, g in enumerate(GREEN_PHASES):
                served = _lanes_green_in_phase(tls, g)
                assert served == [lane_ids[tl][k]], \
                    f"{grid} {tl}: action {k} (phase {g}) serves {served}, slot {k} is {lane_ids[tl][k]}"
    print("PASS 1 - lane order: slot k is the lane action k turns green (2x2 and 4x4)")


def _load_metrics():
    src = open(f"{REPO}/scripts/evaluate_4x4.py").read()
    body = re.search(r"^def metrics\b.*?(?=^def )", src, re.S | re.M).group(0)
    ns = {"ET": ET, "END": END}
    exec(body, ns)
    return ns["metrics"]


def test_never_entered():
    metrics = _load_metrics()
    depart = 200.0
    with tempfile.TemporaryDirectory() as d:
        route, trip = os.path.join(d, "r.rou.xml"), os.path.join(d, "t.xml")
        open(route, "w").write(f'<routes><vehicle id="V" depart="{depart:.2f}"><route edges="a b"/></vehicle></routes>')
        open(trip, "w").write("<tripinfos></tripinfos>")   # V never entered
        m = metrics(trip, route)
    assert m["entered"] == 0 and m["generated"] == 1, m
    assert abs(m["true_delay"] - (END - depart)) < 1e-6, \
        f"never-entered car got {m['true_delay']}, expected {END - depart}"
    print("PASS 2 - never-entered cars: delay = 3600 - planned departure")


def test_neighbor_slots():
    neigh = build_neighbors_from_net(NET_4X4, TL_IDS)
    counts = {}
    for tl in TL_IDS:
        counts[len(neigh[tl])] = counts.get(len(neigh[tl]), 0) + 1
    assert counts == {2: 4, 3: 8, 4: 4}, f"4x4 neighbour counts {counts}, expected 4 corner/8 edge/4 inner"

    env = MARLTrafficEnvE1({"net_path": NET_4X4})
    slots = env.neighbor_slots
    net = sumolib.net.readNet(NET_4X4)
    for tl in TL_IDS:
        for k, lane in enumerate(LANE_IDS[tl]):
            up = net.getLane(lane).getEdge().getFromNode().getID()
            expected = up if (up in TL_IDS and up != tl) else None
            assert slots[tl][k] == expected, f"{tl} slot {k}: {slots[tl][k]} != upstream of lane {lane} ({expected})"

    fake = {tl: np.full(OBS_DIM, i, dtype=np.float32) for i, tl in enumerate(TL_IDS)}
    bundled = env._bundle_with_neighbors(fake)
    for tl in TL_IDS:
        full = bundled[tl]
        block = full[OBS_DIM:OBS_DIM + MAX_NEIGHBORS * OBS_DIM].reshape(MAX_NEIGHBORS, OBS_DIM)
        mask = full[OBS_DIM + MAX_NEIGHBORS * OBS_DIM:]
        for k, n in enumerate(slots[tl]):
            if n is None:
                assert mask[k] == 0.0 and np.allclose(block[k], 0.0)
            else:
                assert mask[k] == 1.0 and np.allclose(block[k], fake[n])
    print("PASS 3 - E1 neighbour slots, mask, and 4/8/4 light structure")


def test_min_green():
    assert MIN_GREEN == 5, f"needs MARL_MIN_GREEN=5 (got {MIN_GREEN})"
    import traci
    env = TraCIEnv()
    env.reset(route_file=f"{REPO}/configs/routes_4x4_uniform_seed0.rou.xml")
    tl0, seq = TL_IDS[0], []
    for _ in range(150):
        env.step([(env.green[tl] // 2 + 1) % len(GREEN_PHASES) for tl in TL_IDS])  # always switch
        seq.append(traci.trafficlight.getPhase(tl0))
    try:
        traci.close()
    except Exception:
        pass
    runs, cur, ln = [], seq[0], 1
    for p in seq[1:]:
        if p == cur:
            ln += 1
        else:
            runs.append((cur, ln)); cur, ln = p, 1
    runs.append((cur, ln))
    for i, (phase, length) in enumerate(runs):
        if phase % 2 == 0 and 0 < i < len(runs) - 1:   # complete green
            assert length >= 5, f"a green lasted {length}s (<5) with MARL_MIN_GREEN=5"

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "random.csv")
        env_vars = dict(os.environ); env_vars["MARL_MIN_GREEN"] = "0"
        subprocess.run([sys.executable, f"{REPO}/scripts/evaluate_4x4.py", "--cond", "random",
                        "--grid", "4x4", "--scenarios", "uniform", "--seeds", "0",
                        "--min-green", "0", "--out", out], cwd=REPO, env=env_vars, check=True)
        got = float(list(csv.DictReader(open(out)))[0]["true_delay"])
    assert abs(got - RANDOM_UNIFORM_SEED0_DELAY) < 1e-3, \
        f"random uniform seed0 gave {got}, expected {RANDOM_UNIFORM_SEED0_DELAY}"
    print("PASS 4 - minimum green: greens >= 5 s; random uniform seed0 unchanged")


if __name__ == "__main__":
    failed = 0
    for t in [test_lane_order, test_never_entered, test_neighbor_slots, test_min_green]:
        try:
            t()
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}:")
            traceback.print_exc()
    print(f"\n{'ALL TESTS PASSED ✅' if not failed else str(failed) + ' test(s) FAILED'}")
    sys.exit(1 if failed else 0)
