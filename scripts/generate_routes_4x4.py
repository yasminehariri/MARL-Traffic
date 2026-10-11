"""
Seeded route generation for the 4x4 grid: Uniform, Peak-hour, Asymmetric.

All three scenarios are built the same way:
  1. Vehicles enter only from fringe (boundary) edges, leave only via fringe edges.
  2. Arrivals at each entry follow a seeded Poisson process (random gaps),
     so different seeds give genuinely different traffic -> real variance
     across evaluation episodes (needed for paired t-tests).
  3. duarouter turns origin/destination trips into full shortest-path routes.

Demand is matched to the 2x2 setup: ~112.5 veh/hr per fringe entry
(2x2 had ~900 veh/hr over 8 entries; 4x4 has 16 entries -> ~1800 veh/hr).

Vehicle mix (4x4 only): 80% cars, 10% delivery vans, 5% buses, 5% trucks,
each with its own length, acceleration and top speed. The type of each
vehicle is drawn from a SEPARATE random stream, so arrival times and
destinations are identical to the cars-only files for the same seed.

Usage in Colab (from the repo root):
  !python scripts/generate_routes_4x4.py --check          # seed 0 of each scenario + sanity check
  !python scripts/generate_routes_4x4.py --all            # all 3 scenarios x 10 seeds + sanity check
  !python scripts/generate_routes_4x4.py --train          # Uniform training seeds 100-104 + sanity check
  !python scripts/generate_routes_4x4.py --train-all      # Peak + Asymmetric training seeds 100-104 + sanity check
"""

import argparse
import os
import random
import re
import subprocess
import xml.etree.ElementTree as ET

import sumolib

CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
NET = os.path.join(CONFIGS, "mygrid4x4.net.xml")

EPISODE_END = 3600          # seconds, matches the 3600-step episodes
PER_ENTRY_RATE = 112.5      # veh/hr per fringe entry (matched to 2x2)
PEAK_WINDOW = (720, 1800)   # seconds where demand doubles (20%-50% of the episode)
PEAK_MULTIPLIER = 2.0
ASYM_RATIO = 4.0            # north entries get 4x the per-entry rate of the others
SEEDS = range(10)               # evaluation only, never seen in training
TRAIN_SEEDS = range(100, 105)   # training only (Uniform), kept separate from evaluation
SCENARIOS = ["uniform", "peak", "asym"]

# id: (share, vType attributes). Lengths and speeds follow SUMO's own class
# defaults for each vehicle class, so they are standard values, not invented.
VEHICLE_MIX = {
    "car":   (0.80, dict(vClass="passenger", guiShape="passenger", length="4.5",  minGap="2.5",
                         accel="2.6", decel="4.5", maxSpeed="13.89", color="70,110,200")),
    "van":   (0.10, dict(vClass="delivery",  guiShape="delivery",  length="6.5",  minGap="2.5",
                         accel="2.0", decel="4.0", maxSpeed="13.89", color="230,230,230")),
    "bus":   (0.05, dict(vClass="bus",       guiShape="bus",       length="12.0", minGap="3.0",
                         accel="1.2", decel="4.0", maxSpeed="11.11", color="230,170,40")),
    "truck": (0.05, dict(vClass="truck",     guiShape="truck",     length="10.0", minGap="3.0",
                         accel="1.3", decel="4.0", maxSpeed="11.11", color="150,70,60")),
}
TYPE_SEED_OFFSET = 10_000  # vehicle types use their own random stream


def fringe_edges(net):
    """Entry edges start at a dead_end node; exit edges end at one."""
    entries, exits = [], []
    for e in net.getEdges():
        if e.getFromNode().getType() == "dead_end":
            entries.append(e)
        if e.getToNode().getType() == "dead_end":
            exits.append(e)
    return entries, exits


def entry_rates(entries, scenario):
    """Per-entry veh/hr. Asymmetric keeps the same total as Uniform, just reshaped."""
    if scenario != "asym":
        return {e.getID(): PER_ENTRY_RATE for e in entries}
    total = PER_ENTRY_RATE * len(entries)
    weights = {e.getID(): (ASYM_RATIO if e.getFromNode().getID().startswith("top") else 1.0)
               for e in entries}
    scale = total / sum(weights.values())
    return {eid: w * scale for eid, w in weights.items()}


def rate_at(base_rate, t, scenario):
    if scenario == "peak" and PEAK_WINDOW[0] <= t < PEAK_WINDOW[1]:
        return base_rate * PEAK_MULTIPLIER
    return base_rate


def make_trips(net, scenario, seed, trips_path):
    rng = random.Random(seed)
    entries, exits = fringe_edges(net)
    rates = entry_rates(entries, scenario)
    trips = []

    for entry in entries:
        from_node = entry.getFromNode().getID()
        # Destination: any exit except straight back out to where it came in
        candidates = [x for x in exits if x.getToNode().getID() != from_node]
        t = 0.0
        while True:
            lam = rate_at(rates[entry.getID()], t, scenario) / 3600.0  # veh/s
            t += rng.expovariate(lam)
            if t >= EPISODE_END:
                break
            dest = rng.choice(candidates)
            trips.append((round(t, 2), entry.getID(), dest.getID()))

    trips.sort(key=lambda x: x[0])
    root = ET.Element("routes")
    for tid, (_, attrs) in VEHICLE_MIX.items():
        ET.SubElement(root, "vType", id=tid, attrib=attrs)

    type_rng = random.Random(seed + TYPE_SEED_OFFSET)
    type_ids = list(VEHICLE_MIX)
    shares = [VEHICLE_MIX[t][0] for t in type_ids]
    for i, (depart, frm, to) in enumerate(trips):
        vtype = type_rng.choices(type_ids, weights=shares)[0]
        ET.SubElement(root, "trip", id=f"{scenario}_{seed}_{i}", type=vtype, depart=f"{depart:.2f}",
                      attrib={"from": frm, "to": to, "departLane": "best", "departSpeed": "max"})
    ET.ElementTree(root).write(trips_path)
    return len(trips)


def route(trips_path, routes_path):
    subprocess.run(["duarouter", "-n", NET, "-t", trips_path, "-o", routes_path,
                    "--ignore-errors", "--no-step-log", "--no-warnings"],
                   check=True, capture_output=True, text=True)


def sanity_check(routes_path, label):
    tripinfo = os.path.join(CONFIGS, f"tripinfo_check_{label}.xml")
    res = subprocess.run(["sumo", "-n", NET, "-r", routes_path, "--end", str(EPISODE_END),
                          "--tripinfo-output", tripinfo, "--no-step-log"],
                         capture_output=True, text=True)
    teleports = len(re.findall(r"Teleporting", res.stderr))
    root = ET.parse(tripinfo).getroot()
    infos = root.findall("tripinfo")
    waits = [float(t.get("waitingTime", 0)) for t in infos]
    mix = {}
    for t in infos:
        mix[t.get("vType")] = mix.get(t.get("vType"), 0) + 1
    os.remove(tripinfo)
    done = len(waits)
    mean = sum(waits) / done if done else float("nan")
    share = sum(1 for w in waits if w > 0) / done if done else 0
    return done, mean, share, teleports, mix


def build(scenario, seed, check):
    trips_path = os.path.join(CONFIGS, f"trips_4x4_{scenario}_seed{seed}.trips.xml")
    routes_path = os.path.join(CONFIGS, f"routes_4x4_{scenario}_seed{seed}.rou.xml")
    n = make_trips(sumolib.net.readNet(NET), scenario, seed, trips_path)
    route(trips_path, routes_path)
    os.remove(trips_path)
    line = f"{scenario:8s} seed{seed}: {n:5d} vehicles generated"
    if check:
        done, mean, share, tp, mix = sanity_check(routes_path, f"{scenario}_{seed}")
        mix_txt = " ".join(f"{k}:{mix.get(k, 0)}" for k in VEHICLE_MIX)
        line += (f" | completed {done:5d} | waited {share:5.1%} | mean wait {mean:7.2f}s"
                 f" | teleports {tp} | {mix_txt}")
    print(line, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="seed 0 of each scenario, with sanity check")
    g.add_argument("--all", action="store_true", help="all scenarios x 10 seeds, with sanity check")
    g.add_argument("--train", action="store_true", help="Uniform training seeds 100-104, with sanity check")
    g.add_argument("--train-all", dest="train_all", action="store_true",
                   help="Peak and Asymmetric training seeds 100-104 (Uniform ones come from --train)")
    a = p.parse_args()

    if a.train:
        for s in TRAIN_SEEDS:
            build("uniform", s, check=True)
    elif a.train_all:
        for sc in ("peak", "asym"):
            for s in TRAIN_SEEDS:
                build(sc, s, check=True)
    else:
        seeds = [0] if a.check else SEEDS
        for sc in SCENARIOS:
            for s in seeds:
                build(sc, s, check=True)