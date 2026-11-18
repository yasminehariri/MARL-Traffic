"""Checks the sensor-failure option (MARL_BLIND) on the real 4x4 network, for the B1, B2 and E1 wrappers.

For each wrapper the same fixed action sequence is run on the same route file with all sensors
working and with three failure patterns: the FYP test (all detectors of B1), four whole lights
(A2, B1, C2, D1) and four single detectors. Because the actions are identical, the
traffic is identical, so the two observation streams must be equal everywhere EXCEPT exactly
the queue values that come from a failed light:
  * a failed light's own 4 queue values are 0 (phase one-hot and timer unchanged),
  * B2: each neighbour receives that light's queues as 0 inside the averaged message,
  * E1: each neighbour sees that light's queue values as 0 in its neighbour block.
With MARL_BLIND unset the code must behave exactly as before (checked separately by
reproducing published CSV rows with scripts/evaluate_4x4.py).

Run from the repo root:  python tests/test_sensor_failure.py
"""
import json, os, pickle, subprocess, sys, tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = {"FYP test (B1)": "B1", "whole lights": "A2,B1,C2,D1", "single detectors": "A2:0,B1:3,C2:1,D1:2"}
ROUTE = os.path.join(REPO, "configs", "routes_4x4_peak_seed0.rou.xml")
STEPS = 600

CHILD = r"""
import os, pickle, sys
sys.path.insert(0, os.environ["REPO"])
cond, out = sys.argv[1], sys.argv[2]
if cond == "B1":
    from env.rllib_env import MARLTrafficEnv as Env
elif cond == "B2":
    from env.rllib_env_b2 import MARLTrafficEnvB2 as Env
else:
    from env.rllib_env_e1 import MARLTrafficEnvE1 as Env
import env.traci_env as te
env = Env({"route_file": os.environ["ROUTE"]})
obs, _ = env.reset()
frames, extra = [obs], {}
for t in range(int(os.environ["STEPS"])):
    acts = {tl: (t // 13 + i) % 4 for i, tl in enumerate(sorted(obs))}
    obs, *_ = env.step(acts)
    frames.append(obs)
if cond == "E1":
    extra["slots"] = env.neighbor_slots
if cond == "B2":
    extra["neighbors"] = env.neighbors
extra["failed"] = {k: sorted(v) for k, v in te.FAILED.items()}
extra["adjacent"] = te.ADJACENT
import traci
traci.close()
pickle.dump({"frames": frames, **extra}, open(out, "wb"))
"""


def run(cond, spec):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "o.pkl")
        env = dict(os.environ, REPO=REPO, ROUTE=ROUTE, STEPS=str(STEPS), MARL_MIN_GREEN="5", MARL_REWARD="local",
                   MARL_NET_FILE=os.path.join(REPO, "configs", "mygrid4x4.net.xml"),
                   MARL_SUMO_CFG=os.path.join(REPO, "configs", "simulation_4x4.sumocfg"))
        env.pop("MARL_BLIND", None)
        if spec:
            env["MARL_BLIND"] = spec
        subprocess.run([sys.executable, "-c", CHILD, cond, out], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return pickle.load(open(out, "rb"))


def parse(spec):
    out = {}
    for e in spec.split(","):
        tl, _, k = e.partition(":")
        out.setdefault(tl, set()).update({int(k)} if k else {0, 1, 2, 3})
    return out


def check(cond, name, spec, base):
    failed = parse(spec)
    a, b = base, run(cond, spec)
    assert a["failed"] == {} and b["failed"] == {k: sorted(v) for k, v in failed.items()}, b["failed"]
    for s in failed:
        assert not set(a["adjacent"][s]) & set(failed), f"{s} has a failed neighbour"
    n_changed, n_hidden = 0, 0
    for t, (fa, fb) in enumerate(zip(a["frames"], b["frames"])):
        for tl in fa:
            va, vb = np.asarray(fa[tl], dtype=np.float64), np.asarray(fb[tl], dtype=np.float64)
            exp = va.copy()
            for k in failed.get(tl, ()):                      # own failed detectors -> 0
                n_hidden += int(exp[k] > 0)
                exp[k] = 0.0
            if cond == "B2":                                  # averaged neighbour message
                nbrs = a["neighbors"][tl]
                if any(n in failed for n in nbrs):
                    msgs = []
                    for n in nbrs:
                        m = np.asarray(fa[n], dtype=np.float64)[0:4].copy()
                        for k in failed.get(n, ()):
                            m[k] = 0.0
                        msgs.append(m)
                    exp[9:13] = np.mean(msgs, axis=0)
            if cond == "E1":                                  # neighbour blocks
                for j, n in enumerate(a["slots"][tl]):
                    for k in failed.get(n, ()):
                        exp[9 + 9 * j + k] = 0.0
            assert np.allclose(vb, exp, atol=1e-6), f"{cond} {name} t={t} {tl}: max diff {np.abs(vb - exp).max()}"
            n_changed += int(not np.allclose(va, vb))
    # only meaningful if the failed detectors actually hid queues
    assert n_hidden > 50, f"{cond} {name}: failed detectors rarely had queues ({n_hidden})"
    print(f"{cond} {name:17s}: OK  ({len(a['frames'])} steps; {n_changed} observations changed, "
          f"each exactly as expected; {n_hidden} non-zero readings hidden; nothing else changed)")


if __name__ == "__main__":
    for c in ("B1", "B2", "E1"):
        base = run(c, "")
        for name, spec in SPECS.items():
            check(c, name, spec, base)
    print("ALL SENSOR-FAILURE CHECKS PASSED")
