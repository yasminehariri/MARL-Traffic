"""Checks --failsafe of scripts/evaluate_4x4.py with a real checkpoint (needs ray, torch, SUMO).

With --blind B1 --failsafe, at every step of a 400-step peak episode:
  * light B1 must follow the tuned fixed-time cycle exactly: action (t // 13) % 4 (10 s green + 3 s yellow),
  * every other light must receive exactly the action its own agent chose for that observation.

    python tests/test_failsafe.py <checkpoint>   (e.g. .../E1_local_mg5/checkpoints/ckpt_000977)
"""
import json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHILD = r"""
import json, os, sys
ckpt = sys.argv[1]
cond = "E1" if "/E1_" in ckpt else "B2" if "/B2_" in ckpt else "B1"
sys.argv = ["evaluate_4x4.py", "--cond", cond, "--ckpt", ckpt, "--blind", "B1", "--failsafe", "--out", "/tmp/unused.csv"]
ns = {"__name__": "not_main", "__file__": os.path.join(os.environ["REPO"], "scripts", "evaluate_4x4.py")}
exec(open(ns["__file__"]).read(), ns)
Env, choose = ns["build_trained"]()
agent = next(c.cell_contents for c in choose.__closure__ if callable(c.cell_contents))  # the agents' own choice
env = Env({"route_file": os.path.join(os.environ["REPO"], "configs", "routes_4x4_peak_seed100.rou.xml")})
obs, _ = env.reset()
log = []
for t in range(400):
    a = choose(obs, t)
    log.append([{k: int(v) for k, v in a.items()}, {k: int(v) for k, v in agent(obs, t).items()}])
    obs, *_ = env.step(a)
import traci; traci.close()
print("JSON" + json.dumps(log))
"""

if __name__ == "__main__":
    out = subprocess.run([sys.executable, "-c", CHILD, sys.argv[1]], env=dict(os.environ, REPO=REPO),
                         capture_output=True, text=True, check=True).stdout
    log = json.loads(out.split("JSON", 1)[1])
    overridden = 0
    for t, (used, own) in enumerate(log):
        assert used["B1"] == (t // 13) % 4, f"t={t}: B1 got {used['B1']}, expected {(t // 13) % 4}"
        overridden += int(used["B1"] != own["B1"])
        for k in used:
            if k != "B1":
                assert used[k] == own[k], f"t={t}: {k} got {used[k]}, its agent chose {own[k]}"
    assert overridden > 0, "the agent never disagreed with the fixed cycle, test not informative"
    print(f"FAIL-SAFE OK: B1 followed the fixed-time cycle at all {len(log)} steps (its agent was overruled "
          f"{overridden} times); the other 15 lights got exactly their agent's action at every step")
