"""
Final evaluation on the held-out EVALUATION seeds (0-9) for all three scenarios.

One row per (condition, scenario, seed) is appended to a CSV on Drive. The script
skips rows that already exist, so after a disconnect you just run it again.

Conditions:
    B1, B2, E1   trained checkpoints (need --ckpt)
    fixed        SUMO's default fixed-time program (92 s cycle) -- the scope's B0
    tuned        fixed-time cycle with GREEN s greens (chosen on TRAINING seeds only)
    random       uniformly random action every second

Metrics (every generated vehicle counts):
    entered, generated, finished, wait_finished (mean waiting of finished trips),
    true_delay (waiting + insertion delay; a car that never entered counts as
    waiting from its planned departure to 3600 s), mean_travel_time (finished),
    teleports.

Minimum green: trained checkpoints are evaluated with the minimum green they were trained with
(read from run_info.json). Runs with a minimum green go to <grid>_eval_mg5.csv by default, so
version 1 results (4x4_eval.csv) are never mixed with them.

Colab (one evaluation per runtime: runs in the same runtime would share tripinfo.xml):
    !python scripts/evaluate_4x4.py --cond fixed
    !python scripts/evaluate_4x4.py --cond tuned
    !python scripts/evaluate_4x4.py --cond random
    !python scripts/evaluate_4x4.py --cond E1 --ckpt /content/drive/MyDrive/MARL_checkpoints/4x4/E1_local/checkpoints/ckpt_000977
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = "/content/MARL-Traffic"
END = 3600

p = argparse.ArgumentParser()
p.add_argument("--cond", required=True, choices=["B1", "B2", "E1", "fixed", "tuned", "random"])
p.add_argument("--ckpt", default=None)
p.add_argument("--label", default=None, help="name written in the CSV (default: --cond)")
p.add_argument("--scenarios", default="uniform,peak,asym")
p.add_argument("--seeds", default="0-9")
p.add_argument("--green", type=int, default=10,
               help="green seconds for --cond tuned (10 s chosen on training seeds 100-104, all scenarios)")
p.add_argument("--explore", action="store_true", help="sample actions instead of the most likely one")
p.add_argument("--out", default=None,
               help="CSV to append to (default: MARL_results/<grid>_eval.csv, or <grid>_eval_mg<N>.csv with a minimum green)")
p.add_argument("--min-green", type=int, default=None,
               help="trained conditions: read automatically from the checkpoint's run_info.json (pass only to "
                    "double-check). tuned/random: default 5. fixed: not used (SUMO's own program)")
p.add_argument("--grid", default="4x4", choices=["4x4", "2x2"],
               help="grid to evaluate ON (a 2x2-trained E1 can be evaluated on 4x4 for RQ3)")
args = p.parse_args()

if args.cond in ("B1", "B2", "E1") and not args.ckpt:
    sys.exit("--ckpt is required for trained conditions")
LABEL = args.label or args.cond
a, b = (args.seeds.split("-") + [None])[:2]
SEEDS = list(range(int(a), int(b) + 1)) if b else [int(s) for s in args.seeds.split(",")]
SCENARIOS = args.scenarios.split(",")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgutil import min_green_for, set_min_green, run_info_for  # noqa: E402

if args.cond in ("B1", "B2", "E1"):
    MIN_GREEN = min_green_for(args.ckpt, args.min_green)
    info = run_info_for(args.ckpt) or {}
    if info.get("grid", args.grid) != args.grid:
        print(f"NOTE: checkpoint trained on {info['grid']}, evaluated on {args.grid} (zero-shot transfer)", flush=True)
elif args.cond == "fixed":
    MIN_GREEN = 0  # SUMO's own fixed program; greens are already long, the env rule does not apply
else:
    MIN_GREEN = 5 if args.min_green is None else args.min_green
set_min_green(MIN_GREEN)
MG_TAG = f"_mg{MIN_GREEN}" if MIN_GREEN > 0 else ""
if args.out is None:
    # fixed-time does not depend on the rule: its rows go to the same file as the run they belong to
    args.out = f"/content/drive/MyDrive/MARL_results/{args.grid}_eval{MG_TAG}.csv"
print(f"Writing to {args.out}", flush=True)

NET, CFG = {"4x4": ("mygrid4x4.net.xml", "simulation_4x4.sumocfg"),
            "2x2": ("mygrid2x2.net.xml", "simulation.sumocfg")}[args.grid]
os.environ["MARL_NET_FILE"] = f"{REPO}/configs/{NET}"
os.environ["MARL_SUMO_CFG"] = f"{REPO}/configs/{CFG}"
sys.path.insert(0, REPO)
TRIPINFO_ENV = f"{REPO}/configs/tripinfo.xml"  # written by both sumocfg files
FIELDS = ["label", "cond", "scenario", "seed", "generated", "entered", "finished",
          "wait_finished", "true_delay", "mean_travel_time", "teleports", "explore", "ckpt", "time",
          "min_green", "switches_per_light_h", "grid"]


def metrics(tripinfo, route):
    infos = ET.parse(tripinfo).getroot().findall("tripinfo")
    planned = {v.get("id"): float(v.get("depart")) for v in ET.parse(route).getroot().iter("vehicle")}
    planned = {k: d for k, d in planned.items() if d < END}
    seen = {t.get("id"): t for t in infos}
    fin = [t for t in infos if float(t.get("arrival", -1)) >= 0]
    true = []
    for vid, dep in planned.items():
        t = seen.get(vid)
        true.append(END - dep if t is None else float(t.get("waitingTime", 0)) + float(t.get("departDelay", 0)))
    return dict(generated=len(planned), entered=len(seen), finished=len(fin),
                wait_finished=round(sum(float(t.get("waitingTime", 0)) for t in fin) / max(1, len(fin)), 3),
                true_delay=round(sum(true) / len(true), 3),
                mean_travel_time=round(sum(float(t.get("duration", 0)) for t in fin) / max(1, len(fin)), 3))


def existing_header():
    if not os.path.exists(args.out):
        return None
    with open(args.out) as f:
        return next(csv.reader(f), None)


def done_rows():
    if not os.path.exists(args.out):
        return set()
    with open(args.out) as f:
        rows = list(csv.DictReader(f))
    if rows and args.cond != "fixed":
        file_mg = {int(r.get("min_green") or 0) for r in rows if r["cond"] != "fixed"}
        if file_mg and file_mg != {MIN_GREEN}:
            sys.exit(f"{args.out} holds rows with min_green {sorted(file_mg)}, this run uses {MIN_GREEN}. "
                     f"Use a different --out so the two rule sets never mix.")
    return {(r["label"], r["scenario"], int(r["seed"])) for r in rows}


def write_row(row):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    header = existing_header()
    fields = header or FIELDS
    missing = [k for k in row if k not in fields]
    if missing and (MIN_GREEN != 0 or args.grid != "4x4"):
        sys.exit(f"{args.out} is an old-format file without {missing}; use a new --out for these runs.")
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if header is None:
            w.writeheader()
        w.writerow(row)


def run_fixed(route):
    out = f"{REPO}/configs/tripinfo_eval_fixed.xml"
    res = subprocess.run(["sumo", "-n", os.environ["MARL_NET_FILE"], "-r", route, "--end", str(END),
                          "--tripinfo-output", out, "--tripinfo-output.write-unfinished", "--no-step-log"],
                         capture_output=True, text=True, check=True)
    m = metrics(out, route)
    os.remove(out)
    m["teleports"] = res.stderr.count("Teleporting vehicle")
    m["switches_per_light_h"] = ""
    return m


def run_env(env, choose, route):
    import traci
    obs, _ = env.reset()
    done, t, teleports = False, 0, 0
    while not done:
        obs, _, term, trunc, _ = env.step(choose(obs, t))
        teleports += traci.simulation.getStartingTeleportNumber()
        t += 1
        done = term.get("__all__", False) or trunc.get("__all__", False)
    try:
        traci.close()
    except Exception:
        pass
    time.sleep(1)
    m = metrics(TRIPINFO_ENV, route)
    m["teleports"] = teleports
    sw = env._env.switches
    m["switches_per_light_h"] = round(sum(sw.values()) / len(sw) * 3600 / END, 1)
    return m


def build_trained():
    import ray, torch
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog
    ray.init(num_cpus=2, object_store_memory=500 * 1024**2, include_dashboard=False, ignore_reinit_error=True)
    training = dict(lr=3e-4, clip_param=0.2, vf_loss_coeff=0.5, entropy_coeff=0.01,
                    gamma=0.99, lambda_=0.95, train_batch_size=512, num_epochs=4)
    policy_cls = None
    if args.cond == "B1":
        from env.rllib_env import MARLTrafficEnv as Env
    elif args.cond == "B2":
        from env.rllib_env_b2 import MARLTrafficEnvB2 as Env
    else:
        from env.rllib_env_e1 import MARLTrafficEnvE1 as Env
        from env.comm_model import E1CommModel
        from env.diversity_policy import E1DiversityPPOPolicy
        ModelCatalog.register_custom_model("e1_comm_model_v3", E1CommModel)
        policy_cls = E1DiversityPPOPolicy
        training["model"] = {"custom_model": "e1_comm_model_v3"}
    cfg = (PPOConfig()
           .environment(Env, env_config={"route_file": None})
           .framework("torch")
           .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
           .multi_agent(policies={"shared_policy": (policy_cls, None, None, {})},
                        policy_mapping_fn=lambda aid, *a, **k: "shared_policy")
           .training(**training)
           .resources(num_gpus=1 if torch.cuda.is_available() else 0)
           .env_runners(num_env_runners=0))
    algo = cfg.build()
    torch.set_num_threads(1)
    algo.restore(args.ckpt)
    print(f"Restored {args.cond} from {args.ckpt}", flush=True)
    choose = lambda obs, t: {aid: algo.compute_single_action(o, policy_id="shared_policy", explore=args.explore)
                             for aid, o in obs.items()}
    return Env, choose


def main():
    have = done_rows()
    todo = [(sc, s) for sc in SCENARIOS for s in SEEDS if (LABEL, sc, s) not in have]
    print(f"{LABEL}: {len(todo)} episodes to run ({len(SCENARIOS) * len(SEEDS) - len(todo)} already in {args.out})",
          flush=True)
    if not todo:
        return

    Env = choose = None
    if args.cond in ("B1", "B2", "E1"):
        Env, choose = build_trained()
    elif args.cond in ("tuned", "random"):
        from env.rllib_env import MARLTrafficEnv as Env
        rng = random.Random(0)
        if args.cond == "tuned":
            period = args.green + 3  # green + 3 s yellow inserted by the env
            choose = lambda obs, t: {aid: (t // period) % 4 for aid in obs}
        else:
            choose = lambda obs, t: {aid: rng.randrange(4) for aid in obs}

    for sc, s in todo:
        route = f"{REPO}/configs/routes_{args.grid}_{sc}_seed{s}.rou.xml"
        t0 = time.time()
        if args.explore:  # reproducible sampling: same seed -> same sampled actions
            import numpy as np, torch
            torch.manual_seed(s); np.random.seed(s); random.seed(s)
        if args.cond == "fixed":
            m = run_fixed(route)
        else:
            env = Env({"route_file": route})
            m = run_env(env, choose, route)
        row = dict(label=LABEL, cond=args.cond, scenario=sc, seed=s, explore=int(args.explore),
                   ckpt=args.ckpt or (f"green={args.green}" if args.cond == "tuned" else ""),
                   time=time.strftime("%Y-%m-%d %H:%M:%S"), min_green=MIN_GREEN, grid=args.grid, **m)
        write_row(row)
        print(f"{LABEL:8s} {sc:8s} seed{s}: entered {m['entered']}/{m['generated']} | finished {m['finished']} | "
              f"true delay {m['true_delay']:7.2f}s | wait(fin) {m['wait_finished']:7.2f}s | "
              f"teleports {m['teleports']} | switches/light/h {m['switches_per_light_h']} | "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
