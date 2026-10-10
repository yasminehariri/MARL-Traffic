"""
Restore a 4x4 checkpoint and run ONE deterministic episode on one route file.

Purpose: (1) prove the checkpoint loads correctly (the "durable policy name"
warning), (2) get a first average-wait number for the trained policy, next to
the fixed-time number for the same traffic. This is a sanity check on a TRAINING
seed only; the real evaluation on seeds 0-9 comes later.

Colab:
    !python scripts/check_ckpt_4x4.py --cond E1 \
        --ckpt /content/drive/MyDrive/MARL_checkpoints/4x4/E1/checkpoints/ckpt_000977 \
        --route /content/MARL-Traffic/configs/routes_4x4_uniform_seed100.rou.xml
"""

import argparse
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = "/content/MARL-Traffic"

p = argparse.ArgumentParser()
p.add_argument("--cond", required=True, choices=["B1", "B2", "E1"])
p.add_argument("--ckpt", required=True)
p.add_argument("--route", required=True)
p.add_argument("--explore", action="store_true",
               help="sample actions from the policy (as in training) instead of always taking the most likely one")
args = p.parse_args()

os.environ["MARL_NET_FILE"] = f"{REPO}/configs/mygrid4x4.net.xml"
os.environ["MARL_SUMO_CFG"] = f"{REPO}/configs/simulation_4x4.sumocfg"
sys.path.insert(0, REPO)

import ray  # noqa: E402
import torch  # noqa: E402
import traci  # noqa: E402
from ray.rllib.algorithms.ppo import PPOConfig  # noqa: E402
from ray.rllib.models import ModelCatalog  # noqa: E402

TRIPINFO = f"{REPO}/configs/tripinfo.xml"  # written by simulation_4x4.sumocfg


def wait_stats(path, route=None):
    """Returns mean wait of finished vehicles, #finished, #entered, and TRUE delay:
    waiting + insertion delay for every generated vehicle; a vehicle that never
    entered counts as waiting from its planned departure until the end (3600 s)."""
    infos = ET.parse(path).getroot().findall("tripinfo")
    done = [float(t.get("waitingTime", 0)) for t in infos if float(t.get("arrival", -1)) >= 0]
    seen = {t.get("id"): float(t.get("waitingTime", 0)) + float(t.get("departDelay", 0)) for t in infos}
    planned = {v.get("id"): float(v.get("depart"))
               for v in ET.parse(route or args.route).getroot().iter("vehicle")}
    true = [seen.get(v, 3600.0 - d) for v, d in planned.items() if d < 3600]
    return (sum(done) / len(done), len(done), len(seen), sum(true) / len(true), len(true))


def fixed_time(route):
    out = f"{REPO}/configs/tripinfo_fixed_check.xml"
    res = subprocess.run(["sumo", "-n", os.environ["MARL_NET_FILE"], "-r", route, "--end", "3600",
                          "--tripinfo-output", out, "--tripinfo-output.write-unfinished",
                          "--no-step-log"], capture_output=True, text=True, check=True)
    s = wait_stats(out, route)
    os.remove(out)
    return s + (res.stderr.count("Teleporting"),)


def build(cond):
    training = dict(lr=3e-4, clip_param=0.2, vf_loss_coeff=0.5, entropy_coeff=0.01,
                    gamma=0.99, lambda_=0.95, train_batch_size=512, num_epochs=4)
    policy_cls = None
    if cond == "B1":
        from env.rllib_env import MARLTrafficEnv as Env
    elif cond == "B2":
        from env.rllib_env_b2 import MARLTrafficEnvB2 as Env
    else:
        from env.rllib_env_e1 import MARLTrafficEnvE1 as Env
        from env.comm_model import E1CommModel
        from env.diversity_policy import E1DiversityPPOPolicy
        ModelCatalog.register_custom_model("e1_comm_model_v3", E1CommModel)
        policy_cls = E1DiversityPPOPolicy
        training["model"] = {"custom_model": "e1_comm_model_v3"}
    cfg = (PPOConfig()
           .environment(Env, env_config={"route_file": args.route})
           .framework("torch")
           .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
           .multi_agent(policies={"shared_policy": (policy_cls, None, None, {})},
                        policy_mapping_fn=lambda aid, *a, **k: "shared_policy")
           .training(**training)
           .resources(num_gpus=1 if torch.cuda.is_available() else 0)
           .env_runners(num_env_runners=0))
    return cfg, Env


def main():
    ray.init(num_cpus=2, object_store_memory=500 * 1024**2,
             include_dashboard=False, ignore_reinit_error=True)
    cfg, Env = build(args.cond)
    algo = cfg.build()
    torch.set_num_threads(1)
    algo.restore(args.ckpt)
    print(f"Restored {args.cond} from {args.ckpt}", flush=True)

    env = Env({"route_file": args.route})
    obs, _ = env.reset()
    t0, steps, done, teleports = time.time(), 0, False, 0
    while not done:
        acts = {aid: algo.compute_single_action(o, policy_id="shared_policy", explore=args.explore)
                for aid, o in obs.items()}
        obs, _, term, trunc, _ = env.step(acts)
        teleports += traci.simulation.getStartingTeleportNumber()
        steps += 1
        done = term.get("__all__", False) or trunc.get("__all__", False)
    try:
        traci.close()
    except Exception:
        pass
    time.sleep(1)

    a_fin, a_n, a_in, a_true, gen = wait_stats(TRIPINFO)
    f_fin, f_n, f_in, f_true, _, f_tp = fixed_time(args.route)
    mode = "sampled (as trained)" if args.explore else "deterministic"
    print(f"\nRoute: {os.path.basename(args.route)} | {steps} steps | {time.time()-t0:.0f}s | policy {mode}")
    print(f"{'':12s} {'entered':>9s} {'finished':>9s} {'wait (finished)':>16s} {'TRUE delay (all)':>17s} {'teleports':>10s}")
    print(f"{'Fixed-time':12s} {f_in:5d}/{gen:<4d} {f_n:9d} {f_fin:15.2f}s {f_true:16.2f}s {f_tp:10d}")
    print(f"{args.cond:12s} {a_in:5d}/{gen:<4d} {a_n:9d} {a_fin:15.2f}s {a_true:16.2f}s {teleports:10d}")
    print(f"Change vs fixed-time (TRUE delay, every vehicle): {100*(a_true-f_true)/f_true:+.1f}%")
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()