"""
Train one condition (B1, B2 or E1) on the 4x4 (or 2x2) grid, with safe checkpoints.

Runs as its own process, so every run starts clean (no stale imports, no
leftover Ray or TraCI state). Saves to Drive as it goes and can resume from
the last checkpoint after a Colab disconnect.

Colab usage (after the setup cell: repo pulled, packages installed, Drive mounted):

    # 1) short smoke run first (~2 minutes)
    !python scripts/train_4x4.py --cond E1 --steps 2048 --out /content/drive/MyDrive/MARL_checkpoints/4x4_smoke

    # 2) the real run
    !python scripts/train_4x4.py --cond E1 --steps 500000

    # 3) after a disconnect: same command plus --resume
    !python scripts/train_4x4.py --cond E1 --steps 500000 --resume

Options added after the lane-order fix:
    --reward team|local|neighbourhood   (default team, the original design)
    --train-scenarios uniform|uniform,peak,asym   (default uniform)
    --grid 4x4|2x2        (default 4x4)

Outputs (default: /content/drive/MyDrive/MARL_checkpoints/<GRID>/<COND>_<REWARD>/):
    checkpoints/ckpt_000010, ckpt_000020, ...   model checkpoints (last 3 kept, plus every 100th)
    progress.csv                                one row per training iteration
    state.json                                  iteration, total env steps, last checkpoint
    run_info.json                               versions, settings, route files used
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

REPO = "/content/MARL-Traffic"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", required=True, choices=["B1", "B2", "E1"])
    p.add_argument("--steps", type=int, default=500_000, help="total simulation (env) steps")
    p.add_argument("--out", default=None, help="output folder (default: Drive/MARL_checkpoints/4x4/<cond>)")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in --out")
    p.add_argument("--ckpt-every", type=int, default=10, help="save a checkpoint every N iterations")
    p.add_argument("--seed", type=int, default=0, help="training seed (RLlib + route picking)")
    p.add_argument("--train-seeds", default="100,101,102,103,104", help="Uniform route seeds used for training")
    p.add_argument("--reward", default="team", choices=["team", "local", "neighbourhood"],
                   help="team: global/N (original); local: own-lane waiting; neighbourhood: own + 0.5 x neighbours")
    p.add_argument("--train-scenarios", default="uniform",
                   help="comma list of scenarios whose training seeds are used, e.g. uniform,peak,asym")
    p.add_argument("--grid", default="4x4", choices=["4x4", "2x2"])
    return p.parse_args()


args = parse_args()
OUT = args.out or f"/content/drive/MyDrive/MARL_checkpoints/{args.grid}/{args.cond}_{args.reward}"
CKPT_DIR = os.path.join(OUT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# --- Grid selection MUST happen before any env import ---
GRID = {"4x4": ("mygrid4x4.net.xml", "simulation_4x4.sumocfg", 16),
        "2x2": ("mygrid2x2.net.xml", "simulation.sumocfg", 4)}[args.grid]
os.environ["MARL_NET_FILE"] = f"{REPO}/configs/{GRID[0]}"
os.environ["MARL_SUMO_CFG"] = f"{REPO}/configs/{GRID[1]}"
os.environ["MARL_ROUTE_SEED"] = str(args.seed)
os.environ["MARL_REWARD"] = args.reward
sys.path.insert(0, REPO)

import ray  # noqa: E402
import torch  # noqa: E402
from ray.rllib.algorithms.ppo import PPOConfig  # noqa: E402
from ray.rllib.models import ModelCatalog  # noqa: E402

TRAIN_ROUTES = [f"{REPO}/configs/routes_{args.grid}_{sc}_seed{s}.rou.xml"
                for sc in args.train_scenarios.split(",")
                for s in args.train_seeds.split(",")]
missing = [r for r in TRAIN_ROUTES if not os.path.exists(r)]
if missing:
    sys.exit(f"Missing training route files: {missing}")

NUM_GPUS = 1 if torch.cuda.is_available() else 0


def build_config():
    from env.traci_env import NUM_AGENTS, SUMO_CFG, REWARD_MODE, LANE_IDS
    assert NUM_AGENTS == GRID[2], f"Expected {GRID[2]} agents on the {args.grid} grid, got {NUM_AGENTS}"
    assert SUMO_CFG.endswith(GRID[1]), f"Wrong SUMO config: {SUMO_CFG}"
    assert REWARD_MODE == args.reward, f"Reward mode mismatch: {REWARD_MODE} vs {args.reward}"
    # LANE_IDS is built in action order and validated at import; print one light as a visible check
    first = sorted(LANE_IDS)[0]
    print(f"[{args.cond}] grid {args.grid} | reward {REWARD_MODE} | {first} lanes in action order: {LANE_IDS[first]}", flush=True)

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
        # Register AFTER ray.init, otherwise a fresh Ray instance forgets it
        ModelCatalog.register_custom_model("e1_comm_model_v3", E1CommModel)
        policy_cls = E1DiversityPPOPolicy
        training["model"] = {"custom_model": "e1_comm_model_v3"}

    return (PPOConfig()
            .environment(Env, env_config={"route_file": TRAIN_ROUTES})
            .framework("torch")
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .multi_agent(policies={"shared_policy": (policy_cls, None, None, {})},
                         policy_mapping_fn=lambda aid, *a, **k: "shared_policy")
            .training(**training)
            .resources(num_gpus=NUM_GPUS)
            .env_runners(num_env_runners=0)
            .debugging(seed=args.seed))


def checkpoints():
    return sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*")))


def prune_checkpoints():
    """Keep the last 3 checkpoints, plus every 100th iteration, to save Drive space."""
    cks = checkpoints()
    for c in cks[:-3]:
        it = int(c.rsplit("_", 1)[1])
        if it % 100 != 0:
            shutil.rmtree(c, ignore_errors=True)


def load_state():
    path = os.path.join(OUT, "state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"iteration": 0, "total_steps": 0, "last_checkpoint": None}


def save_state(state):
    tmp = os.path.join(OUT, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, os.path.join(OUT, "state.json"))


def ep_return(r):
    v = r.get("env_runners", {}).get("episode_return_mean")
    if v is None:
        v = r.get("episode_reward_mean")
    return v


def main():
    state = load_state()
    if state["iteration"] > 0 and not args.resume:
        sys.exit(f"{OUT} already has a run at iteration {state['iteration']}. "
                 f"Add --resume to continue it, or choose a different --out.")

    # Ray startup that avoids the torch 2.11 / Ray 2.59 crash seen on Colab
    ray.init(num_cpus=2, object_store_memory=500 * 1024**2,
             include_dashboard=False, ignore_reinit_error=True)

    algo = build_config().build()
    torch.set_num_threads(1)

    if args.resume:
        if not state["last_checkpoint"] or not os.path.exists(state["last_checkpoint"]):
            cks = checkpoints()
            if not cks:
                sys.exit("--resume given but no checkpoint found in " + CKPT_DIR)
            state["last_checkpoint"] = cks[-1]
        algo.restore(state["last_checkpoint"])
        print(f"Resumed from {state['last_checkpoint']} "
              f"(iteration {state['iteration']}, {state['total_steps']} steps)", flush=True)

    with open(os.path.join(OUT, "run_info.json"), "w") as f:
        json.dump({"cond": args.cond, "grid": args.grid, "reward": args.reward,
                   "train_scenarios": args.train_scenarios,
                   "target_steps": args.steps, "seed": args.seed,
                   "train_routes": TRAIN_ROUTES, "gpus": NUM_GPUS,
                   "torch": torch.__version__, "ray": ray.__version__,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    csv_path = os.path.join(OUT, "progress.csv")
    new_csv = not os.path.exists(csv_path)
    t_start = time.time()

    with open(csv_path, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        if new_csv:
            writer.writerow(["iteration", "total_steps", "episode_return_mean", "iter_seconds", "wall_time"])

        while state["total_steps"] < args.steps:
            t0 = time.time()
            r = algo.train()
            steps_this_iter = r.get("num_env_steps_sampled_this_iter") or r.get("num_env_steps_sampled", 512)
            state["iteration"] += 1
            state["total_steps"] += int(steps_this_iter)
            ret = ep_return(r)
            writer.writerow([state["iteration"], state["total_steps"], ret,
                             round(time.time() - t0, 1), time.strftime("%H:%M:%S")])
            fcsv.flush()

            if state["iteration"] % 5 == 0:
                done = state["total_steps"] / args.steps
                elapsed = time.time() - t_start
                print(f"[{args.cond}] iter {state['iteration']:5d} | steps {state['total_steps']:8d} "
                      f"({done:5.1%}) | return {ret} | {elapsed/60:5.1f} min", flush=True)

            if state["iteration"] % args.ckpt_every == 0:
                path = os.path.join(CKPT_DIR, f"ckpt_{state['iteration']:06d}")
                algo.save(path)
                state["last_checkpoint"] = path
                save_state(state)
                prune_checkpoints()

    final = os.path.join(CKPT_DIR, f"ckpt_{state['iteration']:06d}")
    if not os.path.exists(final):
        algo.save(final)
    state["last_checkpoint"] = final
    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    algo.stop()
    ray.shutdown()

    ok = os.path.exists(final) and any(os.scandir(final))
    print(f"\n[{args.cond}] DONE: {state['total_steps']} steps, {state['iteration']} iterations, "
          f"{(time.time() - t_start)/3600:.2f} h")
    print(f"Final checkpoint: {final}  ({'saved OK' if ok else 'NOT FOUND, check Drive!'})")


if __name__ == "__main__":
    main()