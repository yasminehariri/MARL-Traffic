"""
Diagnose direction starvation on the 4x4 grid.

Runs a trained checkpoint (deterministic) and, separately, random actions, for
N steps on one route file, and measures directly in SUMO:
  - seconds of green each incoming lane received, per traffic light
  - which actions each agent chose, and which SUMO phase each action led to
  - whether the agent's lane list (LANE_IDS) covers every lane the light controls

Colab:
    !python scripts/diag_green_4x4.py --cond E1 \
        --ckpt /content/drive/MyDrive/MARL_checkpoints/4x4/E1/checkpoints/ckpt_000977 \
        --route /content/MARL-Traffic/configs/routes_4x4_uniform_seed100.rou.xml --steps 900
"""

import argparse
import os
import random
import sys
from collections import Counter, defaultdict

REPO = "/content/MARL-Traffic"
p = argparse.ArgumentParser()
p.add_argument("--cond", required=True, choices=["B1", "B2", "E1"])
p.add_argument("--ckpt", required=True)
p.add_argument("--route", required=True)
p.add_argument("--steps", type=int, default=900)
args = p.parse_args()

os.environ["MARL_NET_FILE"] = f"{REPO}/configs/mygrid4x4.net.xml"
os.environ["MARL_SUMO_CFG"] = f"{REPO}/configs/simulation_4x4.sumocfg"
sys.path.insert(0, REPO)

import ray  # noqa: E402
import torch  # noqa: E402
import traci  # noqa: E402
from ray.rllib.algorithms.ppo import PPOConfig  # noqa: E402
from ray.rllib.models import ModelCatalog  # noqa: E402

SHOW = ["A0", "B0", "A1", "B1"]  # bottom row (starved) + one interior row for contrast


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


def run(env, choose, label):
    obs, _ = env.reset()
    green = defaultdict(Counter)          # tl -> lane -> green seconds
    acts = defaultdict(Counter)           # tl -> action -> count
    act_phase = defaultdict(lambda: defaultdict(Counter))  # tl -> action -> phase seen next step
    lanes_of = {}
    for t in range(args.steps):
        a = choose(obs)
        obs, _, term, trunc, _ = env.step(a)
        for tl in a:
            acts[tl][int(a[tl])] += 1
            act_phase[tl][int(a[tl])][traci.trafficlight.getPhase(tl)] += 1
            if tl not in lanes_of:
                lanes_of[tl] = traci.trafficlight.getControlledLanes(tl)
            state = traci.trafficlight.getRedYellowGreenState(tl)
            for lane, s in zip(lanes_of[tl], state):
                if s in "Gg":
                    green[tl][lane] += 1
        if term.get("__all__") or trunc.get("__all__"):
            break
    try:
        traci.close()
    except Exception:
        pass

    print(f"\n================ {label} ({args.steps} steps) ================")
    for tl in SHOW:
        lanes = sorted(set(lanes_of[tl]))
        print(f"\n[{tl}] actions chosen: {dict(sorted(acts[tl].items()))}")
        print(f"[{tl}] action -> SUMO phase: "
              + "; ".join(f"a{k}->{dict(v)}" for k, v in sorted(act_phase[tl].items())))
        for lane in lanes:
            secs = green[tl][lane] // lanes_of[tl].count(lane)  # a lane can have several links
            print(f"    green on {lane:14s}: {secs:5d} s")
    starved = []
    for tl in lanes_of:
        for lane in set(lanes_of[tl]):
            if green[tl][lane] == 0:
                starved.append(f"{tl}:{lane}")
    print(f"\nLanes with ZERO green over the run ({len(starved)}): {starved}")


def main():
    from env.traci_env import LANE_IDS, TL_IDS
    print("Agents:", TL_IDS)
    for tl in SHOW:
        print(f"LANE_IDS[{tl}] (what the agent observes): {LANE_IDS[tl]}")

    ray.init(num_cpus=2, object_store_memory=500 * 1024**2,
             include_dashboard=False, ignore_reinit_error=True)
    cfg, Env = build(args.cond)
    algo = cfg.build()
    torch.set_num_threads(1)
    algo.restore(args.ckpt)
    env = Env({"route_file": args.route})

    # Coverage check: does the agent observe every lane its light controls?
    env.reset()
    print("\nCoverage of observed lanes vs lanes the light controls:")
    for tl in TL_IDS:
        controlled = set(traci.trafficlight.getControlledLanes(tl))
        observed = set(LANE_IDS[tl])
        missing = controlled - observed
        if missing or tl in SHOW:
            print(f"  {tl}: controls {sorted(controlled)} | observed-but-not-controlled "
                  f"{sorted(observed - controlled)} | MISSING {sorted(missing)}")
    try:
        traci.close()
    except Exception:
        pass

    trained = lambda obs: {aid: algo.compute_single_action(o, policy_id="shared_policy", explore=False)
                           for aid, o in obs.items()}
    rnd = random.Random(0)
    n_actions = env.action_space.n if hasattr(env, "action_space") and hasattr(env.action_space, "n") else 4
    randomly = lambda obs: {aid: rnd.randrange(n_actions) for aid in obs}

    run(env, trained, f"{args.cond} trained, deterministic")
    run(env, randomly, "Random actions (control test)")
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
