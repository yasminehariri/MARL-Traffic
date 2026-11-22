"""
Record one simulation run for the 3D demo viewer (demo/index.html).

Saves, for every simulated second inside a time window: every vehicle's position, heading,
type and speed, every traffic light's phase, the queue at each light, network totals, and
(for E1) how strongly each light is "talking" (the size of the message it sends).

Also exports the road network once (--net-only) as demo/net_4x4.json.

Colab (one per runtime, after !git pull). Examples:
    !python scripts/record_demo_4x4.py --net-only
    !python scripts/record_demo_4x4.py --cond E1 --ckpt /content/drive/MyDrive/MARL_checkpoints/4x4/E1_local_mg5/checkpoints/ckpt_000977
    !python scripts/record_demo_4x4.py --cond fixed
    !python scripts/record_demo_4x4.py --cond E1 --ckpt <E1 checkpoint> --blind B1               # sensor failure
    !python scripts/record_demo_4x4.py --cond E1 --ckpt <E1 checkpoint> --blind B1 --failsafe    # ... with fallback
Defaults: peak scenario, evaluation seed 0, window 900-1500 s (the middle of the rush hour).
    !python scripts/record_demo_4x4.py --cond E1 --ckpt <E1 checkpoint> --ambulances             # ambulances, no priority
    !python scripts/record_demo_4x4.py --cond E1 --ckpt <E1 checkpoint> --ambulances --preempt   # ... with signal priority
--blind, --failsafe, --ambulances and --preempt work exactly as in evaluate_4x4.py (same code).
Output: /content/drive/MyDrive/MARL_results/demo/run_<label>.json  (a few MB)
"""

import argparse
import json
import math
import os
import sys
import time

REPO = "/content/MARL-Traffic"
TYPES = ["car", "van", "bus", "truck", "ambulance"]
TYPE_SIZE = {"car": (4.5, 1.8), "van": (6.5, 2.1), "bus": (12.0, 2.5), "truck": (10.0, 2.4), "ambulance": (6.5, 2.2)}

p = argparse.ArgumentParser()
p.add_argument("--net-only", action="store_true", help="only export the road network to --net-out")
p.add_argument("--cond", choices=["fixed", "tuned", "random", "B1", "B2", "E1"])
p.add_argument("--ckpt", default=None)
p.add_argument("--label", default=None)
p.add_argument("--scenario", default="peak")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--start", type=int, default=900, help="first recorded second")
p.add_argument("--duration", type=int, default=600, help="recorded seconds")
p.add_argument("--green", type=int, default=10, help="green seconds for --cond tuned")
p.add_argument("--min-green", type=int, default=None)
p.add_argument("--explore", action="store_true")
p.add_argument("--repo", default=REPO)
p.add_argument("--out", default=None)
p.add_argument("--net-out", default=None)
p.add_argument("--blind", default="",
               help="sensor failure, as in evaluate_4x4.py: 'B1' = all 4 queue detectors of light B1 read 0. "
                    "Trained conditions only")
p.add_argument("--failsafe", action="store_true",
               help="with --blind: the lights with failed detectors run the tuned fixed-time cycle "
                    "(--green s greens) instead of their agent, as in evaluate_4x4.py")
p.add_argument("--ambulances", action="store_true",
               help="add the 11 ambulances of evaluate_4x4.py --ambulances (same schedule for this scenario and seed)")
p.add_argument("--preempt", action="store_true", help="with --ambulances: signal preemption, as in evaluate_4x4.py")
args = p.parse_args()
if args.ckpt:
    args.ckpt = os.path.abspath(args.ckpt)   # RLlib needs an absolute path (relative ones fail to restore)
if args.ambulances:
    if args.cond not in ("B1", "B2", "E1", "tuned"):
        sys.exit("--ambulances: B1, B2, E1 or tuned (as in evaluate_4x4.py)")
    if args.blind:
        sys.exit("--ambulances is a separate experiment: do not combine it with --blind")
    print("Ambulances: 11 per episode; signal preemption " + ("ON" if args.preempt else "OFF"), flush=True)
elif args.preempt:
    sys.exit("--preempt needs --ambulances")
if args.blind:
    if args.cond not in ("B1", "B2", "E1"):
        sys.exit("--blind only makes sense for trained controllers (fixed, tuned and random use no sensors)")
    os.environ["MARL_BLIND"] = args.blind  # read by env/traci_env.py when it is imported below
    print(f"Sensor failure: queue detectors OFF at {args.blind}", flush=True)
else:
    os.environ.pop("MARL_BLIND", None)     # a normal recording must never inherit a failure from the shell
FAILED_LIGHTS = sorted({e.partition(":")[0] for e in args.blind.split(",") if e})
if args.failsafe:
    if not args.blind:
        sys.exit("--failsafe needs --blind (it decides what the lights with failed detectors do)")
    print(f"Fail-safe: {','.join(FAILED_LIGHTS)} run fixed-time ({args.green} s greens) instead of their agent",
          flush=True)
REPO = args.repo
NET = f"{REPO}/configs/mygrid4x4.net.xml"
os.environ["MARL_NET_FILE"] = NET
os.environ["MARL_SUMO_CFG"] = f"{REPO}/configs/simulation_4x4.sumocfg"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def export_net(path):
    import sumolib
    net = sumolib.net.readNet(NET, withPrograms=True)
    xmin, ymin, xmax, ymax = net.getBoundary()
    edges = []
    for e in net.getEdges():
        edges.append({"id": e.getID(), "from": e.getFromNode().getID(), "to": e.getToNode().getID(),
                      "lanes": [{"id": l.getID(), "width": l.getWidth(),
                                 "shape": [[round(x, 2), round(y, 2)] for x, y in l.getShape()]}
                                for l in e.getLanes()]})
    junctions = [{"id": n.getID(), "type": n.getType(), "x": n.getCoord()[0], "y": n.getCoord()[1],
                  "shape": [[round(x, 2), round(y, 2)] for x, y in n.getShape()]} for n in net.getNodes()]
    tls = []
    for t in net.getTrafficLights():
        node = net.getNode(t.getID())
        prog = list(t.getPrograms().values())[0]
        lanes = {}
        for inl, outl, idx in t.getConnections():
            lanes.setdefault(inl.getID(), []).append(idx)
        lane_info = []
        for lid, idxs in sorted(lanes.items()):
            sh = net.getLane(lid).getShape()
            (x0, y0), (x1, y1) = sh[-2], sh[-1]
            d = math.hypot(x1 - x0, y1 - y0) or 1.0
            lane_info.append({"lane": lid, "links": sorted(idxs), "stop": [round(x1, 2), round(y1, 2)],
                              "dir": [round((x1 - x0) / d, 4), round((y1 - y0) / d, 4)],
                              "from": net.getLane(lid).getEdge().getFromNode().getID()})
        tls.append({"id": t.getID(), "x": node.getCoord()[0], "y": node.getCoord()[1], "lanes": lane_info,
                    "phases": [ph.state for ph in prog.getPhases()]})
    ids = {t["id"] for t in tls}
    neighbors = {t["id"]: sorted({l["from"] for l in t["lanes"] if l["from"] in ids}) for t in tls}
    data = {"bounds": [xmin, ymin, xmax, ymax], "edges": edges, "junctions": junctions, "tls": tls,
            "neighbors": neighbors, "types": TYPES, "typeSize": [TYPE_SIZE[t] for t in TYPES]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Network saved to {path} ({os.path.getsize(path) / 1e3:.0f} kB)")


if args.net_only:
    export_net(args.net_out or "/content/drive/MyDrive/MARL_results/demo/net_4x4.json")
    sys.exit(0)
if not args.cond:
    sys.exit("--cond is required (or --net-only)")

from mgutil import min_green_for, set_min_green, no_comm_for  # noqa: E402

if args.cond in ("B1", "B2", "E1"):
    if not args.ckpt:
        sys.exit("--ckpt is required for trained agents")
    if args.cond == "E1" and no_comm_for(args.ckpt):
        sys.exit("This checkpoint is the E1 no-communication control: the demo shows E1's messages, which the "
                 "control does not use. Record E1 or B1 instead.")
    MG = min_green_for(args.ckpt, args.min_green)
elif args.cond == "fixed":
    MG = 0
else:
    MG = 5 if args.min_green is None else args.min_green
set_min_green(MG)
LABEL = args.label or (args.cond + ("_blind" if args.blind else "") + ("_failsafe" if args.failsafe else "")
                        + ("_amb" if args.ambulances else "") + ("_preempt" if args.preempt else "")
                        + ("_explore" if args.explore else ""))
OUT = args.out or f"/content/drive/MyDrive/MARL_results/demo/run_{LABEL}.json"
ROUTE = f"{REPO}/configs/routes_4x4_{args.scenario}_seed{args.seed}.rou.xml"
END = args.start + args.duration

import traci  # noqa: E402


def snapshot(t, tl_ids, tl_lanes, all_lanes):
    vs = []
    for vid in traci.vehicle.getIDList():
        x, y = traci.vehicle.getPosition(vid)
        vt = traci.vehicle.getTypeID(vid)
        vs.append([vid_index(vid), round(x * 10), round(y * 10), round(traci.vehicle.getAngle(vid)),
                   TYPES.index(vt) if vt in TYPES else 0, round(traci.vehicle.getSpeed(vid) * 10)])
    halt = {l: traci.lane.getLastStepHaltingNumber(l) for l in all_lanes}
    return {"t": t, "v": vs,
            "ph": [traci.trafficlight.getPhase(tl) for tl in tl_ids],
            "q": [sum(halt[l] for l in tl_lanes[tl]) for tl in tl_ids],
            "halted": sum(halt.values()), "arrived": arrived[0]}


_vid = {}


def vid_index(v):
    if v not in _vid:
        _vid[v] = len(_vid)
    return _vid[v]


arrived = [0]
frames = []
msg_frames = []
t0 = time.time()

if args.cond == "fixed":
    traci.start(["sumo", "-n", NET, "-r", ROUTE, "--end", str(END), "--no-step-log", "--no-warnings"])
    env = None
    model = None
else:
    import random
    if args.cond in ("tuned", "random"):
        from env.rllib_env import MARLTrafficEnv as Env
        rng = random.Random(0)
        period = args.green + 3
        choose = (lambda obs, t: {a: (t // period) % 4 for a in obs}) if args.cond == "tuned" else \
                 (lambda obs, t: {a: rng.randrange(4) for a in obs})
        model = None
    else:
        import ray
        import torch
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
        cfg = (PPOConfig().environment(Env, env_config={"route_file": None}).framework("torch")
               .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
               .multi_agent(policies={"shared_policy": (policy_cls, None, None, {})},
                            policy_mapping_fn=lambda aid, *a, **k: "shared_policy")
               .training(**training).resources(num_gpus=1 if torch.cuda.is_available() else 0)
               .env_runners(num_env_runners=0))
        algo = cfg.build()
        torch.set_num_threads(1)
        algo.restore(args.ckpt)
        if args.explore:
            torch.manual_seed(args.seed)
        model = algo.get_policy("shared_policy").model if args.cond == "E1" else None
        choose = lambda obs, t: {a: algo.compute_single_action(o, policy_id="shared_policy", explore=args.explore)
                                 for a, o in obs.items()}
        if args.failsafe:
            # identical to evaluate_4x4.py: the agents still choose for every light, then the failed
            # lights' choices are replaced by the tuned cycle
            period = args.green + 3  # green + 3 s yellow inserted by the env, as for --cond tuned
            agent_choose = choose
            choose = lambda obs, t: {aid: ((t // period) % 4 if aid in FAILED_LIGHTS else a)
                                     for aid, a in agent_choose(obs, t).items()}
    env = Env({"route_file": ROUTE})
    if args.blind:
        import env.traci_env as te
        assert te.FAILED and set(te.FAILED) == set(FAILED_LIGHTS), "sensor failure was not applied"
    obs, _ = env.reset()
    pre = None
    if args.ambulances:
        from env import ambulance
        from env.traci_env import LANE_IDS
        ambulance.add_to_simulation(ambulance.schedule(NET, "4x4", args.scenario, args.seed))
        pre = ambulance.Preemption(LANE_IDS)
        if args.preempt:
            choose = ambulance.wrap_choose(choose, pre, env._env)

tl_ids = sorted(traci.trafficlight.getIDList())
tl_lanes = {tl: sorted(set(traci.trafficlight.getControlledLanes(tl))) for tl in tl_ids}
all_lanes = [l for l in traci.lane.getIDList() if not l.startswith(":")]

t = 0
while t < END:
    if env is None:
        traci.simulationStep()
    else:
        obs, _, _, _, _ = env.step(choose(obs, t))
    arrived[0] += traci.simulation.getArrivedNumber()
    t += 1
    if t >= args.start:
        fr = snapshot(t, tl_ids, tl_lanes, all_lanes)
        if model is not None:
            import torch
            own = torch.tensor([list(obs[tl][:9]) for tl in tl_ids], dtype=torch.float32)
            with torch.no_grad():
                m = model.encoder(own.to(next(model.parameters()).device)).cpu()
            fr["msg"] = [round(float(x), 2) for x in m.norm(dim=1)]
        if args.preempt:   # lights that served an ambulance in this second (indices into tls)
            fr["pre"] = [tl_ids.index(tl) for tl in sorted(pre.serving)]
        frames.append(fr)
    if t % 300 == 0:
        print(f"  t={t}s  vehicles {traci.vehicle.getIDCount()}  ({time.time() - t0:.0f}s)", flush=True)
try:
    traci.close()
except Exception:
    pass

data = {"label": LABEL, "cond": args.cond, "scenario": args.scenario, "seed": args.seed, "min_green": MG,
        "explore": bool(args.explore), "start": args.start, "duration": args.duration, "tls": tl_ids,
        "blind": args.blind, "failsafe": bool(args.failsafe),
        "ambulances": bool(args.ambulances), "preempt": bool(args.preempt),
        "types": TYPES, "frames": frames}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(data, f, separators=(",", ":"))
mean_halted = sum(fr["halted"] for fr in frames) / len(frames)
print(f"Saved {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB): {len(frames)} s, {len(_vid)} vehicles, "
      f"mean halted {mean_halted:.0f}, arrived {arrived[0]}")
