"""
E1 message check on the 4x4 grid: are the learned messages diverse, do they carry
information about traffic, and does the policy actually use them?

Three questions, three tests:
  1. COLLAPSE?     Mean pairwise cosine similarity of all sent messages (the same metric
                   used on the 2x2: 0.9991 collapsed before the diversity fix, 0.2773 after),
                   plus same-moment similarity across the 16 agents, dead dimensions and
                   effective dimensionality (PCA participation ratio, max 8).
  2. MEANINGFUL?   Linear decoding: can the sender's queues / current phase be read back
                   from its 8-number message? (R^2 / accuracy on held-out episodes)
  3. USED?         Ablation: rerun the same episodes with every message replaced by the
                   average message (no information, same scale). If delay gets worse, the
                   policy relies on the messages.

A message is what an agent SENDS: encoder(own 9-dim observation). Its neighbours receive
the masked mean of the messages of their real neighbours.

Colab (fresh runtime, one per check; ~50 min with the defaults = 9 normal + 9 ablated episodes):
    !python scripts/check_messages_4x4.py --ckpt /content/drive/MyDrive/MARL_checkpoints/4x4/E1_local/checkpoints/ckpt_000977

Outputs (default /content/drive/MyDrive/MARL_results/e1_messages_4x4/):
    message_report.json   all numbers below
    episodes.csv          true delay etc. for every normal and ablated episode
    messages_4x4.npz      sampled own observations and messages (for later plots)
    fig_e1_messages.png   PCA of messages coloured by sender queue + cosine histogram
"""

import argparse
import csv
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

REPO = "/content/MARL-Traffic"
END = 3600
OBS_DIM = 9


# ----------------------------------------------------------------------------- metrics
def trip_metrics(tripinfo, route):
    """Same definitions as evaluate_4x4.py, so normal episodes must match the eval CSV."""
    infos = ET.parse(tripinfo).getroot().findall("tripinfo")
    planned = {v.get("id"): float(v.get("depart")) for v in ET.parse(route).getroot().iter("vehicle")}
    planned = {k: d for k, d in planned.items() if d < END}
    seen = {t.get("id"): t for t in infos}
    fin = [t for t in infos if float(t.get("arrival", -1)) >= 0]
    true = [END - dep if seen.get(v) is None else
            float(seen[v].get("waitingTime", 0)) + float(seen[v].get("departDelay", 0))
            for v, dep in planned.items()]
    return dict(generated=len(planned), entered=len(seen), finished=len(fin),
                true_delay=round(sum(true) / len(true), 3))


# ----------------------------------------------------------------------------- analysis (numpy only)
def cosine_matrix(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    u = m / np.clip(n, 1e-9, None)
    return u @ u.T


def mean_offdiag(s):
    k = s.shape[0]
    return float((s.sum() - np.trace(s)) / (k * (k - 1)))


def lstsq_fit(Xtr, Ytr, Xte):
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    W, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
    return np.hstack([Xte, np.ones((len(Xte), 1))]) @ W


def r2(y, yhat):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else float("nan")


def analyze(own, msg, ep, step_key, rng_seed=0):
    """
    own: (N, 9) sender observations, msg: (N, 8) messages,
    ep: (N,) episode index, step_key: (N,) id of (episode, sampled step) for same-moment groups.
    """
    rng = np.random.default_rng(rng_seed)
    out = {"n_messages": int(len(msg))}

    # 1. collapse
    idx = rng.choice(len(msg), size=min(3000, len(msg)), replace=False)
    cm = cosine_matrix(msg[idx])
    out["mean_pairwise_cosine"] = mean_offdiag(cm)
    # mean |cosine|: messages pointing in opposite directions (+v and -v) give a mean cosine
    # near 0 but are still one direction; |cosine| near 1 exposes that
    out["mean_abs_pairwise_cosine"] = mean_offdiag(np.abs(cm))
    same = [mean_offdiag(cosine_matrix(msg[step_key == k])) for k in np.unique(step_key)
            if (step_key == k).sum() > 1]
    out["same_moment_cosine_across_agents"] = float(np.mean(same))
    sd = msg.std(axis=0)
    out["per_dim_std"] = [round(float(v), 5) for v in sd]
    out["dead_dims"] = int((sd < 1e-3 * max(sd.max(), 1e-12)).sum())
    ev = np.clip(np.linalg.eigvalsh(np.cov(msg.T)), 0, None)[::-1]
    out["pca_var_explained"] = [round(float(v), 4) for v in ev / ev.sum()]
    out["participation_ratio"] = float(ev.sum() ** 2 / (ev ** 2).sum())
    out["mean_message_norm"] = float(np.linalg.norm(msg, axis=1).mean())
    # how much messages vary relative to their size (collapsed ~0; PCA alone can't see that,
    # since tiny isotropic noise also looks "8-dimensional")
    out["spread_ratio"] = float(np.sqrt(ev.sum()) / max(out["mean_message_norm"], 1e-12))

    # 2. meaningful: decode the sender's state from its message, held-out episodes
    eps = np.unique(ep)
    test_eps = eps[len(eps) * 2 // 3:] if len(eps) >= 3 else eps[-1:]
    te = np.isin(ep, test_eps)
    tr = ~te
    if tr.sum() == 0:  # single episode: split in time instead
        cut = int(0.7 * len(msg)); tr = np.arange(len(msg)) < cut; te = ~tr
    q = own[:, :4]
    qsum = q.sum(1)
    out["decode_queue_total_r2"] = r2(qsum[te], lstsq_fit(msg[tr], qsum[tr], msg[te]))
    out["decode_each_queue_r2"] = [r2(q[te, k], lstsq_fit(msg[tr], q[tr, k], msg[te])) for k in range(4)]
    phase = own[:, 4:8].argmax(1)
    onehot = np.eye(4)[phase]
    pred = lstsq_fit(msg[tr], onehot[tr], msg[te]).argmax(1)
    out["decode_phase_accuracy"] = float((pred == phase[te]).mean())
    out["phase_majority_baseline"] = float(np.bincount(phase[te], minlength=4).max() / te.sum())
    out["decode_elapsed_r2"] = r2(own[te, 8], lstsq_fit(msg[tr], own[tr, 8], msg[te]))
    out["test_episodes"] = [int(e) for e in test_eps]
    return out


def verdicts(rep):
    v = []
    c = rep["mean_pairwise_cosine"]
    ac, pr = rep.get("mean_abs_pairwise_cosine", float("nan")), rep["participation_ratio"]
    if c > 0.9:
        state = "COLLAPSED"
    elif pr < 1.5 or ac > 0.9:
        state = "LOW-DIMENSIONAL"  # one direction (possibly +/-), not a rich 8-dim code
    elif c > 0.5:
        state = "partly similar"
    else:
        state = "diverse"
    v.append(("Collapse", state,
              f"mean cosine {c:.4f}, mean |cosine| {ac:.4f}, effective dims {pr:.2f} of 8 "
              f"(2x2: 0.9991 collapsed, 0.2773 after fix); spread/size {rep['spread_ratio']:.3f}"))
    r = rep["decode_queue_total_r2"]
    v.append(("Informative", "strong" if r >= 0.5 else "partial" if r >= 0.2 else "weak",
              f"queue-total R^2 {r:.3f}; phase accuracy {rep['decode_phase_accuracy']:.3f} "
              f"vs majority {rep['phase_majority_baseline']:.3f}"))
    if "ablation" in rep:
        a = rep["ablation"]
        used = "USED" if a["mean_pct_change"] > 5 and a["episodes_worse"] >= 0.75 * a["n"] else \
               "weakly used" if a["mean_pct_change"] > 1 else "NOT used"
        v.append(("Used by policy", used,
                  f"true delay {a['mean_normal']:.1f}s -> {a['mean_ablated']:.1f}s "
                  f"({a['mean_pct_change']:+.1f}%), worse in {a['episodes_worse']}/{a['n']} episodes, "
                  f"paired p={a['p_paired_t']:.4g}"))
    return v


# ----------------------------------------------------------------------------- simulation
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scenarios", default="uniform,peak,asym")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--every", type=int, default=5, help="record messages every N seconds")
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--eval-csv", default="/content/drive/MyDrive/MARL_results/4x4_eval.csv",
                   help="E1 rows here are used to confirm normal episodes reproduce the evaluation")
    p.add_argument("--out", default="/content/drive/MyDrive/MARL_results/e1_messages_4x4")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scen = args.scenarios.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    os.environ["MARL_NET_FILE"] = f"{REPO}/configs/mygrid4x4.net.xml"
    os.environ["MARL_SUMO_CFG"] = f"{REPO}/configs/simulation_4x4.sumocfg"
    sys.path.insert(0, REPO)
    tripinfo = f"{REPO}/configs/tripinfo.xml"

    import ray
    import torch
    import traci
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog

    ray.init(num_cpus=2, object_store_memory=500 * 1024**2, include_dashboard=False, ignore_reinit_error=True)
    from env.rllib_env_e1 import MARLTrafficEnvE1 as Env
    from env.comm_model import E1CommModel
    from env.diversity_policy import E1DiversityPPOPolicy
    ModelCatalog.register_custom_model("e1_comm_model_v3", E1CommModel)
    cfg = (PPOConfig()
           .environment(Env, env_config={"route_file": None})
           .framework("torch")
           .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
           .multi_agent(policies={"shared_policy": (E1DiversityPPOPolicy, None, None, {})},
                        policy_mapping_fn=lambda aid, *a, **k: "shared_policy")
           .training(lr=3e-4, clip_param=0.2, vf_loss_coeff=0.5, entropy_coeff=0.01, gamma=0.99,
                     lambda_=0.95, train_batch_size=512, num_epochs=4, model={"custom_model": "e1_comm_model_v3"})
           .resources(num_gpus=1 if torch.cuda.is_available() else 0)
           .env_runners(num_env_runners=0))
    algo = cfg.build()
    torch.set_num_threads(1)
    algo.restore(args.ckpt)
    model = algo.get_policy("shared_policy").model
    dev = next(model.parameters()).device
    print(f"Restored E1 from {args.ckpt}", flush=True)

    class ConstMsg(torch.nn.Module):
        """Replaces the encoder: every agent 'sends' the same average message."""
        def __init__(self, c):
            super().__init__()
            self.register_buffer("c", c)

        def forward(self, x):
            return self.c.to(x.device).unsqueeze(0).expand(x.shape[0], -1)

    def episode(sc, s, collect):
        route = f"{REPO}/configs/routes_4x4_{sc}_seed{s}.rou.xml"
        env = Env({"route_file": route})
        obs, _ = env.reset()
        agents = sorted(obs)
        own, full, t, tele, done = [], [], 0, 0, False
        while not done:
            if collect and t % args.every == 0:
                own.append(np.stack([obs[a][:OBS_DIM] for a in agents]))
                if t == 600:
                    full.append(np.stack([obs[a] for a in agents]))
            act = {a: algo.compute_single_action(o, policy_id="shared_policy", explore=False) for a, o in obs.items()}
            obs, _, term, trunc, _ = env.step(act)
            tele += traci.simulation.getStartingTeleportNumber()
            t += 1
            done = term.get("__all__", False) or trunc.get("__all__", False)
        try:
            traci.close()
        except Exception:
            pass
        time.sleep(1)
        m = trip_metrics(tripinfo, route)
        m["teleports"] = tele
        return m, (np.stack(own) if collect else None), (full[0] if full else None), agents

    expected = {}
    if os.path.exists(args.eval_csv):
        with open(args.eval_csv) as f:
            for r in csv.DictReader(f):
                if r["label"] == "E1":
                    expected[(r["scenario"], int(r["seed"]))] = float(r["true_delay"])

    rows, own_all, ep_all, key_all, agent_all, full_states = [], [], [], [], [], []
    ep_id = 0
    for sc in scen:
        for s in seeds:
            t0 = time.time()
            m, own, full, agents = episode(sc, s, collect=True)
            exp = expected.get((sc, s))
            match = "" if exp is None else ("  [matches eval CSV]" if abs(exp - m["true_delay"]) < 0.01
                                            else f"  [eval CSV says {exp:.3f}: MISMATCH, check before trusting]")
            print(f"normal  {sc:8s} seed{s}: true delay {m['true_delay']:7.2f}s | teleports {m['teleports']} | "
                  f"{time.time() - t0:.0f}s{match}", flush=True)
            rows.append(dict(mode="normal", scenario=sc, seed=s, **m))
            T = own.shape[0]
            own_all.append(own.reshape(-1, OBS_DIM))
            ep_all.append(np.full(T * len(agents), ep_id))
            key_all.append(ep_id * 100000 + np.repeat(np.arange(T), len(agents)))
            agent_all.append(np.tile(np.arange(len(agents)), T))
            if full is not None:
                full_states.append(full)
            ep_id += 1

    own_all = np.concatenate(own_all).astype(np.float32)
    with torch.no_grad():
        msg_all = model.encoder(torch.from_numpy(own_all).to(dev)).cpu().numpy()
    ep_all, key_all, agent_all = map(np.concatenate, (ep_all, key_all, agent_all))
    rep = analyze(own_all, msg_all, ep_all, key_all)
    rep.update(ckpt=args.ckpt, scenarios=scen, seeds=seeds, every=args.every, agents=agents)
    np.savez_compressed(os.path.join(args.out, "messages_4x4.npz"), own=own_all, msg=msg_all,
                        episode=ep_all, step_key=key_all, agent=agent_all, agents=np.array(agents))

    if not args.no_ablation:
        mean_msg = torch.from_numpy(msg_all.mean(0)).float()
        original = model.encoder
        model.encoder = ConstMsg(mean_msg).to(dev)
        # confirm the swap changes the policy's output on real states (if 0, the swap did nothing)
        if full_states:
            x = torch.from_numpy(np.concatenate(full_states)).float().to(dev)
            with torch.no_grad():
                abl_logits, _ = model.forward({"obs": x}, [], None)
                model.encoder = original
                norm_logits, _ = model.forward({"obs": x}, [], None)
                model.encoder = ConstMsg(mean_msg).to(dev)
            rep["ablation_logit_change"] = float((abl_logits - norm_logits).abs().max())
            rep["ablation_action_change_pct"] = float(100 * (abl_logits.argmax(1) != norm_logits.argmax(1)).float().mean())
            print(f"Ablation check on t=600 states: max logit change {rep['ablation_logit_change']:.4f}, "
                  f"greedy action changes for {rep['ablation_action_change_pct']:.1f}% of agents", flush=True)
        for sc in scen:
            for s in seeds:
                t0 = time.time()
                m, *_ = episode(sc, s, collect=False)
                print(f"ablated {sc:8s} seed{s}: true delay {m['true_delay']:7.2f}s | teleports {m['teleports']} | "
                      f"{time.time() - t0:.0f}s", flush=True)
                rows.append(dict(mode="ablated", scenario=sc, seed=s, **m))
        model.encoder = original

        from scipy import stats
        nrm = {(r["scenario"], r["seed"]): r for r in rows if r["mode"] == "normal"}
        abl = {(r["scenario"], r["seed"]): r for r in rows if r["mode"] == "ablated"}
        keys = sorted(nrm)
        a = np.array([nrm[k]["true_delay"] for k in keys])
        b = np.array([abl[k]["true_delay"] for k in keys])
        rep["ablation"] = dict(n=len(keys), mean_normal=float(a.mean()), mean_ablated=float(b.mean()),
                               mean_pct_change=float(100 * (b - a).mean() / a.mean()),
                               episodes_worse=int((b > a).sum()),
                               p_paired_t=float(stats.ttest_rel(b, a).pvalue) if len(keys) > 1 else float("nan"),
                               per_scenario={sc: dict(normal=float(np.mean([nrm[k]["true_delay"] for k in keys if k[0] == sc])),
                                                      ablated=float(np.mean([abl[k]["true_delay"] for k in keys if k[0] == sc])),
                                                      ablated_teleports=float(np.mean([abl[k]["teleports"] for k in keys if k[0] == sc])))
                                             for sc in scen})

    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "message_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    # figure: PCA of messages coloured by sender queue, and the cosine distribution
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    i = rng.choice(len(msg_all), size=min(6000, len(msg_all)), replace=False)
    c = msg_all - msg_all.mean(0)
    _, _, Vt = np.linalg.svd(c[i], full_matrices=False)
    pc = c[i] @ Vt[:2].T
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sc_ = ax[0].scatter(pc[:, 0], pc[:, 1], c=own_all[i, :4].sum(1) * 20, cmap="Blues", s=6, vmin=-5)
    fig.colorbar(sc_, ax=ax[0], label="Sender's halted vehicles (4 lanes)")
    ax[0].set_xlabel("Message PC1"); ax[0].set_ylabel("Message PC2")
    ax[0].set_title("E1 messages (PCA), coloured by sender queue", loc="left", fontsize=10)
    j = rng.choice(len(msg_all), size=min(1500, len(msg_all)), replace=False)
    cm = cosine_matrix(msg_all[j])
    ax[1].hist(cm[~np.eye(len(j), dtype=bool)], bins=60, color="#2a78d6")
    ax[1].axvline(rep["mean_pairwise_cosine"], color="#0b0b0b", linewidth=1)
    ax[1].set_xlabel("Pairwise cosine similarity between messages"); ax[1].set_ylabel("Pairs")
    ax[1].set_title(f"Mean {rep['mean_pairwise_cosine']:.3f} (collapse would sit near 1.0)", loc="left", fontsize=10)
    for a_ in ax:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_e1_messages.png"), dpi=200)

    print("\n================ E1 MESSAGE CHECK (4x4) ================")
    print(f"messages analysed: {rep['n_messages']}  (every {args.every}s, {len(agents)} agents, {ep_id} episodes)")
    print(f"mean pairwise cosine:            {rep['mean_pairwise_cosine']:.4f}")
    print(f"mean |pairwise cosine|:          {rep['mean_abs_pairwise_cosine']:.4f}   (near 1 = one direction, even if mean cosine is ~0)")
    print(f"same-moment cosine (16 agents):  {rep['same_moment_cosine_across_agents']:.4f}")
    print(f"spread / size of messages:       {rep['spread_ratio']:.3f}   (collapsed would be ~0)")
    print(f"effective dimensions (of 8):     {rep['participation_ratio']:.2f}   dead dims: {rep['dead_dims']}")
    print(f"PCA variance explained:          {rep['pca_var_explained']}")
    print(f"decode queue total R^2:          {rep['decode_queue_total_r2']:.3f}")
    print(f"decode each queue R^2:           {[round(x, 3) for x in rep['decode_each_queue_r2']]}")
    print(f"decode phase accuracy:           {rep['decode_phase_accuracy']:.3f} (majority {rep['phase_majority_baseline']:.3f})")
    print(f"decode elapsed-time R^2:         {rep['decode_elapsed_r2']:.3f}")
    if "ablation" in rep:
        for sc, d in rep["ablation"]["per_scenario"].items():
            print(f"ablation {sc:8s}: {d['normal']:7.2f}s -> {d['ablated']:7.2f}s  (ablated teleports {d['ablated_teleports']:.1f})")
    print("--------------------------------------------------------")
    for name, verdict, why in verdicts(rep):
        print(f"{name:15s} {verdict:15s} {why}")
    print(f"\nSaved to {args.out}")
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
