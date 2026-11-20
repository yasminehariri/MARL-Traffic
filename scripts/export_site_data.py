"""Export the project's results into small JSON files for the results website (site/data/).

The website must never contain hand-typed numbers: every number it shows comes from these files,
and these files come only from the evaluation CSVs and logs in results/. Rerun this script whenever
a result file changes; the format is documented in site/data/README.md.

    python scripts/export_site_data.py            # from the repo root
    python scripts/export_site_data.py --check    # also print a few key numbers to compare with the report

Needs pandas, numpy, scipy and scikit-learn (all available in Colab).
"""
import argparse, datetime, glob, hashlib, json, os, subprocess, sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(REPO, "results")
M = os.path.join(R, "mg5")
OUT = os.path.join(REPO, "site", "data")

# --------------------------------------------------------------------------- what each file is
# (file, experiment, variant, controller, mode, train_seed, grid, label filter or None)
# experiment: normal | surge | sensor_failure | transfer | grid_2x2
# variant:    standard | failure | failure_failsafe | failure_ablated
# mode:       greedy | sampled | none (controllers without a learned policy behave the same in both)
SOURCES = []


def add(path, experiment, variant, controller, mode, train_seed=0, grid="4x4", label=None, required=True):
    SOURCES.append(dict(path=path, experiment=experiment, variant=variant, controller=controller, mode=mode,
                        train_seed=train_seed, grid=grid, label=label, required=required))


for c in ("E1", "B1", "B2"):
    add(f"{M}/4x4_eval_mg5_{c}.csv", "normal", "standard", c, "greedy")
    add(f"{M}/4x4_eval_mg5_explore_{c}.csv", "normal", "standard", c, "sampled")
    add(f"{M}/4x4_surge_mg5_{c}.csv", "surge", "standard", c, "greedy")
    add(f"{M}/4x4_surge_mg5_explore_{c}.csv", "surge", "standard", c, "sampled")
    add(f"{M}/4x4_blind_mg5_{c}.csv", "sensor_failure", "failure", c, "greedy")
    add(f"{M}/4x4_blind_mg5_explore_{c}.csv", "sensor_failure", "failure", c, "sampled")
    add(f"{M}/4x4_blind_failsafe_mg5_{c}.csv", "sensor_failure", "failure_failsafe", c, "greedy")
for c in ("E1", "B1"):
    for s in (1, 2):
        add(f"{M}/4x4_eval_mg5_{c}_s{s}.csv", "normal", "standard", c, "greedy", train_seed=s)
        add(f"{M}/4x4_eval_mg5_explore_{c}_s{s}.csv", "normal", "standard", c, "sampled", train_seed=s)
add(f"{M}/4x4_blind_ablated_mg5_E1.csv", "sensor_failure", "failure_ablated", "E1", "greedy", required=False)
add(f"{M}/4x4_blind_ablated_mg5_explore_E1.csv", "sensor_failure", "failure_ablated", "E1", "sampled", required=False)
# fixed-time and tuned: identical under both rule versions (verified), kept in Hanin's version-1 file
add(f"{R}/4x4_eval_hanin.csv", "normal", "standard", "fixed", "none", label="fixed")
add(f"{R}/4x4_eval_hanin.csv", "normal", "standard", "tuned", "none", label="tuned")
add(f"{M}/4x4_eval_mg5_random.csv", "normal", "standard", "random", "none")
for c in ("fixed", "tuned", "random"):
    add(f"{M}/4x4_surge_mg5_{c}.csv", "surge", "standard", c, "none")
add(f"{M}/4x4_eval_mg5_E1_from2x2.csv", "transfer", "standard", "E1_from2x2", "greedy")
add(f"{M}/2x2_eval_mg5_E1.csv", "grid_2x2", "standard", "E1", "greedy", grid="2x2")
add(f"{M}/2x2_eval_mg5_fixed.csv", "grid_2x2", "standard", "fixed", "none", grid="2x2")

CONTROLLERS = {
    "E1": "E1: learned communication",
    "B1": "B1: no communication",
    "B2": "B2: hand-crafted communication",
    "fixed": "Fixed-time (SUMO default)",
    "tuned": "Tuned fixed-time (10 s greens)",
    "random": "Random actions",
    "E1_from2x2": "E1 trained on 2x2, run on 4x4 (no retraining)",
    "E1_ablated": "E1 with its messages replaced by their average",
}
COLS = ["generated", "entered", "finished", "true_delay", "wait_finished", "mean_travel_time", "teleports",
        "switches_per_light_h"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def load_episodes():
    rows, used, missing = [], [], []
    for s in SOURCES:
        if not os.path.exists(s["path"]):
            missing.append(dict(file=os.path.relpath(s["path"], REPO), required=s["required"]))
            continue
        d = pd.read_csv(s["path"])
        if s["label"]:
            d = d[d["label"] == s["label"]]
        if d.empty:
            sys.exit(f"{s['path']}: no rows for label {s['label']}")
        dup = d.duplicated(["scenario", "seed"]).sum()
        if dup:
            sys.exit(f"{s['path']}: {dup} duplicated (scenario, seed) rows")
        for _, r in d.iterrows():
            row = dict(experiment=s["experiment"], variant=s["variant"], controller=s["controller"], mode=s["mode"],
                       train_seed=s["train_seed"], grid=s["grid"], scenario=r["scenario"], seed=int(r["seed"]))
            for c in COLS:
                v = r.get(c, np.nan)
                row[c] = None if pd.isna(v) else (int(v) if c in ("generated", "entered", "finished", "teleports")
                                                  else round(float(v), 3))
            rows.append(row)
        used.append(dict(file=os.path.relpath(s["path"], REPO), sha256_12=sha(s["path"]),
                         experiment=s["experiment"], variant=s["variant"], controller=s["controller"],
                         mode=s["mode"], train_seed=s["train_seed"], episodes=int(len(d))))
    return pd.DataFrame(rows), used, missing


KEY = ["experiment", "variant", "grid", "controller", "mode", "train_seed", "scenario"]


def summarise(ep):
    out = []
    for k, g in ep.groupby(KEY, sort=False):
        x = g["true_delay"].to_numpy(float)
        n = len(x)
        sd = float(x.std(ddof=1)) if n > 1 else 0.0
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
        out.append(dict(zip(KEY, k), n=n, true_delay_mean=round(float(x.mean()), 3), true_delay_sd=round(sd, 2),
                        true_delay_ci95=[round(float(x.mean()) - half, 2), round(float(x.mean()) + half, 2)],
                        teleports_mean=round(float(g["teleports"].mean()), 2),
                        entered_pct=round(float(100 * g["entered"].sum() / g["generated"].sum()), 2),
                        travel_time_mean=round(float(g["mean_travel_time"].mean()), 2),
                        switches_per_light_h_mean=(None if g["switches_per_light_h"].isna().all()
                                                   else round(float(g["switches_per_light_h"].mean()), 1))))
    return out


def _pair(a, b, **keys):
    """Paired comparison of two {seed: true_delay} series (a = E1); None if fewer than 5 common seeds."""
    seeds = sorted(set(a.index) & set(b.index))
    if len(seeds) < 5:
        return None
    x, y = a.loc[seeds].to_numpy(float), b.loc[seeds].to_numpy(float)
    d = x - y
    w = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    t = stats.ttest_rel(x, y).pvalue if np.std(d) > 0 else 1.0
    return dict(**keys, n=len(seeds), e1_mean=round(float(x.mean()), 3), other_mean=round(float(y.mean()), 3),
                diff_pct=round(float(100 * (x.mean() - y.mean()) / y.mean()), 1),
                e1_better_seeds=int((x < y).sum()), wilcoxon_p=float(w), ttest_p=float(t))


def comparisons(ep):
    """E1 (training seed 0) against every other controller under the same conditions, paired by traffic seed.
    Plus, once the runs exist, E1 under sensor failure with its real messages against E1 with its messages
    replaced by their average (other = "E1_ablated", variant = "failure")."""
    out = []
    e1 = ep[(ep.controller == "E1") & (ep.train_seed == 0)]
    for (exp, var, grid, mode, sc), a in e1.groupby(["experiment", "variant", "grid", "mode", "scenario"]):
        if var == "failure_ablated":
            continue
        same = ep[(ep.experiment == exp) & (ep.variant == var) & (ep.grid == grid) & (ep.scenario == sc)
                  & (ep.train_seed == 0) & (ep.controller != "E1") & ep["mode"].isin([mode, "none"])]
        a = a.set_index("seed")["true_delay"]
        for other, b in same.groupby("controller"):
            r = _pair(a, b.set_index("seed")["true_delay"], experiment=exp, variant=var, grid=grid, mode=mode,
                      scenario=sc, other=other)
            if r:
                out.append(r)
        if var == "failure":
            abl = e1[(e1.experiment == exp) & (e1.variant == "failure_ablated") & (e1.grid == grid)
                     & (e1["mode"] == mode) & (e1.scenario == sc)]
            r = _pair(a, abl.set_index("seed")["true_delay"], experiment=exp, variant=var, grid=grid, mode=mode,
                      scenario=sc, other="E1_ablated") if len(abl) else None
            if r:
                out.append(r)
    return out


def training():
    runs = {"4x4_E1": ("E1", 0, 16), "4x4_E1_seed1": ("E1", 1, 16), "4x4_E1_seed2": ("E1", 2, 16),
            "4x4_B1": ("B1", 0, 16), "4x4_B1_seed1": ("B1", 1, 16), "4x4_B1_seed2": ("B1", 2, 16),
            "4x4_B2": ("B2", 0, 16), "2x2_E1": ("E1 (2x2 grid)", 0, 4)}
    out = []
    for name, (ctrl, ts, agents) in runs.items():
        p = f"{M}/progress_{name}.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        resumes = [int(v) for v in d.total_steps[d.iteration.diff() < 0]]
        d = d.drop_duplicates("iteration", keep="last").sort_values("iteration")
        v = d.dropna(subset=["episode_return_mean"])
        smooth = (v.episode_return_mean / agents).rolling(20, min_periods=1).mean()
        keep = v.iteration % 5 == 0
        out.append(dict(run=name, controller=ctrl, train_seed=ts, grid=name[:3], agents=agents,
                        iterations=int(d.iteration.max()), env_steps=int(d.total_steps.max()),
                        train_hours=round(float(d.iter_seconds.sum() / 3600), 2), resumed_at_steps=resumes,
                        missing_iterations=int(d.iteration.max() - len(d)),
                        steps=[int(s) for s in v.total_steps[keep]],
                        return_per_agent_smoothed=[round(float(s)) for s in smooth[keep]]))
    return out


def messages(n_points=2500):
    from sklearn.manifold import TSNE
    z = np.load(f"{M}/e1_messages_4x4_mg5/messages_4x4.npz")
    own, msg = z["own"], z["msg"]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(own), size=min(n_points, len(own)), replace=False))
    emb = TSNE(n_components=2, perplexity=40, init="pca", random_state=0).fit_transform(msg[idx])
    q = own[idx, 0:4] * 20.0                                   # halted vehicles per approach (obs scaled by 20)
    phase = own[idx, 4:8].argmax(1)
    pc = msg - msg.mean(0)
    _, _, vt = np.linalg.svd(pc[idx], full_matrices=False)
    pc1 = pc[idx] @ vt[0]
    pts = [dict(x=round(float(a), 2), y=round(float(b), 2), green=["N", "E", "S", "W"][int(p)],
                queue=[int(round(v)) for v in qq], queue_total=int(round(qq.sum())), msg_pc1=round(float(c), 3))
           for (a, b), p, qq, c in zip(emb, phase, q, pc1)]
    corr_lines = open(f"{M}/final/messages/e1_correlation.csv").read().splitlines()
    head = corr_lines[0].lstrip("# ")
    rows_lbl = head.split("|")[0].replace("rows:", "").strip().split(";")
    cols_lbl = head.split("|")[1].replace("cols:", "").strip().split(";")
    corr = [[round(float(v), 3) for v in line.split(",")] for line in corr_lines[1:] if line.strip()]
    ab = pd.read_csv(f"{M}/e1_messages_4x4_mg5/episodes.csv")
    piv = ab.pivot_table(index=["scenario", "seed"], columns="mode", values="true_delay")
    ablation = dict(episodes=[dict(scenario=sc, seed=int(s), normal=round(float(r["normal"]), 3),
                                   ablated=round(float(r["ablated"]), 3)) for (sc, s), r in piv.iterrows()],
                    mean_normal=round(float(piv["normal"].mean()), 2), mean_ablated=round(float(piv["ablated"].mean()), 2),
                    worse_count=int((piv["ablated"] > piv["normal"]).sum()), n=int(len(piv)),
                    wilcoxon_p=float(stats.wilcoxon(piv["ablated"], piv["normal"]).pvalue))
    rep = json.load(open(f"{M}/e1_messages_4x4_mg5/message_report.json"))
    keep = {k: rep[k] for k in rep if not isinstance(rep[k], (list, dict)) or k in ("decoding",)}
    return dict(tsne=dict(points=pts, n=len(pts), sampled_from=int(len(own)), perplexity=40, random_state=0),
                correlation=dict(rows=rows_lbl, cols=cols_lbl, r=corr), ablation_normal_traffic=ablation,
                report=keep)


def git_commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print key numbers to compare with the report")
    ap.add_argument("--no-tsne", action="store_true", help="skip the t-SNE (about 1 minute)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ep, used, missing = load_episodes()
    files = {
        "episodes.json": dict(columns=list(ep.columns), rows=ep.astype(object).where(pd.notna(ep), None).values.tolist()),
        "summary.json": summarise(ep),
        "comparisons.json": comparisons(ep),
        "training.json": training(),
    }
    if not args.no_tsne:
        files["messages.json"] = messages()
    files["meta.json"] = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), git_commit=git_commit(),
                              controllers=CONTROLLERS, sources=used, missing_files=missing,
                              units=dict(true_delay="seconds per vehicle (lower is better)",
                                         teleports="vehicles removed by SUMO after 300 s stuck (gridlock indicator)"))
    for name, obj in files.items():
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(obj, f, separators=(",", ":"), allow_nan=False)
        print(f"wrote site/data/{name:18s} {os.path.getsize(os.path.join(OUT, name)) / 1024:7.1f} KB")
    print(f"{len(ep)} episodes from {len(used)} files" + (f"; not yet available: {', '.join(m['file'] for m in missing)}" if missing else ""))
    if any(m["required"] for m in missing):
        sys.exit("ERROR: a required result file is missing (see above)")
    if args.check:
        s = pd.DataFrame(files["summary.json"])
        pick = s[(s.grid == "4x4") & (s.train_seed == 0) & (s.variant.isin(["standard", "failure", "failure_failsafe"]))]
        print(pick.pivot_table(index=["experiment", "variant", "controller", "mode"], columns="scenario",
                               values="true_delay_mean").round(1).to_string())


if __name__ == "__main__":
    main()
