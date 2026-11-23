"""
Final analysis of the three checks added after the main results (version 2, 5 s minimum green, 4x4 grid):

  sensor_failure/   Light B1's 4 queue detectors fail (read 0) for the whole episode (evaluate_4x4.py --blind B1).
                    E1 against B1, B2, E1 with its messages replaced by their average (ablation) and the
                    no-communication control; greedy and sampled. Plus the usual fallback for a flagged
                    detector fault (the failed light runs tuned fixed-time, --failsafe), greedy.
  training_seeds/   Is E1's advantage over B1 one lucky training run? E1 and B1 were each trained 3 times
                    (training seeds 0, 1, 2). Each E1 run is compared with the B1 run of the same training seed
                    on normal traffic (greedy, sampled), sensor failure (greedy) and ambulances (priority off and on).
  control/          E1 against the no-communication control (train_4x4.py --no-comm: E1's network, loss and
                    training settings, neighbour messages replaced by zeros), and the control against B1, on
                    normal traffic and sensor failure, greedy and sampled.

Every controller was run on the same route files, so episodes are paired by (scenario, evaluation seed),
10 seeds per scenario. Tests per scenario: exact Wilcoxon signed-rank and paired t-test; Holm correction
within each family (one family = one test condition, all its comparisons and scenarios; listed in each
tests CSV as `family`). With 10 seeds the smallest possible Wilcoxon p is 2/1024 = 0.00195 before correction.
A pooled row over all 30 episodes ("all") is given for reference and is not part of any Holm family.

    python scripts/analyze_extensions.py                 # from the repo root; writes results/mg5/final/...

Every input file is checked before use: 30 episodes, no duplicates, 3 scenarios x seeds 0-9, the right action
mode, the right kind of run (sensor failure, ablation, ambulances, priority, control) and the right checkpoint.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--results", default=os.path.join(REPO, "results", "mg5"))
p.add_argument("--out", default=os.path.join(REPO, "results", "mg5", "final"))
args = p.parse_args()
M = args.results

SCEN = ["uniform", "peak", "asym"]
NAMES = {"E1": "E1 learned comm", "B1": "B1 no comm", "B2": "B2 hand-crafted comm",
         "E1_ablated": "E1, messages replaced by their average", "E1_nocomm": "E1 control, no communication"}
COLORS = {"E1": "#1f77b4", "B1": "#ff7f0e", "B2": "#2ca02c", "E1_ablated": "#9467bd", "E1_nocomm": "#7f7f7f"}
RUN_DIR = {("E1", 0): "E1_local_mg5", ("E1", 1): "E1_local_mg5_seed1", ("E1", 2): "E1_local_mg5_seed2",
           ("B1", 0): "B1_local_mg5", ("B1", 1): "B1_local_mg5_seed1", ("B1", 2): "B1_local_mg5_seed2",
           ("B2", 0): "B2_local_mg5", ("E1_ablated", 0): "E1_local_mg5", ("E1_nocomm", 0): "E1nocomm_local_mg5"}


# ----------------------------------------------------------------------------------------------- loading
def path(test, ctrl, mode, ts=0):
    """Result file of one controller, test, action mode and training seed (None if that run does not exist)."""
    sfx = f"_s{ts}" if ts else ""
    ex = "explore_" if mode == "sampled" else ""
    c = {"E1_nocomm": "E1nocomm", "E1_ablated": "E1"}.get(ctrl, ctrl)
    if test == "normal":
        return f"{M}/4x4_eval_mg5_{ex}{c}{sfx}.csv"
    if test == "failure":
        return f"{M}/4x4_blind_{'ablated_' if ctrl == 'E1_ablated' else ''}mg5_{ex}{c}{sfx}.csv"
    if test == "failsafe":
        return f"{M}/4x4_blind_failsafe_mg5_{c}{sfx}.csv"
    if test in ("amb_off", "amb_on"):
        return f"{M}/4x4_{'amb_preempt' if test == 'amb_on' else 'amb'}_mg5_{c}{sfx}.csv" if mode == "greedy" else None
    raise ValueError(test)


def load(test, ctrl, mode, ts=0, required=True):
    f = path(test, ctrl, mode, ts)
    if f is None or not os.path.exists(f):
        if required:
            raise SystemExit(f"missing result file: {f}")
        return None
    d = pd.read_csv(f)
    name = os.path.relpath(f, REPO)
    col = lambda c, default=0: d[c] if c in d else pd.Series(default, index=d.index)
    assert len(d) == 30 and not d.duplicated(["scenario", "seed"]).any(), f"{name}: {len(d)} rows or duplicates"
    assert sorted(d.scenario.unique()) == sorted(SCEN) and (d.groupby("scenario").seed.apply(sorted)
                                                           .apply(lambda s: s == list(range(10))).all()), name
    assert (d.explore == int(mode == "sampled")).all(), f"{name}: wrong action mode"
    assert (d.min_green == 5).all() and (col("grid", "4x4") == "4x4").all(), f"{name}: not the version-2 4x4 rules"
    blind = col("blind", "").fillna("").astype(str)
    assert ((blind == "B1") if test in ("failure", "failsafe") else (blind == "")).all(), f"{name}: wrong --blind"
    assert (col("failsafe") == int(test == "failsafe")).all(), f"{name}: wrong --failsafe"
    assert (col("ablate_messages") == int(ctrl == "E1_ablated")).all(), f"{name}: wrong --ablate-messages"
    assert (col("ambulances") == int(test.startswith("amb"))).all(), f"{name}: wrong --ambulances"
    assert (col("preempt") == int(test == "amb_on")).all(), f"{name}: wrong --preempt"
    assert (col("no_comm") == int(ctrl == "E1_nocomm")).all(), f"{name}: wrong model (control or not)"
    run = d.ckpt.str.extract(r"([^/]+)/checkpoints/ckpt_\d+/?$")[0]          # the run folder of the checkpoint
    want = RUN_DIR[(ctrl, ts)]
    assert (run == want).all(), f"{name}: checkpoint {set(run)} is not {want}"
    return d.set_index(["scenario", "seed"]).sort_index()


# ----------------------------------------------------------------------------------------------- statistics
def holm(ps):
    ps = np.asarray(ps, float)
    adj, run = np.empty_like(ps), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def paired(a, b, metric="true_delay"):
    """a and b: same index (scenario, seed). Returns means, % difference of a vs b, seed counts and p-values."""
    assert a.index.equals(b.index), "unpaired episodes"
    x, y = a[metric].to_numpy(float), b[metric].to_numpy(float)
    dlt = x - y
    return dict(n=len(x), mean_a=x.mean(), mean_b=y.mean(),
                diff_pct=100 * (x.mean() - y.mean()) / y.mean() if y.mean() else np.nan,
                a_lower=int((x < y).sum()), a_higher=int((x > y).sum()),
                wilcoxon_p=stats.wilcoxon(x, y).pvalue if np.any(dlt != 0) else 1.0,
                ttest_p=stats.ttest_rel(x, y).pvalue if np.std(dlt) > 0 else 1.0)


def compare(rows, family, a_name, a, b_name, b, metric="true_delay", **extra):
    for sc in SCEN + ["all"]:
        r = paired(a if sc == "all" else a.loc[[sc]], b if sc == "all" else b.loc[[sc]], metric)
        rows.append(dict(family=family, **extra, a=a_name, b=b_name, scenario=sc, metric=metric, **r))


def finish(rows):
    t = pd.DataFrame(rows)
    t["wilcoxon_p_holm"] = np.nan
    for fam in t.family.unique():
        k = (t.family == fam) & (t.scenario != "all")
        t.loc[k, "wilcoxon_p_holm"] = holm(t.loc[k, "wilcoxon_p"])
    return t


def summary(rows, test, mode, ctrl, ts, d):
    for sc in SCEN + ["all"]:
        g = d if sc == "all" else d.loc[[sc]]
        r = dict(test=test, mode=mode, controller=ctrl, train_seed=ts, scenario=sc, episodes=len(g),
                 true_delay=g.true_delay.mean(), true_delay_sd=g.true_delay.std(ddof=1),
                 entered_pct=100 * g.entered.sum() / g.generated.sum(), teleports=g.teleports.mean(),
                 gridlock_episodes=int((g.teleports > 0).sum()))
        if "amb_travel_time" in g:
            r.update(amb_travel_time=g.amb_travel_time.mean(), amb_arrived=int(g.amb_arrived.sum()),
                     ambulances=int(g.amb_n.sum()))
        rows.append(r)


# ----------------------------------------------------------------------------------------------- Markdown
def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def fmt_p(v):
    return "n/a" if pd.isna(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}")


def tests_md(t, a_label="a", b_label="b"):
    v = t[t.scenario != "all"].copy()
    out = pd.DataFrame({
        "Condition": v.family, a_label: v.a, b_label: v.b, "Scenario": v.scenario,
        f"{a_label} mean": v.mean_a.map(lambda x: f"{x:.1f}"), f"{b_label} mean": v.mean_b.map(lambda x: f"{x:.1f}"),
        "Difference": v.diff_pct.map(lambda x: f"{x:+.1f}%"),
        f"Seeds {a_label} lower": v.a_lower.astype(str) + "/" + v.n.astype(str),
        "p (Holm)": v.wilcoxon_p_holm.map(fmt_p)})
    return md(out)


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def bar_panel(ax, summ, ctrls, title, log=False):
    w = 0.8 / len(ctrls)
    for i, c in enumerate(ctrls):
        s = summ[summ.controller == c].set_index("scenario").loc[SCEN]
        x = np.arange(len(SCEN)) + (i - (len(ctrls) - 1) / 2) * w
        if log:      # a log axis has no zero, so show dots rather than bars
            ax.scatter(x, s.true_delay, s=46, color=COLORS[c], label=NAMES[c], zorder=3, edgecolor="white", lw=1)
        else:
            ax.bar(x, s.true_delay, w * 0.92, color=COLORS[c], label=NAMES[c])
    ax.set_xticks(np.arange(len(SCEN)), SCEN)
    ax.set_title(title)
    ax.set_ylabel("True delay per vehicle (s, log scale)" if log else "True delay per vehicle (s)")
    ax.grid(axis="y", alpha=0.3)
    if log:
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(-0.6, len(SCEN) - 0.4)


# ----------------------------------------------------------------------------------------------- 1. sensor failure
def sensor_failure():
    out = os.path.join(args.out, "sensor_failure")
    os.makedirs(out, exist_ok=True)
    rows, summ, notes = [], [], []
    for mode in ("greedy", "sampled"):
        fam = f"sensor failure, {mode}"
        e1 = load("failure", "E1", mode)
        summary(summ, "failure", mode, "E1", 0, e1)
        summary(summ, "normal", mode, "E1", 0, load("normal", "E1", mode))
        for c in ("B1", "B2", "E1_ablated", "E1_nocomm"):
            d = load("failure", c, mode, required=c != "E1_nocomm")
            if d is None:
                notes.append(f"{c} {mode}: run not available yet")
                continue
            summary(summ, "failure", mode, c, 0, d)
            if c != "E1_ablated":
                summary(summ, "normal", mode, c, 0, load("normal", c, mode))
            compare(rows, fam, "E1", e1, c, d)
    fam = "fallback (failed light runs tuned fixed-time), greedy"
    fs = {c: load("failsafe", c, "greedy") for c in ("E1", "B1", "B2")}
    for c, d in fs.items():
        summary(summ, "failsafe", "greedy", c, 0, d)
    for c in ("B1", "B2"):
        compare(rows, fam, "E1", fs["E1"], c, fs[c])
    t, s = finish(rows), pd.DataFrame(summ)
    s.to_csv(f"{out}/sensor_failure_summary.csv", index=False, float_format="%.4f")
    t.to_csv(f"{out}/sensor_failure_tests.csv", index=False, float_format="%.6g")

    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mode in zip(axes, ("greedy", "sampled")):
        g = s[(s.test == "failure") & (s["mode"] == mode)]
        bar_panel(ax, g, [c for c in ("E1", "E1_ablated", "E1_nocomm", "B1", "B2") if c in set(g.controller)],
                  f"Sensor failure at light B1, {mode}")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_sensor_failure.png", dpi=150)
    plt.close(fig)

    # how much each controller loses when the detectors fail (same controller, same traffic)
    deg = []
    for mode in ("greedy", "sampled"):
        for c in ("E1", "B1", "B2", "E1_nocomm"):
            a, b = s[(s.test == "failure") & (s["mode"] == mode) & (s.controller == c)], \
                s[(s.test == "normal") & (s["mode"] == mode) & (s.controller == c)]
            if a.empty or b.empty:
                continue
            for sc in SCEN:
                fa, nb = a[a.scenario == sc].true_delay.iloc[0], b[b.scenario == sc].true_delay.iloc[0]
                deg.append(dict(mode=mode, controller=c, scenario=sc, normal=f"{nb:.1f}", failure=f"{fa:.1f}",
                                increase=f"+{fa - nb:.1f} s"))
    tab = s[s.scenario != "all"].copy()
    tab = tab[tab.test.isin(["failure", "failsafe"])]
    tab = pd.DataFrame({"Test": tab.test, "Mode": tab["mode"], "Controller": tab.controller, "Scenario": tab.scenario,
                        "True delay (s)": tab.true_delay.map(lambda x: f"{x:.1f}"),
                        "Entered %": tab.entered_pct.map(lambda x: f"{x:.1f}"),
                        "Gridlock episodes": tab.gridlock_episodes.astype(str) + "/10"})
    with open(f"{out}/README.md", "w") as f:
        f.write("# Sensor failure at light B1\n\nAll 4 queue detectors of light B1 read 0 for the whole episode; "
                "every other light works normally. 10 seeds x 3 scenarios, paired with the same traffic. "
                "`failsafe` = the usual fallback for a flagged detector fault: the failed light runs tuned "
                "fixed-time (10 s greens), all other lights keep their agents. Made by "
                "`scripts/analyze_extensions.py`.\n\n")
        if notes:
            f.write("Not yet available: " + "; ".join(notes) + ".\n\n")
        f.write("## Results\n\n" + md(tab) + "\n\n## Paired tests (E1 against each)\n\n"
                + tests_md(t, "E1", "Other") + "\n\n## Cost of the failure (same controller, normal vs failure)\n\n"
                + md(pd.DataFrame(deg)) + "\n\n![sensor failure](fig_sensor_failure.png)\n")
    return s, t


# ----------------------------------------------------------------------------------------------- 2. training seeds
SEED_TESTS = [("normal", "greedy", "true_delay", "normal traffic, greedy"),
              ("normal", "sampled", "true_delay", "normal traffic, sampled"),
              ("failure", "greedy", "true_delay", "sensor failure, greedy"),
              ("amb_off", "greedy", "amb_travel_time", "ambulances, priority off: ambulance travel time"),
              ("amb_on", "greedy", "amb_travel_time", "ambulances, priority on: ambulance travel time"),
              ("amb_off", "greedy", "true_delay", "ambulances, priority off: other traffic"),
              ("amb_on", "greedy", "true_delay", "ambulances, priority on: other traffic")]


def training_seeds():
    out = os.path.join(args.out, "training_seeds")
    os.makedirs(out, exist_ok=True)
    rows, summ = [], []
    for ts in (0, 1, 2):
        for test, mode, metric, label in SEED_TESTS:
            e1, b1 = load(test, "E1", mode, ts), load(test, "B1", mode, ts)
            if metric == "true_delay":
                summary(summ, test, mode, "E1", ts, e1)
                summary(summ, test, mode, "B1", ts, b1)
            compare(rows, f"{label} (training seed {ts})", "E1", e1, "B1", b1, metric, train_seed=ts, test=label)
    t, s = finish(rows), pd.DataFrame(summ)
    s.to_csv(f"{out}/training_seeds_summary.csv", index=False, float_format="%.4f")
    t.to_csv(f"{out}/training_seeds_tests.csv", index=False, float_format="%.6g")

    # one line per test and training seed: per scenario, who is significantly lower (Holm 0.05)
    verdict = []
    for _, _, _, label in SEED_TESTS:
        for ts in (0, 1, 2):
            g = t[(t.test == label) & (t.train_seed == ts) & (t.scenario != "all")].set_index("scenario")
            cells = {}
            for sc in SCEN:
                r = g.loc[sc]
                who = ("E1" if r.mean_a < r.mean_b else "B1") if r.wilcoxon_p_holm < 0.05 else "no sig. difference"
                cells[sc] = f"{r.mean_a:.1f} vs {r.mean_b:.1f} ({who})"
            verdict.append({"Test": label, "Training seed": ts, **cells})
    verdict = pd.DataFrame(verdict)

    plt = plt_setup()
    panels = SEED_TESTS[:5]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 3.8))
    for ax, (test, mode, metric, label) in zip(axes, panels):
        g = t[(t.test == label) & (t.scenario == "peak")]
        x = np.arange(3)
        ax.bar(x - 0.2, g.mean_a, 0.37, color=COLORS["E1"], label="E1")
        ax.bar(x + 0.2, g.mean_b, 0.37, color=COLORS["B1"], label="B1")
        ax.set_xticks(x, [f"seed {i}" for i in range(3)])
        ax.set_title(label.replace(": ", ":\n"), fontsize=8)
        ax.set_ylabel("Ambulance travel time (s)" if metric == "amb_travel_time" else "True delay per vehicle (s)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Peak scenario, E1 vs B1 for each training seed (other scenarios in the tables)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_training_seeds.png", dpi=150)
    plt.close(fig)

    with open(f"{out}/README.md", "w") as f:
        f.write("# Training seeds: is the E1 vs B1 result one lucky run?\n\nE1 and B1 were each trained 3 times "
                "with the same settings and training seeds 0, 1 and 2. Each E1 run is compared with the B1 run of "
                "the same training seed, on the same traffic (10 seeds x 3 scenarios). Values: mean true delay "
                "per vehicle (s) or mean ambulance travel time (s), E1 vs B1, and who is significantly lower "
                "(Wilcoxon, Holm within each test and training seed). Made by `scripts/analyze_extensions.py`.\n\n"
                + md(verdict) + "\n\n## All paired tests\n\n" + tests_md(t, "E1", "B1")
                + "\n\n![training seeds](fig_training_seeds.png)\n")
    return s, t, verdict


# ----------------------------------------------------------------------------------------------- 3. control
def control():
    out = os.path.join(args.out, "control")
    conds = [("normal", "greedy"), ("normal", "sampled"), ("failure", "greedy"), ("failure", "sampled")]
    data = {(test, mode): load(test, "E1_nocomm", mode, required=False) for test, mode in conds}
    if all(v is None for v in data.values()):
        print("control: no runs yet, skipped")
        return None
    os.makedirs(out, exist_ok=True)
    rows, summ, notes = [], [], []
    for (test, mode), nc in data.items():
        label = f"{'normal traffic' if test == 'normal' else 'sensor failure'}, {mode}"
        if nc is None:
            notes.append(f"{label}: run not available yet")
            continue
        e1, b1 = load(test, "E1", mode), load(test, "B1", mode)
        for c, d in (("E1", e1), ("E1_nocomm", nc), ("B1", b1)):
            summary(summ, test, mode, c, 0, d)
        compare(rows, label, "E1", e1, "E1_nocomm", nc, test=label)
        compare(rows, label, "E1_nocomm", nc, "B1", b1, test=label)
    t, s = finish(rows), pd.DataFrame(summ)
    s.to_csv(f"{out}/control_summary.csv", index=False, float_format="%.4f")
    t.to_csv(f"{out}/control_tests.csv", index=False, float_format="%.6g")

    plt = plt_setup()
    avail = [(test, mode) for test, mode in conds if data[(test, mode)] is not None]
    fig, axes = plt.subplots(1, len(avail), figsize=(4.3 * len(avail), 3.8), squeeze=False)
    for ax, (test, mode) in zip(axes[0], avail):
        g = s[(s.test == test) & (s["mode"] == mode)]
        bar_panel(ax, g, ["E1", "E1_nocomm", "B1"],
                  f"{'Normal traffic' if test == 'normal' else 'Sensor failure'}, {mode}", log=True)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_control.png", dpi=150)
    plt.close(fig)

    tab = s[s.scenario != "all"]
    tab = pd.DataFrame({"Test": tab.test, "Mode": tab["mode"], "Controller": tab.controller, "Scenario": tab.scenario,
                        "True delay (s)": tab.true_delay.map(lambda x: f"{x:.1f}"),
                        "Entered %": tab.entered_pct.map(lambda x: f"{x:.1f}"),
                        "Gridlock episodes": tab.gridlock_episodes.astype(str) + "/10"})
    with open(f"{out}/README.md", "w") as f:
        f.write("# No-communication control\n\n`E1_nocomm` has exactly E1's network, loss (including the diversity "
                "penalty) and training settings (same training seed 0, same 500k steps), but the neighbour message "
                "it receives is replaced by zeros, so it cannot communicate (checked by `tests/test_nocomm.py`). "
                "E1 vs the control isolates the effect of communication; the control vs B1 isolates the effect "
                "of the E1 network and loss without communication. Made by `scripts/analyze_extensions.py`.\n\n")
        if notes:
            f.write("Not yet available: " + "; ".join(notes) + ".\n\n")
        f.write("## Results\n\n" + md(tab) + "\n\n## Paired tests\n\n" + tests_md(t, "A", "B")
                + "\n\n![control](fig_control.png)\n")
    return s, t


if __name__ == "__main__":
    sf = sensor_failure()
    print(open(os.path.join(args.out, "sensor_failure", "README.md")).read()[:200], "...\n")
    ts = training_seeds()
    print(ts[2].to_string(index=False))
    ctl = control()
    print("\nWritten: results/mg5/final/sensor_failure/, training_seeds/" + (", control/" if ctl else ""))
