"""
Final 4x4 analysis: merge evaluation CSVs, summary table, paired statistics, figures.

Every condition was evaluated on the SAME route files (evaluation seeds 0-9 per
scenario), so results are paired by (scenario, seed). Tests per scenario:
    paired t-test, Wilcoxon signed-rank (exact, n=10), Cohen's d_z,
    Holm correction across all E1 comparisons (5 opponents x 3 scenarios = 15).

Usage (Colab or locally):
    python scripts/analyze_4x4.py results/4x4_eval_yasmine.csv results/4x4_eval_hanin.csv --out results/4x4_final
    # robustness run later: add the explore CSV and --labels E1_explore,B1_explore,B2_explore
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("csvs", nargs="+")
p.add_argument("--out", default="results/4x4_final")
p.add_argument("--focus", default="E1", help="label compared against every other label")
p.add_argument("--metric", default="true_delay")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

SCEN = ["uniform", "peak", "asym"]
ORDER = ["E1", "B1", "B2", "tuned", "fixed", "random"]
NAMES = {"E1": "E1 learned comm", "B1": "B1 no comm", "B2": "B2 hand-crafted comm",
         "tuned": "Tuned fixed-time (10 s)", "fixed": "Fixed-time (SUMO default)", "random": "Random"}

# ---------- merge and sanity checks ----------
df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True)
dup = df.duplicated(["label", "scenario", "seed"], keep=False)
if dup.any():
    raise SystemExit(f"Duplicate rows:\n{df[dup][['label', 'scenario', 'seed']]}")
labels = [l for l in ORDER if l in set(df.label)] + sorted(set(df.label) - set(ORDER))
counts = df.groupby(["label", "scenario"]).size().unstack()
print("Rows per label x scenario:\n", counts, "\n")
if (counts != 10).any().any():
    print("WARNING: not every label has 10 seeds per scenario\n")
# Pairing check: every label must have used the same route file per (scenario, seed)
g = df.groupby(["scenario", "seed"])["generated"].nunique()
if (g != 1).any():
    raise SystemExit(f"Route mismatch (different 'generated' counts) at:\n{g[g != 1]}")
print("Pairing check OK: all conditions saw identical route files.\n")
df["entered_pct"] = 100 * df.entered / df.generated
df.to_csv(os.path.join(args.out, "4x4_eval_merged.csv"), index=False)

# ---------- summary table ----------
def ci95(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))

rows = []
for sc in SCEN:
    for l in labels:
        d = df[(df.label == l) & (df.scenario == sc)]
        if d.empty:
            continue
        m = d[args.metric]
        rows.append(dict(scenario=sc, label=l, n=len(d), mean=m.mean(), sd=m.std(ddof=1), ci95=ci95(m),
                         median=m.median(), min=m.min(), max=m.max(),
                         wait_finished=d.wait_finished.mean(), travel_time=d.mean_travel_time.mean(),
                         entered_pct=d.entered_pct.mean(), finished=d.finished.mean(),
                         teleports_mean=d.teleports.mean(), teleports_max=d.teleports.max(),
                         runs_with_teleports=int((d.teleports > 0).sum())))
summ = pd.DataFrame(rows)
fixed_mean = summ[summ.label == "fixed"].set_index("scenario")["mean"]
summ["vs_fixed_pct"] = summ.apply(lambda r: 100 * (r["mean"] - fixed_mean.get(r.scenario, np.nan))
                                  / fixed_mean.get(r.scenario, np.nan), axis=1)
summ["rank"] = summ.groupby("scenario")["mean"].rank().astype(int)
summ.round(3).to_csv(os.path.join(args.out, "4x4_summary.csv"), index=False)

# ---------- paired tests ----------
def paired(a, b, sc):
    A = df[(df.label == a) & (df.scenario == sc)].set_index("seed")[args.metric]
    B = df[(df.label == b) & (df.scenario == sc)].set_index("seed")[args.metric]
    s = A.index.intersection(B.index)
    diff = (A[s] - B[s]).values
    t, pt = stats.ttest_rel(A[s], B[s])
    try:
        pw = stats.wilcoxon(A[s], B[s], method="exact").pvalue
    except ValueError:
        pw = np.nan
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.inf
    return dict(scenario=sc, a=a, b=b, n=len(s), mean_a=A[s].mean(), mean_b=B[s].mean(),
                mean_diff=diff.mean(), diff_ci95=ci95(diff), pct_change=100 * diff.mean() / B[s].mean(),
                a_better_seeds=int((diff < 0).sum()), t=t, p_t=pt, p_wilcoxon=pw, cohen_dz=dz)

def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj

tests = [paired(args.focus, b, sc) for sc in SCEN for b in labels if b != args.focus]
T = pd.DataFrame(tests)
T["p_t_holm"] = holm(T.p_t)
T["p_wilcoxon_holm"] = holm(T.p_wilcoxon)
T["significant_0.05"] = T.p_t_holm < 0.05
T.to_csv(os.path.join(args.out, f"4x4_tests_{args.focus}.csv"), index=False)

# secondary question: does hand-crafted communication help over none? (B2 vs B1)
if {"B1", "B2"} <= set(labels):
    T2 = pd.DataFrame([paired("B2", "B1", sc) for sc in SCEN])
    T2["p_t_holm"] = holm(T2.p_t)
    T2.to_csv(os.path.join(args.out, "4x4_tests_B2_vs_B1.csv"), index=False)

pd.set_option("display.width", 200)
print(summ[["scenario", "label", "mean", "sd", "ci95", "vs_fixed_pct", "entered_pct",
            "teleports_mean", "runs_with_teleports", "rank"]].round(1).to_string(index=False), "\n")
print(T[["scenario", "a", "b", "mean_a", "mean_b", "pct_change", "a_better_seeds",
         "p_t_holm", "p_wilcoxon_holm", "cohen_dz"]].round(4).to_string(index=False))

# ---------- figures ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical palette, fixed order by condition (never by rank)
COL = {"E1": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a", "tuned": "#eda100", "fixed": "#e87ba4", "random": "#008300"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF})
TITLE = {"uniform": "Uniform", "peak": "Peak hour", "asym": "Asymmetric"}


def bars(metric_col, err_col, xlabel, fname, title, fmt="{:.0f}"):
    shown = [l for l in ORDER if l in labels]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, sc in zip(axes, SCEN):
        s = summ[summ.scenario == sc].set_index("label").reindex(shown)
        y = np.arange(len(shown))[::-1]
        vals = s[metric_col].values
        err = s[err_col].values if err_col else None
        ax.barh(y, vals, height=0.62, color=[COL.get(l, "#888") for l in shown], zorder=2)
        if err is not None:
            ax.errorbar(vals, y, xerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
        xmax = np.nanmax(vals + (err if err is not None else 0))
        for yi, v, e in zip(y, vals, err if err is not None else np.zeros(len(vals))):
            ax.text(v + e + xmax * 0.02, yi, fmt.format(v), va="center", ha="left", color=INK, fontsize=9)
        ax.set_xlim(0, xmax * 1.22)
        ax.set_title(TITLE[sc], color=INK, fontsize=11, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
        ax.set_xlabel(xlabel)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(shown))[::-1])
    axes[0].set_yticklabels([NAMES[l] for l in shown])
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, fname), dpi=200)
    plt.close(fig)


bars("mean", "ci95", "Mean true delay per vehicle (s)", "fig_4x4_true_delay.png",
     "4x4 grid: mean true delay per vehicle (10 evaluation seeds, 95% CI). Lower is better.")
bars("teleports_mean", None, "Mean teleports per episode (vehicles stuck 300 s)", "fig_4x4_teleports.png",
     "4x4 grid: teleports per episode (gridlock indicator). Lower is better.", fmt="{:.1f}")

# per-seed paired view for the learned controllers
shown = [l for l in ["E1", "B1", "B2"] if l in labels]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
for ax, sc in zip(axes, SCEN):
    for l in shown:
        d = df[(df.label == l) & (df.scenario == sc)].sort_values("seed")
        ax.plot(d.seed, d[args.metric], color=COL[l], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURF, markeredgewidth=1.5, label=NAMES[l], zorder=3)
    for l in ["tuned", "fixed"]:
        if l in labels:
            m = summ[(summ.label == l) & (summ.scenario == sc)]["mean"].iloc[0]
            ax.axhline(m, color=COL[l], linewidth=1.5, linestyle=(0, (4, 3)), zorder=2,
                       label=NAMES[l] + " mean")
    ax.set_title(TITLE[sc], color=INK, fontsize=11, loc="left")
    ax.set_xticks(range(10))
    ax.set_xlabel("Evaluation seed")
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("True delay (s)")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, frameon=False, fontsize=9, loc="lower center", ncol=len(lab))
fig.suptitle("Per-seed true delay of the learned controllers (dashed: fixed-time baselines' means)",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(args.out, "fig_4x4_per_seed.png"), dpi=200)
plt.close(fig)
print(f"\nSaved tables and figures to {args.out}")
