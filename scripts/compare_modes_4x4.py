"""
Greedy vs sampled actions for the trained controllers (E1, B1, B2) on the 4x4 grid.

Greedy  = always the most likely action (labels E1, B1, B2, explore=0).
Sampled = actions sampled from the policy, seeded per episode (labels *_explore, explore=1).
Same route files per (scenario, seed), so each controller's two modes are paired by seed.

    python scripts/compare_modes_4x4.py results/4x4_eval_yasmine.csv results/4x4_eval_hanin.csv \
        results/4x4_eval_explore_*.csv --out results/4x4_modes
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("csvs", nargs="+")
p.add_argument("--out", default="results/4x4_modes")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

SCEN = ["uniform", "peak", "asym"]
CONDS = ["E1", "B1", "B2"]
NAMES = {"E1": "E1 learned comm", "B1": "B1 no comm", "B2": "B2 hand-crafted comm"}
df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True)

rows = []
for sc in SCEN:
    for c in CONDS:
        g = df[(df.label == c) & (df.scenario == sc)].set_index("seed")
        s = df[(df.label == f"{c}_explore") & (df.scenario == sc)].set_index("seed")
        seeds = g.index.intersection(s.index)
        if len(seeds) < 2:
            continue
        a, b = g.loc[seeds, "true_delay"], s.loc[seeds, "true_delay"]
        rows.append(dict(scenario=sc, cond=c, n=len(seeds),
                         greedy_delay=a.mean(), sampled_delay=b.mean(),
                         change_pct=100 * (b.mean() - a.mean()) / a.mean(),
                         p_paired_t=stats.ttest_rel(a, b).pvalue,
                         greedy_teleports=g.loc[seeds, "teleports"].mean(),
                         sampled_teleports=s.loc[seeds, "teleports"].mean(),
                         greedy_entered_pct=100 * g.loc[seeds, "entered"].sum() / g.loc[seeds, "generated"].sum(),
                         sampled_entered_pct=100 * s.loc[seeds, "entered"].sum() / s.loc[seeds, "generated"].sum()))
T = pd.DataFrame(rows)
T.round(4).to_csv(os.path.join(args.out, "4x4_greedy_vs_sampled.csv"), index=False)
pd.set_option("display.width", 200)
print(T.round(2).to_string(index=False))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COL = {"E1": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
TITLE = {"uniform": "Uniform", "peak": "Peak hour", "asym": "Asymmetric"}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, sc in zip(axes, SCEN):
    t = T[T.scenario == sc].set_index("cond").reindex(CONDS)
    y = np.arange(len(CONDS))[::-1]
    xmax = max(t.greedy_delay.max(), t.sampled_delay.max())
    for yi, c in zip(y, CONDS):
        g, s = t.loc[c, "greedy_delay"], t.loc[c, "sampled_delay"]
        ax.plot([g, s], [yi, yi], color=GRID, linewidth=2, zorder=1)
        ax.scatter([s], [yi], s=130, facecolor="none", edgecolor=COL[c], linewidth=2, zorder=3, marker="D")
        ax.scatter([g], [yi], s=45, color=COL[c], edgecolor=SURF, linewidth=1.5, zorder=4, marker="o")
        ax.text(max(g, s) + xmax * 0.06, yi, f"{g:.0f} → {s:.0f} s", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, xmax * 1.3)
    ax.set_title(TITLE[sc], loc="left", color=INK, fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_xlabel("Mean true delay per vehicle (s)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(np.arange(len(CONDS))[::-1])
axes[0].set_yticklabels([NAMES[c] for c in CONDS])
fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=INK2, markersize=8, label="Greedy (most likely action)"),
                    Line2D([], [], marker="D", linestyle="", markerfacecolor="none", markeredgecolor=INK2,
                           markeredgewidth=2, markersize=8, label="Sampled (seeded)")],
           loc="lower center", ncol=2, frameon=False)
fig.suptitle("4x4 grid: greedy vs sampled actions for the learned controllers (10 seeds each)",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(os.path.join(args.out, "fig_4x4_greedy_vs_sampled.png"), dpi=200)
print(f"\nSaved to {args.out}")
