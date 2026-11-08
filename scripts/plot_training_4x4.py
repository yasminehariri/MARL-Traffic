"""Training curves, convergence speed (RQ3) and the training-time table, from progress_*.csv.

    python scripts/plot_training_4x4.py --dir results/mg5 --out results/mg5/final/training
"""
import argparse, os
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--dir", default="results/mg5")
p.add_argument("--out", default="results/mg5/final/training")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
RUNS = {"4x4_E1": ("E1 learned comm, 4x4", "#2a78d6", 16), "4x4_B1": ("B1 no comm, 4x4", "#eb6834", 16),
        "4x4_B2": ("B2 hand-crafted comm, 4x4", "#1baf7a", 16), "2x2_E1": ("E1 learned comm, 2x2", "#86b3ea", 4)}

def load(run):
    d = pd.read_csv(os.path.join(args.dir, f"progress_{run}.csv"))
    resumes = d.total_steps[d.iteration.diff() < 0].tolist()      # a resumed run repeats iterations
    return d.drop_duplicates("iteration", keep="last").sort_values("iteration"), resumes

rows = []
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for run, (name, col, n_agents) in RUNS.items():
    d, resumes = load(run)
    v = d.dropna(subset=["episode_return_mean"])
    x = v.total_steps / 1000
    y = (v.episode_return_mean / n_agents / 1e3).rolling(20, min_periods=1).mean()
    if run.startswith("4x4"):
        axes[0].plot(x, y, color=col, lw=2, label=name)
    frac = (y - y.iloc[0]) / (y.iloc[-1] - y.iloc[0])
    axes[1].plot(x, frac, color=col, lw=2, label=name)
    for r in resumes:   # RLlib's reward average restarts after a resume, so the curve can jump there
        for ax in axes:
            ax.axvline(r / 1000, color=col, ls=":", lw=1.2)
        axes[0].annotate(f"{run.split('_')[1]} run resumed here:\nreward average restarts", xy=(r / 1000, y.min() * 0.52),
                         xytext=(r / 1000 + 60, y.min() * 0.67), color=INK2, fontsize=9,
                         arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    rows.append(dict(run=run, agents=n_agents, iterations=int(d.iteration.max()), env_steps=int(d.total_steps.max()),
                     train_hours=round(d.iter_seconds.sum() / 3600, 2), sec_per_iter=round(d.iter_seconds.mean(), 1),
                     final_return_per_agent=round(y.iloc[-1] * 1e3),
                     steps_to_90pct_of_improvement=int(x[frac >= 0.9].iloc[0] * 1000), resumed_at_steps=resumes))
axes[0].set_title("4x4 training curves (episode reward per agent, smoothed)", loc="left", color=INK, fontsize=11)
axes[0].set_ylabel("Episode return per agent (thousands)")
axes[1].set_title("Convergence speed: share of total improvement reached", loc="left", color=INK, fontsize=11)
axes[1].axhline(0.9, color=INK2, ls="--", lw=1)
axes[1].set_ylabel("Fraction of improvement (0 = start, 1 = end)")
axes[1].set_ylim(-0.1, 1.15)
for ax in axes:
    ax.set_xlabel("Environment steps (thousands)")
    ax.grid(color=GRID, lw=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(args.out, "fig_training_curves.png"), dpi=200)
t = pd.DataFrame(rows)
t.to_csv(os.path.join(args.out, "training_time_table.csv"), index=False)
print(t.to_string(index=False))
print("Saved to", args.out)
