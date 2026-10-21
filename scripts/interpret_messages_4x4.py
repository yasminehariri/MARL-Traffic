"""
Interpretability figures for E1's learned messages on the 4x4 grid (scope Week 9 / D10, RQ2).
Works only from files that check_messages_4x4.py already saved: no simulation, no training.

Inputs:  messages_4x4.npz (own observations + messages), message_report.json (ablation numbers)
Outputs: fig_e1_tsne.png, fig_e1_correlation.png, fig_e1_temporal.png, fig_e1_ablation.png,
         e1_correlation.csv, e1_interpretation.json

    python scripts/interpret_messages_4x4.py --npz results/e1_messages_4x4/messages_4x4.npz \
        --report results/e1_messages_4x4/message_report.json --out results/e1_messages_4x4

Observation layout (traci_env.get_obs): [q_N, q_E, q_S, q_W] halted vehicles / 20,
phase one-hot [N, E, S, W] (which approach has green), time in current phase / 90 s.
Messages are sampled every EVERY seconds (5 by default), so the temporal plot has 5 s resolution.
"""

import argparse
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--npz", required=True)
p.add_argument("--report", required=True)
p.add_argument("--out", default="results/e1_messages_4x4")
p.add_argument("--every", type=int, default=5)
p.add_argument("--tsne-n", type=int, default=4000)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

d = np.load(args.npz, allow_pickle=False)
own, msg, ep, key, agent = d["own"], d["msg"], d["episode"], d["step_key"], d["agent"]
rep = json.load(open(args.report))
phase = own[:, 4:8].argmax(1)
PH = ["N", "E", "S", "W"]
qtot = own[:, :4].sum(1) * 20  # halted vehicles over the 4 approaches
res = {"n_messages": int(len(msg))}

# ---------- structure of the code ----------
norm = np.linalg.norm(msg, axis=1)
res["mean_norm_by_green"] = {PH[k]: float(norm[phase == k].mean()) for k in range(4)}
res["mean_message_by_green"] = {PH[k]: [round(float(x), 2) for x in msg[phase == k].mean(0)] for k in range(4)}
c = msg - msg.mean(0)
_, S, Vt = np.linalg.svd(c, full_matrices=False)
pc = c @ Vt[:3].T
res["pc_var_explained"] = [float(x) for x in (S ** 2 / (S ** 2).sum())[:3]]

# ---------- correlation heatmap ----------
VARS = ["queue N", "queue E", "queue S", "queue W", "queue total",
        "green N", "green E", "green S", "green W", "time in phase"]
X = np.column_stack([own[:, 0], own[:, 1], own[:, 2], own[:, 3], own[:, :4].sum(1),
                     own[:, 4], own[:, 5], own[:, 6], own[:, 7], own[:, 8]])
ROWS = [f"dim {k + 1}" for k in range(8)] + ["PC1", "PC2"]
M = np.column_stack([msg, pc[:, :2]])


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


R = np.array([[corr(M[:, i], X[:, j]) for j in range(X.shape[1])] for i in range(M.shape[1])])
np.savetxt(os.path.join(args.out, "e1_correlation.csv"), R, delimiter=",", fmt="%.4f",
           header="rows: " + ";".join(ROWS) + " | cols: " + ";".join(VARS))
res["max_abs_r"] = float(np.nanmax(np.abs(R[:8])))
res["pairs_abs_r_over_0.5"] = [(ROWS[i], VARS[j], round(float(R[i, j]), 3))
                               for i in range(8) for j in range(len(VARS)) if abs(R[i, j]) > 0.5]
# queue information inside each phase group (does the message carry queues beyond "which green"?)
within = {}
for name, grp in (("N/S green", np.isin(phase, [0, 2])), ("E/W green", np.isin(phase, [1, 3]))):
    within[name] = {"r_PC2_queue_total": corr(pc[grp, 1], qtot[grp]),
                    "r_norm_queue_total": corr(norm[grp], qtot[grp]),
                    "mean_norm": float(norm[grp].mean())}
res["within_phase_group"] = within

# ---------- temporal: message around phase switches ----------
T = {}
offsets = np.arange(-2, 2)  # samples: -10, -5, 0 (first sample with the new green), +5 s
rows_by_seq = {}
order = np.lexsort((key, agent, ep))
for i in order:
    rows_by_seq.setdefault((ep[i], agent[i]), []).append(i)
events = {"to N/S green": [], "to E/W green": []}
for seq in rows_by_seq.values():
    seq = np.array(seq)
    ph = phase[seq]
    sw = np.where(ph[1:] != ph[:-1])[0] + 1
    for s in sw:
        if s + offsets.min() < 0 or s + offsets.max() >= len(seq):
            continue
        kind = "to N/S green" if ph[s] in (0, 2) else "to E/W green"
        events[kind].append(seq[s + offsets])
for kind, ev in events.items():
    ev = np.array(ev)
    T[kind] = {"n_switches": int(len(ev)),
               "norm": [float(norm[ev[:, k]].mean()) for k in range(len(offsets))],
               "queue_total": [float(qtot[ev[:, k]].mean()) for k in range(len(offsets))]}
res["temporal"] = T

# ---------- figures ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# 1. t-SNE
from sklearn.manifold import TSNE
rng = np.random.default_rng(0)
idx = rng.choice(len(msg), size=min(args.tsne_n, len(msg)), replace=False)
emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(msg[idx])
fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0.05, 0.1, 0.42, 0.78])
sc = ax.scatter(emb[:, 0], emb[:, 1], c=qtot[idx], cmap="Blues", s=5, vmin=-8, vmax=np.percentile(qtot, 99))
fig.colorbar(sc, ax=ax, label="Sender's halted vehicles (4 approaches)", fraction=0.046)
ax.set_title("t-SNE of E1 messages, coloured by sender congestion", loc="left", color=INK, fontsize=10)
ax.set_xticks([]); ax.set_yticks([]); clean(ax)
for k in range(4):
    a2 = fig.add_axes([0.55 + 0.22 * (k % 2), 0.52 - 0.42 * (k // 2), 0.2, 0.36])
    on = phase[idx] == k
    a2.scatter(emb[~on, 0], emb[~on, 1], s=3, color=GRID)
    a2.scatter(emb[on, 0], emb[on, 1], s=3, color=BLUE)
    a2.set_title(f"Sender's green: {PH[k]} ({100 * on.mean():.0f}%)", loc="left", color=INK, fontsize=9)
    a2.set_xticks([]); a2.set_yticks([]); clean(a2)
fig.text(0.55, 0.93, "Same map, one panel per current green phase (blue = that phase)", color=INK, fontsize=10)
fig.savefig(os.path.join(args.out, "fig_e1_tsne.png"), dpi=200)
plt.close(fig)

# 2. correlation heatmap (diverging blue <-> red, gray midpoint)
cmap = LinearSegmentedColormap.from_list("div", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#8a1c1c"])
fig, ax = plt.subplots(figsize=(10, 5.2))
im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        v = R[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(v) > 0.6 else INK, fontweight="bold" if abs(v) > 0.5 else "normal")
ax.set_xticks(range(len(VARS))); ax.set_xticklabels(VARS, rotation=35, ha="right", color=INK)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ROWS, color=INK)
ax.axhline(7.5, color=SURF, linewidth=3)
fig.colorbar(im, ax=ax, label="Pearson r", fraction=0.03)
ax.set_title("Correlation between E1 message dimensions and the sender's traffic state (bold: |r| > 0.5)",
             loc="left", color=INK, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(args.out, "fig_e1_correlation.png"), dpi=200)
plt.close(fig)

# 3. temporal
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
x = offsets * args.every
for kind, col in (("to N/S green", BLUE), ("to E/W green", ORANGE)):
    t = T[kind]
    axes[0].plot(x, t["norm"], color=col, linewidth=2, marker="o", markersize=6,
                 markeredgecolor=SURF, markeredgewidth=1.5, label=f"Switch {kind} (n={t['n_switches']})")
    axes[1].plot(x, t["queue_total"], color=col, linewidth=2, marker="o", markersize=6,
                 markeredgecolor=SURF, markeredgewidth=1.5, label=f"Switch {kind}")
for a, yl, tt in ((axes[0], "Message size (norm)", "Message around a phase switch"),
                  (axes[1], "Sender's halted vehicles", "Sender's queue around the same switches")):
    a.axvline(0, color=INK2, linewidth=1)
    a.set_xticks(x); a.set_xlabel("Seconds relative to the first sample with the new green (5 s sampling)")
    a.set_ylabel(yl); a.set_title(tt, loc="left", color=INK, fontsize=10)
    a.grid(axis="y", color=GRID); clean(a)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(args.out, "fig_e1_temporal.png"), dpi=200)
plt.close(fig)

# 4. ablation
if "ablation" in rep:
    ps = rep["ablation"]["per_scenario"]
    sc_names = [s for s in ("uniform", "peak", "asym") if s in ps]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xx = np.arange(len(sc_names))
    w = 0.34
    nb = [ps[s]["normal"] for s in sc_names]
    ab = [ps[s]["ablated"] for s in sc_names]
    ax.bar(xx - w / 2 - 0.01, nb, width=w, color=BLUE, label="E1 with its messages", zorder=2)
    ax.bar(xx + w / 2 + 0.01, ab, width=w, color=ORANGE, label="Messages replaced by the average message", zorder=2)
    for i in range(len(sc_names)):
        ax.text(xx[i] - w / 2, nb[i] + 2, f"{nb[i]:.1f}", ha="center", color=INK, fontsize=9)
        ax.text(xx[i] + w / 2, ab[i] + 2, f"{ab[i]:.1f}\n({100 * (ab[i] - nb[i]) / nb[i]:+.0f}%)",
                ha="center", color=INK, fontsize=9)
    ax.set_xticks(xx); ax.set_xticklabels([s.capitalize() if s != "asym" else "Asymmetric" for s in sc_names], color=INK)
    ax.set_ylabel("Mean true delay (s)"); ax.set_ylim(0, max(ab) * 1.3)
    ax.grid(axis="y", color=GRID, zorder=0); clean(ax)
    a = rep["ablation"]
    ax.set_title(f"Message ablation (seeds 0-2 per scenario): {a['mean_pct_change']:+.1f}% overall, "
                 f"worse in {a['episodes_worse']}/{a['n']} episodes, p={a['p_paired_t']:.1g}",
                 loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_e1_ablation.png"), dpi=200)
    plt.close(fig)

json.dump(res, open(os.path.join(args.out, "e1_interpretation.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
