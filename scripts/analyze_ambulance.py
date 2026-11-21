"""
Ambulance extension: tables, paired statistics and a figure from the evaluate_4x4.py --ambulances runs.

Each controller was run twice on the same traffic and the same 11 ambulances per episode: signal
priority off (results/mg5/4x4_amb_mg5_<c>.csv) and on (4x4_amb_preempt_mg5_<c>.csv). Results are
paired by (scenario, seed); tests are Wilcoxon signed-rank (exact, n = 10 per scenario, n = 30 for all
scenarios together) and the paired t-test, with Holm correction within each metric.

    python scripts/analyze_ambulance.py                      # every controller whose two files exist
    python scripts/analyze_ambulance.py --out results/mg5/final/ambulance

Outputs: ambulance_summary.csv (one row per controller and scenario), ambulance_tests.csv,
fig_ambulance.png, and README.md with the tables in Markdown.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--results", default=os.path.join(REPO, "results", "mg5"))
p.add_argument("--out", default=os.path.join(REPO, "results", "mg5", "final", "ambulance"))
args = p.parse_args()

ORDER = ["E1", "B1", "B2", "tuned"]
NAMES = {"E1": "E1 learned comm", "B1": "B1 no comm", "B2": "B2 hand-crafted comm", "tuned": "Tuned fixed-time"}
SCEN = ["uniform", "peak", "asym"]
METRICS = [("amb_travel_time", "Ambulance travel time (s)"), ("amb_stops", "Ambulance stops"),
           ("amb_wait", "Ambulance waiting (s)"), ("true_delay", "Other traffic: true delay (s/veh)")]


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def load(c):
    f_off = os.path.join(args.results, f"4x4_amb_mg5_{c}.csv")
    f_on = os.path.join(args.results, f"4x4_amb_preempt_mg5_{c}.csv")
    if not (os.path.exists(f_off) and os.path.exists(f_on)):
        return None
    off, on = pd.read_csv(f_off), pd.read_csv(f_on)
    for name, d, want in (("off", off, 0), ("on", on, 1)):
        assert (d.ambulances == 1).all() and (d.preempt == want).all(), f"{c} {name}: wrong kind of rows"
        assert not d.duplicated(["scenario", "seed"]).any(), f"{c} {name}: duplicated episodes"
        assert len(d) == 30, f"{c} {name}: {len(d)} episodes, expected 30"
    return off.set_index(["scenario", "seed"]).sort_index(), on.set_index(["scenario", "seed"]).sort_index()


def main():
    os.makedirs(args.out, exist_ok=True)
    data = {c: d for c in ORDER if (d := load(c)) is not None}
    if not data:
        raise SystemExit(f"no ambulance result pairs in {args.results}")
    rows, tests = [], []
    for c, (off, on) in data.items():
        assert off.index.equals(on.index), f"{c}: off and on runs cover different episodes"
        for sc in SCEN + ["all"]:
            a = off if sc == "all" else off.loc[sc]
            b = on if sc == "all" else on.loc[sc]
            r = dict(controller=c, scenario=sc, episodes=len(a), ambulances=int(a.amb_n.sum()),
                     arrived_off=int(a.amb_arrived.sum()), arrived_on=int(b.amb_arrived.sum()),
                     other_teleports_off=int(a.teleports.sum()), other_teleports_on=int(b.teleports.sum()),
                     preemptions_per_episode=round(float(b.preempt_events.mean()), 1),
                     priority_light_s_per_episode=round(float(b.preempt_light_s.mean()), 1))
            for col, _ in METRICS:
                x, y = a[col].to_numpy(float), b[col].to_numpy(float)
                r[f"{col}_off"], r[f"{col}_on"] = x.mean(), y.mean()          # rounded only when written
                r[f"{col}_change_pct"] = round(100 * (y.mean() - x.mean()) / x.mean(), 1) if x.mean() else np.nan
                d = y - x
                w = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
                t = stats.ttest_rel(x, y).pvalue if np.std(d) > 0 else 1.0
                tests.append(dict(controller=c, scenario=sc, metric=col, n=len(x), mean_off=x.mean(),
                                  mean_on=y.mean(), lower_with_priority=int((y < x).sum()),
                                  higher_with_priority=int((y > x).sum()), wilcoxon_p=w, ttest_p=t))
            rows.append(r)
    summ, tests = pd.DataFrame(rows), pd.DataFrame(tests)
    for m in tests.metric.unique():                    # Holm within each metric, per-scenario tests only
        k = (tests.metric == m) & (tests.scenario != "all")
        tests.loc[k, "wilcoxon_p_holm"] = holm(tests.loc[k, "wilcoxon_p"])
    summ.to_csv(os.path.join(args.out, "ambulance_summary.csv"), index=False, float_format="%.4f")
    tests.to_csv(os.path.join(args.out, "ambulance_tests.csv"), index=False, float_format="%.6g")

    # figure: ambulance travel time and other-traffic delay, off vs on, per controller and scenario
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctrls = list(data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    w = 0.8 / (2 * len(ctrls))
    for i, c in enumerate(ctrls):                      # left: ambulance travel time, off (light) and on (dark)
        s = summ[(summ.controller == c) & (summ.scenario != "all")].set_index("scenario").loc[SCEN]
        x = np.arange(len(SCEN)) + (2 * i - len(ctrls) + 0.5) * w
        axes[0].bar(x, s["amb_travel_time_off"], w, color=f"C{i}", alpha=0.35)
        axes[0].bar(x + w, s["amb_travel_time_on"], w, color=f"C{i}", label=c)
    axes[0].set_ylabel(METRICS[0][1])
    axes[0].set_title("Ambulances: priority off (light) vs on (dark)")
    w = 0.8 / len(ctrls)
    for i, c in enumerate(ctrls):                      # right: change of the other traffic's delay
        s = summ[(summ.controller == c) & (summ.scenario != "all")].set_index("scenario").loc[SCEN]
        x = np.arange(len(SCEN)) + (i - (len(ctrls) - 1) / 2) * w
        axes[1].bar(x, s["true_delay_change_pct"], w, color=f"C{i}", label=c)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Other traffic: change in true delay (%)")
    axes[1].set_title("Cost to the other traffic (priority on vs off)")
    axes[1].legend(ncol=len(ctrls), fontsize=8)
    for ax in axes:
        ax.set_xticks(np.arange(len(SCEN)), SCEN)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_ambulance.png"), dpi=150)

    # Markdown tables
    def md(df):
        cols = list(df.columns)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                         + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])
    t = summ[["controller", "scenario", "amb_travel_time_off", "amb_travel_time_on", "amb_travel_time_change_pct",
              "amb_stops_off", "amb_stops_on", "arrived_off", "arrived_on", "ambulances",
              "true_delay_off", "true_delay_on", "true_delay_change_pct"]].copy()
    for col in ["amb_travel_time_off", "amb_travel_time_on", "true_delay_off", "true_delay_on"]:
        t[col] = t[col].map(lambda v: f"{v:.1f}")
    for col in ["amb_stops_off", "amb_stops_on"]:
        t[col] = t[col].map(lambda v: f"{v:.2f}")
    t.columns = ["Controller", "Scenario", "Amb. travel off (s)", "on (s)", "change %", "Stops off", "on",
                 "Arrived off", "on", "of", "Other delay off (s)", "on (s)", "change %"]
    pv = tests[tests.scenario != "all"].copy()
    pv["p (Holm)"] = pv.wilcoxon_p_holm.map(lambda v: f"{v:.3f}")
    pv["mean_off"], pv["mean_on"] = pv.mean_off.map(lambda v: f"{v:.2f}"), pv.mean_on.map(lambda v: f"{v:.2f}")
    pv = pv[["controller", "scenario", "metric", "mean_off", "mean_on", "lower_with_priority", "p (Holm)"]]
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write("# Ambulances and signal priority\n\n11 ambulances per episode, the same for every controller; "
                "priority off vs on, greedy, 10 seeds x 3 scenarios. Made by `scripts/analyze_ambulance.py`.\n\n")
        f.write("## Summary\n\n" + md(t) + "\n\n## Paired tests (Wilcoxon, Holm-corrected within each metric)\n\n"
                "`lower_with_priority` = number of the 10 seeds where the value is lower with priority.\n\n"
                + md(pv) + "\n\n![ambulance results](fig_ambulance.png)\n")
    print(md(t))
    print(f"\nWritten to {os.path.relpath(args.out, REPO)}/: ambulance_summary.csv, ambulance_tests.csv, fig_ambulance.png, README.md")


if __name__ == "__main__":
    main()
