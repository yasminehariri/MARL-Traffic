"""Check that every video recording reproduces the published evaluation (seed 0).

For each recording made by make_city_videos.sh, the trips completed by the end of the hour
(SUMO's own arrival counter, saved in demo/full/run_<name>.json) must equal `finished` (+ ambulances
arrived) in the evaluation CSV for the same controller, scenario and seed 0.

    python scripts/check_video_runs.py                    -> checks demo/full, writes videos/integrity_check.csv
    python scripts/check_video_runs.py videos/json        -> checks the SUMO-GUI recordings instead
"""
import csv, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(REPO, "demo", "full"); RES = os.path.join(REPO, "results", "mg5")
PAIRS = [("E1", "4x4_eval_mg5_E1.csv", "peak", None), ("B1", "4x4_eval_mg5_B1.csv", "peak", None),
         ("B2", "4x4_eval_mg5_B2.csv", "peak", None), ("fixed", "../4x4_eval_hanin.csv", "peak", "fixed"),
         ("tuned", "../4x4_eval_hanin.csv", "peak", "tuned"), ("E1nocomm", "4x4_eval_mg5_E1nocomm.csv", "peak", None),
         ("E1_blind", "4x4_blind_mg5_E1.csv", "peak", None), ("B1_blind", "4x4_blind_mg5_B1.csv", "peak", None),
         ("E1nocomm_blind", "4x4_blind_mg5_E1nocomm.csv", "peak", None),
         ("E1_blind_failsafe", "4x4_blind_failsafe_mg5_E1.csv", "peak", None),
         ("E1_amb", "4x4_amb_mg5_E1.csv", "peak", None), ("E1_amb_preempt", "4x4_amb_preempt_mg5_E1.csv", "peak", None),
         ("B1_amb_preempt", "4x4_amb_preempt_mg5_B1.csv", "peak", None),
         ("E1_surge", "4x4_surge_mg5_E1.csv", "surge", None), ("B1_surge", "4x4_surge_mg5_B1.csv", "surge", None),
         ("fixed_surge", "4x4_surge_mg5_fixed.csv", "surge", None)]
rows = []
for rid, f, sc, cond in PAIRS:
    path = f"{REC}/run_{rid}.json"
    if not os.path.exists(path):
        print(f"{rid:18s} (no recording yet)"); continue
    r = next(r for r in csv.DictReader(open(f"{RES}/{f}")) if r["scenario"] == sc and r["seed"] == "0"
             and r.get("explore", "0") in ("0", "False", "") and (cond is None or r["cond"] == cond))
    rec = json.load(open(path))["frames"][-1]["arrived"]
    amb = int(float(r.get("amb_arrived") or 0))
    pub = int(float(r["finished"])) + amb
    rows.append({"recording": rid, "scenario": sc, "seed": 0,
                 "evaluation_file": f"results/mg5/{f}".replace("mg5/../", ""), "true_delay_s": r["true_delay"],
                 "trips_finished_eval": r["finished"], "ambulances_arrived_eval": amb,
                 "arrivals_in_recording": rec, "match": rec == pub})
    print(f"{rid:18s} evaluation {pub:5d}   recording {rec:5d}   {'OK' if rec == pub else 'MISMATCH'}")
if rows:
    os.makedirs(os.path.join(REPO, "videos"), exist_ok=True)
    with open(os.path.join(REPO, "videos", "integrity_check.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok = all(r["match"] for r in rows)
print(f"{len(rows)} recordings checked, all match: {ok}")
sys.exit(0 if ok else 1)
