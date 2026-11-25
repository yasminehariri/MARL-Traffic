#!/usr/bin/env bash
# Defence videos in the 3D city: the final trained agents control the 16 lights in SUMO, every
# vehicle is recorded each second, and the 3D city replays exactly those vehicles. Evaluation traffic
# seed 0 (never used in training). Both sides of a video get exactly the same vehicles.
#
# Run from the repo root:          bash scripts/make_city_videos.sh
# Faster on a laptop with a GPU:   GPU=1 bash scripts/make_city_videos.sh   (opens a browser window)
# Needs: eclipse-sumo, ray[rllib], torch, playwright (+ python -m playwright install chromium), ffmpeg.
#
# Step 1  record_demo_4x4.py runs SUMO with each trained controller and saves every vehicle's position,
#         speed and type, and every light's phase, each second -> demo/full/run_<name>.json
# Step 2  a list of those recordings for the 3D viewer -> demo/full/runs.json
# Step 3  render_demo_video.py replays two recordings side by side in the 3D city -> videos/city/*.mp4
# Check   python scripts/check_video_runs.py  (trips completed in each recording = published result)
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)
C=checkpoints/4x4; K=checkpoints/ckpt_000977
E1=$C/E1_local_mg5/$K; B1=$C/B1_local_mg5/$K; B2=$C/B2_local_mg5/$K; CTRL=$C/E1nocomm_local_mg5/$K
FULL=demo/full; OUT=videos/city; mkdir -p $FULL $OUT
rec() {  # rec <name> <start> <duration> <recorder options...>
  local name=$1 start=$2 dur=$3; shift 3
  if [ -f $FULL/run_$name.json ]; then echo "skip recording $name (exists)"; return; fi
  echo "== recording $name"
  python scripts/record_demo_4x4.py --repo "$REPO" --start $start --duration $dur "$@" --out $FULL/run_$name.json
}
# ---- step 1: one SUMO run per controller and situation (the hour from minute 10, 5, 4.7 or 15)
rec E1              600 3000 --cond E1 --ckpt $E1
rec B1              600 3000 --cond B1 --ckpt $B1
rec B2              600 3000 --cond B2 --ckpt $B2
rec fixed           600 3000 --cond fixed
rec tuned           600 3000 --cond tuned
rec E1nocomm        600 3000 --cond E1 --ckpt $CTRL
rec E1_blind        300 3300 --cond E1 --ckpt $E1 --blind B1
rec B1_blind        300 3300 --cond B1 --ckpt $B1 --blind B1
rec E1nocomm_blind  300 3300 --cond E1 --ckpt $CTRL --blind B1
rec E1_blind_failsafe 300 3300 --cond E1 --ckpt $E1 --blind B1 --failsafe
rec E1_amb          280 3320 --cond E1 --ckpt $E1 --ambulances
rec E1_amb_preempt  280 3320 --cond E1 --ckpt $E1 --ambulances --preempt
rec B1_amb_preempt  280 3320 --cond B1 --ckpt $B1 --ambulances --preempt
rec E1_surge        900 2700 --cond E1 --ckpt $E1 --scenario surge
rec B1_surge        900 2700 --cond B1 --ckpt $B1 --scenario surge
rec fixed_surge     900 2700 --cond fixed --scenario surge

# ---- step 2: list of the full-hour recordings for the viewer (same names and colours as the website)
python - <<'EOF'
import json, os
base = json.load(open("demo/runs.json"))
runs = [dict(r, file="full/" + os.path.basename(r["file"])) for r in base["runs"]
        if os.path.exists("demo/full/" + os.path.basename(r["file"]))]
json.dump({**base, "runs": runs}, open("demo/full/runs.json", "w"), indent=1)
print(len(runs), "recordings listed in demo/full/runs.json")
EOF

# ---- step 3: the 3D city, side by side (left vs right). 1600x900, 25 fps.
GPUFLAG=""; [ "${GPU:-0}" = "1" ] && GPUFLAG="--gpu"
render() {  # render <file> <left> <right> <view> <sim from> <sim to> <video seconds>
  local f=$1; if [ -f $OUT/$f.mp4 ]; then echo "skip $f (exists)"; return; fi
  echo "== rendering $f"
  python scripts/render_demo_video.py --manifest full/runs.json --run $2 --vs $3 --view $4 \
      --sim-from $5 --sim-to $6 --seconds $7 --fps 25 --width 1600 --height 900 $GPUFLAG --out $OUT/$f.mp4
}
# peak hour: simulation minute 11:40 to 51:40 in 24 s (100x)
render 01_peak_E1_vs_fixed      E1 fixed    east-avenue 700 3100 24
render 02_peak_E1_vs_B1         E1 B1       east-avenue 700 3100 24
render 03_peak_E1_vs_B2         E1 B2       east-avenue 700 3100 24
render 04_peak_E1_vs_tuned      E1 tuned    east-avenue 700 3100 24
render 05_peak_E1_vs_control    E1 E1nocomm east-avenue 700 3100 24
# sensors of junction B1 fail (red ring): minute 5 to 60 in 24 s (137x)
render 06_failure_E1_vs_B1        E1_blind B1_blind          failure-B1 300 3600 24
render 07_failure_E1_vs_control   E1_blind E1nocomm_blind    failure-B1 300 3600 24
render 08_failure_E1_vs_fallback  E1_blind E1_blind_failsafe failure-B1 300 3600 24
# ambulances (red ring + beacons), 8x: ambulance no. 5 at junction A1, and ambulance no. 7 at junction C0
render 09_ambulance_E1_priority   E1_amb E1_amb_preempt         ambulance-A1 1500 1724 28
render 10_ambulance_E1_vs_B1      E1_amb_preempt B1_amb_preempt ambulance-C0 2120 2320 25
# 3x demand surge (never seen in training): minute 15 to 60 in 24 s (112x)
render 11_surge_E1_vs_fixed     E1_surge fixed_surge east-avenue 900 3600 24
render 12_surge_E1_vs_B1        E1_surge B1_surge    east-avenue 900 3600 24
echo "Done: videos/city/01..12"
