#!/usr/bin/env bash
# Optional proof: the same runs recorded from SUMO's own 2D window (sumo-gui), for anyone who asks to see
# the raw simulator. The defence videos themselves are made in the 3D city: scripts/make_city_videos.sh
#
# Run from the repo root:   bash scripts/make_sumo_gui_videos.sh
# Colab (once per runtime): !apt-get -qq install -y xvfb ffmpeg   then   !bash scripts/make_sumo_gui_videos.sh
# Needs: eclipse-sumo (sumo-gui), ray[rllib], torch, ffmpeg, and xvfb on machines without a screen.
#
# Step 1 runs each controller once with   scripts/record_demo_4x4.py ... --video videos/raw/<name>.mp4
#   (one frame per simulated second, 30 fps = 30x real time; the run's JSON recording is saved too).
# Step 2 joins two of those videos with   scripts/side_by_side_video.py
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)
C=checkpoints/4x4; K=checkpoints/ckpt_000977
E1=$C/E1_local_mg5/$K; B1=$C/B1_local_mg5/$K; B2=$C/B2_local_mg5/$K; CTRL=$C/E1nocomm_local_mg5/$K
RAW=videos/sumo_gui/raw; OUT=videos/sumo_gui; mkdir -p $RAW videos/json
rec() {  # rec <name> <start> <duration> <recorder options...>
  local name=$1 start=$2 dur=$3; shift 3
  if [ -f $RAW/$name.mp4 ]; then echo "skip $name (exists)"; return; fi
  echo "== recording $name"
  python scripts/record_demo_4x4.py --repo "$REPO" --start $start --duration $dur "$@" \
      --out videos/json/run_$name.json --video $RAW/$name.mp4
}
# ---- step 1: one SUMO-GUI recording per controller and situation
# normal peak hour, minutes 10-60
rec E1              600 3000 --cond E1 --ckpt $E1
rec B1              600 3000 --cond B1 --ckpt $B1
rec B2              600 3000 --cond B2 --ckpt $B2
rec fixed           600 3000 --cond fixed
rec tuned           600 3000 --cond tuned
rec E1nocomm        600 3000 --cond E1 --ckpt $CTRL
# sensors of junction B1 fail, minutes 5-60
rec E1_blind        300 3300 --cond E1 --ckpt $E1 --blind B1
rec B1_blind        300 3300 --cond B1 --ckpt $B1 --blind B1
rec E1nocomm_blind  300 3300 --cond E1 --ckpt $CTRL --blind B1
rec E1_blind_failsafe 300 3300 --cond E1 --ckpt $E1 --blind B1 --failsafe
# 11 ambulances (one every 5 min), minutes 4.7-60
rec E1_amb          280 3320 --cond E1 --ckpt $E1 --ambulances
rec E1_amb_preempt  280 3320 --cond E1 --ckpt $E1 --ambulances --preempt
rec B1_amb_preempt  280 3320 --cond B1 --ckpt $B1 --ambulances --preempt
# 3x demand surge (never seen in training), minutes 15-60
rec E1_surge        900 2700 --cond E1 --ckpt $E1 --scenario surge
rec B1_surge        900 2700 --cond B1 --ckpt $B1 --scenario surge
rec fixed_surge     900 2700 --cond fixed --scenario surge

# ---- step 2: side by side (left vs right)
sbs() { python scripts/side_by_side_video.py $RAW/$1.mp4 $RAW/$2.mp4 --title "$3" --out $OUT/$4.mp4; }
sbs E1 fixed          "Peak hour  -  left: E1 (learned communication)   right: fixed-time (SUMO default)" 01_peak_E1_vs_fixed
sbs E1 B1             "Peak hour  -  left: E1 (learned communication)   right: B1 (no communication)" 02_peak_E1_vs_B1
sbs E1 B2             "Peak hour  -  left: E1 (learned communication)   right: B2 (hand-crafted communication)" 03_peak_E1_vs_B2
sbs E1 tuned          "Peak hour  -  left: E1 (learned communication)   right: tuned fixed-time (10 s greens)" 04_peak_E1_vs_tuned
sbs E1 E1nocomm       "Peak hour  -  left: E1   right: E1 control (same network, no messages)" 05_peak_E1_vs_control
sbs E1_blind B1_blind "Sensors of junction B1 failed (red ring)  -  left: E1   right: B1" 06_failure_E1_vs_B1
sbs E1_blind E1nocomm_blind "Sensors of junction B1 failed  -  left: E1   right: E1 control (no messages)" 07_failure_E1_vs_control
sbs E1_blind E1_blind_failsafe "Sensors of junction B1 failed  -  left: E1   right: E1 with fixed-time fallback at B1" 08_failure_E1_vs_fallback
sbs E1_amb E1_amb_preempt "Ambulances (white circles)  -  left: E1 without priority   right: E1 with signal priority" 09_ambulance_E1_priority
sbs E1_amb_preempt B1_amb_preempt "Ambulances with signal priority  -  left: E1   right: B1" 10_ambulance_E1_vs_B1
sbs E1_surge fixed_surge "3x demand surge (never trained)  -  left: E1   right: fixed-time (SUMO default)" 11_surge_E1_vs_fixed
sbs E1_surge B1_surge "3x demand surge (never trained)  -  left: E1   right: B1" 12_surge_E1_vs_B1
echo "Done: videos/01..12 (side by side) and videos/raw (one per controller)."
