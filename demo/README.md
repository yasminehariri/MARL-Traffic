# 3D demo: Traffic Twin 4x4

A 3D replay of real SUMO simulations driven by our trained agents. The page draws only what was recorded: every car position, every light phase, every queue and every E1 message comes from the simulation. Nothing is simulated in the browser.

## What is in this folder

| File | What it is |
|---|---|
| `index.html` | The 3D viewer (three.js). |
| `net_4x4.json` | The road network, exported from `configs/` by `record_demo_4x4.py --net-only`. |
| `run_<id>.json` | One recorded run per controller (you create these, step 1). |

## Step 1: record the runs (Colab, about 10 minutes each)

Each command runs one full SUMO episode (peak scenario, seed 0, same traffic for every controller) and saves seconds 900 to 1500 to `MARL_results/demo/run_<id>.json` on your Drive.

```
!python scripts/record_demo_4x4.py --cond E1 --ckpt /content/drive/MyDrive/MARL_checkpoints/4x4/E1_local_mg5/checkpoints/ckpt_000977
!python scripts/record_demo_4x4.py --cond B1 --ckpt /content/drive/MyDrive/MARL_checkpoints/4x4/B1_local_mg5/checkpoints/ckpt_000977
```

`run_fixed.json` and `run_tuned.json` are already in this folder, recorded with the same script, so only E1 (and optionally B1) need recording. With `run_E1.json` added, you can show E1 alone or side by side with fixed-time. Download the files from `MARL_results/demo/` and put them in this `demo/` folder.

The B1 checkpoint is on the Drive of whoever trained B1, so run the B1 command from that account.

## Step 2: open the demo on your laptop

Browsers block a page opened by double-click from reading its JSON files, so serve the folder:

```
cd demo
python -m http.server 8000
```

Then open http://localhost:8000 in Chrome or Edge. Stop the server with Ctrl+C.

Controls: choose the controller, **Split** for a side-by-side comparison, camera buttons (fly-over, skyline, aerial, junction, street, follow a car), overlays (detection rings, green flow, queue halos, agent messages, light IDs), playback speed and the time slider.

## Step 3: make a video (optional)

```
pip install playwright imageio-ffmpeg
python -m playwright install chromium
python scripts/render_demo_video.py --run E1 --out demo_E1.mp4
python scripts/render_demo_video.py --run E1 --vs fixed --cam aerial --out demo_E1_vs_fixed.mp4
```

Run these from the repo root. The script opens the page in an invisible browser, draws every frame exactly (`window.renderAt`), takes a screenshot of each, and joins them into an MP4 with ffmpeg. Useful options:
- `--seconds 40` (video length)
- `--sim-speed 8` (simulated seconds per video second)
- `--cam` (one of cinematic, skyline, aerial, junction, street, follow)
- `--show-dock` (keep the playback bar, which is hidden in videos by default)
- `--width/--height` (default 1920x1080)

For a quick test, use `--seconds 5 --width 1280 --height 720`. It works on Colab too, just slower, because there is no GPU there.

## How to describe it

"The simulation is real SUMO output from our trained agents. We record it to JSON and replay it in a three.js visualiser. The visualiser only draws what happened; it doesn't simulate anything itself."
