# 3D demo: Traffic Twin 4x4

This is a 3D replay of real SUMO simulations driven by our trained agents. The page draws only what was recorded: every car position, light phase, queue and E1 message comes from the simulation. Nothing is simulated in the browser.

## What is in this folder

| File | What it is |
|---|---|
| `index.html` | The 3D viewer (three.js) |
| `vendor/three/` | three.js 0.160.0 (MIT licence) and the 12 add-on files the viewer uses, so the page works with no internet |
| `net_4x4.json` | The road network, exported from `configs/` by `record_demo_4x4.py --net-only` |
| `runs.json` | The list of recordings the viewer offers (id, file, name, colour, group). Files that do not exist yet are skipped |
| `run_<id>.json` | One recorded simulation each: a 10-minute window, every second |

## The recordings

All recordings use evaluation seed 0, so every controller faces exactly the same cars. Each command plays one SUMO episode, saves the window, and takes 2 to 10 minutes on a CPU.

| Group | Recording | Scenario, window | Status |
|---|---|---|---|
| normal | `E1`, `fixed`, `tuned`, `B2` | peak, 900 to 1500 s | done |
| normal | `B1` | peak, 900 to 1500 s | done (Hanin) |
| surge | `E1_surge`, `fixed_surge` | surge, 1500 to 2100 s (inside the 3x demand period) | done |
| surge | `B1_surge` | surge, 1500 to 2100 s | done (Hanin) |
| failure | `E1_blind`, `E1_blind_failsafe`, `B2_blind` | peak, 900 to 1500 s, junction B1's detectors broken | done |
| failure | `B1_blind`, `B1_blind_failsafe` | peak, 900 to 1500 s | done (Hanin) |
| ambulance | `E1_amb`, `E1_amb_preempt` | peak, 900 to 1500 s, the 11 ambulances of `--ambulances` (2 drive in this window, leaving at 900 and 1200 s), priority off and on | done |

Hanin's four commands (Colab, after `%cd /content/MARL-Traffic` and `!git pull`; CPU is fine; they use her B1 checkpoint):
```
!python scripts/record_demo_4x4.py --cond B1 --ckpt /content/drive/MyDrive/MARL_checkpoints/4x4/B1_local_mg5/checkpoints/ckpt_000977
!python scripts/record_demo_4x4.py --cond B1 --ckpt /content/drive/MyDrive/MARL_checkpoints/4x4/B1_local_mg5/checkpoints/ckpt_000977 --scenario surge --start 1500 --label B1_surge
!python scripts/record_demo_4x4.py --cond B1 --ckpt /content/drive/MyDrive/MARL_checkpoints/4x4/B1_local_mg5/checkpoints/ckpt_000977 --blind B1
!python scripts/record_demo_4x4.py --cond B1 --ckpt /content/drive/MyDrive/MARL_checkpoints/4x4/B1_local_mg5/checkpoints/ckpt_000977 --blind B1 --failsafe
```
They save `run_B1.json`, `run_B1_surge.json`, `run_B1_blind.json` and `run_B1_blind_failsafe.json` to `MARL_results/demo/` on her Drive. Each one ends with a line `Saved ... (x MB): 601 s, ...`. Copy the four files into this `demo/` folder and push.

The commands for the recordings marked done are below (rerunning them gives the same frames, second by second: we checked this for `run_E1.json`, recorded once on Colab and once on another machine):
```
# E1=<E1 checkpoint>, B2=<B2 checkpoint> (bash variables)
python scripts/record_demo_4x4.py --cond E1 --ckpt $E1
python scripts/record_demo_4x4.py --cond B2 --ckpt $B2
python scripts/record_demo_4x4.py --cond fixed
python scripts/record_demo_4x4.py --cond tuned
python scripts/record_demo_4x4.py --cond E1 --ckpt $E1 --scenario surge --start 1500 --label E1_surge
python scripts/record_demo_4x4.py --cond fixed --scenario surge --start 1500 --label fixed_surge
python scripts/record_demo_4x4.py --cond E1 --ckpt $E1 --blind B1
python scripts/record_demo_4x4.py --cond E1 --ckpt $E1 --blind B1 --failsafe
python scripts/record_demo_4x4.py --cond B2 --ckpt $B2 --blind B1
python scripts/record_demo_4x4.py --cond E1 --ckpt $E1 --ambulances
python scripts/record_demo_4x4.py --cond E1 --ckpt $E1 --ambulances --preempt
```
`--blind` and `--failsafe` do exactly what they do in `evaluate_4x4.py`: the same environment code and the same fallback cycle. A recording made with `--blind` stores `"blind": "B1"`, and the viewer then marks junction B1 in red ("SENSORS OFF") and shows the failure line under the scenario name.

In an ambulance recording the ambulances are drawn white with a red stripe, flashing beacons and a red ring, and with priority on a blue "PRIORITY" tag appears over every junction while it serves an ambulance. The "Follow vehicle" camera follows an ambulance when one is on the road.

A recording is a 10-minute window. The delay numbers in the report and on the website are for whole one-hour episodes over 10 seeds, so the replay illustrates a result; it is not the result itself.

## Open the viewer on your laptop

Browsers block a page opened by double-click from reading its JSON files, so serve the folder:
```
cd demo
python -m http.server 8000
```
Then open http://localhost:8000. Stop the server with Ctrl+C.

Controls: controller menu, **Side by side** for a comparison, camera buttons (fly-over, skyline, aerial, junction, street, follow a bus), overlays (detection rings, green flow, queue halos, agent messages, light IDs), playback speed and the time slider.

## URL parameters (for links and for embedding in the website)

`index.html?run=E1&vs=fixed&cam=aerial&speed=5&t=120&ov=flow,queues&paused=1`

| Parameter | Values | Default |
|---|---|---|
| `run` | a recording id from `runs.json` | the first that exists |
| `vs` | a second id; opens the side-by-side view | off |
| `cam` | `cinematic`, `skyline`, `aerial`, `junction`, `street`, `follow` | `cinematic` |
| `speed` | simulated seconds per real second (above 0, up to 60) | `2` |
| `t` | start position in seconds from the start of the recording (0 to 600) | `0` |
| `paused=1` | start paused | playing |
| `ov` | the overlays to turn on, comma-separated: `rings`, `flow`, `queues`, `msgs`, `labels`, or `none` | rings, flow, queues, msgs |
| `q=fast` | lower quality for weak laptops and phones | `high` |
| `ui=0` | hide every panel (a clean picture for embedding) | shown |
| `dock=0` | hide only the playback bar | shown |
| `manifest` | another list file instead of `runs.json` | `runs.json` |

An unknown value is ignored and a warning is printed in the browser console.

Only the recordings on screen are downloaded (3 to 8 MB each). The others are fetched when they are chosen.

## Embedding API

In the same page: `window.viewer.setRun('B1')`, `setCompare('fixed')` (or `null` to close the split), `setCam('aerial')`, `play()`, `pause()`, `seek(120)`, `setSpeed(5)`, `setOverlay('msgs', true)`, `setQuality('fast')`, `state()` and `runs()`. `setRun` and `setCompare` return a promise that resolves to `true` or `false`.

From a parent page that shows the viewer in an `<iframe>`:
```js
frame.contentWindow.postMessage({ viewer: 'setRun', args: ['E1_blind'], id: 1 }, '*');
addEventListener('message', e => {
  if (e.data.viewerReady) { /* viewer loaded; e.data.runs lists the recordings */ }
  if (e.data.viewerReply === 1) { /* e.data.result is the return value */ }
});
```

## Check that it works

```
pip install playwright && python -m playwright install chromium
python tests/test_viewer.py
```
This opens the viewer in a hidden browser with the internet blocked. It checks:
- no errors;
- nothing downloaded from outside the folder;
- lazy loading;
- every URL parameter;
- a bad parameter;
- the failure marker;
- the postMessage API;
- the fallback without `runs.json`.

It saves screenshots to `tests/viewer_screens/`.

## Make a video (optional)

```
pip install playwright imageio-ffmpeg
python -m playwright install chromium
python scripts/render_demo_video.py --run E1 --out demo_E1.mp4
python scripts/render_demo_video.py --run E1 --vs fixed --cam aerial --out demo_E1_vs_fixed.mp4
python scripts/render_demo_video.py --run E1_blind --vs E1_blind_failsafe --cam aerial --out demo_failure.mp4
```
Run these from the repo root. The script opens the page in an invisible browser and draws every frame exactly (`window.renderAt`). It takes a screenshot of each frame and joins them into an MP4 with ffmpeg.

Options:
- `--seconds 40`: video length.
- `--sim-speed 8`: simulated seconds per video second.
- `--cam`: the camera.
- `--show-dock`: keep the playback bar.
- `--width` and `--height`: frame size (default 1920x1080).

For a quick test, use `--seconds 5 --width 1280 --height 720`.

More options:
- `--gpu`: use this computer's graphics card. It opens a browser window and is much faster.
- `--manifest full/runs.json`: use the full-hour recordings.
- `--view east-avenue` (or `failure-B1`, `ambulance-A1`, `ambulance-C0`): a fixed camera.
- `--sim-from` and `--sim-to`: the simulation seconds to show.

`bash scripts/make_city_videos.sh` makes all 12 defence videos in one go; see `videos/DEMO_GUIDE.md`.

The **● Record video** button in the viewer records the tab live while it plays and saves the file. In Chrome or Edge, choose this tab when asked.

## How to describe it

"The simulation is real SUMO output from our trained agents. We record it to JSON and replay it in a three.js visualiser. The visualiser only draws what happened; it doesn't simulate anything itself."
