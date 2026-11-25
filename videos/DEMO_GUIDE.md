# Defence videos in the 3D city: how they are made and how to present them

## What these videos are

Every video shows our 3D city replaying a **real SUMO simulation** in which our final trained agents (5 s minimum green, 3 s yellow) control the 16 traffic lights. The traffic is evaluation **seed 0**, which was never used in training. Both sides of a video receive exactly the same vehicles at exactly the same times.

The pipeline:

1. **SUMO runs the simulation.** `scripts/record_demo_4x4.py` starts SUMO with the trained policy controlling every light through TraCI. It is the same environment code as the evaluation (`env/traci_env.py`).
2. **Every second is recorded.** Every simulated second, the script saves each vehicle's position, heading, speed and type, and each light's phase, queue and message, to `demo/full/run_<name>.json`.
3. **The city replays the recording.** The 3D city (`demo/index.html`, three.js) draws exactly those vehicles and light states. It does not simulate or invent anything. Positions between two recorded seconds are interpolated so the motion is smooth.
4. **The replay becomes a video.** `scripts/render_demo_video.py` plays the replay frame by frame in Chromium and ffmpeg joins the frames into an MP4. The text on screen is the viewer's own: controller names, vehicles waiting, simulation clock.

**What to answer if asked "how did you run and record this?"**
> "We ran our trained agents in SUMO through TraCI and recorded every vehicle and light each second. Our 3D city replays that recording, so every car you see is a real vehicle from the SUMO run. The video is rendered frame by frame from that replay. Each recording completes exactly the same number of trips as our published evaluation for that seed. Anyone can regenerate everything with `bash scripts/make_city_videos.sh`."

## How to reproduce them yourself

From the repository root:

```bash
bash scripts/make_city_videos.sh        # records the 16 SUMO runs, then renders the 12 videos into videos/city/
python scripts/check_video_runs.py      # checks every recording against the published results (all 16 match)
```

- On a laptop with a graphics card, run `GPU=1 bash scripts/make_city_videos.sh`. It opens a browser window and uses the GPU, so it takes minutes instead of hours. Without a GPU (Colab, servers) it still works, at about an hour per video.
- Setup, once: `pip install playwright imageio-ffmpeg`, then `python -m playwright install chromium`.
- The script skips anything that already exists. Delete a file to make it again.

One video by hand:

```bash
python scripts/record_demo_4x4.py --cond E1 --ckpt checkpoints/4x4/E1_local_mg5/checkpoints/ckpt_000977 \
    --start 600 --duration 3000 --out demo/full/run_E1.json            # SUMO run, minute 10 to 60
python scripts/render_demo_video.py --manifest full/runs.json --run E1 --vs B1 --view east-avenue \
    --sim-from 700 --sim-to 3100 --seconds 24 --fps 25 --width 1600 --height 900 --out videos/city/02_peak_E1_vs_B1.mp4
```

## Recording live in the meeting

Open the 3D viewer (the website, or `demo/index.html` served locally) and choose the two controllers. Press **Side by side**, then **● Record video**. Chrome or Edge asks which tab to share: choose this tab. Press **■ Stop and save**, and the browser downloads the video (MP4 in recent Chrome, otherwise WebM). This records exactly what is on screen, in real time.

## How to read a video

- **Left and right** are two controllers on the same traffic. The label above each half shows the controller and how many vehicles are waiting in the **whole city** at that moment.
- **"Simulation time"** in the top-left panel is the SUMO clock (minutes:seconds into the hour). The videos are time-lapses: peak-hour videos show 40 simulated minutes in 24 seconds. The ambulance videos run at 8× so you can follow one ambulance.
- **Queue halos** at each junction go from green to red as the queue grows. The traffic lights are the recorded signal states.
- **Ambulances** have flashing beacons and a red ring. A blue **PRIORITY** tag appears on a junction while it is giving an ambulance a green light.
- **A failed junction** has a red ring and a **SENSORS OFF** tag.
- **On E1 runs, the glowing lines between junctions** are the agents' messages. Brightness shows message size.

## The videos

| # | File | Situation, camera | What to watch | Result over 10 seeds (for you to say) |
|---|---|---|---|---|
| 1 | 01_peak_E1_vs_fixed.mp4 | Peak hour, east avenue, min 11:40 to 51:40 | E1 keeps queues shorter than SUMO's fixed-time plan and clears them faster after the peak | E1 92.1 s vs 172.3 s (−46.5%, 10/10 seeds) |
| 2 | 02_peak_E1_vs_B1.mp4 | Peak hour, east avenue | B1's queues spill from junction to junction until the avenue is jammed end to end. E1's avenue stays clear | E1 92.1 s vs B1 685.2 s (−86.6%, 10/10) |
| 3 | 03_peak_E1_vs_B2.mp4 | Peak hour, east avenue | B2 receives its neighbours' raw queue counts and still locks up | E1 92.1 s vs B2 854.9 s (−89.2%, 10/10) |
| 4 | 04_peak_E1_vs_tuned.mp4 | Peak hour, east avenue | E1 against the strongest non-learned plan | E1 92.1 s vs 169.5 s (−45.7%, 10/10) |
| 5 | 05_peak_E1_vs_control.mp4 | Peak hour, east avenue | The control has E1's network without messages, and flows just as well | Control 86.4 s vs E1 92.1 s: the compact network is the source of the peak-hour gain |
| 6 | 06_failure_E1_vs_B1.mp4 | Sensors of B1 fail, around B1, min 5 to 60 | Queues grow around the blind junction, more slowly under E1 | E1 630.9 s vs B1 994.6 s (−36.6%, 10/10) |
| 7 | 07_failure_E1_vs_control.mp4 | Sensors of B1 fail, around B1 | Where the messages matter: E1 vs the same network without messages | E1 630.9 s vs control 935.0 s (−32.5%, 10/10). With E1's messages replaced by their average: +17 to 19% |
| 8 | 08_failure_E1_vs_fallback.mp4 | Sensors of B1 fail, around B1 | With the fixed-time fallback at B1, the network returns to normal | 630.9 s → 93.6 s (normal traffic: 92.1 s) |
| 9 | 09_ambulance_E1_priority.mp4 | Ambulance no. 5 at junction A1, 8× | Without priority (left) it waits 2.5 min at A1's red light. With priority (right) the light turns green and it goes straight through | This ambulance: 873 s → 165 s. All ambulances: 239.9 s → 98.8 s (−59%), stops 7.7 → 0.35, other traffic −4.7% |
| 10 | 10_ambulance_E1_vs_B1.mp4 | Ambulance no. 7 at junction C0, 8×, priority on | Priority cannot clear a gridlock: under B1 the ambulance is stuck behind the queue at C0 | This ambulance: 109 s (E1) vs 987 s (B1). All: 98.8 s vs 427.0 s; under B1 only 74 of 110 ambulances arrive |
| 11 | 11_surge_E1_vs_fixed.mp4 | 3× surge, never trained, east avenue, min 15 to 60 | Under extreme overload the long fixed greens win | Fixed 334.6 s vs E1 481.9 s. Fixed-time is best here |
| 12 | 12_surge_E1_vs_B1.mp4 | 3× surge, east avenue | E1 is the best adaptive controller and keeps more cars entering | E1 481.9 s vs B1 871.2 s (−44.7%). 92% vs 68% of vehicles entered |

The cameras are fixed and identical on both sides. The east avenue (junctions D0 to D3) is where B1 and B2 jam first; the waiting counts above each half cover the whole city, not just the camera's view. The ambulance numbers were picked from the recordings as the clearest examples, and each one's own trip times are given above next to the average over all ambulances.

## Suggested order for Dr. Ali (about 15 minutes)

1. **Video 2** first: it shows the headline result most clearly, and the jammed avenue is impossible to miss.
2. **Video 1**: E1 against what cities use today.
3. **Videos 9 and 10**: ambulances. The road clears with priority, and B1's jams block emergency response.
4. **Videos 5 and 7** together: the attribution story. In normal traffic the compact network is what wins (5). When a junction goes blind, the messages are what keep E1 ahead (7).
5. **Video 8**: a simple standard fallback contains the failure.
6. **Video 11**: the honest boundary. Fixed-time wins under the 3× surge, and video 12 shows E1 is still the best adaptive controller there.

Then open the live 3D viewer for questions. You can move the camera anywhere, or press **● Record video**.

## Live 3D viewer links (after the push)

Base: `https://yasminehariri.github.io/MARL-Traffic/demo/index.html`, followed by the parameters below. Each window is minutes 15 to 25 of the peak hour.
- E1 vs fixed: `?run=E1&vs=fixed&cam=aerial`
- E1 vs B1: `?run=E1&vs=B1&cam=aerial`
- E1 vs control: `?run=E1&vs=E1nocomm&cam=aerial`
- Sensor failure, E1 vs control: `?run=E1_blind&vs=E1nocomm_blind&cam=aerial`
- Sensor failure, E1 vs fallback: `?run=E1_blind&vs=E1_blind_failsafe&cam=aerial`
- Ambulances, no priority vs priority: `?run=E1_amb&vs=E1_amb_preempt&cam=aerial`
- Ambulances with priority, E1 vs B1: `?run=E1_amb_preempt&vs=B1_amb_preempt&cam=aerial`
- Surge, E1 vs fixed: `?run=E1_surge&vs=fixed_surge&cam=aerial`

## If someone asks to see the raw simulator

`bash scripts/make_sumo_gui_videos.sh` records the same runs from SUMO's own 2D window (sumo-gui), in `videos/sumo_gui/`. It is the same simulation, shown without the city. Use it only as proof if asked.

## What to push

- `demo/index.html`: new **● Record video** button.
- `scripts/render_demo_video.py`: full-hour recordings (`--manifest`), fixed cameras (`--view`), simulation time windows (`--sim-from`, `--sim-to`), and `--gpu`.
- `scripts/make_city_videos.sh` and `scripts/check_video_runs.py`: new.
- `scripts/record_demo_4x4.py` and `env/traci_env.py`: the no-communication control, plus the optional sumo-gui recording. Normal runs are unchanged.
- `scripts/make_sumo_gui_videos.sh`, `scripts/side_by_side_video.py` and `demo/sumo_gui_view.xml`: the optional raw-simulator proof.
- `demo/runs.json`, plus three new website recordings: `demo/run_E1nocomm.json`, `demo/run_E1nocomm_blind.json` and `demo/run_B1_amb_preempt.json`.
- `videos/DEMO_GUIDE.md`, `videos/integrity_check.csv` and `.gitignore`. The ignore file keeps the large folders out of git: `demo/full/` and `videos/city/`. Keep the MP4s on a USB stick or Drive.
