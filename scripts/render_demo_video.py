"""Render the 3D city demo (demo/index.html) to an MP4 video, frame by frame.

How it works
  1. Serves the demo/ folder on a local web server (the page loads its JSON files with fetch,
     which browsers block for file:// pages).
  2. Opens the page in Chromium (Playwright) in "?render" mode, where the page does not animate by
     itself.
  3. For every video frame it calls window.renderAt(simTime, wallTime, camera), so the page draws
     exactly that moment of the recorded SUMO run, and takes a screenshot.
  4. ffmpeg joins the screenshots into an H.264 MP4.
  Rendering frame by frame (instead of screen-recording) gives smooth, sharp video on any machine,
  however slow. On a laptop with a graphics card, add --gpu and it is many times faster.

The recorded runs come from scripts/record_demo_4x4.py (SUMO + the trained agents, one JSON file per
run). demo/runs.json lists the 10-minute recordings used by the website; --manifest full/runs.json
uses the full-hour recordings made by scripts/make_city_videos.sh.

Setup (once):
    pip install playwright imageio-ffmpeg
    python -m playwright install chromium        # skip on machines that already have it

Examples:
    python scripts/render_demo_video.py --run E1 --out demo_E1.mp4
    python scripts/render_demo_video.py --run E1 --vs fixed --cam aerial --out demo_E1_vs_fixed.mp4
    # full-hour recordings, simulation minute 12 to 52 shown in 24 s, camera over the east avenue:
    python scripts/render_demo_video.py --manifest full/runs.json --run E1 --vs B1 --view east-avenue \
        --sim-from 700 --sim-to 3100 --seconds 24 --out videos/city/02_peak_E1_vs_B1.mp4
    python scripts/render_demo_video.py --run E1 --seconds 5 --width 1280 --height 720 --out test.mp4   # quick test
"""
import argparse, functools, http.server, json, os, shutil, subprocess, sys, threading, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ["cinematic", "skyline", "aerial", "junction", "street", "follow"]
# Fixed camera positions used for the defence videos: (camera position, point looked at), in the
# viewer's coordinates (metres; x east, y up, z south; the city centre is 0,0,0; junction A0 is at
# -300,0,300 and D3 at 300,0,-300).
VIEWS = {
    "east-avenue":  ((380, 190, 330), (300, 0, -20)),    # looking north up avenue D (D0 to D3)
    "failure-B1":   ((-60, 260, 380), (-60, 0, 140)),    # junctions B0, B1 (sensors failed), C0, C1
    "ambulance-A1": ((-430, 120, 230), (-320, 0, 100)),  # junction A1, ambulances arriving from the west
    "ambulance-C0": ((230, 110, 420), (100, 0, 280)),    # junction C0, ambulances arriving from the north
}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--demo-dir", default=os.path.join(REPO, "demo"))
p.add_argument("--manifest", default="runs.json", help="list of recorded runs, relative to the demo folder "
               "(runs.json = website recordings, full/runs.json = full-hour recordings)")
p.add_argument("--run", default="E1", help="controller to show (an id from the manifest, e.g. E1, B1, fixed)")
p.add_argument("--vs", default=None, help="second controller for a side-by-side split view, e.g. fixed")
p.add_argument("--cam", default="cinematic", choices=CAMS)
p.add_argument("--view", default=None, choices=sorted(VIEWS), help="fixed camera position (overrides --cam)")
p.add_argument("--start", type=float, default=0, help="first recorded second to show (0 = start of the recording)")
p.add_argument("--sim-from", type=float, default=None, help="first simulation second to show (instead of --start)")
p.add_argument("--sim-to", type=float, default=None, help="last simulation second to show; sets the speed from --seconds")
p.add_argument("--seconds", type=float, default=40, help="video length in seconds")
p.add_argument("--sim-speed", type=float, default=8, help="simulated seconds per video second (ignored with --sim-to)")
p.add_argument("--fps", type=int, default=30)
p.add_argument("--width", type=int, default=1920)
p.add_argument("--height", type=int, default=1080)
p.add_argument("--gpu", action="store_true", help="use this computer's graphics card (opens a browser window; much faster)")
p.add_argument("--show-dock", action="store_true", help="keep the playback bar at the bottom (hidden by default)")
p.add_argument("--out", default=None)
args = p.parse_args()

out = os.path.abspath(args.out or f"demo_{args.run}{'_vs_' + args.vs if args.vs else ''}.mp4")
os.makedirs(os.path.dirname(out), exist_ok=True)
_manifest = os.path.join(args.demo_dir, args.manifest)
if not os.path.exists(_manifest):
    sys.exit(f"Missing {_manifest}.")
_m = json.load(open(_manifest))
FILES = {r["id"]: r["file"] for r in (_m["runs"] if isinstance(_m, dict) else _m)}
START = {}
for rid in [args.run] + ([args.vs] if args.vs else []):
    f = os.path.join(args.demo_dir, FILES.get(rid, f"run_{rid}.json"))
    if not os.path.exists(f):
        sys.exit(f"Missing {f}. Record it first (see demo/README.md).")
    with open(f) as fh:  # read only the header fields ("start" comes before the frames)
        head = fh.read(4000)
    START[rid] = json.loads("{" + head[head.index('"start"'):].split(",", 1)[0] + "}")["start"] if '"start"' in head else 0

if args.sim_from is not None:
    args.start = args.sim_from - START[args.run]
if args.sim_to is not None:
    args.sim_speed = (args.sim_to - (args.start + START[args.run])) / args.seconds
if args.vs and START[args.vs] != START[args.run]:
    sys.exit(f"{args.run} and {args.vs} were recorded from different start times; record them with the same --start.")


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found. Run: pip install imageio-ffmpeg")


ffmpeg = find_ffmpeg()

# 1. local web server for the demo folder
class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):  # keep the console clean (missing optional runs give harmless 404s)
        pass


handler = functools.partial(QuietHandler, directory=args.demo_dir)
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
# the page picks the controller(s) from the URL and sets window.__ready once they are loaded
url = (f"http://127.0.0.1:{server.server_address[1]}/index.html?render&manifest={args.manifest}&run={args.run}"
       + (f"&vs={args.vs}" if args.vs else ""))

from playwright.sync_api import sync_playwright

n_frames = int(round(args.seconds * args.fps))
sim0 = START[args.run] + args.start
print(f"Rendering {n_frames} frames ({args.seconds:g} s at {args.fps} fps, {args.width}x{args.height}) of "
      f"{args.run}{' vs ' + args.vs if args.vs else ''}: simulation {sim0:.0f} s to "
      f"{sim0 + args.seconds * args.sim_speed:.0f} s ({args.sim_speed:g}x), camera '{args.view or args.cam}'")

# 4. ffmpeg reads the screenshots from a pipe (no frames are written to disk)
enc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(args.fps), "-i", "-",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                        "-movflags", "+faststart", out + ".part.mp4"], stdin=subprocess.PIPE)

with sync_playwright() as pw:
    if args.gpu:   # a visible browser window drawing with the real graphics card
        browser = pw.chromium.launch(headless=False, args=["--ignore-gpu-blocklist", "--enable-gpu-rasterization"])
    else:          # SwiftShader gives WebGL on machines without a GPU (Colab, servers): slow but works everywhere
        browser = pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("https://fonts.googleapis.com/**", lambda route: route.abort())  # system fonts; three.js is in demo/vendor
    page.goto(url)
    try:
        page.wait_for_function("window.__ready === true", timeout=600_000)
    except Exception:
        msg = page.inner_text("#loadMsg") if page.query_selector("#loadMsg") else ""
        sys.exit(f"The demo page did not finish loading. {msg} {' '.join(errors)}")

    state = page.evaluate("window.viewer.state()")
    if state["run"] != args.run or (args.vs and state["compare"] != args.vs):
        sys.exit(f"The page could not show {args.run}{' vs ' + args.vs if args.vs else ''} (got {state}).")
    if args.vs:
        page.set_viewport_size({"width": args.width, "height": args.height})  # re-layout both halves
    if not args.show_dock:
        page.add_style_tag(content="#dock{display:none!important}")
    cam = args.cam
    if args.view:  # a fixed camera: move the 'aerial' preset there and use it
        pos, tgt = VIEWS[args.view]
        page.evaluate("([p, t]) => { const d = window.__dbg; d.CAMS.aerial.pos.set(...p); d.CAMS.aerial.tgt.set(...t); }", [pos, tgt])
        cam = "aerial"

    t0 = time.time()
    for i in range(n_frames):
        sim_t = args.start + i * args.sim_speed / args.fps
        wall = i / args.fps
        # the first call sets and settles the camera; later calls just advance time
        page.evaluate("([t, w, c]) => window.renderAt(t, w, c)", [sim_t, wall, cam if i == 0 else None])
        enc.stdin.write(page.screenshot(type="png"))
        if i % 25 == 0 or i == n_frames - 1:
            el = time.time() - t0
            eta = el / (i + 1) * (n_frames - i - 1)
            print(f"  frame {i + 1}/{n_frames}  ({el / 60:.1f} min elapsed, about {eta / 60:.0f} min left)", flush=True)
    browser.close()
server.shutdown()
enc.stdin.close()
if enc.wait() != 0:
    sys.exit("ffmpeg failed")
os.replace(out + ".part.mp4", out)   # the final name appears only when the video is complete
if errors:
    print("Page errors during rendering:", *errors[:5], sep="\n  ")
print("Saved", out)
