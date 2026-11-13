"""Render the 3D demo (demo/index.html) to an MP4 video, frame by frame.

How it works
  1. Serves the demo/ folder on a local web server (the page loads its JSON files with fetch,
     which browsers block for file:// pages).
  2. Opens the page in a headless Chromium (Playwright) in "?render" mode, where the page does not
     animate by itself.
  3. For every video frame it calls window.renderAt(simTime, wallTime, camera) so the page draws
     exactly that moment, then saves a screenshot.
  4. ffmpeg joins the screenshots into an H.264 MP4.
  Rendering frame by frame (instead of screen-recording) gives perfectly smooth, sharp video
  on any machine, however slow.

Needs the recorded runs in demo/ (run_E1.json, run_fixed.json, ...; see demo/README.md).

Setup (once):
    pip install playwright imageio-ffmpeg
    python -m playwright install chromium        # skip on machines that already have it

Examples:
    python scripts/render_demo_video.py --run E1 --out demo_E1.mp4
    python scripts/render_demo_video.py --run E1 --vs fixed --cam aerial --out demo_E1_vs_fixed.mp4
    python scripts/render_demo_video.py --run E1 --seconds 5 --width 1280 --height 720 --out test.mp4   # quick test
"""
import argparse, functools, http.server, os, shutil, subprocess, sys, tempfile, threading, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ["cinematic", "skyline", "aerial", "junction", "street", "follow"]

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--demo-dir", default=os.path.join(REPO, "demo"))
p.add_argument("--run", default="E1", help="controller to show: E1, B1, B2, fixed, tuned (needs demo/run_<id>.json)")
p.add_argument("--vs", default=None, help="second controller for a side-by-side split view, e.g. fixed")
p.add_argument("--cam", default="cinematic", choices=CAMS)
p.add_argument("--start", type=float, default=0, help="first recorded second to show (0 = start of the recording)")
p.add_argument("--seconds", type=float, default=40, help="video length in seconds")
p.add_argument("--sim-speed", type=float, default=8, help="simulated seconds per video second")
p.add_argument("--fps", type=int, default=30)
p.add_argument("--width", type=int, default=1920)
p.add_argument("--height", type=int, default=1080)
p.add_argument("--show-dock", action="store_true", help="keep the playback bar at the bottom (hidden by default)")
p.add_argument("--keep-frames", action="store_true", help="keep the PNG frames folder")
p.add_argument("--out", default=None)
args = p.parse_args()

out = os.path.abspath(args.out or f"demo_{args.run}{'_vs_' + args.vs if args.vs else ''}.mp4")
for rid in [args.run] + ([args.vs] if args.vs else []):
    f = os.path.join(args.demo_dir, f"run_{rid}.json")
    if not os.path.exists(f):
        sys.exit(f"Missing {f}. Record it first (see demo/README.md).")


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
url = f"http://127.0.0.1:{server.server_address[1]}/index.html?render"

from playwright.sync_api import sync_playwright

frames_dir = tempfile.mkdtemp(prefix="demo_frames_")
n_frames = int(round(args.seconds * args.fps))
print(f"Rendering {n_frames} frames ({args.seconds:g} s at {args.fps} fps, {args.width}x{args.height}) of "
      f"{args.run}{' vs ' + args.vs if args.vs else ''}, camera '{args.cam}'")

with sync_playwright() as pw:
    # SwiftShader gives WebGL on machines without a GPU (Colab, servers); a real GPU is used when present
    browser = pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    three_dir = os.environ.get("MARL_THREE_DIR")  # optional: serve three.js from a local npm copy (offline machines)
    if three_dir:
        prefix = "https://cdn.jsdelivr.net/npm/three@0.160.0/"
        page.route(prefix + "**", lambda route: route.fulfill(
            path=os.path.join(three_dir, route.request.url[len(prefix):].split("?")[0]), content_type="application/javascript"))
        page.route("https://fonts.googleapis.com/**", lambda route: route.abort())
    page.goto(url)
    try:
        page.wait_for_function("window.__ready === true", timeout=180_000)
    except Exception:
        msg = page.inner_text("#loadMsg") if page.query_selector("#loadMsg") else ""
        sys.exit(f"The demo page did not finish loading. {msg} {' '.join(errors)}")

    # choose controller(s); select_option fires the page's change handlers
    page.select_option("#runSel", args.run)
    if args.vs:
        page.click("#splitBtn")
        page.select_option("#runSel2", args.vs)
        page.set_viewport_size({"width": args.width, "height": args.height})  # re-layout both halves
    if not args.show_dock:
        page.add_style_tag(content="#dock{display:none!important}")

    t0 = time.time()
    for i in range(n_frames):
        sim_t = args.start + i * args.sim_speed / args.fps
        wall = i / args.fps
        # the first call sets and settles the camera; later calls just advance time
        page.evaluate("([t, w, c]) => window.renderAt(t, w, c)", [sim_t, wall, args.cam if i == 0 else None])
        page.screenshot(path=os.path.join(frames_dir, f"{i:05d}.png"))
        if i % 30 == 0 or i == n_frames - 1:
            el = time.time() - t0
            eta = el / (i + 1) * (n_frames - i - 1)
            print(f"  frame {i + 1}/{n_frames}  ({el:.0f} s elapsed, about {eta / 60:.1f} min left)", flush=True)
    browser.close()
server.shutdown()
if errors:
    print("Page errors during rendering:", *errors[:5], sep="\n  ")

# 4. join frames into an MP4
subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", os.path.join(frames_dir, "%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart", out], check=True)
if args.keep_frames:
    print("Frames kept in", frames_dir)
else:
    shutil.rmtree(frames_dir, ignore_errors=True)
print("Saved", out)
