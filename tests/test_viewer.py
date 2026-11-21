"""Checks the 3D viewer (demo/index.html) in a headless browser with the internet blocked.

  * it loads with no JavaScript errors and requests nothing outside the demo folder (three.js is in
    demo/vendor/, fonts fall back to system fonts),
  * only the recordings it shows are downloaded (lazy loading),
  * URL parameters set the controller, comparison, camera, speed, time, overlays and pause,
  * a bad parameter value is ignored without breaking the page,
  * a sensor-failure run shows the red marker on the failed junction and the failure line,
  * an ambulance run with priority draws the ambulances and the PRIORITY tag,
  * the embedding API works from a parent page through postMessage,
  * without runs.json the built-in run list is used.

Run from the repo root (needs: pip install playwright; python -m playwright install chromium):
    python tests/test_viewer.py
Saves screenshots to tests/viewer_screens/ to look at.
"""
import functools, http.server, json, os, shutil, sys, tempfile, threading

from playwright.sync_api import sync_playwright

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(REPO, "demo")
SHOTS = os.path.join(REPO, "tests", "viewer_screens")
os.makedirs(SHOTS, exist_ok=True)
REQUESTS = []


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        REQUESTS.append(("GET", self.path))
        super().do_GET()

    def do_HEAD(self):
        REQUESTS.append(("HEAD", self.path))
        super().do_HEAD()


def serve(folder):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=folder))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


OPEN = []


def open_page(browser, url, outside):
    while OPEN:                      # one page at a time: software WebGL is slow on small machines
        OPEN.pop().close()
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    errors, warnings = [], []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: warnings.append(m.text) if m.type == "warning" else None)

    def block(route):
        u = route.request.url
        if not u.startswith("http://127.0.0.1"):
            outside.append(u)
            return route.abort()
        return route.continue_()
    page.route("**/*", block)
    OPEN.append(page)
    page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_function("window.__ready === true", timeout=180_000)
    return page, errors, warnings


def run_gets():
    return sorted({p.split("?")[0].lstrip("/") for m, p in REQUESTS if m == "GET" and "/run_" in p})


def main():
    have = {f for f in os.listdir(DEMO) if f.startswith("run_") and f.endswith(".json")}
    assert {"run_E1.json", "run_fixed.json"} <= have, "demo/run_E1.json and demo/run_fixed.json are needed"
    srv, base = serve(DEMO)
    outside = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])

        # 1. default page
        REQUESTS.clear()
        page, errors, _ = open_page(browser, base + "/index.html", outside)
        page.wait_for_timeout(1500)
        st = page.evaluate("window.viewer.state()")
        assert not errors, errors
        assert st["run"] == "E1" and st["compare"] is None and st["playing"], st
        assert run_gets() == ["run_E1.json"], f"only the shown run should download, got {run_gets()}"
        listed = [r["id"] for r in page.evaluate("window.viewer.runs()")]
        manifest = [r["id"] for r in json.load(open(os.path.join(DEMO, "runs.json")))["runs"]]
        expected = [i for i in manifest if f"run_{i}.json" in have]
        assert listed == expected, (listed, expected)
        page.screenshot(path=os.path.join(SHOTS, "1_default.png"), timeout=180_000)
        print(f"1 default page: OK (E1 shown, only run_E1.json downloaded, {len(listed)} runs listed: {', '.join(listed)})")

        # 2. URL parameters
        REQUESTS.clear()
        page, errors, _ = open_page(browser, base + "/index.html?run=fixed&vs=E1&cam=aerial&speed=5&t=120&ov=flow,queues&paused=1", outside)
        page.wait_for_timeout(2500)
        st = page.evaluate("window.viewer.state()")
        assert not errors, errors
        assert st["run"] == "fixed" and st["compare"] == "E1" and st["cam"] == "aerial" and st["speed"] == 5, st
        assert not st["playing"] and abs(st["t"] - 120) < 1e-9, st
        assert st["overlays"] == {"rings": False, "flow": True, "queues": True, "msgs": False, "labels": False}, st
        assert run_gets() == ["run_E1.json", "run_fixed.json"], run_gets()
        assert page.is_visible("#runSel2") and page.is_visible("#divider")
        page.screenshot(path=os.path.join(SHOTS, "2_params_fixed_vs_E1.png"), timeout=180_000)
        print("2 URL parameters: OK (run, vs, cam, speed, t, ov, paused all applied)")

        # 3. bad values are ignored, page still works
        page, errors, warnings = open_page(browser, base + "/index.html?run=nope&cam=moon&speed=-3&ov=sparkles", outside)
        st = page.evaluate("window.viewer.state()")
        assert not errors, errors
        assert st["run"] == "E1" and st["cam"] == "cinematic" and st["speed"] == 2, st
        assert any("ignored unknown URL parameter" in w for w in warnings), warnings
        print("3 bad parameters: OK (ignored with a console warning)")

        # 4. ui=0 hides all panels
        page, errors, _ = open_page(browser, base + "/index.html?ui=0", outside)
        assert not errors and not any(page.is_visible(s) for s in ("#brand", "#stats", "#dock"))
        page.screenshot(path=os.path.join(SHOTS, "4_ui0.png"), timeout=180_000)
        print("4 ui=0: OK (panels hidden)")

        # 5. sensor-failure run
        if "run_E1_blind.json" in have:
            page, errors, _ = open_page(browser, base + "/index.html?run=E1_blind&cam=aerial&t=500&paused=1", outside)
            page.wait_for_timeout(2500)
            assert not errors, errors
            line = page.inner_text("#failLine")
            assert page.is_visible("#failLine") and line.lower() == "sensor failure at junction b1 · no fallback", line
            page.screenshot(path=os.path.join(SHOTS, "5_failure_E1_blind.png"), timeout=180_000)
            if "run_E1_blind_failsafe.json" in have:
                page.evaluate("window.viewer.setRun('E1_blind_failsafe')")
                page.wait_for_timeout(1500)
                assert page.inner_text("#failLine").lower().endswith("fixed-time fallback on")
                page.evaluate("window.viewer.setRun('E1')")
                page.wait_for_timeout(500)
                assert not page.is_visible("#failLine"), "failure line must disappear for a normal run"
            print("5 sensor-failure run: OK (failure line shown, cleared for a normal run)")
        else:
            print("5 sensor-failure run: skipped (no demo/run_E1_blind.json yet)")

        # 5b. ambulance run with priority: ambulances drawn, PRIORITY tag shown while a light serves one
        if "run_E1_amb_preempt.json" in have:
            page, errors, _ = open_page(browser, base + "/index.html?run=E1_amb_preempt&cam=aerial&paused=1", outside)
            assert not errors, errors
            got = page.evaluate("() => window.__dbg.findPri()")
            assert got and got['t'] >= 0 and got['tags'] >= 1 and got['amb'] >= 1, got
            assert page.inner_text("#failLine").lower() == "ambulances on the road · signal priority on"
            page.evaluate(f"window.viewer.seek({got['t']})"); page.wait_for_timeout(1500)
            page.screenshot(path=os.path.join(SHOTS, "5b_ambulance_priority.png"), timeout=180_000)
            print(f"5b ambulance run: OK (second {got['t']}: {got['amb']} ambulance(s) drawn, {got['tags']} PRIORITY tag(s) shown)")
        else:
            print("5b ambulance run: skipped (no demo/run_E1_amb_preempt.json yet)")

        # 6. embedding API through postMessage from a parent page
        parent = f"""<!doctype html><meta charset=utf-8><iframe id=f src="{base}/index.html?dock=0" width=900 height=500></iframe>
<script>window.got=[];addEventListener('message',e=>window.got.push(e.data));
window.send=(m,a,id)=>document.getElementById('f').contentWindow.postMessage({{viewer:m,args:a,id}},'*');</script>"""
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "parent.html"), "w").write(parent)
            srv2, base2 = serve(d)
            while OPEN:
                OPEN.pop().close()
            pg = browser.new_page()
            pg.route("**/*", lambda r: r.continue_() if r.request.url.startswith("http://127.0.0.1") else (outside.append(r.request.url), r.abort()))
            pg.goto(base2 + "/parent.html", wait_until="domcontentloaded", timeout=120_000)
            pg.wait_for_function("window.got.some(m => m.viewerReady)", timeout=180_000)
            pg.evaluate("send('setRun', ['fixed'], 1); send('setCam', ['junction'], 2); send('pause', [], 3)")
            pg.wait_for_function("window.got.filter(m => m.viewerReply).length === 3", timeout=60_000)
            pg.evaluate("send('state', [], 4)")
            pg.wait_for_function("window.got.some(m => m.viewerReply === 4)", timeout=10_000)
            st = pg.evaluate("window.got.find(m => m.viewerReply === 4).result")
            assert st["run"] == "fixed" and st["cam"] == "junction" and not st["playing"], st
            pg.close()
            srv2.shutdown()
        print("6 embedding API (postMessage): OK")

        # 7. no runs.json -> built-in list
        with tempfile.TemporaryDirectory() as d:
            for f in ("index.html", "net_4x4.json", "run_E1.json", "run_fixed.json"):
                shutil.copy(os.path.join(DEMO, f), d)
            shutil.copytree(os.path.join(DEMO, "vendor"), os.path.join(d, "vendor"))
            srv3, base3 = serve(d)
            page, errors, _ = open_page(browser, base3 + "/index.html", outside)
            assert not errors and [r["id"] for r in page.evaluate("window.viewer.runs()")] == ["E1", "fixed"]
            srv3.shutdown()
        print("7 without runs.json: OK (built-in list)")
        browser.close()
    srv.shutdown()
    other = sorted({u for u in outside if not u.startswith("https://fonts.g")})
    assert not other, f"the page tried to reach the internet: {other}"
    print(f"Nothing requested from the internet except Google Fonts ({len(outside)} font requests, optional).")
    print(f"ALL VIEWER CHECKS PASSED. Screenshots: {os.path.relpath(SHOTS, REPO)}/")


if __name__ == "__main__":
    sys.exit(main())
