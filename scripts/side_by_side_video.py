"""Put two SUMO-GUI videos (from record_demo_4x4.py --video) side by side, with a title bar.

Only ffmpeg is used: the two recordings are placed next to each other unchanged (same simulated
seconds, same speed) and a title line is written above them.

    python scripts/side_by_side_video.py videos/raw/E1.mp4 videos/raw/B1.mp4 \
        --title "Peak hour: E1 (left) vs B1 (right)" --out videos/02_peak_E1_vs_B1.mp4
"""
import argparse, os, shutil, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument("left"); p.add_argument("right")
p.add_argument("--title", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
if not shutil.which("ffmpeg"):
    sys.exit("ffmpeg is needed (Colab: !apt-get -qq install -y ffmpeg)")
font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
fontopt = f"fontfile={font}:" if os.path.exists(font) else ""
title = a.title.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")
vf = (f"[0:v][1:v]hstack=inputs=2:shortest=1[s];"
      f"[s]pad=iw:ih+70:0:70:color=0x0e1218,"
      f"drawtext={fontopt}text='{title}':x=(w-tw)/2:y=20:fontsize=30:fontcolor=white[v]")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.left, "-i", a.right, "-filter_complex", vf,
                "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast",
                "-movflags", "+faststart", a.out], check=True)
print("Saved", a.out)
