#!/usr/bin/env python3
"""One frame per beat from a rendered episode MP4, tiled into labelled contact sheets (12 per sheet), for QA.

Usage: python3 -I tools/beat_sheet.py episodes/<ep> <video.mp4> <out_dir> [fraction_of_beat=0.75]

Timings come from <ep>/props.json with the same layout as the compositor: a 1.6 s sting after the intro segment.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw
ep, mp4, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.75
os.makedirs(outdir, exist_ok=True)
props = json.load(open(os.path.join(ep, "props.json")))
STING = 1.6
t, sting, times = 0.0, -1, []
segs = props["segments"]
for i, s in enumerate(segs):
    bt = t
    for j, b in enumerate(s["beats"]):
        times.append((f"{s['id']}[{j}] {b['visual']['type']}", bt + b["duration"] * frac, b["duration"]))
        bt += b["duration"]
    t += s["duration"]
    if s["kind"] == "title" and sting < 0 and i < len(segs) - 1:
        sting = t; t += STING
print("beats", len(times), "total", round(t + 1.6, 1))
files = []
for k, (lab, at, d) in enumerate(times):
    f = os.path.join(outdir, f"b{k:03d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", mp4, "-frames:v", "1", "-vf", "scale=640:-1", f], check=True)
    files.append((f, f"{lab} {d:.1f}s @{at:.1f}"))
for sh in range(0, len(files), 12):
    chunk = files[sh:sh + 12]
    S = Image.new("RGB", (3 * 640, 4 * 380), "white"); D = ImageDraw.Draw(S)
    for i, (f, lab) in enumerate(chunk):
        x, y = (i % 3) * 640, (i // 3) * 380
        S.paste(Image.open(f), (x, y + 20)); D.text((x + 4, y + 4), lab, fill="black")
    S.save(os.path.join(outdir, f"sheet{sh // 12}.png"))
