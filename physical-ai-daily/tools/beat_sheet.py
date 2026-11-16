#!/usr/bin/env python3
"""One frame per beat from a rendered episode MP4, tiled into contact sheets (12 per sheet, 3x4), for QA.

Usage: python3 -I tools/beat_sheet.py episodes/<ep> <video.mp4> <out_dir> [fraction_of_beat=0.75]

Timings come from <ep>/props.json with the same layout as the compositor: a 1.6 s sting after the intro segment.
Writes b000.png..., sheet0.png..., and labels.txt (sheet, cell, beat, visual type, duration, time) to read beside them.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

STING_S = 1.6  # matches STING_S in video/src/brand.tsx
PER_SHEET = 12


def beat_times(props: dict, frac: float) -> list[tuple[str, float, float]]:
    t, sting, out = 0.0, False, []
    segs = props["segments"]
    for i, s in enumerate(segs):
        bt = t
        for j, b in enumerate(s["beats"]):
            out.append((f"{s['id']}[{j}] {b['visual']['type']}", bt + b["duration"] * frac, b["duration"]))
            bt += b["duration"]
        t += s["duration"]
        if s["kind"] == "title" and not sting and i < len(segs) - 1:
            sting, t = True, t + STING_S
    return out


def main() -> None:
    ep, mp4, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.75
    out.mkdir(parents=True, exist_ok=True)
    times = beat_times(json.loads((ep / "props.json").read_text()), frac)
    labels = []
    for k, (lab, at, d) in enumerate(times):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", mp4, "-frames:v", "1",
                        "-vf", "scale=640:-2", str(out / f"b{k:03d}.png")], check=True)
        labels.append(f"sheet{k // PER_SHEET} cell{k % PER_SHEET + 1:>2}  {lab}  {d:.1f}s  @{at:.1f}s")
    for sh in range(0, len(times), PER_SHEET):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-start_number", str(sh), "-i", str(out / "b%03d.png"),
                        "-frames:v", "1", "-vf", "tile=3x4:padding=6:color=white",
                        str(out / f"sheet{sh // PER_SHEET}.png")], check=True)
    (out / "labels.txt").write_text("\n".join(labels) + "\n")
    print(f"{len(times)} beats -> {out}/sheet*.png (labels.txt lists each cell)")


if __name__ == "__main__":
    main()
