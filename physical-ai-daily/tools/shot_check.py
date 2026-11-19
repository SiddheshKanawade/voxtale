#!/usr/bin/env python3
"""Find shots too short to see: hard cuts closer together than --min seconds.

Usage: python3 tools/shot_check.py <video.mp4> [--min 1.5]

A sliver of another shot (a presenter's face, half a lower-third) flashing for a few frames reads as a glitch, not an
edit. They come from source footage cut at the wrong point. tools/clip.py runs this on every cut and refuses one
with a short shot unless it is passed --allow-short-shots "<why>"; render.sh and render_short.sh run it on the
final MP4 and warn. Fast cuts are fine when they are the point (a montage, a before/after snap): say why and keep them.
Exit 1 when a short shot is found.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

MIN_SHOT_S = 1.5  # under this a viewer can't take a shot in (user, 2026-11-19)
SCENE = 0.3  # ffmpeg scene-change score for a hard cut


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, check=True, text=True)
    return float(out.stdout)


def cuts(path: Path) -> list[float]:
    """Times of hard cuts, from ffmpeg's scene score on a downscaled copy."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-vf", f"scale=480:-2,select='gt(scene,{SCENE})',showinfo",
                          "-f", "null", "-"], capture_output=True, text=True)
    return [float(t) for t in re.findall(r"pts_time:([0-9.]+)", out.stderr)]


def short_shots(path: Path, min_s: float = MIN_SHOT_S) -> list[tuple[float, float]]:
    """(start, end) of every shot shorter than min_s, including a sliver at either end of the file."""
    edges = [0.0, *cuts(path), duration(path)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b - a < min_s]


def longest_shot(path: Path) -> tuple[float, float]:
    edges = [0.0, *cuts(path), duration(path)]
    return max(zip(edges, edges[1:]), key=lambda s: s[1] - s[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--min", type=float, default=MIN_SHOT_S, help="shortest shot in seconds")
    a = ap.parse_args()
    found = short_shots(a.video, a.min)
    for s, e in found:
        print(f"  SHORT SHOT {s:.2f}-{e:.2f}s ({e - s:.2f}s) in {a.video.name}")
    if not found:
        print(f"shots ok (none under {a.min}s): {a.video.name}")
    raise SystemExit(1 if found else 0)


if __name__ == "__main__":
    main()
