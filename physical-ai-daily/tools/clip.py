#!/usr/bin/env python3
"""Cut a short, muted, credited video clip into an episode's assets/ for a "clip" beat.

Usage: python3 tools/clip.py <episode-or-short-dir> <source.mp4> <name> --start S --dur S --credit "Video: NVIDIA" --url URL

<source.mp4> is a local file you already have. Getting it is a separate step: downloading needs the user's OK, and
the footage rule (docs/video-requirements.md 7.2) decides what may be used: official footage the company or lab
published first; other public footage only when that doesn't show it; wire agencies last. Every clip is a few
seconds, muted, credited on screen and listed in the description with its URL.

Writes assets/<name>.mp4 (H.264, no audio, source resolution up to 3840 px wide for 4K renders, 30 fps), assets/<name>_last.png (the last frame,
held if the beat outlasts the clip) and assets/<name>.json (size, duration, credit, url), which tools/voice.py
inlines like an image's capture metadata.

Every cut is checked for shots under 1.5 s (tools/shot_check.py): official videos are edited fast, and a few frames of
the next shot at either end flash past as a glitch. A cut with one is refused, with the longest clean shot printed to
re-cut from. Pass --allow-short-shots "<why>" only when the fast cut is the point (a montage, a before/after snap).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_check import longest_shot, short_shots  # noqa: E402

MAX_CLIP_S = 10  # "a few seconds": longer is re-publishing, not commentary (checklist 7.2)


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-show_entries", "format=duration", "-of", "json", str(path)], capture_output=True, check=True, text=True)
    j = json.loads(out.stdout)
    return {"width": j["streams"][0]["width"], "height": j["streams"][0]["height"], "duration": float(j["format"]["duration"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ep_dir", type=Path)
    ap.add_argument("source", type=Path)
    ap.add_argument("name")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the source")
    ap.add_argument("--dur", type=float, required=True, help="clip length in seconds")
    ap.add_argument("--credit", required=True, help='on-screen credit, e.g. "Video: Google DeepMind"')
    ap.add_argument("--url", required=True, help="where the footage was published (goes in the description)")
    ap.add_argument("--allow-short-shots", metavar="WHY", help="keep shots under 1.5 s on purpose, and say why")
    a = ap.parse_args()
    if a.dur > MAX_CLIP_S:
        raise SystemExit(f"--dur {a.dur}s is over {MAX_CLIP_S}s: use a few seconds and let the narration carry the beat")

    assets = a.ep_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    mp4, last = assets / f"{a.name}.mp4", assets / f"{a.name}_last.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a.start), "-i", str(a.source), "-t", str(a.dur), "-an",
                    "-vf", "scale='min(3840,iw)':-2,fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14",
                    "-movflags", "+faststart", str(mp4)], check=True)
    found = short_shots(mp4)
    if found and not a.allow_short_shots:
        s, e = longest_shot(mp4)
        mp4.unlink()
        raise SystemExit("\n".join([f"  SHORT SHOT {x:.2f}-{y:.2f}s ({y - x:.2f}s) in the cut" for x, y in found] + [
            f"refused: re-cut inside one shot, e.g. --start {a.start + s + 0.05:.2f} --dur {e - s - 0.1:.2f}",
            '(or --allow-short-shots "<why>" when the fast cut is the point)']))
    if found:
        print(f"  keeping {len(found)} short shot(s): {a.allow_short_shots}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.1", "-i", str(mp4), "-frames:v", "1", "-update", "1", str(last)], check=True)
    meta = {**probe(mp4), "highlight": None, "last": f"assets/{last.name}", "credit": a.credit, "url": a.url}
    (assets / f"{a.name}.json").write_text(json.dumps(meta, indent=2))
    print(f"{mp4} ({meta['duration']:.1f}s, {meta['width']}x{meta['height']}) credit '{a.credit}'")


if __name__ == "__main__":
    main()
