#!/usr/bin/env python3
"""Render an episode's thumbnail variants (episode.json "thumbnails") for YouTube Test & Compare.

Usage: python3 tools/thumbnails.py episodes/<episode-dir> [design ...]

Each variant is rendered with the Remotion "Thumbnail" still (video/src/Thumbnail.tsx) in every requested design
(default: the episode's own design) to out/thumb_<n>_<design>.png at 1280x720. Also writes
out/thumbs_preview.png: each thumbnail at full size next to the 168x94 size YouTube uses in the sidebar, to check
the words are still readable there (docs/thumbnails.md).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(ep_dir: Path, designs: list[str]) -> None:
    ep_dir = ep_dir.resolve()  # remotion runs from video/, so paths must be absolute
    ep = json.loads((ep_dir / "episode.json").read_text())
    variants = ep.get("thumbnails") or sys.exit("episode.json has no \"thumbnails\" list")
    designs = designs or [ep.get("design", "charcoal")]
    out = ep_dir / "out"
    out.mkdir(exist_ok=True)
    made = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, v in enumerate(variants, 1):
            for d in designs:
                props = Path(tmp) / f"{n}_{d}.json"
                props.write_text(json.dumps({**v, "design": d}))
                png = out / f"thumb_{n}_{d}.png"
                subprocess.run(["npx", "remotion", "still", "src/index.ts", "Thumbnail", str(png), f"--props={props}",
                                f"--public-dir={ep_dir}"], cwd=ROOT / "video", check=True, capture_output=True)
                made.append(png)
                print(png.name)
    # Preview: full size (scaled to 640) beside the 168x94 sidebar size, one row per thumbnail.
    inputs = [a for p in made for a in ("-i", str(p))]
    rows = "".join(f"[{i}]split[f{i}][s{i}];[f{i}]scale=640:360[F{i}];[s{i}]scale=168:94,pad=200:360:16:133:white[S{i}];"
                   f"[F{i}][S{i}]hstack[r{i}];" for i in range(len(made)))
    graph = rows + "".join(f"[r{i}]" for i in range(len(made))) + f"vstack={len(made)}" if len(made) > 1 else rows + "[r0]null"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, str(out / "thumbs_preview.png")], check=True)
    print("preview:", out / "thumbs_preview.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
