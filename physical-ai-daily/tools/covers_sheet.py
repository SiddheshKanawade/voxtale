#!/usr/bin/env python3
"""Lettered contact sheet of a Short's covers: <short>/out/covers-sheet.png from out/cover-<A..>-<design>.png.

Usage: python3 tools/covers_sheet.py <episode>/shorts/<slug> [--design dark]

The user picks covers by letter, so every Short gets one sheet (promo-shorts skill, "Wide options"). This ffmpeg has
no drawtext filter, so the letters are drawn with Pillow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TW, TH, GAP, TOP = 495, 880, 8, 48  # one cover at about a quarter of 1080x1920


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("short", type=Path)
    ap.add_argument("--design", default="dark")
    a = ap.parse_args()
    covers = sorted((a.short / "out").glob(f"cover-?-{a.design}.png"))
    if not covers:
        raise SystemExit(f"no out/cover-?-{a.design}.png in {a.short}")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except OSError:
        font = ImageFont.load_default()
    sheet = Image.new("RGB", (len(covers) * (TW + GAP) + GAP, TH + TOP + GAP), "white")
    draw = ImageDraw.Draw(sheet)
    for i, c in enumerate(covers):
        x = GAP + i * (TW + GAP)
        sheet.paste(Image.open(c).convert("RGB").resize((TW, TH), Image.LANCZOS), (x, TOP))
        draw.text((x + 4, 2), c.name.split("-")[1], fill="black", font=font)
    out = a.short / "out" / "covers-sheet.png"
    sheet.save(out)
    print(f"{out}: {', '.join(c.name.split('-')[1] for c in covers)}")


if __name__ == "__main__":
    main()
