#!/usr/bin/env python3
"""Crop a region of an image (fractions of width/height) and write the capture metadata json.

Usage: python3 tools/crop.py <src.png> <out.png> <x0> <y0> <x1> <y1>
"""
import json
import sys
from pathlib import Path

from PIL import Image

src, out, *box = sys.argv[1:]
x0, y0, x1, y1 = map(float, box)
im = Image.open(src).convert("RGB")
w, h = im.size
crop = im.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
crop.save(out)
Path(out).with_suffix(".json").write_text(json.dumps({"width": crop.width, "height": crop.height, "highlight": None}))
print(f"OK {out} {crop.size}")
