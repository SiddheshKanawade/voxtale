#!/usr/bin/env python3
"""Set an episode's thumbnail receipts from OCR: crop each receipt to the lines between two phrases and box a phrase.

Usage: python3 -I tools/thumb_receipts.py episodes/<ep> <spec.json>
Needs the OCR helper (macOS Vision): swiftc -O tools/ocr.swift -o tools/ocr   (the binary is git-ignored).

spec.json is the episode's "thumbnails" list (see the youtube-thumbnails skill). A receipt may give
{"src", "source", "from": "<text in its first line>", "to": "<text in its last line>", "mark": "<phrase to box>",
"pad": 0.012, "x0"/"x1": fixed left/right fractions, "span"/"shift": a narrower window centred on the mark}.
Lines from..to (inclusive) form the crop; the mark is located inside its OCR line by character position
(approximate) and boxed in red over the capture's own highlight. Writes the list into <ep>/episode.json.
"""
from __future__ import annotations

import json, subprocess, sys
from pathlib import Path
OCR = Path(__file__).with_name("ocr")
ep = Path(sys.argv[1]); spec = json.loads(Path(sys.argv[2]).read_text())

def lines(png):
    out = []
    for l in subprocess.run([str(OCR), str(png)], capture_output=True, text=True, check=True).stdout.splitlines():
        nums, text = l.split(" | ", 1); out.append((*map(float, nums.split()), text))
    return out

for t in spec:
    r = t.get("receipt")
    if not r or "from" not in r: continue
    L = lines(ep / r["src"])
    fr = r.pop("from"); to = r.pop("to"); i0 = next(i for i, l in enumerate(L) if fr in l[4])
    i1 = next(i for i, l in enumerate(L) if i >= i0 and to in l[4])
    sel = L[i0:i1 + 1]; pad = r.pop("pad", 0.012)
    x0 = r.pop("x0", min(l[0] for l in sel) - pad); x1 = r.pop("x1", max(l[0] + l[2] for l in sel) + pad)
    span = r.pop("span", None)  # a narrower window (fraction of page width) centred on the mark: bigger words, ragged edges
    if span and r.get("mark"):
        l = next(l for l in sel if r["mark"] in l[4]); n = len(l[4]); k = l[4].index(r["mark"])
        c = l[0] + l[2] * (k + len(r["mark"]) / 2) / n + r.pop("shift", 0)
        x0 = max(x0, c - span / 2); x1 = x0 + span
    y0 = sel[0][1] - pad * 1.6; y1 = sel[-1][1] + sel[-1][3] + pad * 1.6
    crop = dict(x=max(0, x0), y=max(0, y0), w=min(1, x1) - max(0, x0), h=y1 - y0)
    r["crop"] = {k: round(v, 4) for k, v in crop.items()}
    m = r.pop("mark", None)
    if m:
        l = next(l for l in sel if m in l[4]); n = len(l[4]); k = l[4].index(m)
        bx = l[0] + l[2] * k / n; bw = l[2] * len(m) / n
        r["mark"] = {"x": round((bx - crop["x"]) / crop["w"] - 0.006, 4), "y": round((l[1] - crop["y"]) / crop["h"] - 0.02, 4),
                     "w": round(bw / crop["w"] + 0.012, 4), "h": round(l[3] / crop["h"] + 0.04, 4)}
d = json.loads((ep / "episode.json").read_text()); d["thumbnails"] = spec
(ep / "episode.json").write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
for i, t in enumerate(spec, 1): print(i, t["lines"], t.get("receipt", {}).get("crop"), t.get("receipt", {}).get("mark"))
