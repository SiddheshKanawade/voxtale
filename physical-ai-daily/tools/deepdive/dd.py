#!/usr/bin/env python3
"""Deep-dive helpers for the render scripts (the visual types are documented in video/src/deepdive/DeepDive.tsx).

Usage:
  python3 tools/deepdive/dd.py check <dd-dir>             after voice.py: cues that aren't in their beat's words, chapter
                                                         cards over 4 s, unknown node ids, regions off the image...
  python3 tools/deepdive/dd.py frames <dd-dir> <spec>...  frame numbers for stills: N, or <seg>:<beat>[@frac] (beat is
                                                         0-based; frac of the way into it, default 0.95: all revealed)
  python3 tools/deepdive/dd.py total <dd-dir>             the composition's length in frames
  python3 tools/deepdive/dd.py shots <dd-dir>             writes <dd-dir>/qa/shots.json for the video critic: one entry per
                                                         beat [{id: c<seg>b<beat>, t0, t1, narration, visual, chapter,
                                                         pause}] plus the sting, timed exactly as the composition

check exits 1 on errors (they render wrong) and prints warnings otherwise. The timing mirrors DeepDive.tsx: the
digest's timeline() with the sting after the first "open" segment and the end tail after the last, on beat durations
that include each beat's "pause" (silence after its narration).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

FPS, STING_S, END_S = 30, 1.6, 1.6  # video/src/brand.tsx
CHAPTER_CARD_MAX_S = 4
HOLD_S = 1.0  # every reveal lands by the beat's end - 1 s (kit.tsx cuer clamps later ones)
EARLY_S = 1.5  # a first element is on screen from the cut: warn when its word comes later than this
QUOTE_MAX_WORDS = 25
STAGE_W, STAGE_H = 1680, 660


def jsround(x: float) -> int:  # Math.round: halves up (Python's round() goes to even)
    return math.floor(x + 0.5)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def props(d: Path) -> dict:
    p = d / "props.json"
    if not p.exists():
        sys.exit(f"{p} missing: run tools/voice.py {d} first")
    return json.loads(p.read_text())


def on_screen(b: dict) -> float:
    """A beat's time on screen: its narration plus its pause."""
    return b["duration"] + b.get("pause", 0)


def starts(segs: list[dict]) -> tuple[list[float], float]:
    t, sting, out = 0.0, False, []
    for i, s in enumerate(segs):
        out.append(t)
        t += sum(on_screen(b) for b in s["beats"])
        if s["kind"] == "open" and not sting and i < len(segs) - 1:
            sting, t = True, t + STING_S
    return out, t + END_S


def beat_frames(segs: list[dict]) -> dict[tuple[str, int], tuple[int, int]]:
    """(segment id, beat index) -> (first frame, length), rounded as DeepDive.tsx rounds them."""
    st, _ = starts(segs)
    out = {}
    for s, t in zip(segs, st):
        f = jsround(t * FPS)
        for i, b in enumerate(s["beats"]):
            n = jsround(on_screen(b) * FPS)
            out[(s["id"], i)] = (f, n)
            f += n
    return out


def cue_time(words: list[dict], cue, beat: dict | None = None) -> float | None:
    """Seconds into the beat, as kit.tsx cuer: a number, "word", "word#n" (nth word starting with it), "pause" (the
    pause's last second) or "pause+X"."""
    if isinstance(cue, (int, float)):
        return float(cue)
    pm = re.match(r"^pause(?:\+([\d.]+))?$", cue)
    if pm:
        if beat is None:
            return None
        return beat["duration"] + (float(pm.group(1)) if pm.group(1) else max(0.0, beat.get("pause", 0) - HOLD_S))
    m = re.match(r"^(.+)#(\d+)$", cue)
    stem, nth = (m.group(1), int(m.group(2))) if m else (cue, 1)
    hits = [w for w in words if norm(w["text"]).startswith(norm(stem))]
    return hits[nth - 1]["start"] if len(hits) >= nth else None


def cues(v: dict) -> list[tuple[str, object]]:
    """Every (where, cue) a visual waits for."""
    out = []
    add = lambda where, c: c is not None and out.append((where, c))
    for k in ("cue", "questionCue", "noteCue", "revealCue", "markCue", "cycleCue", "centerCue", "badgeCue", "split", "outcomeCue"):
        add(k, v.get(k))
    for key in ("nodes", "edges", "focus", "regions", "options", "items", "bars"):
        for i, x in enumerate(v.get(key) or []):
            if isinstance(x, dict):
                for k in ("cue", "reveal", "outcomeCue"):
                    add(f"{key}[{i}].{k}", x.get(k))
                if isinstance(x.get("check"), dict):
                    add(f"{key}[{i}].check", x["check"].get("cue"))
    for key in ("compare", "back", "baseline", "freeze", "check"):
        if isinstance(v.get(key), dict):
            add(key, v[key].get("cue"))
    return out


def fit_h(w: float, label: str, sub: str | None, icon: str | None) -> float:
    """A build box's height for its text, as fitH in video/src/deepdive/diagrams.tsx (38 px label, 28 px sub)."""
    tw = (w - 48 - (74 if icon else 0)) * 0.92
    lines = lambda s, px: max(1, math.ceil(len(s) * px / tw))
    return max(100, 46 + 42 * lines(label, 38 * 0.545) + (6 + 34 * lines(sub, 28 * 0.525) if sub else 0))


def overlap(a: dict, b: dict) -> bool:
    return abs(a["x"] - b["x"]) < (a["w"] + b["w"]) / 2 and abs(a["y"] - b["y"]) < (a["h"] + b["h"]) / 2


def check(d: Path) -> int:
    p = props(d)
    errors, warns = [], []
    chapters = [s for s in p["segments"] if s["kind"] == "chapter"]
    if p["segments"] and p["segments"][0]["kind"] != "open":
        warns.append("first segment isn't kind 'open': no sting after a cold open")
    for s in p["segments"]:
        if s["kind"] not in ("open", "chapter", "outro"):
            errors.append(f"{s['id']}: kind {s['kind']!r} (open | chapter | outro)")
        if s["kind"] == "chapter" and "chapter" not in s:
            warns.append(f"{s['id']}: chapter segment without a 'chapter' number (header and card show none)")
        for i, b in enumerate(s["beats"]):
            v, words, at = b["visual"], b.get("words", []), f"{s['id']}[{i}] {b['visual'].get('type')}"
            end = on_screen(b)
            for where, c in cues(v):
                ct = cue_time(words, c, b)
                if ct is None:
                    errors.append(f"{at} {where}: cue {c!r} not in the narration: {b['narration'][:70]!r}")
                elif ct > end - HOLD_S:
                    warns.append(f"{at} {where}: cue {c!r} at {ct:.1f}s, within {HOLD_S:.0f} s of the beat's end ({end:.1f}s):"
                                 f" it lands at {max(0, end - HOLD_S):.1f}s instead (hold rule)")
            # the first content element is on screen from the cut, whatever its cue (DeepDive.tsx "Cuts")
            first = {"build": "nodes", "loop": "nodes", "steps": "nodes", "timeline": "items", "bars": "bars", "grid": "items"}.get(v.get("type"))
            if first and not v.get("cont"):
                ts = [cue_time(words, x["cue"], b) for x in v.get(first, []) if isinstance(x, dict) and x.get("cue") is not None]
                ts = [t for t in ts if t is not None]
                if ts and min(ts) > EARLY_S:
                    warns.append(f"{at}: its first element shows from the cut, {min(ts):.1f}s before its word: open the beat on it")
            if v.get("type") == "number" and v.get("cue") is not None and (vt := cue_time(words, v["cue"], b)) and vt > EARLY_S:
                warns.append(f"{at}: the number shows from the cut, {vt:.1f}s before its word: open the beat on it")
            if v.get("type") == "predict" and not b.get("pause"):
                warns.append(f"{at}: no pause: give the beat a 2-3 s \"pause\" and revealCue \"pause\" so the viewer can guess")
            t = v.get("type")
            if t == "chapter" and b["duration"] > CHAPTER_CARD_MAX_S:
                warns.append(f"{at}: chapter card holds {b['duration']:.1f}s > {CHAPTER_CARD_MAX_S}s: shorten its narration")
            if t == "build":
                ids = {n["id"] for n in v.get("nodes", [])}
                boxes = {n["id"]: {"x": n["x"], "y": n["y"], "w": n.get("w", 400),
                                   "h": n.get("h") or fit_h(n.get("w", 400), n["label"], n.get("sub"), n.get("icon"))}
                         for n in v.get("nodes", [])}
                eids = {e.get("id", f"{e['from']}>{e['to']}") for e in v.get("edges", [])}
                shown = {n["id"]: cue_time(words, n["cue"], b) if n.get("cue") is not None else None for n in v.get("nodes", [])}
                for e in v.get("edges", []):
                    for end in (e["from"], e["to"]):
                        if end not in ids:
                            errors.append(f"{at}: edge {e['from']}>{e['to']}: no node {end!r}")
                        elif e.get("cue") is not None and None not in (ct := cue_time(words, e["cue"], b), shown[end]) and ct < shown[end]:
                            warns.append(f"{at}: edge {e['from']}>{e['to']} cue {e['cue']!r} comes before node {end!r}: the arrow waits for it")
                for f in v.get("focus", []):
                    for x in f.get("ids", []):
                        if x not in ids | eids:
                            errors.append(f"{at}: focus id {x!r} is neither a node nor an edge ('from>to')")
                ks = list(boxes)
                if ks:  # the auto-fit centres the layout; only one bigger than the stage shrinks text below 38/28 px
                    bw = max(b["x"] + b["w"] / 2 for b in boxes.values()) - min(b["x"] - b["w"] / 2 for b in boxes.values())
                    bh = max(b["y"] + b["h"] / 2 for b in boxes.values()) - min(b["y"] - b["h"] / 2 for b in boxes.values())
                    if bw > STAGE_W or bh > STAGE_H:
                        warns.append(f"{at}: boxes span {bw:.0f} x {bh:.0f}, more than the stage ({STAGE_W} x {STAGE_H}): the fit shrinks the text")
                for j, a in enumerate(ks):
                    bx = boxes[a]
                    for c in ks[j + 1:]:
                        if overlap(bx, boxes[c]):
                            errors.append(f"{at}: nodes {a!r} and {c!r} overlap")
                        elif overlap({**bx, "w": bx["w"] + 40, "h": bx["h"] + 40}, boxes[c]):
                            warns.append(f"{at}: nodes {a!r} and {c!r} are under 20 px apart (box heights fit 38/28 px text): space them out")
            if t == "figure":
                m = v.get("src", {})
                for j, r in enumerate(v.get("regions", [])):
                    if r["x"] < 0 or r["y"] < 0 or r["x"] + r["w"] > m.get("width", 1e9) + 1 or r["y"] + r["h"] > m.get("height", 1e9) + 1:
                        errors.append(f"{at}: regions[{j}] leaves the image ({m.get('width')}x{m.get('height')}, the .json's space)")
            if t == "number":
                if not re.search(r"\d", str(v.get("value", ""))):
                    warns.append(f"{at}: value {v.get('value')!r} has no digits (numbers as digits on screen)")
            if t == "predict":
                n = len(v.get("options", []))
                if not 2 <= n <= 3:
                    warns.append(f"{at}: {n} options (2-3)")
                # answer null: the question beat stays unanswered; a later beat (cont) ticks it on its own word
                if v.get("answer") is not None and (not isinstance(v["answer"], int) or not 0 <= v["answer"] < n):
                    errors.append(f"{at}: answer {v.get('answer')!r} isn't an option index")
                if v.get("answer") is not None and "revealCue" not in v:
                    warns.append(f"{at}: no revealCue: the answer shows at 80% of the beat")
            if t in ("pair", "video", "grid"):
                sides = v.get("items", []) if t != "video" else [{**(v.get("src") or {}), **v}]
                for j, it in enumerate(sides):
                    w, h, cr = it.get("width", 1e9), it.get("height", 1e9), it.get("crop")
                    if cr and (cr["x"] < 0 or cr["y"] < 0 or cr["x"] + cr["w"] > w + 1 or cr["y"] + cr["h"] > h + 1):
                        errors.append(f"{at}: side {j} crop leaves the clip ({w}x{h}, clip pixels)")
                    if it.get("from") and it.get("duration") and it["from"] >= it["duration"]:
                        errors.append(f"{at}: side {j} from={it['from']} is past the clip's end ({it['duration']}s)")
                for m in (v.get("freeze") or {}).get("marks", []):
                    side = sides[m.get("side", 0)] if m.get("side", 0) < len(sides) else {}
                    if not (0 <= m["x"] <= side.get("width", 1e9) and 0 <= m["y"] <= side.get("height", 1e9)):
                        errors.append(f"{at}: freeze mark at {m['x']},{m['y']} is off side {m.get('side', 0)}'s clip")
            if t == "pair":
                if len(v.get("items", [])) != 2:
                    errors.append(f"{at}: pair needs 2 items")
                if v.get("mode") != "sequence":
                    for j, it in enumerate(v.get("items", [])):
                        if "cue" in it:
                            warns.append(f"{at}: items[{j}].cue is ignored in a split pair (both sides show from the start): use "
                                         "reveal to hold a side back, focus to light it")
            if t == "grid" and not 2 <= len(v.get("items", [])) <= 6:
                errors.append(f"{at}: grid takes 2-6 items (2x2 or 2x3)")
            if t == "bars":
                bars = v.get("bars", [])
                if not 1 <= len(bars) <= 8:
                    warns.append(f"{at}: {len(bars)} bars (1-8)")
                for j, br in enumerate(bars):
                    if br.get("value") is None and not br.get("range"):
                        errors.append(f"{at}: bars[{j}] has neither value nor range")
                    if br.get("value") is not None and br.get("display") and not re.search(r"\d", br["display"]):
                        warns.append(f"{at}: bars[{j}] display {br['display']!r} has no digits")
            if t == "quote":
                nw = len(str(v.get("text", "")).split())
                if nw > QUOTE_MAX_WORDS:
                    warns.append(f"{at}: quote is {nw} words (<= {QUOTE_MAX_WORDS})")
                if v.get("mark") and v["mark"] not in v.get("text", ""):
                    errors.append(f"{at}: mark {v['mark']!r} isn't part of the quote text")
            if t == "timeline" and not 3 <= len(v.get("items", [])) <= 6:
                warns.append(f"{at}: {len(v.get('items', []))} items (3-6)")
            if t in ("loop", "steps") and not 2 <= len(v.get("nodes", [])) <= 6:
                warns.append(f"{at}: {len(v.get('nodes', []))} nodes (3-6)")
    for w in warns:
        print(f"  WARN {w}")
    for e in errors:
        print(f"  ERROR {e}")
    _, total = starts(p["segments"])
    print(f"check: {len(chapters)} chapters, {total / 60:.1f} min, {len(errors)} errors, {len(warns)} warnings")
    return 1 if errors else 0


def frames(d: Path, specs: list[str]) -> None:
    p = props(d)
    bf = beat_frames(p["segments"])
    out = []
    for spec in specs:
        if spec.isdigit():
            out.append(int(spec))
            continue
        m = re.match(r"^([^:]+):(\d+)(?:@([\d.]+))?$", spec)
        if not m or (m.group(1), int(m.group(2))) not in bf:
            sys.exit(f"bad frame spec {spec!r}: N or <segment id>:<beat index>[@fraction]; beats: "
                     + ", ".join(f"{s}:{i}" for s, i in bf))
        f0, n = bf[(m.group(1), int(m.group(2)))]
        out.append(f0 + min(n - 1, math.floor(n * float(m.group(3) or 0.95))))
    print(" ".join(map(str, out)))


def shots(d: Path) -> None:
    """The cut list as the composition plays it, for the video critic's packet (qa/shots.json)."""
    p = props(d)
    segs = p["segments"]
    bf = beat_frames(segs)
    st, total = starts(segs)
    out = []
    for k, s in enumerate(segs):
        for i, b in enumerate(s["beats"]):
            f0, n = bf[(s["id"], i)]
            out.append({"id": f"c{k}b{i:02d}", "t0": round(f0 / FPS, 3), "t1": round((f0 + n) / FPS, 3), "narration": b["narration"],
                        "visual": b["visual"].get("type"), "chapter": s["headline"], "pause": b.get("pause", 0)})
        if s["kind"] == "open" and k < len(segs) - 1 and not any(x["id"] == "sting" for x in out):
            t0 = out[-1]["t1"]
            out.append({"id": "sting", "t0": t0, "t1": round(jsround(st[k + 1] * FPS) / FPS, 3), "narration": "", "visual": "sting",
                        "chapter": "", "pause": 0})
    out[-1]["t1"] = round(math.ceil(total * FPS) / FPS, 3)  # the last shot holds through the end tail (sonic logo)
    (d / "qa").mkdir(exist_ok=True)
    (d / "qa" / "shots.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"shots: {len(out)} -> {d / 'qa' / 'shots.json'} ({out[-1]['t1']:.2f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, d = sys.argv[1], Path(sys.argv[2])
    if cmd == "check":
        sys.exit(check(d))
    elif cmd == "frames":
        frames(d, sys.argv[3:])
    elif cmd == "total":
        print(math.ceil(starts(props(d)["segments"])[1] * FPS))
    elif cmd == "shots":
        shots(d)
    else:
        sys.exit(__doc__)
