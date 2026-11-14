#!/usr/bin/env python3
"""Measure how varied (vs monotone) a narration sounds.

Usage: python tools/prosody.py <audio>... [--words N]

Metrics, following the charisma/narration literature (Niebuhr et al.; Rosenberg & Hirschberg):
  f0_med      median pitch (Hz)
  f0_sd_st    pitch variability: SD of F0 in semitones around the median   (higher = less monotone)
  f0_rng_st   pitch range: 5th-95th percentile spread in semitones           (higher = more melodic)
  int_sd_db   loudness variability (dB SD over voiced frames)                (emphasis)
  pause_pct   share of time in pauses >= 150 ms
  phr_f0_sd   phrase-level variety: SD of each phrase's median pitch (st)    (higher = less predictable)
  phr_db_sd   phrase-level loudness variety (dB)
  wpm         words per minute (needs --words)
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import parselmouth


def to_wav(path: str) -> str:
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", "22050", out], check=True)
    return out


def measure(path: str, words: int | None) -> dict:
    snd = parselmouth.Sound(to_wav(path))
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=65, pitch_ceiling=450)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    med = float(np.median(voiced))
    st = 12 * np.log2(voiced / med)
    inten = snd.to_intensity(time_step=0.01)
    db = inten.values[0]
    loud = db > (np.percentile(db, 95) - 25)  # speech frames
    # pauses: runs of quiet frames at least 150 ms long
    pause, run = 0, 0
    for q in ~loud:
        run = run + 1 if q else 0
        if run == 15:
            pause += 15
        elif run > 15:
            pause += 1
    dur = snd.duration
    # Phrase-level variety: split speech at pauses, then measure how much each phrase's pitch level and
    # loudness differ from the others. A voice that repeats one sentence melody scores low here even if
    # its within-phrase pitch range is wide (the "predictable TTS cadence" problem).
    f0_frames = np.interp(np.arange(len(db)) * 0.01, pitch.xs(), f0)
    phrases, start, quiet = [], None, 0
    for i, is_loud in enumerate(loud):
        if is_loud:
            start = i if start is None else start
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= 15:
                phrases.append((start, i - quiet))
                start, quiet = None, 0
    if start is not None:
        phrases.append((start, len(db)))
    lv, ld = [], []
    for a, b in phrases:
        seg = f0_frames[a:b]
        seg = seg[seg > 0]
        if len(seg) > 20:
            lv.append(np.median(12 * np.log2(seg / med)))
            ld.append(np.mean(db[a:b]))
    return {
        "f0_med": round(med),
        "f0_sd_st": round(float(np.std(st)), 2),
        "f0_rng_st": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 1),
        "int_sd_db": round(float(np.std(db[loud])), 1),
        "pause_pct": round(100 * pause / len(db), 1),
        "phrases": len(lv),
        "phr_f0_sd": round(float(np.std(lv)), 2) if len(lv) > 1 else None,
        "phr_db_sd": round(float(np.std(ld)), 2) if len(ld) > 1 else None,
        "wpm": round(words / dur * 60) if words else None,
        "dur_s": round(dur, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--words", type=int)
    a = ap.parse_args()
    rows = [(Path(f).stem, measure(f, a.words)) for f in a.files]
    rows.sort(key=lambda r: -r[1]["f0_sd_st"])
    keys = list(rows[0][1])
    print(f"{'voice':44}" + "".join(f"{k:>11}" for k in keys))
    for name, m in rows:
        print(f"{name[:44]:44}" + "".join(f"{str(m[k]):>11}" for k in keys))
