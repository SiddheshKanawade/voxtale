#!/usr/bin/env python3
"""Synthesize the channel's sonic identity (original, generated here, so no licensing questions).

Usage: python3 tools/brand_audio.py <out-dir>
Writes:
  sonic_logo.wav  1.6 s, timed to the sting in video/src/brand.tsx: four "pixel" blips rising up an A-major
                  arpeggio (one per pixel pop), a short rising whoosh as the arrow drives in, then a soft chord.
Brand research behind it: docs/brand.md.
"""
import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100


def note(f: float, dur: float, decay: float, harm: float = 0.25) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1, t / 0.005) * np.exp(-t / decay)
    return env * (np.sin(2 * np.pi * f * t) + harm * np.sin(2 * np.pi * 2 * f * t) + 0.5 * harm * np.sin(2 * np.pi * 3 * f * t))


def place(buf: np.ndarray, sig: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(at * SR)
    n = min(len(sig), len(buf) - i)
    buf[i:i + n] += gain * sig[:n]


def whoosh(dur: float) -> np.ndarray:
    rng = np.random.default_rng(7)  # fixed seed: the identity must sound identical in every video
    n = int(SR * dur)
    noise = rng.standard_normal(n)
    out, y = np.zeros(n), 0.0
    for k in range(n):  # one-pole low-pass whose cutoff sweeps up: a rising "air" sound
        a = 0.02 + 0.25 * (k / n) ** 2
        y += a * (noise[k] - y)
        out[k] = y
    return out * np.sin(np.pi * np.arange(n) / n) ** 2


def write(path: Path, x: np.ndarray, peak_db: float) -> None:
    x = x / np.abs(x).max() * 10 ** (peak_db / 20)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((x * 32767).astype(np.int16).tobytes())


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    A4 = 440.0
    logo = np.zeros(int(SR * 1.6))
    for i, semis in enumerate([0, 4, 7, 12]):  # A, C#, E, A: pixels pop at frames 2, 5, 8, 11 (30 fps)
        place(logo, note(A4 * 2 ** (semis / 12), 0.25, 0.06), (2 + 3 * i) / 30, 0.55)
    place(logo, whoosh(0.32), 0.45, 0.9)  # arrow slides in over frames 15-23
    for semis in [-12, -5, 0, 4, 7]:  # resolving A-major chord as the arrow lands
        place(logo, note(A4 * 2 ** (semis / 12), 0.85, 0.32, 0.15), 0.76, 0.32)
    write(out / "sonic_logo.wav", logo, -3)
    print("wrote", out / "sonic_logo.wav")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
