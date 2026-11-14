"""Loudness-match samples to -14 LUFS (two-pass loudnorm, then a gain trim and limiter) into voicelab/leveled/.

Usage: python3 voicelab/level.py voicelab/samples/<name>.mp3 [...]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

TARGET = -14.0
OUT = Path(__file__).parent / "leveled"


def ff(*args):
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args], capture_output=True, text=True).stderr


def lufs(path):
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", ff("-i", str(path), "-af", "ebur128", "-f", "null", "-"))[-1])


for src in map(Path, sys.argv[1:]):
    log = ff("-i", str(src), "-af", f"loudnorm=I={TARGET}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-")
    m = json.loads(log[log.rindex("{"):log.rindex("}") + 1])
    tmp, dst = OUT / f"_{src.stem}.wav", OUT / f"{src.stem}.wav"
    ff("-y", "-i", str(src), "-af", f"loudnorm=I={TARGET}:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
       f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,aresample=24000",
       "-ac", "1", str(tmp))
    ff("-y", "-i", str(tmp), "-af", f"volume={TARGET - lufs(tmp):.2f}dB,alimiter=limit=0.89:level=false", str(dst))
    tmp.unlink()
    print(f"{dst.name}: {lufs(dst):.1f} LUFS")
