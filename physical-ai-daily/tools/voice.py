#!/usr/bin/env python3
"""Synthesize narration for each segment of an episode and write Remotion props.

Usage: .venv/bin/python tools/voice.py episodes/<episode-dir>

Reads <episode-dir>/episode.json, writes audio/<segment>_<beat>.wav and props.json (episode + per-beat
audio path, duration, word timings for captions, and inlined visual metadata).

Direction (docs/voice-research.md): a beat may carry "direct": [[style, text, pause_after_s], ...]. Each phrase is
voiced separately with that style's rate/pitch, trimmed to its words, and joined with the given pause. The phrase
texts must join back into the narration exactly. Beats without "direct" are voiced as one "neutral" phrase.
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

import edge_tts

TICKS_PER_SECOND = 10_000_000  # edge-tts reports offsets in 100ns units
TAIL_PAD = 0.6  # seconds of breathing room at the end of a segment
BEAT_GAP = 0.25  # pause between beats inside a segment
HOLD_CAP_S = 7  # longest a single static visual should hold (matches HOLD_CAP_S in video/src/scenes.tsx)
STATIC_TYPES = {"photo", "shot", "kinetic"}
SR = 24000  # edge-tts output rate
LEAD, TAIL, LAST_TAIL = 0.04, 0.12, 0.3  # seconds kept around each phrase's words when trimming
# Per delivery style: (rate offset %, pitch offset %) relative to "neutral". Shapes come from the voicelab test
# (voicelab/gen_directed.py): emphasis = slower (duration is a main prominence cue), punchlines also lower,
# questions higher. STRENGTH scales every offset; PAUSE_SCALE scales the directed pauses.
BASE_RATE = -4
VOICE_F0 = 112  # Hz, measured median pitch of en-US-AndrewNeural (tools/prosody.py); edge-tts only takes pitch in Hz
STRENGTH = float(os.environ.get("VOICE_STRENGTH", 3))
PAUSE_SCALE = float(os.environ.get("PAUSE_SCALE", 1.5))
STYLES = {
    "neutral": (0, 0),
    "hook": (10, 3),     # openers, payoffs said with energy
    "explain": (-4, -1), # how-it-works sentences
    "build": (-8, -2),   # tension before a reveal
    "lift": (-4, 4),     # emphasis: slower and higher
    "punch": (-10, -4),  # punchline: slower and lower
    "question": (0, 5),
    "caveat": (-6, -3),
    "aside": (8, -2),    # throwaway clauses, said lighter
}


def prosody_args(style: str) -> tuple[str, str]:
    rate, pitch = STYLES[style]
    return f"{BASE_RATE + STRENGTH * rate:+.0f}%", f"{STRENGTH * pitch * VOICE_F0 / 100:+.0f}Hz"


async def synth(text: str, voice: str, out: Path, rate: str = "+0%", pitch: str = "+0Hz", attempts: int = 10) -> list[dict]:
    # The edge-tts service intermittently returns no audio; retry with backoff.
    for i in range(attempts):
        try:
            return await synth_once(text, voice, out, rate, pitch)
        except edge_tts.exceptions.NoAudioReceived:
            if i == attempts - 1:
                raise
            await asyncio.sleep(3 + 2 * i)


async def synth_once(text: str, voice: str, out: Path, rate: str, pitch: str) -> list[dict]:
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, boundary="WordBoundary")
    words, audio = [], bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            words.append({
                "text": chunk["text"],
                "start": chunk["offset"] / TICKS_PER_SECOND,
                "end": (chunk["offset"] + chunk["duration"]) / TICKS_PER_SECOND,
            })
    out.write_bytes(audio)
    return words


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


async def voice_beat(beat: dict, voice: str, out: Path) -> list[dict]:
    """Voice each directed phrase, trim it to its words, join with the directed pauses; return shifted word timings."""
    phrases = beat.get("direct") or [["neutral", beat["narration"], 0]]
    joined = " ".join(t for _, t, _ in phrases)
    if joined.split() != beat["narration"].split():
        raise ValueError(f"direct does not match narration:\n  {joined}\n  {beat['narration']}")
    pcm, words, t = [], [], 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for k, (style, text, pause) in enumerate(phrases):
            rate, pitch = prosody_args(style)
            mp3 = Path(tmp) / f"{k}.mp3"
            ws = await synth(text, voice, mp3, rate, pitch)
            audio = decode(mp3)
            last = k == len(phrases) - 1
            a = max(0.0, ws[0]["start"] - LEAD)
            b = min(len(audio) / SR, ws[-1]["end"] + (LAST_TAIL if last else TAIL))
            pcm.append(audio[int(a * SR):int(b * SR)])
            words += [{**w, "start": round(w["start"] - a + t, 3), "end": round(w["end"] - a + t, 3)} for w in ws]
            t += b - a
            if not last:
                pcm.append(np.zeros(int(pause * PAUSE_SCALE * SR), dtype=np.int16))
                t += pause * PAUSE_SCALE
    with wave.open(str(out), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(np.concatenate(pcm).tobytes())
    return words


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def restore_punctuation(words: list[dict], narration: str) -> list[dict]:
    """Boundary events drop punctuation; walk the script tokens in step to put it back for captions."""
    tokens, j, rest = narration.split(), 0, None
    for w in words:
        if j >= len(tokens):
            break
        rest = norm(tokens[j]) if rest is None else rest
        nw = norm(w["text"])
        if nw and rest.startswith(nw):
            rest = rest[len(nw):]
            if not rest:  # token fully consumed: show it as written, punctuation included
                w["text"] = tokens[j] if norm(tokens[j]) == nw else w["text"] + re.sub(r"^.*?([^\w]*)$", r"\1", tokens[j])
                j, rest = j + 1, None
    return words


def media_meta(ep_dir: Path, src: str) -> dict:
    """Capture metadata (size, highlight box) for an asset, so the compositor can frame it."""
    return {"src": src, **json.loads((ep_dir / src).with_suffix(".json").read_text())}


def inline_media(ep_dir: Path, visual: dict) -> dict:
    for key in ("src", "side"):
        if key in visual:
            visual[key] = media_meta(ep_dir, visual[key])
    for item in visual.get("items", []):
        if "src" in item:
            item.update(media_meta(ep_dir, item["src"]))
    return visual


async def main(ep_dir: Path) -> None:
    ep = json.loads((ep_dir / "episode.json").read_text())
    (ep_dir / "audio").mkdir(exist_ok=True)
    for seg in ep["segments"]:
        for i, beat in enumerate(seg["beats"]):
            out = ep_dir / "audio" / f"{seg['id']}_{i:02d}.wav"
            beat["words"] = restore_punctuation(await voice_beat(beat, ep["voice"], out), beat["narration"])
            beat["audio"] = f"audio/{out.name}"
            last = i == len(seg["beats"]) - 1
            beat["duration"] = round(duration(out) + (TAIL_PAD if last else BEAT_GAP), 3)
            inline_media(ep_dir, beat["visual"])
        seg["duration"] = round(sum(b["duration"] for b in seg["beats"]), 3)
        for i, b in enumerate(seg["beats"]):
            # Static visuals have no word-cued reveals; the compositor punch-cuts them past 7 s, but a new
            # visual reads better, so flag them for a script split.
            if b["visual"]["type"] in STATIC_TYPES and b["duration"] > HOLD_CAP_S:
                print(f"  WARN {seg['id']}[{i}] {b['visual']['type']} holds {b['duration']:.1f}s > {HOLD_CAP_S}s: consider splitting")
        print(f"{seg['id']}: {seg['duration']:.1f}s, {len(seg['beats'])} beats")
    (ep_dir / "props.json").write_text(json.dumps(ep, indent=2, ensure_ascii=False))
    print(f"total {sum(s['duration'] for s in ep['segments']):.1f}s -> {ep_dir / 'props.json'}")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
