#!/usr/bin/env python3
"""Synthesize narration for each segment of an episode and write Remotion props.

Usage: .venv/bin/python tools/voice.py episodes/<episode-dir>

Reads <episode-dir>/episode.json, writes audio/<segment>_<beat>.wav and props.json (episode + per-beat
audio path, duration, word timings for captions, and inlined visual metadata).

Direction (docs/voice-research.md): a beat may carry "direct": [[style, text, pause_after_s], ...]. Each phrase is
voiced separately with that style's rate/pitch, trimmed to its words, and joined with the given pause. The phrase
texts must join back into the narration exactly. Beats without "direct" are voiced as one "neutral" phrase.

ElevenLabs: with "tts": "elevenlabs" in episode.json, "voice" is an ElevenLabs voice id and "model" defaults to eleven_v3
(the blind-test winner, docs/voice-research.md section 8). Each beat is one call on its narration, so pauses come from
punctuation, not splices. Word timings come from the API's character timestamps. ELEVENLABS_API_KEY is read from the
environment or the project's .env. Audio is cached per beat in audio/, so re-runs don't spend characters.

Speed: an optional top-level "speed" (e.g. 1.175 for Shorts) time-stretches every beat after synthesis, with pitch
kept and word timings scaled. It costs no TTS characters.
"""
import base64
import hashlib
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

import edge_tts

TICKS_PER_SECOND = 10_000_000  # edge-tts reports offsets in 100ns units
TAIL_PAD = 0.6  # seconds of breathing room at the end of a segment
BEAT_GAP = 0.25  # pause between beats inside a segment
HOLD_CAP_S = 7  # longest a single static visual should hold (matches HOLD_CAP_S in video/src/scenes.tsx)
STATIC_TYPES = {"photo", "shot", "kinetic"}
CLIP_HOLD_S = 2  # a clip beat may outlast its footage by this much (the last frame holds); more reads as a freeze
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


async def synth(text: str, voice: str, out: Path, rate: str = "+0%", pitch: str = "+0Hz", attempts: int = 40) -> list[dict]:
    # The edge-tts service intermittently returns no audio or stalls (worse with several episodes voicing at once);
    # time out and retry with capped backoff.
    for i in range(attempts):
        try:
            return await asyncio.wait_for(synth_once(text, voice, out, rate, pitch), timeout=60)
        except (edge_tts.exceptions.NoAudioReceived, asyncio.TimeoutError):
            if i == attempts - 1:
                raise
            print(f"  retry {i + 1}: {text[:40]}", flush=True)
            await asyncio.sleep(min(4 + 2 * i, 20))


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


def api_key() -> str:
    if "ELEVENLABS_API_KEY" in os.environ:
        return os.environ["ELEVENLABS_API_KEY"]
    env = Path(__file__).resolve().parent.parent / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("ELEVENLABS_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("ELEVENLABS_API_KEY not set (put it in .env)")


def eleven_words(alignment: dict) -> list[dict]:
    """Group the API's per-character timings into words, skipping [audio tags]."""
    words, cur, depth = [], None, 0
    for ch, a, b in zip(alignment["characters"], alignment["character_start_times_seconds"],
                        alignment["character_end_times_seconds"]):
        depth += (ch == "[") - (ch == "]")
        if ch.isspace() or depth or ch == "]":
            cur = None
            continue
        if cur is None:
            cur = {"text": "", "start": a, "end": b}
            words.append(cur)
        cur["text"] += ch
        cur["end"] = b
    return words


def eleven_beat(beat: dict, voice: str, model: str, out: Path) -> list[dict]:
    """Voice a beat in one ElevenLabs call (cached by voice, model and text); trim to its words; return word timings."""
    text = beat.get("eleven_text", beat["narration"])  # optional variant with [audio tags]
    key = hashlib.sha1(f"{voice}|{model}|{text}".encode()).hexdigest()[:16]
    mp3, meta = out.with_suffix(".el.mp3"), out.with_suffix(".el.json")
    if not (meta.exists() and json.loads(meta.read_text())["key"] == key):
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps?output_format=mp3_44100_128",
            data=json.dumps({"text": text, "model_id": model}).encode(),
            headers={"xi-api-key": api_key(), "Content-Type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            sys.exit(f"ElevenLabs HTTP {e.code}: {e.read().decode()[:300]}")
        mp3.write_bytes(base64.b64decode(d["audio_base64"]))
        meta.write_text(json.dumps({"key": key, "alignment": d["alignment"]}))
    ws = eleven_words(json.loads(meta.read_text())["alignment"])
    audio = decode(mp3)
    a = max(0.0, ws[0]["start"] - LEAD)
    b = min(len(audio) / SR, ws[-1]["end"] + LAST_TAIL)
    with wave.open(str(out), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(audio[int(a * SR):int(b * SR)].tobytes())
    return [{**w, "start": round(w["start"] - a, 3), "end": round(w["end"] - a, 3)} for w in ws]


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


def speed_up(wav: Path, words: list[dict], factor: float) -> list[dict]:
    """Time-stretch a voiced beat by `factor` (pitch kept, ffmpeg atempo) and scale its word timings to match.
    Applied after synthesis, so it costs no TTS credits and the per-beat caches keep the original-speed audio."""
    if abs(factor - 1) < 1e-3:
        return words
    tmp = wav.with_suffix(".speed.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", f"atempo={factor}", str(tmp)], check=True)
    tmp.replace(wav)
    return [{**w, "start": round(w["start"] / factor, 3), "end": round(w["end"] / factor, 3)} for w in words]


def check_clip(seg_id: str, i: int, beat: dict) -> None:
    """A clip is always credited on screen (checklist HARD 7.2); the credit comes from tools/clip.py's metadata."""
    v = beat["visual"]
    if v["type"] != "clip":
        return
    v.setdefault("credit", v["src"].get("credit"))
    if not v.get("credit") or not v["src"].get("url"):
        raise SystemExit(f"{seg_id}[{i}] clip {v['src']['src']} has no credit or url: cut it with tools/clip.py")
    if beat["duration"] > v["src"]["duration"] + CLIP_HOLD_S:
        print(f"  WARN {seg_id}[{i}] clip is {v['src']['duration']:.1f}s, beat {beat['duration']:.1f}s: split the beat or cut a longer clip")


async def main(ep_dir: Path) -> None:
    ep = json.loads((ep_dir / "episode.json").read_text())
    (ep_dir / "audio").mkdir(exist_ok=True)
    for seg in ep["segments"]:
        for i, beat in enumerate(seg["beats"]):
            out = ep_dir / "audio" / f"{seg['id']}_{i:02d}.wav"
            if ep.get("tts") == "elevenlabs":
                words = eleven_beat(beat, ep["voice"], ep.get("model", "eleven_v3"), out)
            else:
                # Cache edge beats by voice and text, so a rerun only re-voices edited or failed beats.
                key = hashlib.sha1(f"{ep['voice']}|{json.dumps(beat.get('direct'))}|{beat['narration']}".encode()).hexdigest()[:16]
                cw, cj = ep_dir / "audio" / f".edge_{key}.wav", ep_dir / "audio" / f".edge_{key}.json"
                if cw.exists() and cj.exists():
                    words = json.loads(cj.read_text())
                else:
                    words = await voice_beat(beat, ep["voice"], cw)
                    cj.write_text(json.dumps(words))
                out.write_bytes(cw.read_bytes())
            words = speed_up(out, words, float(ep.get("speed", 1)))
            beat["words"] = restore_punctuation(words, beat["narration"])
            beat["audio"] = f"audio/{out.name}"
            last = i == len(seg["beats"]) - 1
            beat["duration"] = round(duration(out) + (TAIL_PAD if last else BEAT_GAP), 3)
            inline_media(ep_dir, beat["visual"])
            check_clip(seg["id"], i, beat)
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
