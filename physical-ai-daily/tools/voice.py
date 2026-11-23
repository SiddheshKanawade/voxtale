#!/usr/bin/env python3
"""Synthesize narration for each segment of an episode and write Remotion props.

Usage: .venv/bin/python tools/voice.py episodes/<episode-dir>

Reads <episode-dir>/episode.json, writes audio/<segment>_<beat>.wav and props.json (episode + per-beat
audio path, duration, word timings for captions, and inlined visual metadata).

One continuous take per segment (user, 2026-11-18: "always get complete audio, never in pieces"). Each segment (a
whole Short, or one story of an episode) is voiced in a single request on its full narration, so prosody, loudness
and pauses flow like one read. The take is then cut into beats at the natural pause between sentences, using the
TTS word timings, so the beat files join back into the original take exactly: no added silences, no re-trimming.
The visuals are timed to those cuts. Per-beat "direct" phrases and per-beat "eleven_text" are ignored.

ElevenLabs: with "tts": "elevenlabs" in episode.json, "voice" is an ElevenLabs voice id and "model" defaults to eleven_v3
(the blind-test winner, docs/voice-research.md section 8). One optional delivery tag for the whole take goes in
top-level (or per-segment) "eleven_tag", e.g. "[excited]"; it is said in the tone, never read out. Word timings come
from the API's character timestamps. A segment longer than ELEVEN_MAX characters is split at beat boundaries into a
few long takes (the API's per-request limit). ELEVENLABS_API_KEY is read from the environment or the project's .env.
Takes are cached in audio/, so re-runs don't spend characters.

Speed: an optional top-level "speed" (1.15 for episodes, 1.175 for Shorts) time-stretches the take after synthesis, with pitch
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
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

import edge_tts

TICKS_PER_SECOND = 10_000_000  # edge-tts reports offsets in 100ns units
TAIL_PAD = 0.6  # seconds of breathing room at the end of a segment
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
    # time out and retry with capped backoff. The timeout scales with the text: a whole-segment take streams at about
    # 17 characters a second (2,435 characters took 140 s), so a fixed 60 s timed out every attempt.
    timeout = 60 + len(text) / 5
    for i in range(attempts):
        try:
            return await asyncio.wait_for(synth_once(text, voice, out, rate, pitch), timeout=timeout)
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


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def restore_punctuation(words: list[dict], narration: str) -> list[dict]:
    """Boundary events drop punctuation; walk the script tokens in step to put it back for captions.

    A token the TTS speaks as several boundary words ("GPT-6.1" -> "GPT-6", ".", "1"; "two-eighty" -> "two",
    "eighty") becomes one caption word spanning them, shown as written, so captions don't split a name.
    """
    tokens, j, rest, cur, out = narration.split(), 0, None, None, []
    for w in words:
        if j >= len(tokens):
            out.append(w)
            continue
        rest = norm(tokens[j]) if rest is None else rest
        nw = norm(w["text"])
        if cur is not None and (not nw or rest.startswith(nw)):  # another piece of the token being spoken
            cur["end"] = w["end"]
        elif nw and rest.startswith(nw):
            cur = {**w}
            out.append(cur)
        else:
            out.append(w)
            continue
        rest = rest[len(nw):]
        if not rest:  # token fully consumed: show it as written, punctuation included
            cur["text"] = tokens[j]
            j, rest, cur = j + 1, None, None
    return out


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


ELEVEN_MAX = 2800  # characters per ElevenLabs request; longer segments are split at beat boundaries


def eleven_take(text: str, voice: str, model: str, cache: Path) -> tuple[np.ndarray, list[dict]]:
    """One ElevenLabs request for a whole text (cached by voice, model and text). Returns audio at SR and word timings."""
    key = hashlib.sha1(f"{voice}|{model}|{text}".encode()).hexdigest()[:16]
    mp3, meta = cache / f".take_{key}.mp3", cache / f".take_{key}.json"
    if not (mp3.exists() and meta.exists()):
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps?output_format=mp3_44100_128",
            data=json.dumps({"text": text, "model_id": model}).encode(),
            headers={"xi-api-key": api_key(), "Content-Type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
        except urllib.error.HTTPError as e:
            sys.exit(f"ElevenLabs HTTP {e.code}: {e.read().decode()[:300]}")
        mp3.write_bytes(base64.b64decode(d["audio_base64"]))
        meta.write_text(json.dumps({"alignment": d["alignment"]}))
    return decode(mp3), eleven_words(json.loads(meta.read_text())["alignment"])


async def edge_take(text: str, voice: str, cache: Path) -> tuple[np.ndarray, list[dict]]:
    """One edge-tts request for a whole text (cached by voice and text)."""
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16]
    mp3, meta = cache / f".take_{key}.mp3", cache / f".take_{key}.json"
    if not (mp3.exists() and meta.exists()):
        rate, pitch = prosody_args("neutral")
        meta.write_text(json.dumps(await synth(text, voice, mp3, rate, pitch)))
    return decode(mp3), json.loads(meta.read_text())


def chunks(beats: list[dict], limit: int) -> list[list[int]]:
    """Beat indices grouped into consecutive runs whose narration fits one request."""
    out, cur, n = [], [], 0
    for i, b in enumerate(beats):
        if cur and n + len(b["narration"]) + 1 > limit:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += len(b["narration"]) + 1
    return out + ([cur] if cur else [])


def assign_words(words: list[dict], beats: list[dict]) -> list[list[dict]]:
    """Give each TTS word to its beat by walking the normalized characters of the narration."""
    ends, total = [], 0
    for b in beats:
        total += len(norm(b["narration"]))
        ends.append(total)
    per, pos, k = [[] for _ in beats], 0, 0
    for w in words:
        while k < len(beats) - 1 and pos >= ends[k]:
            k += 1
        per[k].append(w)
        pos += len(norm(w["text"]))
    return per


def stretch(audio: np.ndarray, words: list[dict], factor: float) -> tuple[np.ndarray, list[dict]]:
    """Time-stretch a whole take (pitch kept) and scale its word timings."""
    if abs(factor - 1) < 1e-3:
        return audio, words
    out = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                          "-af", f"atempo={factor}", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-"],
                         input=audio.tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16), [{**w, "start": w["start"] / factor, "end": w["end"] / factor} for w in words]


async def voice_segment(ep: dict, seg: dict, ep_dir: Path) -> None:
    """Voice a segment as continuous take(s), cut it into beat files at the pauses between beats, fill beat timings."""
    beats, cache = seg["beats"], ep_dir / "audio"
    pieces, words, t0 = [], [], 0.0
    eleven = ep.get("tts") == "elevenlabs"
    for run in chunks(beats, ELEVEN_MAX if eleven else 10_000):
        text = " ".join(beats[i]["narration"] for i in run)
        if eleven:
            tag = seg.get("eleven_tag") or ep.get("eleven_tag") or ""
            audio, ws = eleven_take(f"{tag} {text}".strip(), ep["voice"], ep.get("model", "eleven_v3"), cache)
        else:
            audio, ws = await edge_take(text, ep["voice"], cache)
        pieces.append(audio)
        words += [{**w, "start": w["start"] + t0, "end": w["end"] + t0} for w in ws]
        t0 += len(audio) / SR
    audio, words = stretch(np.concatenate(pieces), words, float(ep.get("speed", 1)))
    per = assign_words(words, beats)
    if any(not ws for ws in per):
        raise SystemExit(f"{seg['id']}: TTS words didn't line up with the beats' narration")
    # Cut halfway through the pause between one beat's last word and the next beat's first word.
    cuts = [max(0.0, per[0][0]["start"] - LEAD)]
    cuts += [(per[i][-1]["end"] + per[i + 1][0]["start"]) / 2 for i in range(len(beats) - 1)]
    cuts.append(min(len(audio) / SR, per[-1][-1]["end"] + LAST_TAIL))
    for i, beat in enumerate(beats):
        a, b = cuts[i], cuts[i + 1]
        out = cache / f"{seg['id']}_{i:02d}.wav"
        with wave.open(str(out), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SR)
            f.writeframes(audio[int(a * SR):int(b * SR)].tobytes())
        ws = [{**w, "start": round(w["start"] - a, 3), "end": round(w["end"] - a, 3)} for w in per[i]]
        beat["words"] = restore_punctuation(ws, beat["narration"])
        beat["audio"] = f"audio/{out.name}"
        beat["duration"] = round(b - a + (TAIL_PAD if i == len(beats) - 1 else 0), 3)


async def main(ep_dir: Path) -> None:
    ep = json.loads((ep_dir / "episode.json").read_text())
    (ep_dir / "audio").mkdir(exist_ok=True)
    for seg in ep["segments"]:
        await voice_segment(ep, seg, ep_dir)
        for i, beat in enumerate(seg["beats"]):
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
