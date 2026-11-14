"""Edge-TTS with per-phrase direction: rate/pitch chosen per phrase, deliberate pauses before punchlines.

Usage: .venv/bin/python voicelab/gen_directed.py <voice>
"""
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

import edge_tts

OUT = Path(__file__).parent / "samples"

# (text, rate, pitch, pause_after_s)
DIRECTED = [
    ("Robots are leaving the lab,", "+8%", "+3Hz", 0.15),
    ("and this week we got two very different examples of what that looks like.", "+3%", "+0Hz", 0.55),
    ("Their solution works like a mousetrap.", "-6%", "-3Hz", 0.5),
    ("A special material warms up in the light, and slowly bends a tiny curved strip, storing up energy,", "-12%", "-2Hz", 0.12),
    ("until the strip suddenly snaps the other way, and releases it all at once!", "+12%", "+4Hz", 0.7),
    ("The result?", "-4%", "+5Hz", 0.45),
    ("A ninety-six percent success rate.", "-14%", "-4Hz", 0.3),
]


async def phrase(text, voice, rate, pitch, out):
    for i in range(10):
        try:
            await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(out)
            return
        except edge_tts.exceptions.NoAudioReceived:
            await asyncio.sleep(3 + 2 * i)
    raise RuntimeError("edge-tts gave no audio")


async def main(voice):
    tmp = Path(tempfile.mkdtemp())
    parts = []
    for i, (text, rate, pitch, pause) in enumerate(DIRECTED):
        mp3 = tmp / f"{i}.mp3"
        await phrase(text, voice, rate, pitch, str(mp3))
        parts.append(f"file '{mp3}'")
        sil = tmp / f"{i}_sil.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(pause), "-q:a", "9", str(sil)], check=True)
        parts.append(f"file '{sil}'")
    (tmp / "list.txt").write_text("\n".join(parts))
    out = OUT / f"directed_{voice}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"), "-ar", "24000", "-ac", "1", str(out)], check=True)
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "en-US-AndrewNeural"))
