"""Generate the test passage with local TTS models (Kokoro, Chatterbox) for the voice comparison.

Usage: .venv-tts/bin/python voicelab/gen_local.py
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).parent
OUT = HERE / "samples"
TEXT = (HERE / "passage.txt").read_text().strip()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def kokoro(voices: list[str]) -> None:
    from kokoro import KPipeline
    for v in voices:
        pipe = KPipeline(lang_code="b" if v.startswith("b") else "a")
        audio = np.concatenate([a for _, _, a in pipe(TEXT, voice=v, speed=1.0)])
        sf.write(OUT / f"kokoro_{v}.wav", audio, 24000)
        print("kokoro", v)


def chatterbox(settings: list[tuple[float, float]]) -> None:
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device=DEVICE)
    for exaggeration, cfg in settings:
        # Generate sentence by sentence; long inputs drift. Short gaps keep the rhythm natural.
        parts = []
        for sentence in [s.strip() for s in TEXT.replace("? ", "?|").replace(". ", ".|").split("|") if s.strip()]:
            wav = model.generate(sentence, exaggeration=exaggeration, cfg_weight=cfg)
            parts += [wav.squeeze(0).cpu().numpy(), np.zeros(int(model.sr * 0.18))]
        sf.write(OUT / f"chatterbox_ex{exaggeration}_cfg{cfg}.wav", np.concatenate(parts), model.sr)
        print("chatterbox", exaggeration, cfg)


def chatterbox_directed() -> None:
    """Chatterbox with per-phrase direction: exaggeration for energy, cfg for pace, explicit pauses."""
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device=DEVICE)
    # (text, exaggeration, cfg_weight, pause_after_s); lower cfg = faster, higher exaggeration = more intense
    script = [
        ("Robots are leaving the lab,", 0.7, 0.4, 0.15),
        ("and this week we got two very different examples of what that looks like.", 0.5, 0.5, 0.55),
        ("Their solution works like a mousetrap.", 0.6, 0.5, 0.5),
        ("A special material warms up in the light, and slowly bends a tiny curved strip, storing up energy,", 0.4, 0.7, 0.12),
        ("until the strip suddenly snaps the other way, and releases it all at once!", 1.0, 0.3, 0.7),
        ("The result?", 0.7, 0.5, 0.45),
        ("A ninety-six percent success rate.", 0.6, 0.7, 0.3),
    ]
    parts = []
    for text, ex, cfg, pause in script:
        wav = model.generate(text, exaggeration=ex, cfg_weight=cfg)
        parts += [wav.squeeze(0).cpu().numpy(), np.zeros(int(model.sr * pause))]
    sf.write(OUT / "chatterbox_directed.wav", np.concatenate(parts), model.sr)
    print("chatterbox directed")


if __name__ == "__main__":
    if "--skip-kokoro" not in sys.argv:
        kokoro(["am_michael", "am_fenrir", "am_puck", "af_heart", "bm_george"])
    if "--directed-only" in sys.argv:
        chatterbox_directed()
        sys.exit()
    chatterbox([(0.5, 0.5), (0.7, 0.3), (0.9, 0.3)])
    chatterbox_directed()
