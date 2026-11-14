"""Generate the test passage with ElevenLabs v3 in one pass (pause and audio tags inline) and report its timestamps.

Usage: set -a; . ./.env; set +a; python3 voicelab/gen_eleven.py <voice_id> <out.mp3> [stability]
stability: 0.0 Creative, 0.5 Natural (default), 1.0 Robust.
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request

TEXT = ("Robots are leaving the lab, and this week we got two very different examples of what that looks like. "
        "Their solution works like a mousetrap. A special material warms up in the light and slowly bends a tiny curved strip, "
        "storing up energy, until the strip suddenly [excited] snaps the other way and releases it all at once. "
        "The result? [short pause] A ninety-six percent success rate.")

voice, out = sys.argv[1], sys.argv[2]
body = {"text": TEXT, "model_id": "eleven_v3"}
if len(sys.argv) > 3:
    body["voice_settings"] = {"stability": float(sys.argv[3])}
req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps?output_format=mp3_44100_128",
                             data=json.dumps(body).encode(),
                             headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"})
try:
    d = json.load(urllib.request.urlopen(req))
except urllib.error.HTTPError as e:
    sys.exit(f"HTTP {e.code}: {e.read().decode()[:500]}")
open(out, "wb").write(base64.b64decode(d["audio_base64"]))
a = d["alignment"]
s = "".join(a["characters"])
print(f"{out}: {a['character_end_times_seconds'][-1]:.2f} s, {len(s)} aligned chars; "
      f"'result' at {a['character_start_times_seconds'][s.find('result')]:.2f} s, 'ninety' at {a['character_start_times_seconds'][s.find('ninety')]:.2f} s")
