#!/usr/bin/env python3
"""Check every paid source's key and remaining credit before a harvest or a voice run.

Usage: python3 -I tools/credits.py

A key that exists can still be useless: a spent twitterapi.io balance or a NewsData key at its daily cap returns errors
mid-harvest, and the episode then quietly runs on fewer sources. This prints one line per source (OK / LOW / OUT /
MISSING) and exits 1 if any harvest source is OUT or MISSING, so the research skill can say up front which source will
run degraded. Never prints a key.

| Source       | How it's checked                                         | LOW below                        |
|--------------|----------------------------------------------------------|----------------------------------|
| twitterapi.io | /oapi/my/info -> recharge_credits + total_bonus_credits | ~2 weekly runs (sources.json)   |
| NewsData.io  | one 1-result query per pooled key -> X-API-Limit-Remaining (costs 1 credit) | one run's pages |
| ElevenLabs   | /v1/user/subscription -> characters left this cycle      | one episode's narration (~6k)    |
| Render       | /v1/owners (only needed for sync_p2a_topics.py)          | n/a                              |
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discover import ROOT, key, keys  # noqa: E402

# twitterapi.io bills 15 credits per tweet returned; a weekly harvest returns ~1,000-1,500 tweets plus chatter
# searches, so ~30k credits a run.
X_RUN_CREDITS = 30_000
EPISODE_CHARS = 6_000


def call(url: str, headers: dict) -> tuple[int, dict, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **headers})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), {}


def main() -> int:
    cfg = json.loads((ROOT / "sources.json").read_text())
    rows, harvest_bad = [], []

    k = key("TWITTERAPI_IO_KEY")
    if not k:
        rows.append(("X (twitterapi.io)", "MISSING", "falls back to the public timeline (our accounts only)"))
        harvest_bad.append("x")
    else:
        code, _, d = call("https://api.twitterapi.io/oapi/my/info", {"X-API-Key": k})
        credits = (d.get("recharge_credits") or 0) + (d.get("total_bonus_credits") or 0)
        state = "OUT" if code != 200 or credits < X_RUN_CREDITS // 3 else "LOW" if credits < 2 * X_RUN_CREDITS else "OK"
        rows.append(("X (twitterapi.io)", state, f"{credits:,} credits (~{credits // X_RUN_CREDITS} weekly runs)"
                     if code == 200 else f"HTTP {code}"))
        if state == "OUT":
            harvest_bad.append("x")

    pool = keys("NEWSDATAIO")
    need = len(cfg["news_queries"]) * cfg.get("news_pages", 3)
    if not pool:
        rows.append(("News (NewsData.io)", "MISSING", "skipped; cover news with web search"))
        harvest_bad.append("news")
    else:
        left = []
        for i, nk in enumerate(pool, 1):
            code, h, _ = call(f"https://newsdata.io/api/1/latest?apikey={nk}&qInTitle=robot&language=en&size=1", {})
            remaining = h.get("X-API-Limit-Remaining") or h.get("x_api_limit_remaining")
            left.append(int(remaining) if code == 200 and remaining else 0)
        total = sum(left)
        state = "OUT" if total == 0 else "LOW" if total < need else "OK"
        rows.append(("News (NewsData.io)", state, f"{total} credits left today across {len(pool)} keys "
                     f"({', '.join(map(str, left))}); a run uses up to {need}; resets daily"))
        if state == "OUT":
            harvest_bad.append("news")

    ek = key("ELEVENLABS_API_KEY")
    if not ek:
        rows.append(("Voice (ElevenLabs)", "MISSING", "needed to voice the episode, not for research"))
    else:
        code, _, d = call("https://api.elevenlabs.io/v1/user/subscription", {"xi-api-key": ek})
        if code == 200:
            left = d["character_limit"] - d["character_count"]
            reset = datetime.fromtimestamp(d["next_character_count_reset_unix"], timezone.utc).date()
            state = "OUT" if left < 500 else "LOW" if left < EPISODE_CHARS else "OK"
            rows.append(("Voice (ElevenLabs)", state, f"{left:,} of {d['character_limit']:,} characters left "
                         f"({d.get('tier')} tier), resets {reset}"))
        else:
            rows.append(("Voice (ElevenLabs)", "OUT", f"HTTP {code}"))

    rk = key("RENDER_API_KEY")
    code = call("https://api.render.com/v1/owners?limit=1", {"Authorization": f"Bearer {rk}"})[0] if rk else 0
    rows.append(("Render (topic sync)", "OK" if code == 200 else "MISSING" if not rk else "OUT",
                 "can read the P2A digest topics" if code == 200 else f"sync_p2a_topics.py will fail (HTTP {code})"))
    rows.append(("arXiv, Semantic Scholar, Hugging Face, YouTube", "OK", "free, no key"))

    width = max(len(r[0]) for r in rows)
    for name, state, note in rows:
        print(f"{name:<{width}}  {state:<7}  {note}")
    if harvest_bad:
        print(f"degraded harvest sources: {', '.join(harvest_bad)}", file=sys.stderr)
    return 1 if harvest_bad else 0


if __name__ == "__main__":
    sys.exit(main())
