"""Objective screen of every voice sample (docs/voice-research.md section 6), then a blind listening page.

Reads loudness-matched samples from voicelab/leveled/ (-14 LUFS). Criteria:
  variety    phrase-level pitch variety (st)                               -> avoids one repeated sentence melody
  emph_snap  prominence of "snaps": z-scores of duration/syllable, loudness and pitch peak vs the speaker's own words
  emph_96    prominence of "ninety-six", same measure                      -> emphasis on the punchline
  beat_96    longest silence just before "ninety-six" (s), from the loudness envelope (Whisper timestamps fold pauses in);
             credit is capped at BEAT_CAP: a beat helps, dead air doesn't (the cap is our assumption, not from a study)
  pace       distance from 150 wpm (industry convention for explainers 130-150, audiobooks 150-160; not research-backed)
  wer        Whisper word error rate; > 10% is disqualified

Usage: .venv-tts/bin/python voicelab/waveforms.py
Writes voicelab/analysis/{scores.json, <name>.png, shortlist.png, blind_key.json} and voicelab/blind.html.
"""
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import jiwer
import matplotlib
import numpy as np
import parselmouth

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "tools"))
from prosody import measure  # noqa: E402

SRC = HERE / "leveled"
OUT = HERE / "analysis"
BLIND = HERE / "blind"
TEXT = (HERE / "passage.txt").read_text().strip()
WORDS = len(TEXT.split())
ANCHOR = "edge_en-US-AndrewNeural"  # the current voice always goes in the shortlist, as the reference point
TARGET_WPM = 150
BEAT_CAP = 0.6


def norm(s: str) -> str:
    s = s.lower().replace("96%", "ninety-six percent").replace("96", "ninety-six")
    return re.sub(r"[^a-z\- ]", " ", s).replace("-", " ")


def syllables(word: str) -> int:
    return max(1, sum(len(re.findall(r"[aeiouy]+", w)) for w in norm(word).split()))


def analyse(path: Path, asr) -> dict:
    segs, _ = asr.transcribe(str(path), language="en", word_timestamps=True)
    words = [w for s in segs for w in s.words]
    wer = jiwer.wer(norm(TEXT), norm(" ".join(w.word for w in words)))

    snd = parselmouth.Sound(str(path))
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=65, pitch_ceiling=450)
    f0 = pitch.selected_array["frequency"]
    med = np.median(f0[f0 > 0])
    f0_st = np.where(f0 > 0, 12 * np.log2(np.where(f0 > 0, f0, 1) / med), np.nan)
    inten = snd.to_intensity(time_step=0.01)
    db, db_t = inten.values[0], inten.xs()
    loud = db > np.percentile(db, 95) - 25  # same speech/silence split as tools/prosody.py

    def span(w):
        """Word interval with any absorbed leading/trailing silence trimmed off, using the loudness envelope."""
        idx = np.where((db_t >= w.start) & (db_t <= w.end) & loud)[0]
        return (db_t[idx[0]], db_t[idx[-1]]) if len(idx) else (w.start, w.end)

    feats = []  # per word: seconds per syllable, mean dB, 90th-percentile pitch (st)
    for w in words:
        a, b = span(w)
        p = f0_st[(pitch.xs() >= a) & (pitch.xs() <= b)]
        p = p[~np.isnan(p)]
        d = db[(db_t >= a) & (db_t <= b)]
        feats.append(((b - a) / syllables(w.word), float(np.mean(d)) if len(d) else np.nan,
                      float(np.percentile(p, 90)) if len(p) else np.nan))
    feats = np.array(feats)
    mu, sd = np.nanmean(feats, 0), np.nanstd(feats, 0)

    def prominence(i):
        if i is None:
            return 0.0, (0, 0, 0)
        z = (feats[i] - mu) / sd
        z = np.nan_to_num(z)
        return round(float(np.mean(z)), 2), tuple(round(float(v), 2) for v in z)

    def find(pred):
        return next((i for i, w in enumerate(words) if pred(w.word.lower())), None)
    i_snap = find(lambda w: "snap" in w)
    i_96 = find(lambda w: "96" in w or "ninety" in w)

    # The beat sits after "The result?", before "A ninety-six"; anchor on "result", not the word before 96.
    i_res = find(lambda w: "result" in w)
    beat, beat_span = 0.0, None
    if i_96 and i_res is not None:
        a, b = words[i_res].start, span(words[i_96])[0] + 0.02
        win = np.where((db_t >= a) & (db_t <= b))[0]
        best, run, end = 0, 0, None
        for k in win:
            run = run + 1 if not loud[k] else 0
            if run > best:
                best, end = run, db_t[k]
        if best:
            beat, beat_span = round(best * 0.01, 2), (end - best * 0.01, end)

    m = measure(str(path), WORDS)
    snap, snap_z = prominence(i_snap)
    n96, n96_z = prominence(i_96)
    return {
        "name": path.stem, "variety": m["phr_f0_sd"] or 0, "wpm": m["wpm"], "wer": round(wer, 3),
        "emph_snap": snap, "snap_z(dur,db,pitch)": snap_z, "emph_96": n96, "n96_z(dur,db,pitch)": n96_z,
        "beat_96": beat, "pace_off": abs(m["wpm"] - TARGET_WPM),
        "_plot": (snd, pitch, f0_st, words, span, i_snap, i_96, beat_span),
    }


def score(rows):
    """Average of per-criterion ranks (0 = worst, 1 = best); equal weights, since no study gives better ones."""
    crit = {"variety": 1, "emph_snap": 1, "emph_96": 1, "beat_96": 1, "pace_off": -1}
    for key, sign in crit.items():
        vals = np.array([sign * (min(r[key], BEAT_CAP) if key == "beat_96" else r[key]) for r in rows])
        for r, v in zip(rows, vals):  # tie-aware rank: equal values share the average position
            pos = (np.sum(vals < v) + (np.sum(vals == v) - 1) / 2) / (len(rows) - 1)
            r.setdefault("ranks", {})[key] = round(float(pos), 2)
    for r in rows:
        r["score"] = round(float(np.mean(list(r["ranks"].values()))), 3) if r["wer"] <= 0.10 else 0.0
    rows.sort(key=lambda r: -r["score"])


def draw(ax, r, title=True):
    snd, pitch, f0_st, words, span, i_snap, i_96, beat_span = r["_plot"]
    hop = int(snd.sampling_frequency * 0.01)  # 10 ms peak envelope; raw samples render as invisible slivers
    y = np.abs(snd.values[0][: len(snd.values[0]) // hop * hop]).reshape(-1, hop).max(1)
    ax.fill_between(np.arange(len(y)) * 0.01, y, -y, color="#7d8fb3", lw=0, alpha=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 24)
    ax2 = ax.twinx()
    ax2.plot(pitch.xs(), f0_st, color="#ffe14d", lw=1.6)
    ax2.set_ylim(-12, 14)
    ax2.axhline(0, color="#ffe14d", lw=0.5, alpha=0.3)
    for i, lab in ((i_snap, "snaps"), (i_96, "96%")):
        if i is not None:
            a, b = span(words[i])
            ax.axvspan(a, b, color="#ff5c7a", alpha=0.35)
            ax.text(a, 0.82, lab, color="#ff5c7a", fontsize=9, weight="bold")
    if beat_span:
        ax.axvspan(*beat_span, color="#3ddc97", alpha=0.45)
    if title:
        ax.set_title(f"{r['name']}   score {r['score']}  ·  variety {r['variety']} st  ·  emphasis snaps {r['emph_snap']} / 96% {r['emph_96']}"
                     f"  ·  beat {r['beat_96']} s  ·  {r['wpm']} wpm  ·  WER {r['wer']:.0%}", color="white", fontsize=10, loc="left")
    ax.set_facecolor("#151a24")
    ax2.patch.set_visible(False)  # otherwise the pitch axis background hides the waveform
    for a in (ax, ax2):
        a.tick_params(colors="#8b93a3", labelsize=7)
        for s in a.spines.values():
            s.set_visible(False)


def figure(rows, out):
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 2.4 * len(rows)), facecolor="#0b0e14", squeeze=False)
    for ax, r in zip(axes[:, 0], rows):
        draw(ax, r)
    axes[-1, 0].set_xlabel("seconds   grey = waveform · yellow = pitch (semitones vs the speaker's median) · red = key word · green = pause before punchline",
                           color="#8b93a3")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def blind_page(shortlist):
    """Shuffled, unnamed, loudness-matched samples A-F with 0-100 sliders (MUSHRA-style, no reference shown)."""
    BLIND.mkdir(exist_ok=True)
    order = shortlist[:]
    random.Random(20261114).shuffle(order)
    key = {}
    for letter, r in zip("ABCDEF", order):
        shutil.copy(SRC / f"{r['name']}.wav", BLIND / f"{letter}.wav")
        key[letter] = r["name"]
    (OUT / "blind_key.json").write_text(json.dumps(key, indent=1))
    cards = "".join(
        f'<div class="s"><div class="n">Sample {L}</div><audio controls preload="auto" src="blind/{L}.wav"></audio>'
        f'<label>How engaging? <input type="range" min="0" max="100" value="50" data-l="{L}"> <b>50</b></label></div>'
        for L in key)
    (HERE / "blind.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Blind voice test</title>
<style>
:root {{ --bg:#0b0e14; --panel:#151a24; --ink:#f2f4f8; --dim:#8b93a3; --accent:#ffe14d; }}
body {{ background:var(--bg); color:var(--ink); font:16px -apple-system,Helvetica,sans-serif; max-width:820px; margin:32px auto; padding:0 16px; }}
.s {{ background:var(--panel); border-radius:10px; padding:12px 16px; margin:10px 0 }} .n {{ font-weight:700; margin-bottom:6px }}
audio {{ width:100% }} label {{ display:flex; gap:12px; align-items:center; margin-top:8px; color:var(--dim) }} input {{ flex:1 }}
button {{ background:var(--accent); border:0; border-radius:8px; padding:10px 18px; font-weight:700; cursor:pointer }}
pre {{ background:var(--panel); padding:12px; border-radius:8px; white-space:pre-wrap }} li {{ color:var(--dim); margin:4px 0 }}
</style>
<h1>Blind voice test</h1>
<ol>
<li>All six read the same passage and are matched to the same loudness. Names are hidden and the order is shuffled.</li>
<li>Imagine each as the narrator of a 4-minute explainer. Listen to all six at least once before rating.</li>
<li>Rate how much you'd want to keep listening (0 = would click away, 100 = hooked). Replay as often as you like.</li>
<li>Click the button and paste the result back to Claude, with a word or two on your top pick.</li>
</ol>
{cards}
<p><button onclick="show()">Show my result</button></p><pre id="out"></pre>
<script>
document.querySelectorAll('input').forEach(i => i.oninput = () => i.nextElementSibling.textContent = i.value);
document.querySelectorAll('audio').forEach(a => a.onplay = () => document.querySelectorAll('audio').forEach(o => o !== a && o.pause()));
function show() {{
  const r = [...document.querySelectorAll('input')].map(i => [i.dataset.l, +i.value]).sort((a, b) => b[1] - a[1]);
  document.getElementById('out').textContent = 'Blind test: ' + r.map(([l, v]) => l + '=' + v).join(', ');
}}
</script>
""")


def main():
    from faster_whisper import WhisperModel
    OUT.mkdir(exist_ok=True)
    asr = WhisperModel("small", device="cpu", compute_type="int8")
    rows = [analyse(p, asr) for p in sorted(SRC.glob("*.wav"))]
    score(rows)
    shortlist = [r for r in rows if r["score"] > 0][:5]
    if ANCHOR not in [r["name"] for r in shortlist]:
        shortlist.append(next(r for r in rows if r["name"] == ANCHOR))
    else:
        shortlist.append(next(r for r in rows if r["score"] > 0 and r not in shortlist))
    figure(shortlist, OUT / "shortlist.png")
    figure(rows, OUT / "all.png")
    blind_page(shortlist)
    (OUT / "scores.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "_plot"} for r in rows], indent=1))

    print(f"{'voice':32}{'score':>7}{'variety':>9}{'snap':>7}{'96':>7}{'beat96':>8}{'wpm':>6}{'wer':>7}   snap z(dur,db,pitch)   96 z(dur,db,pitch)")
    for r in rows:
        print(f"{r['name'][:32]:32}{r['score']:>7}{r['variety']:>9}{r['emph_snap']:>7}{r['emph_96']:>7}{r['beat_96']:>8}{r['wpm']:>6}{r['wer']:>7}"
              f"   {str(r['snap_z(dur,db,pitch)']):22} {r['n96_z(dur,db,pitch)']}")
    print("\nshortlist:", [r["name"] for r in shortlist])


if __name__ == "__main__":
    main()
