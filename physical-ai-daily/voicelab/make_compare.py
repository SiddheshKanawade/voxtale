"""Build voicelab/compare.html: every sample with a player and its prosody metrics, for blind-ish listening.

Usage: .venv/bin/python voicelab/make_compare.py
"""
import html
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
from prosody import measure  # noqa: E402

GROUPS = [
    ("Current voice", ["edge_en-US-AndrewNeural"]),
    ("Same voice, directed (per-phrase pace/pitch + pauses)", ["directed_en-US-AndrewNeural", "directed_en-US-EmmaNeural"]),
    ("Expressive local model: Chatterbox (MIT)", None),  # filled from samples/chatterbox_*
    ("Other free Edge voices", ["edge_en-US-EmmaNeural", "edge_en-US-GuyNeural", "edge_en-US-ChristopherNeural", "edge_en-US-BrianNeural"]),
    ("Local model: Kokoro (Apache-2.0)", ["kokoro_am_fenrir", "kokoro_am_michael", "kokoro_af_heart", "kokoro_bm_george", "kokoro_am_puck"]),
]

words = len((HERE / "passage.txt").read_text().split())
samples = {p.stem: p for p in sorted((HERE / "samples").iterdir())}
GROUPS[2] = (GROUPS[2][0], sorted((s for s in samples if s.startswith("chatterbox")), key=lambda s: s != "chatterbox_directed"))

rows = []
for title, names in GROUPS:
    rows.append(f"<h2>{html.escape(title)}</h2>")
    for n in names:
        if n not in samples:
            continue
        m = measure(str(samples[n]), words)
        rows.append(
            f'<div class="s"><div class="n">{html.escape(n)}</div><audio controls preload="none" src="samples/{samples[n].name}"></audio>'
            f'<div class="m">pitch SD {m["f0_sd_st"]} st · range {m["f0_rng_st"]} st · phrase variety {m["phr_f0_sd"]} st · {m["wpm"]} wpm · pauses {m["pause_pct"]}%</div></div>'
        )

page = f"""<!doctype html><meta charset="utf-8"><title>Voice comparison</title>
<style>
:root {{ --bg:#0b0e14; --panel:#151a24; --ink:#f2f4f8; --dim:#8b93a3; --accent:#ffe14d; }}
body {{ background:var(--bg); color:var(--ink); font:16px -apple-system,Helvetica,sans-serif; max-width:900px; margin:32px auto; padding:0 16px; }}
h1 {{ font-size:28px }} h2 {{ color:var(--accent); font-size:18px; margin-top:28px }}
.s {{ background:var(--panel); border-radius:10px; padding:12px 16px; margin:8px 0 }}
.n {{ font-weight:700; margin-bottom:6px }} .m {{ color:var(--dim); font-size:13px; margin-top:6px }}
audio {{ width:100% }} blockquote {{ color:var(--dim); border-left:3px solid var(--accent); margin:0; padding-left:12px }}
</style>
<h1>Voice comparison: same passage, every candidate</h1>
<blockquote>{html.escape((HERE / "passage.txt").read_text().strip())}</blockquote>
<p style="color:var(--dim)">Listen for: does it sound <b>excited</b> at "snaps", is there a <b>beat before</b> "ninety-six percent", does each sentence sound <b>different</b>?</p>
{''.join(rows)}
"""
(HERE / "compare.html").write_text(page)
print("wrote", HERE / "compare.html")
