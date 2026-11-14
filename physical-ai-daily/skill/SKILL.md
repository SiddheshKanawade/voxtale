---
name: physical-ai-daily
description: Produce an episode of the "Physical AI Daily" YouTube channel — discover the day's physical-AI stories (news, X posts, arXiv papers), verify each against its primary source, write a narration script, capture highlighted screenshots of the sources, synthesize voice, and render a 1080p video with Remotion on this Mac. Use when the user asks to make today's episode, a sample episode, add a story, re-render, or change the channel's video format.
---

# Physical AI Daily

Project: this folder (`physical-ai-daily/`). Everything runs locally on the Mac.

```
sources.json                 accounts, arXiv categories, news queries (user-editable)
tools/discover.py            arXiv + X candidates -> JSON on stdout
tools/capture.mjs            page / tweet / element / video-frame screenshots via installed Chrome, in-page highlight
tools/crop.py                crop a figure panel (fractions) + write its metadata json
tools/voice.py               episode.json -> audio/<seg>_<beat>.mp3 + props.json (word timings, punctuation, media metadata)
tools/render.sh              voice -> QA contact sheet -> MP4 -> loudness-normalized final (-14 LUFS)
video/src/Episode.tsx        chrome: top bar, chapter progress bar, story header, credit chip, captions
video/src/scenes.tsx         scene catalog (below) + word-cue timing + hold-cap punch-in
video/src/diagrams.tsx       custom animated explainers (kitting, snap-through)
episodes/<date>[-slug]/      episode.json, assets/, audio/, qa/, out/
```

**Reference episode: `episodes/2026-11-13-sample-v2/`** (2 stories, 3:54, 36 beats). Copy its `episode.json` shape.
v1 (`2026-11-13-sample/`, one static screenshot per story) is obsolete: the user dropped off midway. Too little
motion, and too technical.

## Audience and pacing (user feedback, 2026-11-13)

- **The viewer is a product manager:** high-level understanding, no robotics background. Define every term on screen
  (an acronym scene for "VLA"), use one everyday analogy per hard idea ("works like a mousetrap"), and end each story
  with *why it matters* plus a reality check.
- **Keep it moving.** No single static visual holds for more than **~5–7 s** (`HOLD_CAP_S = 7`). Cut points follow the
  audio: each beat is one narration chunk and one visual, and elements inside a scene appear on the word that names them.
  This is a cap, not a fixed rhythm. Short beats are fine.
- Prefer real source imagery (video stills from the announcement, paper figure panels) and animated explainers over
  page screenshots. Screenshots appear only for the headline and the highlighted proof sentence.
- Length is driven by content. Aim for 3–5 stories at 60–110 s each. Don't pad toward a target length.

## Pipeline

1. **Discover.** `python3 -I tools/discover.py --hours 36 > <ep>/candidates.json`, plus WebSearch over `news_queries` in `sources.json`.
2. **Verify every story** (rules below). Drop anything that fails. Never pad to reach 10.
3. **Rank and pick.** Default: real-world deployment > model/product release > funding > paper > opinion. Prefer stories with a strong visual (figure, video still, chart). Show the shortlist to the user before scripting unless they said to run unattended.
4. **Write `episode.json`.** Segments (`kind` story/title, `label` NEWS / ON X / PAPER, `headline` ≤ 40 chars,
   `source`, `sourceDate`, `urls`), each holding `beats: [{narration, visual}]`. Split narration at sentence or clause
   boundaries so static beats stay ≤ 7 s. `voice.py` prints `WARN` for photo/shot/kinetic beats over the cap.
   Give every list item a `cue` (a word in that beat's narration) so it appears when spoken.
5. **Capture** each visual:
   - `node tools/capture.mjs page <url> <ep>/assets/x.png [--find "<exact sentence>"]`: a highlight makes the video zoom onto that sentence
   - `node tools/capture.mjs tweet <post_id> <ep>/assets/x.png`
   - `node tools/capture.mjs frames <url> <ep>/assets/x.png --selector video --nth N --times "2,8,15"`: stills from a video playing on the source page (find N by probing each `nth` once; some lazy videos never load, so skip them)
   - arXiv figures: check the license on the abs page (CC BY means reusable with credit), download `arxiv.org/html/<id>v1/figures/*.png`, then `python3 tools/crop.py <src> <out> x0 y0 x1 y1` for panels
   Look at every PNG before rendering. A `WARN: text not found` means the quote isn't on the page verbatim, so fix the quote.
6. **Render.** `tools/render.sh <ep> --stills-only` first. Better: render once, then pull one frame per beat at 75% of its
   length into contact sheets and read them. Fix, then run the full render (about 3.5 min for a 4-min episode).

## Scene catalog (`visual.type`)

| type | use | key fields |
|---|---|---|
| `shot` | page screenshot; zooms to `--find` highlight | `src`, `stamp` (e.g. "Self-reported") |
| `photo` | full-bleed Ken Burns, or `fit:"contain"` card for charts | `src`, `pan`, `overlay`, `badge`+`badgeCue`, `tags` |
| `split` | two images side by side | `items[{src,label,cue}]` |
| `panels` | 3–4 images revealed in turn (sequences, variants) | `items[{src,label,cue}]` |
| `kinetic` | big words, chapter turns ("Lab → Factory") | `text`, `sub` |
| `acronym` | define jargon letter by letter | `letters[{letter,word}]` |
| `flow` | step diagram; `arrows:false` for a plain icon row | `steps[{icon,label,sub,cue}]` |
| `hub` | one thing → many (one brain, many bodies) | `center`, `spokes[{icon,label,sub,cue}]` |
| `stat` | counting number(s); optional `side` image (e.g. the post) | `items[{to,prefix,suffix,label,cue}]` |
| `versus` | old vs new, problem vs need | `left/right{icon,title,sub,cue}` |
| `checklist` | reality check (ok / no / warn) | `title`, `items[{status,text,cue}]` |
| `kitting`, `snap` | bespoke explainers in `diagrams.tsx`; write a new one when a mechanism needs it | cue words in code |
| `end` | end card; Subscribe appears on `subscribeCue` | — |

Icons are Apple emoji, which render in Remotion on this Mac. Any static photo or shot past 7 s punch-cuts to a tighter
frame at the clause break nearest its middle.
7. **Hand off.** Report the final path, the duration, and a description block listing every source URL. Uploading is the user's action, so never upload.

## Verification rules (learned the hard way)

The first sample's searches returned two plausible but false items: a "$500M Agile Robots Series F" that no primary source confirms, and Runway Praxis-1 presented as today's news when it was announced in September. Search summaries mix dates and companies (Agile Robots vs Agility Robotics).

- Every story needs a **primary source**: the company blog or press release, the paper, or the author's own post. Secondary coverage can lead you to a story but can't confirm it.
- Check the **date on the primary source**. It has to fall within the episode window, or the script has to say it's older.
- Every number in the narration must appear verbatim on a captured page. Use that sentence as the `--find` highlight.
- Label vendor-reported results as such ("DeepMind's own numbers").
- Say "proof of concept", "pilot", or "preprint" when that's what it is.
- Read the figure captions, not only the abstract. SunLightBots' abstract says "near 1 Sun", but the robot demos ran
  at 1.5 suns (swim, crawl) and 2 suns (jump). v1 overstated this; v2 states it.
- Attribute quotes to the person named on the page (the "few weeks of post-training" line is Apptronik CEO Jeff Cardenas).

## Script rules

- Narration ≈ 150 words/min; a story is 150–280 words over 12–18 beats. Spell numbers as words ("ninety-six percent"), because the TTS reads digits unreliably. On-screen numbers use digits.
- Story arc: hook (what happened) → plain-English explainer (analogy + diagram) → evidence (source visual, highlighted sentence) → why it matters → reality check. Don't use hype words like "shocking" or "changes everything".
- The intro teases the stories with visuals in about 20 s. The outro points to the sources and cues Subscribe.

## Source access (tested 2026-11-13)

| Source | Method | Status |
|---|---|---|
| arXiv | export.arxiv.org API (`discover.py`), abs/html pages | Reliable. arXiv content is often CC BY; check the license line. |
| News/blogs | `capture.mjs page` (headless Chrome; only ever clicks reject/necessary-only consent buttons) | Reliable for public pages. Paywalls are not bypassed. |
| Single X post | `capture.mjs tweet <id>` (official embed); `cdn.syndication.twimg.com/tweet-result?id=` for JSON | Reliable without login |
| X account timelines | `syndication.twitter.com/srv/timeline-profile/...` | **Unreliable:** rate limit 30 per window (got 429s) and sometimes stale (one account returned posts that were months old). Treat as hints. For dependable discovery, use the user's logged-in Chrome (claude-in-chrome) or the paid X API. |
| x.com directly | headless browser | Blank page without login |

## Voice

`edge-tts` with `en-US-AndrewNeural` (free Microsoft neural voice via an unofficial endpoint). It intermittently returns no audio, and `voice.py` retries up to 10×. If it's down, fall back to macOS `say`, which sounds robotic, or a paid TTS (ElevenLabs, Higgsfield `generate_audio`) once the user has a budget. Multilingual voices failed in testing.

## Synthetic images

Higgsfield MCP has image models (Nano Banana Pro/2, GPT Image 2.6, FLUX), but the account had **0 credits** on 2026-11-13. Check `balance` first. Use synthetic images only for explainer diagrams or concepts with no source visual, and never for something that looks like a real event or product.

## Format (current: v0.2, 16:9 long-form)

Dark background with a drifting dot grid, yellow accent `#ffe14d` (also the highlight colour), top bar plus a chapter
progress bar, label chip + headline + "STORY n/N", credit chip on every beat, punctuated word-synced captions grouped
at clause breaks. A Short (9:16) version isn't built yet.

## Copyright

Screenshots of articles and posts with on-screen attribution, plus commentary, is the standard news-review practice, but it isn't a guarantee. Keep each source on screen only as long as the narration discusses it. Video *stills* captured from a source page are used like screenshots. Downloading and re-cutting another party's video clips still needs the user's explicit decision (not yet given). Always list sources in the description.
