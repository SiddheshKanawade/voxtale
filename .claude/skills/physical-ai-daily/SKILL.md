---
name: physical-ai-daily
description: Produce an episode of the Pixels2Actions YouTube channel (project name physical-ai-daily) — discover the day's physical-AI stories (news, X posts, arXiv papers), verify each against its primary source, write a narration script, capture highlighted screenshots of the sources, synthesize voice, and render a 1080p video with Remotion on this Mac. Use when the user asks to make today's episode, a sample episode, add a story, re-render, or change the channel's video format.
---

# Pixels2Actions (project: physical-ai-daily)

The channel is named **Pixels2Actions**, after the user's brand (pixelstoactions.com). The folder, skill and
repo names stay `physical-ai-daily`.

Project: `physical-ai-daily/` in the voxtale repo. Everything runs locally on the Mac.

```
sources.json                 accounts, arXiv categories, news queries (user-editable)
tools/discover.py            arXiv + X + news candidates with signals and priors -> deduped JSON (physical-ai-research skill)
tools/credits.py             remaining credit per paid source (twitterapi.io, NewsData, ElevenLabs, Render)
tools/triage_batches.py      harvest -> reading batches for triage subagents
tools/chatter.py             what X, HN and YouTube say about one story
tools/news_daily.sh          scheduled NewsData pull (launchd via tools/install_news_schedule.sh) into news/
tools/capture.mjs            page / tweet / element / video-frame screenshots via installed Chrome, in-page highlight
tools/crop.py                crop a figure panel (fractions) + write its metadata json
tools/voice.py               episode.json -> audio/<seg>_<beat>.wav + props.json (word timings, punctuation, media metadata)
tools/render.sh              voice -> QA contact sheet -> MP4 -> loudness-normalized final (-14 LUFS)
video/src/Episode.tsx        chrome: top bar, chapter progress bar, story header, credit chip, captions
video/src/scenes.tsx         scene catalog (below) + word-cue timing + hold-cap punch-in
video/src/diagrams.tsx       custom animated explainers (kitting, snap-through)
video/src/brand.tsx          Pixels2Actions logo mark, wordmark, lockup, and the sting after the cold open
episodes/<date>[-slug]/      episode.json, assets/, audio/, qa/, out/
```

**Reference episode: `episodes/2026-11-14-sample-v7-blueprint/`** (frozen 2026-11-14: 2 stories, 4:04, 36 beats).
Copy its `episode.json` shape, including `design`, `tts`/`voice`/`model`, per-beat `section`, and `thumbnails`.
`episodes/2026-11-14-sample-v7-blueprint-light/` is the same episode in the light design.

**Remotion know-how:** Remotion's official agent skill `remotion-best-practices` (animation, timing, audio, fonts,
captions, rendering, docs lookup) is in the repo as a pinned git submodule (`vendor/remotion-skills`, not copied, because its
license isn't open source); `tools/link_skills.sh` sets up every skill on a new machine (`docs/skills.md`). Load it when changing `video/src/`. Where it differs from this
skill, **this skill wins**: render with `tools/render.sh` and its QA steps (it renders on our say-so, without opening
Studio first), and the frozen v1.0 format rules stand.

**The tools are starting points, not limits.** If one doesn't fit the job, change it or write your own script; a missing tool or option is never a reason to stop. Keep the data formats between steps and the rules intact (`physical-ai-daily/docs/tools.md`).

## Audience and pacing (user feedback, 2026-11-13)

- **The viewer has a slight technical background but little knowledge of the field** (updated 2026-11-14; was "a
  product manager"). Hard concepts are fine when explained so that viewer can follow. Define every term on screen
  (an acronym scene for "VLA"), use one everyday analogy per hard idea ("works like a mousetrap"), and end each story
  with *why it matters* plus a reality check.
- **Keep it moving.** No single static visual holds for more than **~5–7 s** (`HOLD_CAP_S = 7`). Cut points follow the
  audio: each beat is one narration chunk and one visual, and elements inside a scene appear on the word that names them.
  This is a cap, not a fixed rhythm. Short beats are fine.
- Prefer real source imagery (video stills from the announcement, paper figure panels) and animated explainers over
  page screenshots. Screenshots appear only for the headline and the highlighted proof sentence.
- Length is driven by content. Aim for 3–5 stories at 60–110 s each. Don't pad toward a target length.
- **Cadence (user plan, 2026-11-14):** one long-form video a week with the week's top 3–5 stories, explained as now.
  Shorts cut from it promote it. This is unlike the P2A digest, which is daily.

## Pipeline

1. **Research.** Run the `physical-ai-research` skill: credit check, harvest (`tools/discover.py`: arXiv, X,
   NewsData.io), subagent triage, web search, what people are saying, scoring, and verification. It writes `<ep>/candidates.json` and a scored shortlist in
   `<ep>/research.md`. Never pad to reach a story count.
2. **Pick.** Take the shortlist's top 3–5. Show it to the user before scripting unless they said to run unattended.
3. **Re-check** each pick against the verification rules below while capturing; research doesn't replace them.
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
- The intro teases the stories with visuals in about 20 s. The outro opens with "That's Pixels to Actions." (written
  that way so the TTS says it naturally), points to the sources and cues Subscribe. Never mention the website or the
  digest in the video: the brand is meant to be a subtle presence.

## Source access (tested 2026-11-13)

| Source | Method | Status |
|---|---|---|
| arXiv | export.arxiv.org API (`discover.py`: cs.RO plus cs.CV/LG/AI/eess.SY filtered by our terms and the digest keywords), abs/html pages | Reliable. arXiv content is often CC BY; check the license line. |
| News discovery | NewsData.io `/latest` (`discover.py`, key `NEWSDATAIO`) plus web search | Needs a key; free tier is credit-limited. Without it, web search only. |
| Hacker News, YouTube | `chatter.py` per shortlisted story (HN Algolia; yt-dlp search) | No key. Not harvested weekly: mostly off-topic. YouTube captions are rate-limited at times |
| Paper signals | alphaXiv metadata (orgs, votes, visits), Hugging Face Daily Papers, GitHub (`discover.py`) | No key; alphaXiv's endpoint is undocumented |
| News/blogs | `capture.mjs page` (headless Chrome; only ever clicks reject/necessary-only consent buttons) | Reliable for public pages. Paywalls are not bypassed. |
| Single X post | `capture.mjs tweet <id>` (official embed); `cdn.syndication.twimg.com/tweet-result?id=` for JSON | Reliable without login |
| X discovery | twitterapi.io `advanced_search` (`discover.py`, key `TWITTERAPI_IO_KEY`): batched `from:` queries plus keyword search with a likes floor | Paid per tweet. Without the key, falls back to `syndication.twitter.com` timelines: **unreliable** (30 per window, sometimes stale) |
| x.com directly | headless browser | Blank page without login |

## Voice (frozen 2026-11-14)

**ElevenLabs `eleven_v3`, voice Eric (`cjVigY5qzO86Huf0OWal`).** Set in episode.json as `"tts": "elevenlabs"`, `"voice"`,
`"model"`. The user picked it twice in blind tests, as the most realistic voice, over Edge Andrew, directed Andrew and
six Eleven v4 voices (`docs/voice-research.md` sections 7b–9).
- Each beat is one call on its narration. Pauses come from punctuation; don't split beats into spliced phrases (the
  user found the directed splices unnatural). An optional `eleven_text` can add v3 audio tags like `[short pause]`.
- Word timings come from the API's character timestamps; `voice.py` aborts if they're missing.
- `ELEVENLABS_API_KEY` lives in `physical-ai-daily/.env` (git-ignored; never print or commit it). Free tier:
  10,000 characters a month. An episode is about 3,600. Beats are cached in `audio/*.el.*`, so edits re-voice only the
  changed beats. Check quota with `GET /v1/user/subscription`.
- Free accounts can't use Voice Library voices over the API; Eric is a built-in voice.
- Fallback if ElevenLabs is unavailable: remove `tts` and set `"voice": "en-US-AndrewNeural"` (edge-tts, free).

## Synthetic images

Higgsfield MCP has image models (Nano Banana Pro/2, GPT Image 2.6, FLUX), but the account had **0 credits** on 2026-11-13. Check `balance` first. Use synthetic images only for explainer diagrams or concepts with no source visual, and never for something that looks like a real event or product.

## Format (FROZEN 2026-11-14: v1.0, 16:9 long-form; the user approved it)

Don't change the identity without the user's say-so. The research behind it is in `docs/brand.md` (identity),
`docs/attention.md` (what holds attention) and `docs/thumbnails.md`.

- **Design:** `"design": "blueprint"` (default): near-black `#0a0a0b` with a fine **static** engineering grid.
  `"blueprint-light"` is the approved alternative: warm paper `#fbfaf7`, ink grid and text; photos become framed cards.
  The other candidates in `video/src/design.ts` (charcoal, paper, signal, aurora) are unused.
- **Brand:** top bar = Pixels2Actions lockup (logo mark + wordmark, `brand.tsx`) + date. Red `#f43f2e` for label chips
  and the chapter progress bar; yellow `#f0c945` for accents and the spoken-word caption highlight.
- **Sonic logo** (`tools/brand_audio.py`, generated, original): plays only in silent gaps, with the 1.6 s logo sting
  after the cold open and on a 1.6 s end-card tail. Never under the voice.
- **Named sections** recur in every story, as a visual chip only (no sound): set `"section"` on the beat where it
  starts. Values: `plain` (In plain English), `evidence` (The evidence), `matters` (Why it matters), `reality`
  (Reality check). The caveat beat's narration opens with "Reality check."
- **Every beat:** label chip + headline + "STORY n/N", credit chip, and **captions (required, the user's call)**:
  punctuated, word-synced, grouped at clause breaks.
- **Attention rules** (`docs/attention.md`): no moving or decorative backgrounds; no music or effects under speech;
  one focal element per beat; a new visual every 5–7 s; the first item on screen within 0.4 s; episodes under about
  6 min.
- **End card:** lockup, "Sources linked in the description", Subscribe. Never the website, URL or digest. The outro
  narration opens "That's Pixels to Actions."
- **Thumbnails:** use the `youtube-thumbnails` skill. It researches first, makes 5–7 candidates, rubric-scores them,
  picks one, and keeps two runners-up for Test & Compare.
- **Light-mode QA:** after any visual change, sample about 15 frames across the render (dark and light) and look for
  empty stages, washed-out media and colours that only work on dark.
- A Short (9:16) version isn't built yet.

## Copyright

Screenshots of articles and posts with on-screen attribution, plus commentary, is the standard news-review practice, but it isn't a guarantee. Keep each source on screen only as long as the narration discusses it. Video *stills* captured from a source page are used like screenshots. Downloading and re-cutting another party's video clips still needs the user's explicit decision (not yet given). Always list sources in the description.
