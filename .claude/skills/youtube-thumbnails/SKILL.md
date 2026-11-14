---
name: youtube-thumbnails
description: Design and render evidence-led YouTube thumbnails for the Pixels2Actions channel (project physical-ai-daily): one honest curiosity hook, a subject "face", and a tilted source "receipt" with the key claim highlighted, in the episode's design (blueprint / blueprint-light). Produces 3 variants for YouTube Test & Compare plus a 168x94 readability check. Use when the user asks for a thumbnail, thumbnail variants, packaging, or to improve click-through.
---

# YouTube thumbnails (Pixels2Actions)

Research and sources: `physical-ai-daily/docs/thumbnails.md`. Template: `physical-ai-daily/video/src/Thumbnail.tsx`.
Renderer: `physical-ai-daily/tools/thumbnails.py`.

## What the evidence supports (and how strongly)

1. **Faces and short text set breakout thumbnails apart; contrast alone doesn't.** vidIQ, 500 breakouts, July 2026
   (correlational). Our "face" is the robot or the main subject, shown big.
2. **A curiosity gap invites the click, but only if the video closes it.** Loewenstein's information-gap theory;
   clickbait studies show misleading packaging raises clicks while repeated exposure breeds scepticism, and satisfaction
   follows expectation vs delivery. So the hook asks a question **the video answers**. A literal "WATCH NOW" has no
   evidence behind it and reads as spam: the open question *is* the call to watch.
3. **Source cues raise clicks and trust.** High-reputation sources get clicked more regardless of content (Winter &
   Krämer); expert quotes raise trust. Credibility *labels* alone don't change behaviour, so show the actual receipt:
   the source's own sentence with the number highlighted, not just a logo.
4. **Thumbnail and title split one promise.** The thumbnail is the visual hook; the title adds context. Never repeat the
   title's words (practitioner consensus: vidIQ, Hooklab, Overseeros).
5. **Readable at 168×94** (the sidebar size); 1280×720, under 2 MB.
6. **Arrows and circles: mixed evidence.** Use them only to point at something small. The receipt's highlighter is our
   one pointer.
7. **Test, don't guess:** YouTube Studio's Test & Compare runs up to 3 thumbnails and picks on watch time.

## The template ("evidence" layout)

```
+--------------------------------------------------+
| [mark]                         |                  |
| HOOK LINE ONE                  |   SUBJECT        |
| *HOOK TWO*  (accent colour)    |   (face/robot,   |
|                                |    big, right)   |
|  SOURCE: <WHO>                 |                  |
|  /receipt crop, tilted -3deg/  |                  |
|  key number highlighted        |                  |
+--------------------------------------------------+
```

- **Brand mark** top-left only (YouTube draws the duration at bottom-right).
- **Hook:** 2 lines, ≤ ~4 words. The colour line is the curiosity gap. Text auto-fits and never wraps.
- **Subject:** right ~55%, faded into the background; `focus` picks the crop and `zoom` cuts figure labels.
- **Receipt:** a crop of a page we already captured for the episode (`assets/*.png` + its `.json`). The tag names the
  source ("GOOGLE DEEPMIND", "ARXIV PAPER"). If the capture has no highlight, add `mark` (fractions of the crop).
- **Design** follows the episode: `blueprint` (dark) or `blueprint-light`.

## Workflow

1. **Find the promise.** What's the one surprising, true thing in the episode? Find where the video answers it (often
   the reality-check beat).
2. **Write 3 hooks, one per angle:** *tension* ("96% success / *but...*", answered by the reality check), *question*
   ("Robot brain / *2× better?*", answered with the 30–100% range), *mystery* ("No battery. / *How?*", answered by the
   mousetrap explainer).
3. **Honesty check** (the channel's script rules apply): every number on the thumbnail is in the source receipt and the
   video. Questions must not imply a false answer. No "shocking", no fake urgency, no implying a deployment that was a
   pilot.
4. **Pick the subject image:** a robot face or close-up beats a wide shot. Prefer CC BY figures (arXiv) or our own
   art; press frames (company videos) are the user's call, since a thumbnail is promotional.
5. **Pick the receipt:** the captured sentence or post that states the number. Crop tightly so the highlighted words are
   large.
6. Add the variants to `episode.json`:

```json
"thumbnails": [
  {"layout": "evidence", "lines": ["96% success", "*but...*"], "image": "assets/dm_merc_36s.png", "focus": "55% 35%",
   "receipt": {"src": "assets/s2_tweet.png", "source": "GOOGLE DEEPMIND",
               "crop": {"x": 0, "y": 0, "w": 1, "h": 0.34}, "mark": {"x": 0.845, "y": 0.665, "w": 0.095, "h": 0.12}}}
]
```

7. **Render:** `python3 tools/thumbnails.py episodes/<dir> blueprint blueprint-light` → `out/thumb_<n>_<design>.png`
   and `out/thumbs_preview.png`.
8. **Look at the preview** (Read the PNG): is the hook legible at 168 px? Is the receipt's highlighted number legible
   at full size? Is the face clear of the text? Fix crops (`crop`, `mark`, `focus`, `zoom`) and re-render.
9. Give the user the 3 for Test & Compare. Once results come in, note which angle won in `docs/thumbnails.md`.
