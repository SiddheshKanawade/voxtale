# Video requirements: when an episode is done

The acceptance checklist for every Pixels2Actions episode. An episode isn't ready to hand off until every **MUST**
passes; a **SHOULD** can be broken on purpose, with the reason written in the hand-off. This file collects rules that
live in the `physical-ai-daily` skill, `brand.md`, `attention.md`, `thumbnails.md` and the tools; where they
disagree, the skill wins and this file gets fixed. Each item says how to check it.

## 1. Content and sourcing

| # | Requirement | Level | How to check |
|---|---|---|---|
| 1.1 | Five stories from the week's `research.md` picks (fewer only if the week truly has fewer; never padded) | MUST | Compare `episode.json` segments with `research.md` |
| 1.2 | At least 2 of the 5 are **learning** stories (new capability, method or experiment with a result), and they come first | MUST | `kind` in `judge/stories.json`; segment order |
| 1.3 | Every fact, number, quote and opinion in the narration comes from a source in `research.md` or a page captured for the episode; nothing is generated | MUST | Read each beat against the story's quotes |
| 1.4 | Every number spoken appears verbatim on a captured page, highlighted with `capture.mjs --find` | MUST | No `WARN: text not found` from `capture.mjs`; the `shot` beat shows the highlight |
| 1.5 | Each story has a primary source dated inside the episode window, or the narration says it's older | MUST | `sourceDate` per segment vs the window |
| 1.6 | Epistemic status said out loud as the source states it: pilot, proof of concept, preprint, finalist, preview, plan, in talks, self-reported / vendor-reported | MUST | Script read-through against the gaps in `research.md` |
| 1.7 | Opinions are attributed to the person or organisation named on the page; a single post is never "people say" | MUST | Script read-through |
| 1.8 | Each number stays attached to its exact setting ("96% in a pilot on Mercedes lines", not "96%") | MUST | Script read-through |

## 2. Story structure and script

| # | Requirement | Level | How to check |
|---|---|---|---|
| 2.1 | Intro teases the stories with visuals in about 20 s | MUST | Intro segment duration in `props.json` |
| 2.2 | Each story follows: hook → in plain English (analogy + diagram) → evidence (source visual + highlighted sentence) → why it matters → reality check | MUST | Beat `section` values: `plain`, `evidence`, `matters`, `reality` appear in order |
| 2.3 | Each story opens "Story one / two …"; the caveat beat opens "Reality check."; the outro opens "That's Pixels to Actions." | MUST | Narration text |
| 2.4 | Every term is defined on screen when first used (an `acronym` scene for VLA and the like); one everyday analogy per hard idea | MUST | Script read-through |
| 2.5 | Story length about 60–90 s (150–280 words, 12–18 beats); episode about 7–8 min, cap ~9 min | SHOULD | `props.json` durations |
| 2.6 | No hype words ("shocking", "changes everything", "insane") | MUST | Search the narration |
| 2.7 | Numbers spelled as words in narration ("ninety-six percent"); digits on screen | MUST | Narration text |
| 2.8 | No mention of the website, its URL, the digest or pricing anywhere in the video | MUST | Script and end card |

## 3. Pacing and attention (`attention.md`)

| # | Requirement | Level | How to check |
|---|---|---|---|
| 3.1 | No static visual (photo, shot, kinetic) holds longer than 7 s; longer beats are split or punch-cut | MUST | `voice.py` prints no `WARN … holds` |
| 3.2 | A new visual every 5–7 s; one focal element per beat | SHOULD | Per-beat contact sheets |
| 3.3 | First element on screen within 0.4 s of a beat starting; list items appear on the word that names them (`cue`) | MUST | Frames at the start of beats |
| 3.4 | No moving or decorative background; no arrows pointing at things; no stacked visual + verbal cues | MUST | Contact sheets |
| 3.5 | No music or sound effects under speech; the sonic logo only in silent gaps (sting after the cold open, end tail) | MUST | Listen to the sting and end card; nothing else in the mix |

## 4. Visuals

| # | Requirement | Level | How to check |
|---|---|---|---|
| 4.1 | Real source imagery first (video stills, paper figures), animated explainers next; page screenshots only for the headline and the highlighted proof sentence | SHOULD | Count of `shot` beats vs others |
| 4.2 | Every captured PNG looked at before rendering: right page, not a consent wall, not blank, text legible | MUST | Look at `assets/` |
| 4.3 | Every beat shows the label chip, headline, "STORY n/N", credit chip, and captions | MUST | Per-beat contact sheets |
| 4.4 | No empty stages, washed-out media, or colours that work only on one design; text readable at 1080p | MUST | Light-mode QA: ~15 frames across the render, both designs |
| 4.5 | Synthetic images only for explainer concepts with no source visual, never for something that looks like a real event or product | MUST | Review any generated image |

## 5. Captions and voice

| # | Requirement | Level | How to check |
|---|---|---|---|
| 5.1 | Captions on every spoken beat: punctuated, word-synced, grouped at clause breaks | MUST | Frames mid-sentence |
| 5.2 | Voice: ElevenLabs `eleven_v3` "Eric" for published episodes; the free edge-tts "Andrew" only for tests or when credits are short (say so in the hand-off) | MUST | `episode.json` `tts`/`voice` |
| 5.3 | Word timings present for every beat (captions and cues depend on them) | MUST | `voice.py` completes without aborting |

## 6. Identity (frozen v1.0, `brand.md`)

| # | Requirement | Level | How to check |
|---|---|---|---|
| 6.1 | Design `blueprint-light` by default; `blueprint` (dark) when a dark version is asked for | MUST | `episode.json` `design` |
| 6.2 | Top bar: Pixels2Actions lockup and the episode date; red `#f43f2e` chips and progress bar; yellow `#f0c945` accents and caption highlight | MUST | Any frame |
| 6.3 | Logo sting (1.6 s) after the cold open; end card with lockup, "Sources linked in the description" and Subscribe on `subscribeCue` | MUST | Frames at the sting and the end |
| 6.4 | Identity changes only with the user's say-so | MUST | — |

## 7. Copyright

| # | Requirement | Level | How to check |
|---|---|---|---|
| 7.1 | Each source shown only while the narration discusses it, with on-screen attribution (credit chip) | MUST | Per-beat contact sheets |
| 7.2 | Video stills captured from source pages are fine; downloading or re-cutting another party's video clips is not (until the user decides otherwise) | MUST | `assets/` contains stills only |
| 7.3 | arXiv figures only when the abs page shows CC BY (or another reuse licence), credited | MUST | License line on the abs page |
| 7.4 | Every source listed in the description | MUST | Hand-off description block |

## 8. Technical output

| # | Requirement | Level | How to check |
|---|---|---|---|
| 8.1 | 1920×1080, 30 fps, H.264 video, AAC audio | MUST | `ffprobe out/*.mp4` |
| 8.2 | Loudness about −14 LUFS integrated, true peak ≤ −1.5 dBTP (`render.sh` masters it) | MUST | `ffmpeg -i out/x.mp4 -af ebur128 -f null -` |
| 8.3 | Files named `out/physical-ai-daily-<episode>-light.mp4` and `-dark.mp4` when both designs are made | SHOULD | `ls out/` |
| 8.4 | No missing media at render (no blank frames, no fallback boxes) | MUST | Per-beat contact sheets of the final file |

## 9. QA before hand-off

1. `tools/render.sh <ep> --stills-only` and read `qa/contact_sheet.png`.
2. After the full render: one frame per beat at ~75% of its length, tiled into contact sheets, every sheet read.
3. Light-mode QA: ~15 frames across the render in each design made.
4. Listen to the first 30 s, one full story, and the end card.
5. Fix, re-render, re-check what changed.

## 10. Hand-off (what the producer delivers)

- Path to each MP4 and its duration.
- Facts dropped or changed during re-verification, and why.
- What QA caught and what was done.
- Any requirement above that wasn't met, and why.
- A YouTube description block: title idea, chapters with timestamps, every source URL.
- Never upload: that's the user's action.
