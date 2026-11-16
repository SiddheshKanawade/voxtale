# Video requirements: when an episode is done

Guidance for every Pixels2Actions episode: what a good episode looks like, and how to check it. The producer
(an Opus-level model) follows it by default and uses judgement where a check doesn't fit the episode in front of it.

- **`GUIDE`**: the default. Follow it unless there's a good reason not to; in the rare case you depart from it, do
  what serves the viewer and the facts best, and say in the hand-off what you did and why.
- **`HARD`**: a firm rule set by the user (the sourcing rule, copyright, the identity, what never appears in the
  video). Don't depart from these; if one truly can't be met, stop and ask the user.

This file collects rules that live in the `physical-ai-daily` skill, `brand.md`, `attention.md`, `thumbnails.md` and
the tools; where they disagree, the skill wins and this file gets fixed.

## 1. Content and sourcing

- [ ] **1.1** `GUIDE` Five stories from the week's `research.md` picks (fewer only if the week truly has fewer; never padded)  
  *Check:* Compare `episode.json` segments with `research.md`
- [ ] **1.2** `GUIDE` At least 2 of the 5 are **learning** stories (new capability, method or experiment with a result), and they come first  
  *Check:* `kind` in `judge/stories.json`; segment order
- [ ] **1.3** `HARD` Every fact, number, quote and opinion in the narration comes from a source in `research.md` or a page captured for the episode; nothing is generated  
  *Check:* Read each beat against the story's quotes
- [ ] **1.4** `GUIDE` Every number spoken appears verbatim on a captured page, highlighted with `capture.mjs --find`  
  *Check:* No `WARN: text not found` from `capture.mjs`; the `shot` beat shows the highlight
- [ ] **1.5** `GUIDE` Each story has a primary source dated inside the episode window, or the narration says it's older  
  *Check:* `sourceDate` per segment vs the window
- [ ] **1.6** `GUIDE` Epistemic status said out loud as the source states it: pilot, proof of concept, preprint, finalist, preview, plan, in talks, self-reported / vendor-reported  
  *Check:* Script read-through against the gaps in `research.md`
- [ ] **1.7** `HARD` Opinions are attributed to the person or organisation named on the page; a single post is never "people say"  
  *Check:* Script read-through
- [ ] **1.8** `GUIDE` Each number stays attached to its exact setting ("96% in a pilot on Mercedes lines", not "96%")  
  *Check:* Script read-through

## 2. Story structure and script

- [ ] **2.1** `GUIDE` Intro teases the stories with visuals in about 20 s  
  *Check:* Intro segment duration in `props.json`
- [ ] **2.2** `GUIDE` Each story follows: hook → in plain English (analogy + diagram) → evidence (source visual + highlighted sentence) → why it matters → reality check  
  *Check:* Beat `section` values: `plain`, `evidence`, `matters`, `reality` appear in order
- [ ] **2.3** `GUIDE` Each story opens "Story one / two …"; the caveat beat opens "Reality check."; the outro opens "That's Pixels to Actions."  
  *Check:* Narration text
- [ ] **2.4** `GUIDE` Every term is defined on screen when first used (an `acronym` scene for VLA and the like); one everyday analogy per hard idea  
  *Check:* Script read-through
- [ ] **2.5** `GUIDE` Each story is as long as its explanation needs (no per-story time target); no padding. Episode: `HARD` cap 20 min  
  *Check:* `props.json` durations
- [ ] **2.6** `GUIDE` No hype words ("shocking", "changes everything", "insane")  
  *Check:* Search the narration
- [ ] **2.7** `GUIDE` Numbers spelled as words in narration ("ninety-six percent"); digits on screen  
  *Check:* Narration text
- [ ] **2.8** `HARD` No mention of the website, its URL, the digest or pricing anywhere in the video  
  *Check:* Script and end card

## 3. Pacing and attention (`attention.md`)

- [ ] **3.1** `GUIDE` No static visual (photo, shot, kinetic) holds longer than 7 s; longer beats are split or punch-cut  
  *Check:* `voice.py` prints no `WARN … holds`
- [ ] **3.2** `GUIDE` A new visual every 5–7 s; one focal element per beat  
  *Check:* Per-beat contact sheets
- [ ] **3.3** `GUIDE` First element on screen within 0.4 s of a beat starting; list items appear on the word that names them (`cue`)  
  *Check:* Frames at the start of beats
- [ ] **3.4** `GUIDE` No moving or decorative background; no arrows pointing at things; no stacked visual + verbal cues  
  *Check:* Contact sheets
- [ ] **3.5** `GUIDE` No music or sound effects under speech; the sonic logo only in silent gaps (sting after the cold open, end tail)  
  *Check:* Listen to the sting and end card; nothing else in the mix

## 4. Visuals

- [ ] **4.1** `GUIDE` Real source imagery first (video stills, paper figures), animated explainers next; page screenshots only for the headline and the highlighted proof sentence  
  *Check:* Count of `shot` beats vs others
- [ ] **4.2** `GUIDE` Every captured PNG looked at before rendering: right page, not a consent wall, not blank, text legible  
  *Check:* Look at `assets/`
- [ ] **4.3** `GUIDE` Every beat shows the label chip, headline, "STORY n/N", credit chip, and captions  
  *Check:* Per-beat contact sheets
- [ ] **4.4** `GUIDE` No empty stages, washed-out media, or colours that work only on one design; text readable at 1080p  
  *Check:* Light-mode QA: ~15 frames across the render, both designs
- [ ] **4.5** `HARD` Anything we generate (Higgsfield images or video, Blender animation) is used rarely, only to explain an idea when no real footage or diagram can, is labelled on screen ("Illustration" / "AI-generated"), and is never shown or narrated as real footage of an event, product or person  
  *Check:* Every generated asset has the label; the narration doesn't call it real

## 5. Captions and voice

- [ ] **5.1** `GUIDE` Captions on every spoken beat: punctuated, word-synced, grouped at clause breaks  
  *Check:* Frames mid-sentence
- [ ] **5.2** `GUIDE` Voice: ElevenLabs `eleven_v3` "Eric" for published episodes; the free edge-tts "Andrew" only for tests or when credits are short (say so in the hand-off)  
  *Check:* `episode.json` `tts`/`voice`
- [ ] **5.3** `GUIDE` Word timings present for every beat (captions and cues depend on them)  
  *Check:* `voice.py` completes without aborting

## 6. Identity (frozen v1.0, `brand.md`)

- [ ] **6.1** `GUIDE` Design `blueprint-light` by default; `blueprint` (dark) when a dark version is asked for  
  *Check:* `episode.json` `design`
- [ ] **6.2** `GUIDE` Top bar: Pixels2Actions lockup and the episode date; red `#f43f2e` chips and progress bar; yellow `#f0c945` accents and caption highlight  
  *Check:* Any frame
- [ ] **6.3** `GUIDE` Logo sting (1.6 s) after the cold open; end card with lockup, "Sources linked in the description" and Subscribe on `subscribeCue`  
  *Check:* Frames at the sting and the end
- [ ] **6.4** `HARD` Identity changes only with the user's say-so  
  *Check:* —

## 7. Copyright

- [ ] **7.1** `GUIDE` Each source shown only while the narration discusses it, with on-screen attribution (credit chip)  
  *Check:* Per-beat contact sheets
- [ ] **7.2** `HARD` Video clips from public sources are allowed **with proper citation**, preferring, in order: (1) official footage the company or lab itself published (newsroom, press kit, its own page or channel); (2) other public footage (news outlets, other creators), only when (1) doesn't show it. Every clip: a few seconds, our narration discussing it the whole time (commentary, not re-publishing), its own audio muted, credited on screen ("Video: Reuters") and listed in the description with its URL. Never re-use another creator's commentary or a whole segment. Wire-agency footage (AP, Reuters, Getty, Storyful) last, because Content ID claims are likely  
  *Check:* Each clip's source URL and owner in the hand-off; credit chip present; clip length (`tools/clip.py` enforces ≤ 10 s, muted, credit and URL; `voice.py` refuses a clip without them)
- [ ] **7.3** `HARD` arXiv figures only when the abs page shows CC BY (or another reuse licence), credited  
  *Check:* License line on the abs page
- [ ] **7.4** `GUIDE` Every source listed in the description  
  *Check:* Hand-off description block

## 8. Technical output

- [ ] **8.1** `GUIDE` 1920×1080, 30 fps, H.264 video, AAC audio  
  *Check:* `ffprobe out/*.mp4`
- [ ] **8.2** `GUIDE` Loudness about −14 LUFS integrated, true peak ≤ −1.5 dBTP (`render.sh` masters it)  
  *Check:* `ffmpeg -i out/x.mp4 -af ebur128 -f null -`
- [ ] **8.3** `GUIDE` Files named `out/physical-ai-daily-<episode>-light.mp4` and `-dark.mp4` when both designs are made  
  *Check:* `ls out/`
- [ ] **8.4** `GUIDE` No missing media at render (no blank frames, no fallback boxes)  
  *Check:* Per-beat contact sheets of the final file

## 9. QA before hand-off

- [ ] `tools/render.sh <ep> --stills-only` and read `qa/contact_sheet.png`.
- [ ] After the full render: one frame per beat at ~75% of its length, tiled into contact sheets, every sheet read.
- [ ] Light-mode QA: ~15 frames across the render in each design made.
- [ ] Listen to the first 30 s, one full story, and the end card.
- [ ] Fix, re-render, re-check what changed.

## 10. Hand-off (what the producer delivers)

- [ ] Path to each MP4 and its duration.
- [ ] Facts dropped or changed during re-verification, and why.
- [ ] What QA caught and what was done.
- [ ] Any requirement above that wasn't met, and why.
- [ ] A YouTube description block: title idea, chapters with timestamps, every source URL.
- [ ] `HARD` Never upload: that's the user's action.
