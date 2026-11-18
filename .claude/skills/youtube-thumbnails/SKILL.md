---
name: youtube-thumbnails
description: Design and render evidence-led YouTube thumbnails for the Pixels2Actions channel (project physical-ai-daily): one honest curiosity hook, a subject "face", and a tilted source "receipt" with the key claim highlighted, in the episode's design (blueprint / blueprint-light). Researches current thumbnail guidance on the web first, renders 8-10 candidate thumbnails, scores them against a rubric and picks one (keeping the next two for YouTube Test & Compare). Use when the user asks for a thumbnail, thumbnail variants, packaging, or to improve click-through.
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

**The tools are starting points, not limits.** If one doesn't fit the job, change it or write your own script; a missing tool or option is never a reason to stop. Keep the data formats between steps and the rules intact (`physical-ai-daily/docs/tools.md`).

## Workflow

### 0. Research first, every time

Thumbnail practice and YouTube's tools change, so don't rely only on the evidence above. Before designing:

1. **Search aggressively, not a token check.** Decide how many searches based on the episode's topics and what
   turns up. Send them in parallel batches, and use `extended` mode for anything recent or niche. Cover all of these
   angles:
   - new thumbnail studies and data (`YouTube thumbnail study <current year>`, `thumbnail CTR analysis`, vidIQ / 1of10
     / eden.so research)
   - YouTube's own changes: Test & Compare, the Studio packaging and title-test features, Creator Insider, YouTube Help
   - the episode's topic niche (`robotics YouTube thumbnails`, `AI explainer thumbnails what works`), plus what the
     current top videos on this story use (search the story name + YouTube)
   - curiosity, clickbait and trust research (new papers on misleading thumbnails or source cues)
   - text, faces and readability on mobile and TV home feeds
   - anything that contradicts our rules: search for counter-evidence on purpose (`thumbnail text hurts CTR`,
     `faces don't matter thumbnails`)

   Open the strongest sources (primary studies, YouTube docs) and read them rather than trusting search snippets. Keep
   going until new searches stop turning up new findings.
2. Prefer data (coded samples, A/B results, YouTube's own docs) over opinion blogs. Note each source's strength the way
   `docs/thumbnails.md` does.
3. Compare with the evidence list above. **If something is new or contradicts it, append a dated row to
   `docs/thumbnails.md`** and say how it changes this run (e.g. a new layout, more or less text).
4. Tell the user in one or two lines what the research changed, or that it confirmed the current approach.

### Rules from the user's reviews (2026-11-16/17)

- **One candidate per story at least.** Each Short ends on the thumbnail of its own story (promo-shorts skill), so
  the candidates double as the Shorts' end cards: cover every story the Shorts will use, not just the top pick.
- **Image labels are small and subtle** ("Simulation", "Self-reported"): they inform, they don't pull the eye. The
  hook stays big.
- **"AI-generated" only on media we generated ourselves.** Another lab's generated footage (e.g. Genie 4) gets no
  stamp; say what it is in the hook or credit ("Not filmed.", "Video: Google DeepMind · generated by Genie 4").
- **Newcomer-friendly hooks**, inform rather than sell (no pitchy phrasing).
- **Title and cover text work together, never repeat each other** (user, 2026-11-17). The title carries one part
  (e.g. the fact or score: "GPT-6 Astra vs. a power strip: 0 for 11"), the cover carries another (e.g. the contrast
  and the open question: "GPT-6 Astra moves blocks just fine. Why can't it plug in?"). Viewers see both at once, so
  repeated words are wasted space. Write them as a pair and check them side by side.

### 1. Generate options: 8-10 candidates

**A hook must say what happened and to what, not just a number** (user, 2026-11-18). Rejected: "Learned from
13.2 minutes" (learned what? from what?) and "Doubling every 54 days?" (what is doubling?). Accepted: "Learned to
climb / from 13 min of human moves" and "Getting better fast. / Still under 50%." (answers the title's question).
Test each hook on someone who hasn't seen the video: could they say in one sentence what it claims?

**Wide options, up front (user, 2026-11-18: "we iterate a lot here"):** at least 8 titles, each tagged with its angle (current / plainer / question / number-led / contrast), one marked ★; 8–10 rendered thumbnails or covers that vary the hook text (2–3 hooks on the same image) as well as the image; one lettered contact sheet of all of them; every title names the thumbnail it pairs with. All of it goes in an "Options" section of publish.md.

1. **Find the promise:** the one surprising, true thing in the episode, and where the video answers it (often the
   reality-check beat).
2. **Write 8-10 candidates that differ in idea, not just colour.** Cover several angles:
   - *tension* ("96% success / *but...*", answered by the reality check)
   - *question* ("Robot brain / *2× better?*", answered by the 30–100% range)
   - *mystery* ("No battery. / *How?*", answered by the explainer)
   - *stakes* (what changes for the viewer or the industry, if the video says so)
   - *contrast* (old way vs new, using the `split` layout)
   - *number* (the single most striking verified number, using the `stat` layout)

   Vary the subject image and receipt too, where the episode has more than one good option. Any layout in
   `Thumbnail.tsx` is allowed; `evidence` is the default.
3. **Honesty check on every candidate** (the channel's script rules apply): every number on the thumbnail is in the
   source receipt and the video. Questions must not imply a false answer. No "shocking", no fake urgency, no implying a
   deployment that was a pilot. Drop any candidate that fails; don't fix it by blurring the claim.
4. **Images:** a robot face or close-up beats a wide shot. Prefer CC BY figures (arXiv) or our own art; press frames
   (company videos) are the user's call, since a thumbnail is promotional.
5. **Receipt:** the captured sentence or post that states the number. Crop tightly so the highlighted words are large.

### 2. Render all candidates

Put every candidate in `episode.json` → `"thumbnails"`, then render them in the episode's design:

```
python3 tools/thumbnails.py episodes/<dir> <design>
```

This writes `out/thumb_<n>_<design>.png` plus `out/thumbs_preview.png` (each one beside its 168×94 sidebar size).

### 3. Choose one: score, don't guess

**Look at** `out/thumbs_preview.png` and each full-size PNG (Read them). Score every candidate 0-2 on each criterion,
and show the table to the user:

| Criterion | 2 = | Evidence |
|---|---|---|
| Readable at 168×94 | hook legible in the sidebar size | practitioner consensus |
| One clear focal subject | face/robot big, not covered by text | vidIQ breakout study |
| Honest curiosity gap | raises a question the video answers | information-gap theory, clickbait studies |
| Visible proof | receipt shows the claim, number highlighted and legible at full size | source-cue experiments |
| Complements the title | doesn't repeat the title's words | practitioner consensus |
| Series consistency | design, brand mark top-left, bottom-right clear | brand rules (`docs/brand.md`) |
| Distinct at a glance | the idea reads in about a second | practitioner consensus |

Rules:
- Any candidate that failed the honesty check is out, whatever it scores.
- **Pick the highest total.** On a tie, prefer the one whose question the video answers earliest.
- Keep the runners-up 2 and 3 for YouTube **Test & Compare** (it tests up to 3).
- If every candidate scores 1 or less on readability or focal subject, revise crops (`crop`, `mark`, `focus`, `zoom`) or
  hooks and render again, rather than picking the least bad.

### 4. Deliver

Copy the winner to `out/thumbnail.png` and the runners-up to `out/thumbnail_test_2.png` / `_3.png`. Report:
- what the research changed
- the score table
- why the winner won
- where the files are

After Test & Compare results come in, record which angle won in `docs/thumbnails.md`, so future runs can weight it.

### Example `episode.json` entry

```json
"thumbnails": [
  {"layout": "evidence", "lines": ["96% success", "*but...*"], "image": "assets/dm_merc_36s.png", "focus": "55% 35%",
   "receipt": {"src": "assets/s2_tweet.png", "source": "GOOGLE DEEPMIND",
               "crop": {"x": 0, "y": 0, "w": 1, "h": 0.34}, "mark": {"x": 0.845, "y": 0.665, "w": 0.095, "h": 0.12}}}
]
```
