---
name: youtube-thumbnails
description: Write the titles and render the thumbnails for the Pixels2Actions channel (project physical-ai-daily). One lead story, packaged as a robot doing something concrete; the thumbnail shows that story's decisive moment (a real frame) with 0–3 words that add to the title and a label naming the robot, in the episode's design (blueprint / blueprint-light). Researches current guidance first, renders 8–10 candidates on a lettered contact sheet, scores them against a rubric and picks one. Use when the user asks for a title, a thumbnail, thumbnail variants, packaging, or better click-through.
---

# Titles and thumbnails (Pixels2Actions)

Research: `physical-ai-daily/docs/packaging.md`, on content (2026-11-22), which supersedes the receipt approach.
Design rounds 1–3: `physical-ai-daily/docs/thumbnails.md`. Template: `physical-ai-daily/video/src/Thumbnail.tsx`.
Renderer: `physical-ai-daily/tools/thumbnails.py`.

**Why this changed (user, 2026-11-22):** "I am not liking the titles and thumbnails that you are generating. It's
not a design problem, rather a content." The noun-list roundup titles and the sentence-plus-receipt thumbnails were the
problem.

## What the evidence supports (and how strongly)

Details and sources are in `docs/packaging.md`.

1. **The idea matters more than the wording.** In 24,333 Upworthy A/B pairs, wording picked the winner only 54% of the
   time. YouTube: "You need a compelling idea, or everything else really won't matter". So the first job is choosing
   **the lead story**: the single most surprising true result that a newcomer can picture.
2. **One story, one promise.** No niche video that beat its channel's size was a roundup. Noun lists ("A, B, and 3
   more") sit at the bottom.
3. **The promise pays off early.** Test & Compare picks on watch time, and YouTube measures viewer satisfaction. The
   title's lead story is the **first** story in the episode.
4. **Concrete, plain words, a protagonist acting, names last** ("This Robot Learned Cornhole From a Video of a Person",
   not "Gemini 4 Argon for Robotics"). Simpler headlines won across about 31,000 tests.
5. **Give the setup and withhold the outcome, the "how" or the catch.** Curiosity needs some knowledge plus an open
   gap. A complete factual sentence leaves no gap.
6. **Honest negativity beats hype.** In about 105,000 headline tests, each negative word added +2.3% CTR and each
   positive word cost 1.0%. A real limit from the source is a strong angle; "insane" and "breakthrough" are not.
7. **Statements over questions.** Questions underperformed in several datasets.
8. **The thumbnail shows the moment** (action, outcome or tension), not a portrait or a setup shot, with **0–3 words**.
   YouTube: titles and thumbnails "work together to create interest and tell a story".
9. **No source screenshot on the thumbnail.** No evidence says it helps, it's unreadable at feed size, and it's a
   second idea. Proof goes in the video's first minute and in the description.
10. **Readable at 168×94** (the sidebar); 1280×720, under 2 MB.
11. **Test, don't guess:** Test & Compare runs up to 3 options and picks on watch time.

## The template ("moment" layout, the default)

```
+--------------------------------------------------+
| [mark]                                            |
|  [ROBOT NAME + TYPE]  (red chip)                  |
|  0-3 BIG WORDS          the decisive instant,     |
|  (outcome / catch /     full-bleed real frame,    |
|   how)                  subject on the right      |
+--------------------------------------------------+
```

- **Image:** one real frame of the decisive instant: the bag leaving the hand, the foot on the cube, the robot holding
  the ball. A frame from the source's own video or a CC BY figure. Never a generated or composited image passed off as
  real.
- **Words:** 0–3, adding what the title doesn't say: "7 in a row", "No hands", "A person steers". Break them over two
  lines to get bigger type.
- **Label (`label`):** names the robot and its type ("APOLLO HUMANOID", "UNITREE G1", "ANYMAL-D"), so the words can be
  short and still say what happened to what.
- **Brand mark** top-left only; YouTube draws the duration at bottom-right.
- **Design** follows the episode: `blueprint` (dark) or `blueprint-light`.
- Other layouts stay available: `split` (two real frames with `labels`, for a true before/after or two-story
  contrast), `stat` (only when the number *is* the story), `face`. The `evidence` (receipt) layout is retired as a
  default.

**The tools are starting points, not limits.** If one doesn't fit the job, change it or write your own script; a missing tool or option is never a reason to stop. Keep the data formats between steps and the rules intact (`physical-ai-daily/docs/tools.md`).

## Titles

- **Lead with one story.** Write "[protagonist] [did concrete thing]", with the plain word first and the name last or in
  parentheses: "This Robot Learned Cornhole From a Video of a Person (Gemini 4 Argon)". A roundup may add "(plus 4
  more)". Never a list of names.
- **Plain words a newcomer knows:** "robot dog", "humanoid", "robot arm". No acronyms or model names up front.
- **Statements, not questions.** No hype words. An honest catch from the source is allowed and often strongest:
  "Google Says Its Robot Sank 7 in a Row. It Didn't Say How Many Tries".
- About 70 characters or fewer; no date suffix (the week goes in the description). No "Robotics this week:" prefix.
- **Honesty:** every claim is in the source and the video, and the video pays the title off in its first story.

## Workflow

### 0. Research first, every time

Practice and YouTube's tools change, so don't rely only on the evidence above. Before writing:

1. **Search properly, not a token check.** Send searches in parallel batches, and use `extended` mode for anything
   recent or niche. Cover:
   - new title and thumbnail studies (vidIQ / 1of10 / headline A/B research)
   - YouTube's own changes (Test & Compare, Creator Insider, YouTube Help, the Liaison)
   - what the top videos on this week's lead story use: search the story name plus YouTube, and check views against
     channel size
   - counter-evidence to our rules, searched for on purpose

   Open the strongest sources and read them; don't trust snippets.
2. **Pull niche data when the doc is stale** (over ~60 days). Use yt-dlp metadata of recent robotics videos, rank them by
   views ÷ subscribers, and look at the outliers' titles and thumbnails. Study thumbnails privately and delete them
   after.
3. Append new or contradicting findings, dated, to `docs/packaging.md` and say how they change this run.
4. Tell the user in a line or two what the research changed or confirmed.

### Rules from the user's reviews (still in force)

- **One candidate per story at least.** Each Short ends on its own story's thumbnail (promo-shorts skill), so cover
  every story the Shorts use.
- **Labels are small and factual;** the words stay big. "AI-generated" only on media we generated. Another lab's
  generated footage is identified in the credit, never stamped.
- **Title and thumbnail work together and never repeat each other** (2026-11-17). Write them as a pair and check them
  side by side.
- **Words + label + title together say what happened and to what** (2026-11-18). Rejected for vagueness: "Learned
  from 13.2 minutes", "Doubling every 54 days?", "AI-run robot arms". Test on someone who hasn't seen the video: could
  they say in one sentence what it claims?
- **No acronyms a newcomer can't read** ("WBD holds on" was rejected, 2026-11-21). Name the robot or company.
- **Wide options, up front** (2026-11-18, "we iterate a lot here"):
  - At least 8 titles, each tagged with its angle (plainer / current / protagonist / honest catch / how / number-led /
    contrast), one marked ★.
  - 8–10 rendered thumbnails varying the image as well as the words.
  - One lettered contact sheet; every title names the thumbnail it pairs with.
  - All of it goes in an "Options" section of publish.md, and in chat at hand-off.

### 1. Generate options: 8–10 candidates

1. **Pick the lead story** before anything else (in a weekly package, before the episode is scripted, so it opens the
   episode): the most surprising true result a newcomer can picture, with a showable moment.
2. **Find each story's moment.** Scan the clips at 2–6 fps (`ffmpeg -vf "fps=6,scale=300:-1,tile=6x4"`), pick the
   decisive frame, and extract it at full resolution. Measure where the subject sits (its x-range) before setting
   `focus`; don't guess. Crop off burned-in source labels ("Solving – Step 3/15", "1x").
3. **Write 8–10 candidates that differ in idea**, mostly on the lead story, plus at least one per other story (for the
   Shorts' end cards). Angles for the words: *outcome* ("7 in a row"), *catch* ("A person steers"), *how* ("No hands"),
   *contrast* (`split` with two real frames), *number* (`stat`, only when the number is the story).
4. **Honesty check on every candidate:** every word and number is in the source and the video; no implied result the
   video doesn't show; no fake urgency. Drop failures; don't blur the claim.
5. **Images:** prefer the source's own footage frames or CC BY figures. Press frames are promotional use; the user
   decides.

**Then the A/B test set**, when the user wants one (YouTube Test & Compare, "Title and thumbnail" mode). Read
`physical-ai-daily/docs/research-ab-testing.md` first. Build 3 title + thumbnail pairs, each a different *idea* about
the lead story (outcome vs. catch vs. how), or another story only if it plays in the first ~2 minutes. ★ goes first.
Write them to `out/ab-test/`. The rest of the rules are in the weekly-package skill under "A/B test set".

### 2. Render all candidates

Put every candidate in `episode.json` → `"thumbnails"`, then render in the episode's design:

```
python3 tools/thumbnails.py episodes/<dir> <design>
```

This writes `out/thumb_<n>_<design>.png` plus `out/thumbs_preview.png` (each beside its 168×94 size).

### 3. Choose one: score, don't guess

**Look at** `out/thumbs_preview.png` and each full-size PNG. Score every candidate 0–2 per criterion and show the table:

| Criterion | 2 = | Evidence |
|---|---|---|
| Shows the moment | the decisive instant (action, outcome, tension), not a portrait or setup | YouTube guidance; niche outliers |
| One focal subject | the robot big and clear, not covered by words | vidIQ breakout study |
| Words add to the title | 0–3 words carrying the outcome, catch or how; no repeat of the title | YouTube guidance; practitioner consensus |
| Honest curiosity gap | the open part is answered in the video's first minute | information-gap research; Test & Compare on watch time |
| Plain to a newcomer | label + words + title say what happened to what, no jargon | headline simplicity tests; user's rules |
| Readable at 168×94 | words and subject legible in the sidebar size | practitioner consensus |
| Series consistency | design, brand mark top-left, bottom-right clear | `docs/brand.md` |

- Anything that failed the honesty check is out, whatever it scores.
- **Pick the highest total.** On a tie, prefer the lead story.
- If every candidate scores 1 or less on "Shows the moment", find better frames rather than picking the least bad.

### 4. Deliver

Copy the winner to `out/thumbnail.png`, and the A/B pairs (if any) to `out/ab-test/`. Report what the research
changed, the score table, why the winner won, and where the files are. After a Test & Compare result, record which idea
won in `docs/youtube-feedback.md`.

### Example `episode.json` entry

```json
"thumbnails": [
  {"layout": "moment", "lines": ["7 in a row"], "label": "APOLLO HUMANOID",
   "image": "assets/g_moment_release.png", "focus": "50% 50%"}
]
```

Paired title: "This Robot Learned Cornhole From a Video of a Person (Gemini 4 Argon)".
