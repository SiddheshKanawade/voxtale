# How a deep dive should open (research, 2026-11-25)

**Why we did this:** the user reviewed the first deep dive (RoboBuilder) and pointed out that its first 30 s never said
what the video was about. The study wasn't named until 0:45, and the video never said what it would cover. We needed
an opening format backed by evidence.

**Method.** We ran two strands and checked one against the other:
- **Measured (niche).** We coded the first 90 s of 25 outlier videos from 15 robotics, physical-AI and AI-research
  channels.
  - Each video was 10–30 min, about one topic, at 2.3–74.9× the channel's median views.
  - We also coded a general-science reference group of 3 videos.
  - Sample, method and per-video tables: `research-deep-dive-intro-sample.md`. Data: `data/intro-sample.json`.
- **Sourced (web).** Findings W1–W35 come from:
  - YouTube's official guidance;
  - creator and strategist breakdowns;
  - journalism style guides (the "nut graf");
  - documentary craft;
  - learning science: curiosity, prequestions, advance organizers and signaling.

  Every finding and its source is in `research-deep-dive-intro-web.md`.

**Limits:**
- No study has tested opening structure against retention on YouTube or any online video. The web strand is official
  guidance, style guides, practitioner opinion and lab learning science.
- The measured strand is descriptive and survivor-biased (it only looks at outliers). It shows what working videos do,
  not what causes them to work.
- Caption timings are approximate (±2 s; ±3–5 s for 4 videos). One person coded the videos, from speech only.

## Where the two strands agree

| # | Finding | Measured (niche, n=25) | Sourced |
|---|---|---|---|
| A1 | **Speech starts at once** | First spoken word: median 0.1 s, latest 8.5 s | W1 (get straight to the value); W8 (only 2.2% of openings in ≥1M-view videos are greetings) |
| A2 | **The subject is named within seconds** | 23/25 name it within 90 s, at a median of 5 s. Research videos: median 16 s (IQR 3–20). Subject before the question in 13 of 16 | W3 (YouTube, the first 15 s: "Make it clear what your video is about early on"); W4 (the viewer knows what the video is within 5 s); W9 (Cleo Abram's first line names the subject); W18 (NPR names the study in paragraph 2, after a one-paragraph scene) |
| A3 | **Name the source early when the work is someone else's** | 12/16 outside-work videos name it within 90 s, at a median of 18 s (IQR 5–39) | W18 (in a study story, the study and journal come in paragraph 2 and the researcher in paragraph 3) |
| A4 | **Research videos may open on the problem, then name the subject** | 6/11 research videos put the problem or stakes first (0/8 builds). Example: Kevin Wood's AGIBOT video has footage, then the problem at ~13 s, stakes at ~18 s, and subject and source at ~20 s | W18 (scene, then nut graf); W21 (an evocative scene first, context after); W23 (a violated expectation triggers curiosity) |
| A5 | **Say why it matters early** | 22/25 within 90 s, at a median of 13 s | W16 (the nut graf says why care and why now) |
| A6 | **Make a promise by about 30 s** | 15/25 make one within 90 s, at a median of 29 s; 8/25 within 30 s | W1, W2 (the intro must keep the title and thumbnail's promise, judged at 30 s); W12 (Winston: "start with empowerment promise"); W16 (the nut graf "delivers a promise of the story's content") |
| A7 | **Keep a roadmap short and spoken as questions** | Only 9/25 have one, at a median of 34 s and 2 items. Kevin Wood: 3 items at ~25 s | W9 (questions work "almost like a table of contents"); L3/W28 (signaling d ≈ 0.69, and outlines help only with complex material); W17 (don't give away the ending) |
| A8 | **Put branding after the promise, or leave it out** | 2/25 have a channel line | W3 (branding comes later); W8 (greetings are rare) |
| A9 | **For a study, reveal the headline finding early, but not the resolution** | Kilcher summarizes the whole paper within ~55 s; Welch Labs states the mystery at 0–2 s | W18 (study stories give the finding by the nut graf); W17 (keep the ending); W23 (curiosity pays only when viewers expect an answer soon) |

## Where they conflict, and how this format resolves it

- **"No intro, open on a mystery" (Muller, C1) vs "say what it's about early" (W1, W3, W12, W16–18).** The data
  settles this for our niche: 18/25 name the subject first, and the research videos that open on a problem still name
  it by a median of 16 s. **Resolution:** a scene of at most 10 s, then the nut graf.
- **"Reveal the finding" (W18) vs "never give away the ending" (W17).** **Resolution:** reveal the anomaly the title
  promises (the finding), and hold the fix for later (the resolution).
- **Roadmaps help with complex material (W28) but are a minority practice (9/25).** **Resolution:** 2–3 questions in
  the order of the chapters, in no more than ~8 s, with no answers.
- **Rhetorical prequestions are untested.** The benefit is measured only when learners answer (W26). Treat the spoken
  questions as signposts, not as a learning intervention.

## The format (chapter 0, about 35–45 s)

| Slot | By | Job | Basis |
|---|---|---|---|
| 1. Scene | 0–10 s | The decisive moment on footage, outcome first. Narration starts in the first second; at most 2 sentences | A1, A4 |
| 2. Nut graf | ≤ 20 s | **What** (the study or project, by name), **who** (lab or company), **when or where** (date, venue), and what they did, in one or two sentences | A2, A3, A5 |
| 3. Finding | ≤ 30 s | The headline finding the title promises, with its denominator. This is the stake for an engineer | A6, A9 |
| 4. Promise | ≤ 40 s | 2–3 questions in chapter order (≤ 8 s), unanswered | A6, A7 |
| 5. Channel line | after 4 | "This is Pixels to Actions. We track the latest work in robotics and physical AI." (the user's line, once) | A8 |

**Rules:**
- The study is named by 20 s. Never open with a greeting or "In this video, I…".
- Time order: at most one jump, marked in words ("In the first round…", "Later…"). A clip that is not the event being
  described carries its label (clip audit).
- Chapter 1 does not repeat the nut graf.
- **Critic check:** from the first 45 s of the transcript alone, a fresh reader must be able to answer three questions
  what is this about and who did it (by 30 s), and what will I get from watching (by the end of slot 4).

## Order chosen by the user (2026-11-25)

After seeing the format above, the user chose this order: scene → finding → questions → intro line → "Today we're
looking at RoboBuilder, a preprint from MIT's DeCoDE Lab, posted on November eighteenth…". The study is named at about
32 s. The data supports it:
- The topic is clear from the scene (a fresh reviewer had it at ~8 s).
- The study's name isn't in the title or thumbnail, so the title's promise is still kept within 30 s.
- 7 of 25 niche videos open problem-first and name the subject at ~15–40 s. Welch Labs' 9.5× outlier names its
  subject at 24 s and its paper at 78 s.

The skill's opening table follows this order.
