---
name: promo-shorts
description: Make 9:16 promotional YouTube Shorts for a finished Pixels2Actions episode (project physical-ai-daily) — pick the 2–3 strongest stories, write a fresh 20–45 s script per Short (hook on frame 0, one open question, the payoff, one call to action to the full episode), reuse the episode's verified facts and captured visuals, voice and render with tools/render_short.sh in light and dark, and hand off titles, descriptions and the Related-video setting. Use when the user asks for Shorts, promos or vertical cuts of an episode.
---

# Promotional Shorts (Pixels2Actions)

A Short's job is to make someone who has never heard of the channel **stay** (past the first seconds), **get
something real** (the Short pays off its own hook), and **want the rest** (one tap to the full episode). Built like
an editor would by hand: finish the episode, then cut Shorts around it, with new scripts written for the format.

Evidence behind every rule, with sources and strength: `physical-ai-daily/docs/research-shorts-offers.md` (Hormozi,
YouTube's official guidance), `research-shorts-retention.md` (studies), `research-shorts-references.md` (12 reference
Shorts studied). Where they say "unverified" (the "3 seconds", loops), this skill doesn't lean on it.

**The tools are starting points, not limits** (`physical-ai-daily/docs/tools.md`). Shorts follow the channel's
**sourcing rule** and the HARD items of `docs/video-requirements.md` exactly like episodes.

## What YouTube measures (official)

Shorts are ranked on **% of viewers who chose to view** ("Stayed to watch": stayed past the initial seconds),
**average view duration** and **average % viewed**, among viewers who stayed. Shorts can run up to 3 min. The only
clickable path to the episode is the **Related video** link under the channel handle (set in Studio; links in
descriptions and comments aren't clickable). So: hook for Stayed-to-watch, retain for view duration, reward and one
CTA for Related-video clicks.

## 1. Pick 2–3 stories (user's choice: top 2–3 per episode)

From the episode's five, pick the stories with:
1. a **concrete first frame**: a robot doing the thing, a striking image, a number. Real subjects beat screenshots
   and title cards (references: every winner opened on the payoff; screenshots and title slides were losers);
2. a **visible puzzle**: a question the viewer can see and that the story answers;
3. at least one **learning** story (the channel's rule: learning over knowing).

One story per Short. Never a roundup ("so many updates this week…" was among the worst references).

## 2. Write the script (`shorts/<slug>/episode.json`)

**Length:** 20–45 s to start (research: about 20–40 s, then let our own engaged-views data decide; YouTube: "to
what length do you need to tell your story?"). About 5–7 beats, ~150–190 words/min, fast and enthusiastic (edX:
faster, enthusiastic speech held learners).

**Shape (Hormozi's Hook → Retain → Reward, mapped to the value equation):**

| Part | Beats | What it does | Rules |
|---|---|---|---|
| **Hook** | 1 | The question or surprising result, over the story's most striking concrete visual. The same line is on screen from frame 0 (`hook`), so it's also the cover in the Shorts grid | Use at least two of Hormozi's headline components: recency, relevancy, celebrity (a name viewers know), proximity, conflict, unusual, ongoing. Specific and answerable (Loewenstein: curiosity needs a gap that looks closable). Never the ask, never a logo or title card |
| **Retain** | 2–4 | Build toward the answer: who, what it does, how, in plain words with one analogy | One idea per beat; a new visual each beat; every term defined |
| **Reward** | 1–2 | Pay off the hook fully: the answer, the number, the demo. Then the reality check (pilot, preprint, self-reported) | "Give away the information, sell the implementation": the Short gives the *what* completely. The epistemic status survives compression: if the hook's number came from a pilot, say "pilot" |
| **CTA** | 1 | One call to action, spoken and shown: what the episode adds (the *how*, *why it matters*, *what we still don't know*) and where (the link below) | "Clear, not clever"; only one CTA; after the payoff, never before (references: "WATCH THE FULL TALK" from frame one lost) |

**Sourcing (HARD):** every fact and number comes from the episode's `research.md` quotes or its captured pages,
nothing generated. Numbers spelled as words in narration, digits on screen. The CTA's promise must be true of the
episode ("what we still don't know" only if the episode covers it).

**Fields:** copy the episode's `design`, `voice`/`tts`, `date`. Add top-level `"hook"` (≤ ~12 words; wrap one key
word in `*asterisks*` for the accent colour) and `"cta"` (end-card line). One segment `{"id": "short", "kind":
"story", "label": "NEWS" | "PAPER", "headline", "beats"}`.

**Visual types** (renderer `video/src/shorts/Short.tsx`): `photo` (picture band with a slow push-in; `badge`,
`stamp`, `credit`), `shot` (zooms to the `--find` highlight of a captured page), `stat` (counting number,
`items[{to,suffix,label}]`), `split`/`panels` (two stacked images), `kinetic` (big words). Copy the assets you use
from the episode's `assets/` into the Short's `assets/`.

**Motion: official press footage first.** Real robot footage wins in this topic (the three biggest reference
multipliers). Use short clips of **official footage the company or lab itself published** with its announcement (its
newsroom, press kit, own page or channel), then other public footage (news, other creators) with proper citation
when official footage doesn't show it (checklist HARD 7.2): a few seconds each, credited, audio muted, and ideally
the robot doing the thing in frame 0. Wire-agency footage last (Content ID claims). When no official
footage exists, get motion from consecutive stills, the push-in, counting stats and a new visual every beat.
Generated video (Higgsfield, Blender) only rarely, to explain an idea, labelled "Illustration" / "AI-generated" and
never presented as real (HARD 4.5).

## 3. Layout (in the renderer, don't fight it)

1080×1920, 30 fps. Text stays clear of the Shorts UI: right ~150 px (buttons) and bottom ~430 px (title, channel,
Related-video link). Hook at the top, picture band in the middle, captions (2–4 words, burned in: most mobile
viewing is muted) under the picture, end card for 2.2 s after the last word. Light design is the default; render both.

## 4. Render and QA

```
tools/render_short.sh <ep>/shorts/<slug> --stills-only            # voice + frame 0 + contact sheet
tools/render_short.sh <ep>/shorts/<slug> --skip-voice              # light final
tools/render_short.sh <ep>/shorts/<slug> --skip-voice --design blueprint   # dark final
```

Check, every Short:
- [ ] **Frame 0** (`qa/f00000.png`): the concrete subject plus the hook, readable at phone size; it is the cover
- [ ] The hook's question is answered before the CTA
- [ ] Nothing important in the bottom ~430 px or right ~150 px
- [ ] Captions on every spoken word; no beat without a visual
- [ ] Every number on screen matches its source; epistemic status said
- [ ] 20–45 s (or a written reason); one CTA, at the end

## 5. Hand off (never upload)

Per Short: the MP4 paths (light, dark), duration, and:
- **Title** (≤ 40 chars, the hook's question or result, no hype words) and a 1–2 line **description** with the sources.
- **Related video**: "set this Short's Related video to the episode in Studio" (the only clickable path).
- **When**: publish after the episode is live, so the link works.
- **What to watch** after a week (`docs/youtube-feedback.md`): Stayed-to-watch %, average view duration and %,
  Related-video clicks. Early-swipe patterns are specific to each video (Dashlet), so our own numbers decide what
  changes next.
