---
name: promo-shorts
description: Make 9:16 promotional YouTube Shorts for a finished Pixels2Actions episode (project physical-ai-daily) — make 3–5 Shorts from the strongest stories (a strong story may get two), write a fresh 30–45 s script per Short that explains one idea to a complete newcomer (two-sentence plain opening: background + question; what they did; the result; why it matters as the source says; one caveat; one call to action to the full episode), reuse the episode's verified facts and captured visuals, voice and render with tools/render_short.sh in light and dark, and hand off titles, descriptions and the Related-video setting. Use when the user asks for Shorts, promos or vertical cuts of an episode.
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

**Other tools in the harness (optional, rarely needed):** `yt-dlp`/`ffmpeg` to download and analyse videos (reference
Shorts for private study, deleted after; official footage for credited clips), the Blender and Higgsfield MCPs for an
explanatory illustration. Same rules and limits as episodes (physical-ai-daily skill, "Other tools in the harness").

## What YouTube measures (official)

Shorts are ranked on **% of viewers who chose to view** ("Stayed to watch": stayed past the initial seconds),
**average view duration** and **average % viewed**, among viewers who stayed. Shorts can run up to 3 min. The only
clickable path to the episode is the **Related video** link under the channel handle (set in Studio; links in
descriptions and comments aren't clickable). So: hook for Stayed-to-watch, retain for view duration, reward and one
CTA for Related-video clicks.

## 1. Pick 3–5 Shorts (user, 2026-11-15: more posting helps)

Make 3–5 Shorts per episode. A story worth it may get **two** Shorts, each with its own hook and angle (e.g. the
result, and how it works), never the same script twice. From the episode's stories, favour those with:
1. a **concrete first frame**: a robot doing the thing, a striking image, a number. Real subjects beat screenshots
   and title cards (references: every winner opened on the payoff; screenshots and title slides were losers);
2. a **visible puzzle**: a question the viewer can see and that the story answers;
3. at least one **learning** story (the channel's rule: learning over knowing).

One story per Short (two Shorts may share a story; one Short never covers two). Never a roundup ("so many updates this week…" was among the worst references).

## 2. Write the script (`shorts/<slug>/episode.json`)

**Who it's for (user, 2026-11-16):** someone who has never heard of physical AI. A Short explains **one idea** so
that person understands it, then points to the full episode for everything else. Spend the words on explaining, not
on detail: the result and why it matters, nothing more from the paper.

**Length:** 30–45 s at normal pace, about 6–8 beats, plain spoken English. **Every Short is then played at 1.175×**
(user, 2026-11-16): set top-level `"speed": 1.175` in its episode.json (voice.py time-stretches after synthesis,
pitch kept, no TTS credits), so finals land around 32–40 s.

**Shape (approved by the user, 2026-11-16):**

| # | Part | What it says | Rules |
|---|---|---|---|
| 1 | **Opening** (1 beat, two simple sentences) | Sentence 1: the background, from something the viewer already knows ("Plugging in a cable is easy for us, but it takes a precise touch, and that is hard for robots."). Sentence 2: the question the Short answers ("So what happened when one of today's most advanced AI models took control of a real robot and tried it?") | Must make sense with zero background. No jargon (simulation, GPU, policy, controller, benchmark), no names viewers don't know (model names, startups, papers); Apple, Google, Tesla, NVIDIA are fine. Over the most concrete footage. The on-screen `hook` is a short version of the question (≤ ~12 words) |
| 2 | **What they did** (1 beat) | One plain sentence: who tried what | Say "researchers" or "the company" if the name adds nothing |
| 3 | **The result** (1–2 beats) | The answer, in words a newcomer gets ("mostly worked", "failed every single time") | **At most one number** in the whole Short |
| 4 | **Why it matters** (1–2 beats) | The impact, **as the source or episode states it** ("The researchers' verdict: competent, but not yet practical") | Never our own opinion (sourcing rule). Light rewording is fine if the essence matches the source or the episode's verified narration (user, 2026-11-16). If no source states the impact, say what changed in plain words instead |
| 5 | **Honest caveat** (≤1 beat) | One short line: company's own numbers / early test / preprint | Keep the epistemic status; one line is enough |
| 6 | **CTA** (1 beat) | What the episode adds and where: "How researchers are trying to close that gap is in this week's full episode. Link below." | One CTA, at the end, true of the episode |

**Leave out** (they belong in the episode): method details, extra numbers, benchmark or model names, trial counts,
author lists, secondary results.

**Sample (approved):** astra-plug-insertion: opening as above over the robot at a power strip → "Researchers gave
the AI direct control of real robot arms and asked it to do everyday tasks." → "Picking things up and putting them
somewhere else mostly worked." → "But plugging in the cable failed every single time." → "The researchers' verdict:
these AI models are competent, but not yet practical for real robots." → "Their report says precise, hands-on tasks
like this are still much harder." → CTA.

**Sourcing (HARD):** every fact and number comes from the episode's `research.md` quotes or its captured pages,
nothing generated. Numbers spelled as words in narration, digits on screen. The CTA's promise must be true of the
episode ("what we still don't know" only if the episode covers it).

**Voice:** edge-tts `en-US-AndrewNeural` (user, 2026-11-15: "Andrew's voice is good for this"). If a script runs long,
cut detail first (see "Leave out"); only then voice it faster with `"direct": [["hook", narration, 0]]` and
`VOICE_STRENGTH=2`.

**Point to OUR episode (user, 2026-11-17):** the Short sends viewers to our full episode, not to the original source.
Copy the episode thumbnail candidate **about this Short's own story** (`<ep>/out/thumb_<n>_*.png`, matched by its
receipt/hook; user, 2026-11-17: the astra Short must not end on Machenta's thumbnail) to the Short's
`assets/episode_thumb.png` and set
top-level `"episodeThumb": "assets/episode_thumb.png"`: the renderer then shows that thumbnail with "FULL EPISODE ↓"
during the spoken CTA (the last beat) and on the end card, so viewers recognise what the Related-video link opens.
In the description, the episode link comes first ("Full episode: <url>"), sources after it.
Known trade-off (accepted by the user): the end card shows the Short's own story's thumbnail, while the Related-video
link opens the episode, whose YouTube thumbnail may be a different story; the episode title in the link carries it.
Before rendering, look at the chosen end-card thumbnail: it must follow the current tag rules (small labels; no
"AI-generated" stamp on another lab's footage). Remake the candidate if it doesn't.
**Mapping used for the first three weeks** (thumbnail candidate number from `<ep>/episode.json` "thumbnails"):
Oct 25–Nov 1: astra, zap → 2 · genie4 → 1 · machenta → 3. Nov 1–8: halo-sim-homes → 1 · halo-how-it-trains → 2 ·
robocolosseum → 3 · agility → 6. Nov 8–15: groot2 → 1 · gemini → 2 · cybercab → 3 · both supermimic → 5.

**Fields:** copy the episode's `design`, `voice`/`tts`, `date`. Add top-level `"hook"` (≤ ~12 words; wrap one key
word in `*asterisks*` for the accent colour) and `"cta"` (end-card line). One segment `{"id": "short", "kind":
"story", "label": "NEWS" | "PAPER", "headline", "beats"}`.

**Visual types** (renderer `video/src/shorts/Short.tsx`): `photo` (picture band with a slow push-in; `badge`,
`stamp`, `credit`), `shot` (zooms to the `--find` highlight of a captured page), `stat` (counting number,
`items[{to,suffix,label}]`), `split`/`panels` (two stacked images), `kinetic` (big words), `clip` (a few seconds of
muted public footage filling the band, cut with `tools/clip.py`, credited from its metadata; see Motion below). Copy the assets you use
from the episode's `assets/` into the Short's `assets/`.

**Motion: official press footage first.** Real robot footage wins in this topic (the three biggest reference
multipliers). Use short clips of **official footage the company or lab itself published** with its announcement (its
newsroom, press kit, own page or channel), then other public footage (news, other creators) with proper citation
when official footage doesn't show it (checklist HARD 7.2): a few seconds each, credited, audio muted, and ideally
the robot doing the thing in frame 0. Wire-agency footage last (Content ID claims). Downloading the source needs the
user's OK; cut it with `tools/clip.py` (same as episodes, see the physical-ai-daily skill). When no official
footage exists, get motion from consecutive stills, the push-in, counting stats and a new visual every beat.
Tags on footage stay small and subtle (user, 2026-11-16). "AI-generated" is only for media we generate ourselves:
another lab's generated footage (e.g. Genie 4) is identified by its credit line and the narration, never by a stamp.
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
- [ ] **Newcomer test:** read the script as someone who has never heard of physical AI. Every sentence makes sense
      without background; no jargon, no unknown names; at most one number
- [ ] "Why it matters" is the source's or episode's own statement, not ours
- [ ] 30–45 s (or a written reason); one CTA, at the end

**Cover (thumbnail), every Short (user, 2026-11-16):** not the first frame. Make a cover that provokes curiosity or
raises an eyebrow: a big two- or three-line hook (the company's name when it's newsworthy) over one or two frames that
tell the story at a glance, e.g. "SHOWN ONCE" → "ON ITS OWN". Frames and their labels must be true to the source (use
the source's own on-screen labels where possible). Render in the design being uploaded (the user uploads **dark**):
`shorts/<slug>/cover/props.json` = {"design", "lines", "panels": [{"src", "label"}], "credit"}, then
`npx remotion still src/shorts/index.ts ShortCover <slug>/out/cover-<design>.png --props=<slug>/cover/props.json --public-dir=<slug>`
(renderer `video/src/shorts/Cover.tsx`).

**Title and cover text work together, never repeat each other** (user, 2026-11-17). The title carries one part
(e.g. the fact or score: "GPT-6 Astra vs. a power strip: 0 for 11"), the cover carries another (e.g. the contrast
and the open question: "GPT-6 Astra moves blocks just fine. Why can't it plug in?"). Viewers see both at once, so
repeated words are wasted space. Write them as a pair and check them side by side.

## 5. Hand off (never upload)

Per Short: the MP4 paths (light, dark), duration, and:
- **Title** (≤ 40 chars, the hook's question or result, no hype words) and a 1–2 line **description** with the sources.
- **Related video**: "set this Short's Related video to the episode in Studio" (the only clickable path).
- **When**: publish after the episode is live, so the link works.
- **What to watch** after a week (`docs/youtube-feedback.md`): Stayed-to-watch %, average view duration and %,
  Related-video clicks. Early-swipe patterns are specific to each video (Dashlet), so our own numbers decide what
  changes next.
