---
name: weekly-package
description: One command from a week to a complete Pixels2Actions posting package (project physical-ai-daily) — research the week, make the long-form episode (4K, light and dark), thumbnails, 3–4 promo Shorts with covers, and publish.md with titles, descriptions, chapters, sources and tags. Takes the week (its end date, e.g. "2026-10-18" or "Oct 11–18"). Use when the user says "/weekly-package <week>", "make the package for <week>", or asks for the complete posting package of a week.
---

# Weekly posting package (kickoff)

`/weekly-package <week>`. One argument: the week. Accept `2026-10-18` (the week's **end** date, the folder
convention), `Oct 11–18` or `week of Oct 11`. The window is 8 days inclusive, end minus 7 through end, and the
folder is `physical-ai-daily/episodes/<end>-weekly/`. Example: `episodes/2026-10-25-weekly` covers Oct 18–25. If the
week ended more than ~10 days ago it is a **backlog week**, so use dated framing throughout.

Typing the command means **run unattended**: don't stop for the pick review. Print the picks when they're chosen and
go on. Stop and ask only for a HARD item that can't be met (checklist `docs/video-requirements.md`), or for a
decision that is genuinely the user's.

This skill only orders the work. The rules live in the phase skills; read each one when its phase starts, and
`physical-ai-daily/docs/feedback.md` before scripting:

| Phase | Skill | Output |
|---|---|---|
| 1 Research | `physical-ai-research` | `<ep>/research.md` (5 picks, ≥3 learning on backlog weeks) |
| 2 Episode | `physical-ai-daily` | `<ep>/episode.json`, `<ep>/out/*.mp4` (4K light + dark) |
| 3 Thumbnails | `youtube-thumbnails` | 8–10 candidates on a lettered contact sheet, pick + 2 runners-up, one per story |
| 4 Shorts | `promo-shorts` | 3–4 Shorts in `<ep>/shorts/<slug>/out/` (dark + light), each with a cover |
| 5 Package | (this skill) | `<ep>/publish.md` complete, the hand-off message |

## 0. Preflight (2 minutes, before anything else)

- If `research.md` already exists for the week, skip phase 1 and show the picks. If episode outputs exist, say what's
  there and build only what's missing.
- `python3 tools/credits.py`: twitterapi.io, NewsData, ElevenLabs.
  - Voice: ElevenLabs Eric (`cjVigY5qzO86Huf0OWal`, `eleven_v3`) when there is enough credit. An episode is about
    3,600 characters and a Short about 600. Otherwise use edge-tts `en-US-AndrewNeural` for all of it.
  - Say which voice and why in the hand-off. Never print the key.
- Check every judge model answers (Fable quota runs out; fall back to Haiku, see the research skill).
- `df -h` (4K renders are large), and check `.render.lock` isn't held by a dead process.

## 1–4. Run the phases in order, each in its own subagent

The phases are long, and their tool output would flood one context. Run each as a subagent:
- Brief it with the week, the folder, this skill's rules section and the phase skill to load.
- Wait for it to finish, check its output files yourself, then start the next phase.

Phases 3 and 4 may overlap phase 2's final 4K render. They need `episode.json`, the assets and the per-story
thumbnails, not the final MP4s. `tools/render_both.sh` takes the shared lock, so renders queue.

## Rules that cut across phases (user decisions; keep them)

- **Sourcing:** never generate facts or opinions. Every number in narration, covers and descriptions is on a
  captured page.
- **Backlog framing:**
  - Intro "In the week of October 11th…". On-screen date "Oct 11–18, 2026".
  - Never "this week", "today" or "just announced".
  - Say what changed since, when a source says so.
- **Intro** ends, word for word: "This is Pixels to Actions. Every week, we go through what's new in robotics and
  physical AI." Inform, don't sell.
- **Episode:**
  - 4K (render.sh defaults).
  - No per-story time target; hard cap 20 min.
  - Screenshots hold still.
  - Tags small and subtle.
  - "AI-generated" only on media we generate.
- **Audio:** one continuous take per segment, cut at natural pauses (`voice.py`), never per-beat pieces.
- **Clips:** official footage first, a few seconds each, muted, credited, URL in the description.
  - Raw downloads stay git-ignored and are deleted after cutting.
  - The user OK'd official clips for episodes; ask before any non-official footage.
- **Titles and thumbnails, wide options (the user iterates a lot here):**
  - At least 8 titles per video, each tagged by angle, always including *current* (names companies or models) and
    *plainer* (a newcomer understands); one marked ★; each names the thumbnail/cover it pairs with.
  - Episode: 8–10 thumbnails; each Short: 3–4 covers. Vary the hook text, not only the image. One lettered
    contact sheet per video, in the hand-off.
  - No "Robotics this week:" prefix.
  - Title and cover/thumbnail text complement each other and never repeat.
- **Shorts:**
  - Newcomer format: two-sentence plain opening (background, then question), what they did, result (≤1 number),
    why it matters as the source says it, one caveat, and a CTA.
  - Speed 1.175×.
  - Curiosity cover (ShortCover) naming the company or model, not frame 0.
  - The end card shows **that Short's own story** thumbnail with "FULL EPISODE ↓".
  - The user uploads the **dark** versions.
- **Never upload, never post.** Commit only code, skill and doc changes. Never commit `.env`, `p2a_topics.json`,
  media, or episode working files. The repo is public.

## 5. Package and hand off

Finish `<ep>/publish.md` so the user can post from it alone:
- **Episode:**
  - The "Options" section: all titles with angle and paired thumbnail, the contact sheet, rubric scores.
  - The description: names the week, then chapters with timestamps from the rendered `props.json`, every source
    URL, clip credits, and tags.
  - Thumbnail pick and the 2 runners-up for Test & Compare.
  - MP4 paths (light and dark).
- **Each Short:**
  - 8+ title options and 3–4 covers (contact sheet), paired.
  - Description starting `Full episode: <EPISODE_URL>` (a placeholder until the user posts and sends the link; then
    fill it in).
  - The Related video setting.
  - Cover path and dark MP4 path.
  - A suggested posting order (strongest hook first).

Before handing off, look at the episode's beat sheet, every thumbnail and every Short cover yourself.

The hand-off message lists:
- Every path the user uploads.
- The contact sheets and the ★ title per video (the full lists are in publish.md).
- The voice used.
- What was dropped or unverified, and why.
- Any credits spent (Higgsfield, ElevenLabs).

Then append to `docs/feedback.md` or the right skill anything this run taught (tool failures, workarounds),
so the next clean session has it. Commit.
